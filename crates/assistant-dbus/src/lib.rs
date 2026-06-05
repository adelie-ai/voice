use adele_voice_core::VoiceError;
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway};
use futures::StreamExt;
use tokio::sync::mpsc;
use zbus::Connection;

const DEFAULT_DBUS_SERVICE: &str = "org.desktopAssistant";
const DBUS_CONVERSATIONS_PATH: &str = "/org/desktopAssistant/Conversations";

#[zbus::proxy(interface = "org.desktopAssistant.Conversations")]
trait Conversations {
    async fn create_conversation(&self, title: &str) -> zbus::fdo::Result<String>;

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> zbus::fdo::Result<String>;

    /// Additive method (desktop-assistant #200 follow-up). Absent on an
    /// orchestrator that predates it — the gateway detects that via the
    /// `UnknownMethod` / `UnknownInterface` / `UnknownObject` D-Bus errors
    /// and falls back to `send_prompt`.
    async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> zbus::fdo::Result<String>;

    #[zbus(signal)]
    fn response_chunk(
        &self,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_complete(
        &self,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_error(
        &self,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::fdo::Result<()>;
}

fn resolve_dbus_service_name() -> String {
    std::env::var("DESKTOP_ASSISTANT_DBUS_SERVICE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_DBUS_SERVICE.to_string())
}

/// True when a D-Bus call failed because the method/interface/object
/// isn't exported by the peer — i.e. the orchestrator predates the
/// additive `SendPromptWithSystemRefinement` method and the caller should
/// fall back to `SendPrompt`. Any other error (bus failure, daemon-side
/// `Failed`, etc.) is a genuine error and must propagate.
fn is_unknown_method(err: &zbus::fdo::Error) -> bool {
    matches!(
        err,
        zbus::fdo::Error::UnknownMethod(_)
            | zbus::fdo::Error::UnknownInterface(_)
            | zbus::fdo::Error::UnknownObject(_)
    )
}

/// Fallback prompt composition for an orchestrator without the
/// refinement-aware method: fold the refinement into the prompt the way
/// the voice daemon did before #200 (blank refinement = bare prompt).
/// Mirrors the daemon pipeline's `compose_prompt` so the fallback wire
/// content is byte-identical to the historical behaviour.
fn prepend_refinement(refinement: &str, prompt: &str) -> String {
    if refinement.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{refinement}\n\n{prompt}")
    }
}

pub struct DbusAssistantGateway {
    proxy: ConversationsProxy<'static>,
}

impl DbusAssistantGateway {
    pub async fn connect() -> Result<Self, VoiceError> {
        let connection = Connection::session()
            .await
            .map_err(|e| VoiceError::Assistant(format!("failed to connect to session bus: {e}")))?;

        let service_name = resolve_dbus_service_name();
        let proxy = ConversationsProxy::builder(&connection)
            .destination(service_name)
            .map_err(|e| VoiceError::Assistant(format!("invalid service name: {e}")))?
            .path(DBUS_CONVERSATIONS_PATH)
            .map_err(|e| VoiceError::Assistant(format!("invalid path: {e}")))?
            .build()
            .await
            .map_err(|e| VoiceError::Assistant(format!("failed to build proxy: {e}")))?;

        tracing::info!("connected to desktop-assistant daemon via D-Bus");

        Ok(Self { proxy })
    }
}

impl AssistantGateway for DbusAssistantGateway {
    async fn create_conversation(&self, title: &str) -> Result<String, VoiceError> {
        self.proxy
            .create_conversation(title)
            .await
            .map_err(|e| VoiceError::Assistant(format!("create_conversation failed: {e}")))
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String, VoiceError> {
        self.proxy
            .send_prompt(conversation_id, prompt)
            .await
            .map_err(|e| VoiceError::Assistant(format!("send_prompt failed: {e}")))
    }

    async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> Result<String, VoiceError> {
        match self
            .proxy
            .send_prompt_with_system_refinement(conversation_id, prompt, system_refinement)
            .await
        {
            Ok(request_id) => Ok(request_id),
            // Graceful fallback for an orchestrator that predates the
            // refinement-aware method: the method/interface/object isn't
            // exported, so fold the refinement into the prompt and use the
            // original `send_prompt`. This reproduces the pre-#200 voice
            // behaviour exactly, so an un-upgraded daemon still answers.
            Err(e) if is_unknown_method(&e) => {
                tracing::warn!(
                    "orchestrator lacks SendPromptWithSystemRefinement ({e}); \
                     falling back to prepending the hint and calling send_prompt"
                );
                let composed = prepend_refinement(system_refinement, prompt);
                self.send_prompt(conversation_id, &composed).await
            }
            Err(e) => Err(VoiceError::Assistant(format!(
                "send_prompt_with_system_refinement failed: {e}"
            ))),
        }
    }

    async fn subscribe(&self) -> Result<mpsc::UnboundedReceiver<AssistantEvent>, VoiceError> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut chunk_stream =
            self.proxy.receive_response_chunk().await.map_err(|e| {
                VoiceError::Assistant(format!("failed to subscribe to chunks: {e}"))
            })?;

        let tx_chunk = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = chunk_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_chunk.send(AssistantEvent::Chunk {
                        request_id: args.request_id.to_string(),
                        text: args.chunk.to_string(),
                    });
                }
            }
        });

        let mut complete_stream = self.proxy.receive_response_complete().await.map_err(|e| {
            VoiceError::Assistant(format!("failed to subscribe to completions: {e}"))
        })?;

        let tx_complete = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = complete_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_complete.send(AssistantEvent::Complete {
                        request_id: args.request_id.to_string(),
                        full_response: args.full_response.to_string(),
                    });
                }
            }
        });

        let mut error_stream =
            self.proxy.receive_response_error().await.map_err(|e| {
                VoiceError::Assistant(format!("failed to subscribe to errors: {e}"))
            })?;

        tokio::spawn(async move {
            while let Some(signal) = error_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx.send(AssistantEvent::Error {
                        request_id: args.request_id.to_string(),
                        error: args.error.to_string(),
                    });
                }
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepend_refinement_folds_hint_when_present() {
        assert_eq!(
            prepend_refinement("Respond briefly, by voice.", "what's the weather?"),
            "Respond briefly, by voice.\n\nwhat's the weather?"
        );
    }

    #[test]
    fn prepend_refinement_is_bare_prompt_when_blank() {
        assert_eq!(prepend_refinement("", "hello"), "hello");
        assert_eq!(prepend_refinement("   ", "hello"), "hello");
    }

    /// The three "peer doesn't export this" D-Bus errors must trigger the
    /// fallback; an unrelated daemon-side failure must not.
    #[test]
    fn is_unknown_method_matches_only_absent_method_errors() {
        assert!(is_unknown_method(&zbus::fdo::Error::UnknownMethod(
            "no such method".into()
        )));
        assert!(is_unknown_method(&zbus::fdo::Error::UnknownInterface(
            "no such interface".into()
        )));
        assert!(is_unknown_method(&zbus::fdo::Error::UnknownObject(
            "no such object".into()
        )));
        // A real daemon-side failure is NOT an "absent method" — it must
        // propagate, not silently fall back.
        assert!(!is_unknown_method(&zbus::fdo::Error::Failed("boom".into())));
    }
}
