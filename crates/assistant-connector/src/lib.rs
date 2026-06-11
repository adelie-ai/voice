//! Assistant gateway over desktop-assistant's transport-agnostic `Connector`.
//!
//! Implements the voice [`AssistantGateway`] port by delegating to a
//! [`Connector`], so the voice service reaches the orchestrator over whichever
//! transport its config selects — local UDS (the default), remote WebSocket, or
//! D-Bus — instead of the old bespoke zbus proxy. The single shared transport
//! abstraction also means a new orchestrator capability no longer has to be
//! hand-mirrored onto a per-capability D-Bus surface (voice#31).
//!
//! The orchestrator's streaming `SignalEvent`s are mapped to the voice domain's
//! [`AssistantEvent`]s; only the response-turn signals (chunk / complete /
//! error) are relevant to the voice pipeline.

use std::sync::{Arc, Mutex};

use adele_voice_core::VoiceError;
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway, ClientToolSpec};
use desktop_assistant_api_model::ClientToolRegistration;
use desktop_assistant_client_common::{Connector, SignalEvent};
use tokio::sync::mpsc;

// Re-export so the daemon can build a connection config without depending on
// client-common directly.
pub use desktop_assistant_client_common::{ConnectionConfig, TransportMode};

/// Voice's assistant gateway, backed by a transport-agnostic [`Connector`].
///
/// The connection lives behind a `Mutex<Option<Arc<Connector>>>` so it can be
/// (re)built lazily: a raw UDS/WS socket dies with the peer, so the gateway
/// reconnects on a failed call and the next turn uses a fresh link. (D-Bus
/// re-resolves its well-known name on its own, but reconnecting is harmless
/// there too.)
///
/// The `Option` lets the gateway start in a DISCONNECTED state when the
/// orchestrator isn't up yet (voice#86): the daemon must keep serving — wake
/// word, D-Bus, TTS — and connect lazily once the orchestrator appears, rather
/// than dying at startup and crash-looping under systemd during a session-start
/// race. A call made while disconnected dials on demand; if that fails it
/// returns a recoverable error (the pipeline apologizes and the next turn
/// retries) — never a crash.
pub struct ConnectorAssistantGateway {
    config: ConnectionConfig,
    connector: Mutex<Option<Arc<Connector>>>,
}

impl ConnectorAssistantGateway {
    /// Connect to the orchestrator over the transport named by `config`,
    /// failing if it isn't reachable. Prefer [`connect_or_degrade`] in the
    /// daemon so a missing orchestrator doesn't kill startup (voice#86).
    pub async fn connect(config: &ConnectionConfig) -> Result<Self, VoiceError> {
        let connector = Self::dial(config).await?;
        Ok(Self {
            config: config.clone(),
            connector: Mutex::new(Some(Arc::new(connector))),
        })
    }

    /// Build the gateway WITHOUT requiring the orchestrator to be up (voice#86).
    /// Tries to connect; on failure logs and returns a gateway in the
    /// disconnected state that connects lazily on the first call. The daemon
    /// uses this so an orchestrator that isn't ready at startup degrades the
    /// voice turn (apology, retried next turn) instead of crash-looping the
    /// whole service.
    pub async fn connect_or_degrade(config: &ConnectionConfig) -> Self {
        let connector = match Self::dial(config).await {
            Ok(connector) => Some(Arc::new(connector)),
            Err(e) => {
                tracing::warn!(
                    "orchestrator not reachable at startup ({e}); voice will keep running and \
                     connect when it appears"
                );
                None
            }
        };
        Self {
            config: config.clone(),
            connector: Mutex::new(connector),
        }
    }

    async fn dial(config: &ConnectionConfig) -> Result<Connector, VoiceError> {
        let connector = Connector::connect(config).await.map_err(|e| {
            VoiceError::Assistant(format!("failed to connect to orchestrator: {e}"))
        })?;
        tracing::info!(
            transport = %connector.label(),
            "connected to desktop-assistant orchestrator"
        );
        Ok(connector)
    }

    /// The live connector, dialing lazily if the gateway is still disconnected
    /// (voice#86). On a successful lazy dial the connection is cached for
    /// subsequent calls; if the orchestrator is still down this returns a
    /// recoverable error rather than panicking.
    async fn current(&self) -> Result<Arc<Connector>, VoiceError> {
        if let Some(connector) = self.connector.lock().unwrap().as_ref() {
            return Ok(Arc::clone(connector));
        }
        // Disconnected: dial now (outside the lock — Connector::connect awaits).
        let connector = Arc::new(Self::dial(&self.config).await?);
        let mut guard = self.connector.lock().unwrap();
        // Another task may have connected while we were dialing; keep the first.
        Ok(Arc::clone(guard.get_or_insert(connector)))
    }

    /// Best-effort reconnect after a failed call, so the next turn talks to a
    /// live orchestrator (e.g. one that just restarted). If it's still down the
    /// next call simply fails and retries — the daemon never crashes over it.
    async fn reconnect(&self) {
        match Self::dial(&self.config).await {
            Ok(connector) => *self.connector.lock().unwrap() = Some(Arc::new(connector)),
            Err(e) => tracing::warn!("orchestrator reconnect failed: {e}"),
        }
    }
}

/// Map an orchestrator signal to a voice turn event. The response-turn signals
/// (chunk / complete / error) and the per-turn progress `Status` matter to the
/// voice pipeline; everything else (title, task, scratchpad, disconnect) is
/// ignored.
fn map_signal(event: SignalEvent) -> Option<AssistantEvent> {
    match event {
        SignalEvent::Chunk { request_id, chunk } => Some(AssistantEvent::Chunk {
            request_id,
            text: chunk,
        }),
        // Progress status (turn-start + per tool call). The pipeline uses it as
        // a progress heartbeat and narrates it sparingly (#58).
        SignalEvent::Status {
            request_id,
            message,
        } => Some(AssistantEvent::Status {
            request_id,
            message,
        }),
        SignalEvent::Complete {
            request_id,
            full_response,
        } => Some(AssistantEvent::Complete {
            request_id,
            full_response,
        }),
        SignalEvent::Error { request_id, error } => {
            Some(AssistantEvent::Error { request_id, error })
        }
        // The turn suspended on a client-local tool call (voice#61): the LLM is
        // driving the voice session. Map it through so the pipeline can run the
        // tool and post the result back, resuming the parked turn. The
        // `conversation_id` the signal carries isn't needed here — the pipeline
        // acts on the tool name and replies via the task/tool-call ids.
        SignalEvent::ClientToolCall {
            task_id,
            tool_call_id,
            tool_name,
            arguments,
            ..
        } => Some(AssistantEvent::ClientToolCall {
            task_id,
            tool_call_id,
            tool_name,
            arguments,
        }),
        _ => None,
    }
}

impl AssistantGateway for ConnectorAssistantGateway {
    async fn create_conversation(&self, title: &str) -> Result<String, VoiceError> {
        let connector = self.current().await?;
        match connector.create_conversation(title).await {
            Ok(id) => Ok(id),
            Err(e) => {
                self.reconnect().await;
                Err(VoiceError::Assistant(format!(
                    "create_conversation failed: {e}"
                )))
            }
        }
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String, VoiceError> {
        let connector = self.current().await?;
        match connector.send_prompt(conversation_id, prompt).await {
            Ok(id) => Ok(id),
            Err(e) => {
                self.reconnect().await;
                Err(VoiceError::Assistant(format!("send_prompt failed: {e}")))
            }
        }
    }

    async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> Result<String, VoiceError> {
        // Per-turn idempotency key (#39). If the first send fails — the
        // orchestrator restarted, or the socket dropped mid-turn — we reconnect
        // and retry ONCE with the SAME key, so the orchestrator re-attaches to
        // the still-running turn (or replays a completed reply) instead of
        // re-running the turn and re-processing its tool actions. Recovering the
        // answer beats the old apologize-and-give-up, and the same key makes the
        // retry safe. (The D-Bus transport drops the key — see the Connector
        // method — so this is effective on the default UDS and on WS.)
        let idempotency_key = uuid::Uuid::new_v4().to_string();
        match self
            .current()
            .await?
            .send_prompt_with_system_refinement_idempotent(
                conversation_id,
                prompt,
                system_refinement,
                Some(idempotency_key.clone()),
            )
            .await
        {
            Ok(id) => Ok(id),
            Err(first_err) => {
                tracing::warn!(
                    "send failed ({first_err}); reconnecting and retrying with the same \
                     idempotency key"
                );
                self.reconnect().await;
                let connector = self.current().await.map_err(|retry_err| {
                    VoiceError::Assistant(format!(
                        "send_prompt_with_system_refinement failed; reconnect before retry \
                         also failed: {first_err}; reconnect error: {retry_err}"
                    ))
                })?;
                connector
                    .send_prompt_with_system_refinement_idempotent(
                        conversation_id,
                        prompt,
                        system_refinement,
                        Some(idempotency_key),
                    )
                    .await
                    .map_err(|retry_err| {
                        VoiceError::Assistant(format!(
                            "send_prompt_with_system_refinement failed; retry after reconnect \
                             also failed: {first_err}; retry error: {retry_err}"
                        ))
                    })
            }
        }
    }

    async fn register_client_tools(&self, tools: Vec<ClientToolSpec>) -> Result<usize, VoiceError> {
        let registrations = tools
            .into_iter()
            .map(|t| ClientToolRegistration {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect();
        self.current()
            .await?
            .register_client_tools(registrations)
            .await
            .map_err(|e| VoiceError::Assistant(format!("register_client_tools failed: {e}")))
    }

    async fn submit_client_tool_result(
        &self,
        task_id: &str,
        tool_call_id: &str,
        result: Result<String, String>,
    ) -> Result<(), VoiceError> {
        self.current()
            .await?
            .submit_client_tool_result(task_id, tool_call_id, result)
            .await
            .map_err(|e| VoiceError::Assistant(format!("submit_client_tool_result failed: {e}")))
    }

    async fn subscribe(&self) -> Result<mpsc::UnboundedReceiver<AssistantEvent>, VoiceError> {
        // Take a fresh slice of the current connector's fanned-out signal stream
        // and forward the response-turn events, mapped into the voice domain.
        let mut signals = self.current().await?.subscribe();
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(event) = signals.recv().await {
                if let Some(mapped) = map_signal(event)
                    && tx.send(mapped).is_err()
                {
                    break; // the pipeline dropped this subscription
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
    fn maps_response_turn_signals_into_voice_events() {
        assert!(matches!(
            map_signal(SignalEvent::Chunk { request_id: "r".into(), chunk: "hi".into() }),
            Some(AssistantEvent::Chunk { text, .. }) if text == "hi"
        ));
        assert!(matches!(
            map_signal(SignalEvent::Complete { request_id: "r".into(), full_response: "done".into() }),
            Some(AssistantEvent::Complete { full_response, .. }) if full_response == "done"
        ));
        assert!(matches!(
            map_signal(SignalEvent::Error { request_id: "r".into(), error: "boom".into() }),
            Some(AssistantEvent::Error { error, .. }) if error == "boom"
        ));
    }

    #[test]
    fn maps_status_into_a_voice_status_event() {
        // #58: progress status must reach the pipeline (for narration + as a
        // stall heartbeat), not be dropped.
        assert!(matches!(
            map_signal(SignalEvent::Status {
                request_id: "r".into(),
                message: "checking your calendar".into()
            }),
            Some(AssistantEvent::Status { message, .. }) if message == "checking your calendar"
        ));
    }

    #[test]
    fn maps_client_tool_call_into_a_voice_event() {
        // voice#61: a suspended client-tool call must reach the pipeline so it
        // can run the tool (stop/listen/say) and post the result back.
        assert!(matches!(
            map_signal(SignalEvent::ClientToolCall {
                task_id: "task-1".into(),
                conversation_id: "conv-1".into(),
                tool_call_id: "call-1".into(),
                tool_name: "stop_listening".into(),
                arguments: serde_json::json!({}),
            }),
            Some(AssistantEvent::ClientToolCall { task_id, tool_call_id, tool_name, .. })
                if task_id == "task-1" && tool_call_id == "call-1" && tool_name == "stop_listening"
        ));
    }

    #[test]
    fn ignores_signals_outside_the_voice_turn() {
        assert!(map_signal(SignalEvent::Disconnected { reason: "x".into() }).is_none());
        assert!(
            map_signal(SignalEvent::TitleChanged {
                conversation_id: "c".into(),
                title: "t".into()
            })
            .is_none()
        );
    }
}
