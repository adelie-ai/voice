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

use adele_voice_core::VoiceError;
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway};
use desktop_assistant_client_common::{Connector, SignalEvent};
use tokio::sync::mpsc;

// Re-export so the daemon can build a connection config without depending on
// client-common directly.
pub use desktop_assistant_client_common::{ConnectionConfig, TransportMode};

/// Voice's assistant gateway, backed by a transport-agnostic [`Connector`].
pub struct ConnectorAssistantGateway {
    connector: Connector,
}

impl ConnectorAssistantGateway {
    /// Connect to the orchestrator over the transport named by `config`.
    pub async fn connect(config: &ConnectionConfig) -> Result<Self, VoiceError> {
        let connector = Connector::connect(config).await.map_err(|e| {
            VoiceError::Assistant(format!("failed to connect to orchestrator: {e}"))
        })?;
        tracing::info!(
            transport = %connector.label(),
            "connected to desktop-assistant orchestrator"
        );
        Ok(Self { connector })
    }
}

/// Map an orchestrator signal to a voice turn event. Only response-turn signals
/// matter to the voice pipeline; everything else (status, title, task,
/// scratchpad, disconnect) is ignored.
fn map_signal(event: SignalEvent) -> Option<AssistantEvent> {
    match event {
        SignalEvent::Chunk { request_id, chunk } => Some(AssistantEvent::Chunk {
            request_id,
            text: chunk,
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
        _ => None,
    }
}

impl AssistantGateway for ConnectorAssistantGateway {
    async fn create_conversation(&self, title: &str) -> Result<String, VoiceError> {
        self.connector
            .create_conversation(title)
            .await
            .map_err(|e| VoiceError::Assistant(format!("create_conversation failed: {e}")))
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String, VoiceError> {
        self.connector
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
        self.connector
            .send_prompt_with_system_refinement(conversation_id, prompt, system_refinement)
            .await
            .map_err(|e| {
                VoiceError::Assistant(format!("send_prompt_with_system_refinement failed: {e}"))
            })
    }

    async fn subscribe(&self) -> Result<mpsc::UnboundedReceiver<AssistantEvent>, VoiceError> {
        // Take a fresh slice of the connector's fanned-out signal stream and
        // forward the response-turn events, mapped into the voice domain.
        let mut signals = self.connector.subscribe();
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
    fn ignores_signals_outside_the_voice_turn() {
        assert!(map_signal(SignalEvent::Disconnected { reason: "x".into() }).is_none());
        assert!(
            map_signal(SignalEvent::Status {
                request_id: "r".into(),
                message: "thinking".into()
            })
            .is_none()
        );
    }
}
