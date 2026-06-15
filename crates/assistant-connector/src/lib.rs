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
/// We hold a SINGLE durable [`Connector`] for the gateway's whole life and never
/// swap it out once it exists (voice#83). The Connector already owns a
/// persistent, auto-reconnecting signal fan-out (#246) plus a supervisor that
/// reconnects its transport *in place* on a drop and keeps feeding the same
/// stream. Swapping in a fresh Connector on a failed call (the old behaviour)
/// built a NEW fan-out and orphaned the pipeline's existing event subscription —
/// so a recovered turn streamed invisibly and the user heard the stall apology
/// though it succeeded. Keeping one Connector, letting its supervisor heal the
/// link, and handing out a [`subscribe`](AssistantGateway::subscribe) receiver
/// that re-registers across a drop keeps the subscription live through a
/// reconnect.
///
/// The connector lives behind a `Mutex<Option<Arc<Connector>>>` only so it can
/// start ABSENT and be dialed LAZILY (voice#86): the gateway may begin in a
/// DISCONNECTED state when the orchestrator isn't up yet, so the daemon keeps
/// serving — wake word, D-Bus, TTS — and connects once the orchestrator appears,
/// rather than dying at startup and crash-looping under systemd during a
/// session-start race. The `Option` is filled AT MOST ONCE (the first
/// successful [`dial`](Self::dial)); after that the single durable Connector is
/// never replaced, so any subscription taken from it stays valid across the
/// supervisor's in-place reconnects. A call made while still disconnected dials
/// on demand; if that fails it returns a recoverable error (the pipeline
/// apologizes and the next turn retries) — never a crash.
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
    /// subsequent calls and never replaced thereafter (voice#83 — a stable Arc
    /// keeps every subscription valid across the supervisor's in-place
    /// reconnects); if the orchestrator is still down this returns a recoverable
    /// error rather than panicking.
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

    /// Test-only hook to drive the reconnect path directly (voice#83 regression
    /// test). Production code reaches `reconnect` via a failed send.
    #[doc(hidden)]
    pub async fn reconnect_for_test(&self) {
        self.reconnect().await;
    }

    /// Give the orchestrator a moment to come back after a failed call, so a
    /// retry (or the next turn) talks to a live connection — e.g. after the
    /// orchestrator restarted.
    ///
    /// We deliberately do NOT force a transport reconnect here (voice#83): the
    /// `Connector` already runs its own reconnect supervisor that reconnects the
    /// SAME transport in place on a drop and keeps the persistent event fan-out
    /// alive. Forcing a second, concurrent reconnect from here would race that
    /// supervisor (two sockets, swapped writers) and lose the streamed reply.
    /// Instead we back off briefly and let the supervisor heal the link; the
    /// retry then rides the reconnected transport, and the durable subscription
    /// handed out by [`subscribe`](AssistantGateway::subscribe) re-registers
    /// across the drop so the reply still reaches the pipeline. If the
    /// orchestrator is still down the retry simply fails and the next call tries
    /// again — the daemon never crashes over it.
    async fn reconnect(&self) {
        // Once we hold a live Connector we deliberately do NOT swap it out here
        // (voice#83): a fresh Connector would build a NEW fan-out and orphan the
        // pipeline's existing subscription. We just back off and let the
        // Connector's own supervisor reconnect the SAME transport in place. The
        // ONLY case we still dial from here is when the gateway started degraded
        // and never connected (voice#86) — there is no Connector yet, so there is
        // no subscription to orphan, and a lazy dial is exactly right.
        if self.connector.lock().unwrap().is_some() {
            tokio::time::sleep(RECONNECT_GRACE).await;
            return;
        }
        match Self::dial(&self.config).await {
            Ok(connector) => {
                let mut guard = self.connector.lock().unwrap();
                // Another task may have connected while we were dialing; keep
                // the first so the single durable Connector is never replaced.
                let _ = guard.get_or_insert_with(|| Arc::new(connector));
            }
            Err(e) => tracing::warn!("orchestrator reconnect failed: {e}"),
        }
    }
}

/// Brief pause after a failed call before retrying, giving the Connector's
/// reconnect supervisor time to heal the transport in place (voice#83).
const RECONNECT_GRACE: std::time::Duration = std::time::Duration::from_millis(250);

/// Map an orchestrator signal to a voice turn event. The response-turn signals
/// (chunk / complete / error) and the per-turn progress `Status` matter to the
/// voice pipeline; everything else (title, task, scratchpad, disconnect) is
/// ignored.
fn map_signal(event: SignalEvent) -> Option<AssistantEvent> {
    match event {
        SignalEvent::Chunk {
            request_id, chunk, ..
        } => Some(AssistantEvent::Chunk {
            request_id,
            text: chunk,
        }),
        // Progress status (turn-start + per tool call). The pipeline uses it as
        // a progress heartbeat and narrates it sparingly (#58).
        SignalEvent::Status {
            request_id,
            message,
            ..
        } => Some(AssistantEvent::Status {
            request_id,
            message,
        }),
        SignalEvent::Complete {
            request_id,
            full_response,
            ..
        } => Some(AssistantEvent::Complete {
            request_id,
            full_response,
        }),
        SignalEvent::Error {
            request_id, error, ..
        } => Some(AssistantEvent::Error { request_id, error }),
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
        // Hand the pipeline ONE durable subscription that survives a reconnect
        // (voice#83).
        //
        // The Connector keeps a single persistent, auto-reconnecting fan-out —
        // but on every transport drop its supervisor DRAINS the current
        // subscribers (sending each a terminal `Disconnected`) on purpose, since
        // a turn waiting across the drop is lost. So a raw `Connector::subscribe`
        // receiver goes dead the instant the socket drops. The pipeline
        // subscribes *before* sending, so an idempotent send-retry that
        // reconnects mid-turn would orphan that subscription and the recovered
        // turn would stream invisibly (the user hears the stall apology even
        // though the turn succeeded).
        //
        // We forward through a task that, on a `Disconnected`, RE-REGISTERS a
        // fresh slice of the (reconnected-in-place) fan-out and keeps going — so
        // the pipeline holds one receiver for the whole turn and never sees the
        // gap.
        //
        // `current()` dials lazily if the gateway started degraded (voice#86):
        // once it returns the single durable Connector, that Arc is never
        // replaced, so this subscription stays bound to the same fan-out the
        // supervisor heals in place.
        let connector = self.current().await?;
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut signals = connector.subscribe();
            loop {
                match signals.recv().await {
                    Some(SignalEvent::Disconnected { reason }) => {
                        // The fan-out unstuck us on a drop/stall. The Connector's
                        // supervisor reconnects the transport in place and keeps
                        // feeding the same persistent stream; re-register so the
                        // new connection's events reach the pipeline. (Swallowed,
                        // not surfaced: a `Disconnected` maps to nothing and would
                        // otherwise just be skipped — but we must re-subscribe, or
                        // the drained registration delivers no further events.)
                        tracing::debug!(
                            %reason,
                            "assistant event stream disconnected; re-subscribing across reconnect"
                        );
                        // Yield briefly so a same-tick supervisor reconnect can
                        // make progress before we re-register; the fan-out is
                        // persistent, so re-registering even slightly early is
                        // safe (we still receive the post-reconnect events).
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        signals = connector.subscribe();
                    }
                    Some(event) => {
                        if let Some(mapped) = map_signal(event)
                            && tx.send(mapped).is_err()
                        {
                            break; // the pipeline dropped this subscription
                        }
                    }
                    None => break, // fan-out gone (Connector dropped) — terminal
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
            map_signal(SignalEvent::Chunk { conversation_id: "c".into(), request_id: "r".into(), chunk: "hi".into() }),
            Some(AssistantEvent::Chunk { text, .. }) if text == "hi"
        ));
        assert!(matches!(
            map_signal(SignalEvent::Complete { conversation_id: "c".into(), request_id: "r".into(), full_response: "done".into() }),
            Some(AssistantEvent::Complete { full_response, .. }) if full_response == "done"
        ));
        assert!(matches!(
            map_signal(SignalEvent::Error { conversation_id: "c".into(), request_id: "r".into(), error: "boom".into() }),
            Some(AssistantEvent::Error { error, .. }) if error == "boom"
        ));
    }

    #[test]
    fn maps_status_into_a_voice_status_event() {
        // #58: progress status must reach the pipeline (for narration + as a
        // stall heartbeat), not be dropped.
        assert!(matches!(
            map_signal(SignalEvent::Status {
                conversation_id: "c".into(),
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
