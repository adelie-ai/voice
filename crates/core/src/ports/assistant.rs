use crate::VoiceError;
use tokio::sync::mpsc;

/// Events received from the assistant daemon.
#[derive(Debug, Clone)]
pub enum AssistantEvent {
    /// A chunk of the response text.
    Chunk { request_id: String, text: String },
    /// The response is complete.
    Complete {
        request_id: String,
        full_response: String,
    },
    /// An error occurred while generating the response.
    Error { request_id: String, error: String },
}

/// Outbound port for communicating with the desktop-assistant daemon.
pub trait AssistantGateway: Send + Sync {
    /// Create a new conversation, returning its ID.
    fn create_conversation(
        &self,
        title: &str,
    ) -> impl std::future::Future<Output = Result<String, VoiceError>> + Send;

    /// Send a prompt to an existing conversation, returning a request ID.
    fn send_prompt(
        &self,
        conversation_id: &str,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, VoiceError>> + Send;

    /// Send a prompt with a per-request `system_refinement` — a one-turn
    /// addition to the assistant's system prompt (empty = none).
    ///
    /// The voice daemon uses this to attach a spoken-response hint
    /// ("respond briefly, by voice") to a turn dictated into an existing
    /// chat WITHOUT prepending it to the user's message, so the visible
    /// transcript records only the clean `prompt`. The orchestrator
    /// appends the refinement to the system prompt for this LLM call only
    /// and never stores it.
    ///
    /// Implementations talking to an orchestrator that predates the
    /// refinement-aware D-Bus method must fall back gracefully (prepend
    /// the refinement to the prompt and call [`send_prompt`]) so an
    /// un-upgraded daemon still answers. Returns a request ID like
    /// [`send_prompt`].
    fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> impl std::future::Future<Output = Result<String, VoiceError>> + Send;

    /// Subscribe to streaming response events from the daemon.
    fn subscribe(
        &self,
    ) -> impl std::future::Future<
        Output = Result<mpsc::UnboundedReceiver<AssistantEvent>, VoiceError>,
    > + Send;
}
