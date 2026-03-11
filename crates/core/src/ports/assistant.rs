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

    /// Subscribe to streaming response events from the daemon.
    fn subscribe(
        &self,
    ) -> impl std::future::Future<
        Output = Result<mpsc::UnboundedReceiver<AssistantEvent>, VoiceError>,
    > + Send;
}
