use crate::VoiceError;
use tokio::sync::mpsc;

/// Events received from the assistant daemon.
#[derive(Debug, Clone)]
pub enum AssistantEvent {
    /// A chunk of the response text.
    Chunk { request_id: String, text: String },
    /// A human-readable progress status for the in-flight turn (e.g. "checking
    /// your calendar…"), emitted by the orchestrator at turn-start and per tool
    /// call. The voice pipeline narrates it sparingly and uses every status —
    /// like every chunk — as a progress heartbeat that resets its stall
    /// deadline (#58).
    Status { request_id: String, message: String },
    /// The response is complete.
    Complete {
        request_id: String,
        full_response: String,
    },
    /// An error occurred while generating the response.
    Error { request_id: String, error: String },
    /// The orchestrator's turn has suspended on a client-local tool call
    /// (voice#61). The LLM decided to drive the voice session — stop listening,
    /// keep listening, or speak a specific line — by calling one of the static
    /// tools the daemon registered at startup. The pipeline runs the named tool
    /// and MUST post the outcome back via
    /// [`AssistantGateway::submit_client_tool_result`] (carrying the same
    /// `task_id` + `tool_call_id`); until then the orchestrator turn is parked.
    ///
    /// Unlike the response-turn events, this is NOT keyed on the turn's
    /// `request_id` — a suspended tool call carries the orchestrator `task_id`
    /// instead, so the pipeline acts on every `ClientToolCall` it sees on the
    /// stream rather than filtering by request id.
    ClientToolCall {
        task_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
}

/// A client-local tool the daemon advertises to the orchestrator so the LLM can
/// drive the voice session (voice#61). Mirrors the orchestrator's
/// `ClientToolRegistration` without the core crate depending on its api-model;
/// the connector translates it on the wire.
#[derive(Debug, Clone)]
pub struct ClientToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's input, forwarded verbatim to the LLM.
    pub input_schema: serde_json::Value,
}

/// Outbound port for communicating with the desktop-assistant daemon.
pub trait AssistantGateway: Send + Sync {
    /// Advertise the set of client-local tools this connection can run
    /// (voice#61). The orchestrator replaces any previously-registered set on
    /// each call, so the full list is sent — re-register on every connect.
    /// Returns the count the daemon accepted.
    fn register_client_tools(
        &self,
        tools: Vec<ClientToolSpec>,
    ) -> impl std::future::Future<Output = Result<usize, VoiceError>> + Send;

    /// Deliver the outcome of a [`AssistantEvent::ClientToolCall`] back to the
    /// orchestrator so the suspended turn can resume (voice#61). `result` is the
    /// tool's textual outcome (`Ok`) or an error message (`Err`); pass the same
    /// `task_id` + `tool_call_id` the event carried.
    fn submit_client_tool_result(
        &self,
        task_id: &str,
        tool_call_id: &str,
        result: Result<String, String>,
    ) -> impl std::future::Future<Output = Result<(), VoiceError>> + Send;

    /// Create a new conversation, returning its ID.
    fn create_conversation(
        &self,
        title: &str,
    ) -> impl std::future::Future<Output = Result<String, VoiceError>> + Send;

    /// Send a prompt with a per-request `system_refinement` — a one-turn
    /// addition to the assistant's system prompt (empty = none).
    ///
    /// The voice daemon uses this to attach a spoken-response hint
    /// ("respond briefly, by voice") to a turn dictated into an existing
    /// chat WITHOUT prepending it to the user's message, so the visible
    /// transcript records only the clean `prompt`. The orchestrator
    /// appends the refinement to the system prompt for this LLM call only
    /// and never stores it. Pass an empty `system_refinement` for an
    /// ordinary prompt. Returns a request ID.
    ///
    /// Implementations talking to an orchestrator that predates the
    /// refinement-aware D-Bus method must fall back gracefully (prepend
    /// the refinement to the prompt on the older send path) so an
    /// un-upgraded daemon still answers.
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
