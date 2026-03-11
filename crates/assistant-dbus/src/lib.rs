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
