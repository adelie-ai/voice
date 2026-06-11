//! Regression test for voice#86 (V-6): the daemon must NOT die if the
//! orchestrator is unreachable at startup. It should degrade — keep running —
//! and connect once the orchestrator appears.
//!
//! At the gateway level that means `connect_or_degrade` never errors (even with
//! nothing listening), a call made while disconnected returns a *recoverable*
//! error instead of panicking, and the first call after the orchestrator comes
//! up connects lazily and succeeds.

use std::path::PathBuf;
use std::sync::Arc;

use adele_voice_assistant_connector::{ConnectionConfig, ConnectorAssistantGateway, TransportMode};
use adele_voice_core::ports::assistant::AssistantGateway;
use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_uds::{UdsAuthValidator, UdsServer, UdsServerConfig};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-uds-iss";
const AUD: &str = "test-uds-aud";

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn mint_test_jwt(signing_key: &str, subject: &str) -> String {
    let now = unix_now();
    let claims = jwt::Claims {
        iss: ISS.into(),
        sub: subject.into(),
        aud: AUD.into(),
        exp: now + 600,
        iat: now,
        nbf: now.saturating_sub(1),
        jti: uuid::Uuid::new_v4().to_string(),
    };
    jwt::encode(&claims, signing_key).expect("encode jwt")
}

/// Acks commands and acks a send (returns a task id) — enough to prove the
/// gateway connected and can talk to the orchestrator.
struct AckHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for AckHandler {
    async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
        Ok(api::CommandResult::Ack)
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        Ok(())
    }

    async fn start_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        _request_id: String,
        _idempotency_key: Option<String>,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        Ok(Some(api::TaskId(uuid::Uuid::new_v4().to_string())))
    }
}

struct StaticJwtAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if path.exists() && UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("uds socket {path:?} did not appear");
}

fn uds_config(socket_path: PathBuf, jwt: String) -> ConnectionConfig {
    ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(socket_path),
        ws_jwt: Some(jwt),
        ..ConnectionConfig::default()
    }
}

fn start_server(socket_path: PathBuf, signing_key: String) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(AckHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
    let config = UdsServerConfig::new(socket_path);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, config);
    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    tx
}

/// `connect_or_degrade` against a socket with no server listening must NOT
/// error (the daemon would otherwise die at startup — voice#86) and a call made
/// while disconnected must return a recoverable error, not panic.
#[tokio::test]
async fn degrades_when_orchestrator_unreachable_at_startup() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    // A path with nothing listening.
    let path = dir.path().join("absent.sock");
    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));

    // Must return a gateway, never an Err — this is the line that used to kill
    // the daemon (`connect(...).await?`).
    let gateway = ConnectorAssistantGateway::connect_or_degrade(&cfg).await;

    // A turn attempted while degraded surfaces a recoverable error (the pipeline
    // apologizes and retries next turn), NOT a crash.
    let result = timeout(Duration::from_secs(3), gateway.send_prompt("conv-1", "hi"))
        .await
        .expect("call must not hang while degraded");
    assert!(
        result.is_err(),
        "a send while the orchestrator is down should error, not succeed"
    );
}

/// After `connect_or_degrade` starts disconnected, the first call once the
/// orchestrator is up connects lazily and succeeds.
#[tokio::test]
async fn connects_lazily_once_orchestrator_appears() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let cfg = uds_config(path.clone(), mint_test_jwt(&signing_key, "dave"));

    // Start degraded: nothing is listening yet.
    let gateway = ConnectorAssistantGateway::connect_or_degrade(&cfg).await;

    // Orchestrator comes up.
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    // The next call connects lazily and succeeds (retry a couple of times to
    // absorb the connect race).
    let mut ok = false;
    for _ in 0..10 {
        if timeout(Duration::from_secs(3), gateway.send_prompt("conv-1", "hi"))
            .await
            .expect("call must not hang")
            .is_ok()
        {
            ok = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        ok,
        "the gateway must connect lazily and send once the orchestrator is up"
    );

    let _ = shutdown.send(());
}
