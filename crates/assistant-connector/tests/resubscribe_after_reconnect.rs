//! Regression test for voice#83 (V-3, HIGH): an event subscription taken from
//! the gateway must keep delivering events AFTER the gateway's reconnect-retry
//! path runs.
//!
//! The bug: the gateway used to reconnect by swapping in a brand-new
//! `Connector`. A subscription handed out before the swap was bound to the OLD
//! connector's fan-out, so it silently stopped receiving events — the turn
//! streamed invisibly and the user heard the stall apology even though the turn
//! succeeded. The fix reconnects the SAME connector's transport in place (its
//! fan-out survives), so the pre-existing subscription keeps flowing.
//!
//! This spins up the real `desktop-assistant-uds` server in-process (reusing the
//! pattern from client-common's `uds_streaming` test), builds the production
//! `ConnectorAssistantGateway`, subscribes, forces the gateway's reconnect, then
//! sends — and asserts the response arrives on the ORIGINAL subscription.

use std::path::PathBuf;
use std::sync::Arc;

use adele_voice_assistant_connector::{ConnectionConfig, ConnectorAssistantGateway, TransportMode};
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway};
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

/// Streams a one-shot "hello" response for every send, mirroring the production
/// registry path (task_id distinct from request_id; events stamped with the
/// request_id).
struct StreamingHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for StreamingHandler {
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
        conversation_id: String,
        _content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        request_id: String,
        _idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        let task_id = api::TaskId(uuid::Uuid::new_v4().to_string());
        tokio::spawn(async move {
            sink.emit(api::Event::AssistantDelta {
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                chunk: "hello".into(),
            })
            .await;
            sink.emit(api::Event::AssistantCompleted {
                conversation_id,
                request_id,
                full_response: "hello".into(),
            })
            .await;
        });
        Ok(Some(task_id))
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
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(StreamingHandler);
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

/// Drain `rx` until a `Complete` (success) or `Error`/closed (failure), so the
/// assertion is "the response reached the original subscription", not a brittle
/// chunk-by-chunk match.
async fn await_complete(rx: &mut tokio::sync::mpsc::UnboundedReceiver<AssistantEvent>) -> String {
    loop {
        match timeout(Duration::from_secs(3), rx.recv()).await {
            Ok(Some(AssistantEvent::Complete { full_response, .. })) => return full_response,
            Ok(Some(AssistantEvent::Error { error, .. })) => {
                panic!("turn errored instead of completing: {error}")
            }
            Ok(Some(_)) => continue,
            Ok(None) => panic!(
                "subscription closed before the Complete event — events were orphaned (voice#83)"
            ),
            Err(_) => panic!(
                "timed out waiting for the Complete event on the original subscription — \
                 the subscription was orphaned by the reconnect (voice#83)"
            ),
        }
    }
}

/// The core voice#83 regression: a subscription taken BEFORE the orchestrator
/// connection drops must still deliver the response of a turn sent AFTER the
/// connection comes back.
///
/// This reproduces the real failure path. The orchestrator restarts (the socket
/// drops); the Connector's reconnect supervisor sees the drop and — by design —
/// sends every CURRENT subscriber a terminal `Disconnected` (a turn waiting
/// across the drop is lost) before reconnecting in place. A naive gateway
/// subscription dies on that drain: the forwarding task ends and the pipeline's
/// receiver closes, so the reply to the next (idempotently-retried) send streams
/// invisibly and the user hears the stall apology even though the turn
/// succeeded. The fixed gateway hands out a DURABLE subscription that
/// re-registers across the drop, so the reply still lands on the pipeline's
/// original receiver.
///
/// To make the regression deterministic we wait out the supervisor's
/// disconnect-and-drain before bringing the server back: a naive subscription is
/// guaranteed dead by then, while the durable one has re-registered.
#[tokio::test]
async fn subscription_survives_an_orchestrator_restart() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");

    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path.clone(), mint_test_jwt(&signing_key, "dave"));
    let gateway = ConnectorAssistantGateway::connect(&cfg)
        .await
        .expect("gateway connects over uds");

    // Subscribe first (the pipeline's ordering) — the subscription that must
    // survive the drop.
    let mut rx = gateway.subscribe().await.expect("subscribe");

    // Drop the orchestrator. The supervisor notices the socket close, drains
    // current subscribers with a terminal `Disconnected`, and starts
    // reconnecting. Wait long enough that the drain has definitely happened (so a
    // naive subscription is provably dead at this point).
    let _ = shutdown.send(());
    tokio::time::sleep(Duration::from_millis(400)).await;

    // Bring a fresh orchestrator back on the SAME socket path; the supervisor
    // reconnects in place.
    let shutdown2 = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    // Send once the gateway has reconnected (it retries on a failed send), then
    // assert the reply lands on the ORIGINAL subscription.
    let mut sent = false;
    for _ in 0..30 {
        if timeout(
            Duration::from_secs(3),
            gateway.send_prompt_with_system_refinement("conv-1", "hi", ""),
        )
        .await
        .expect("send must not hang")
        .is_ok()
        {
            sent = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    assert!(sent, "send never succeeded after the orchestrator restart");

    let response = await_complete(&mut rx).await;
    assert_eq!(
        response, "hello",
        "the original subscription must still receive the turn's response after a reconnect"
    );

    let _ = shutdown2.send(());
}

/// Deterministic reproduction of the orphaning at the gateway boundary. The old
/// gateway reconnected by SWAPPING in a brand-new `Connector` (with its own
/// fan-out); a subscription taken from the old connector was then dead. This
/// drives that exact swap via `reconnect_for_test` between subscribe and send —
/// the old code never delivers the reply here; the fixed gateway (single
/// connector + durable subscription) does.
#[tokio::test]
async fn subscription_survives_a_gateway_reconnect_call() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let gateway = ConnectorAssistantGateway::connect(&cfg)
        .await
        .expect("gateway connects over uds");

    let mut rx = gateway.subscribe().await.expect("subscribe");
    // Drive the reconnect path the idempotent send-retry takes on a failed send.
    gateway.reconnect_for_test().await;

    let _id = timeout(
        Duration::from_secs(3),
        gateway.send_prompt_with_system_refinement("conv-1", "hi", ""),
    )
    .await
    .expect("send should ack")
    .expect("ack ok");

    let response = await_complete(&mut rx).await;
    assert_eq!(
        response, "hello",
        "the original subscription must still receive the reply after a gateway reconnect"
    );

    let _ = shutdown.send(());
}

/// Sanity: without any reconnect, the subscription delivers the response. Guards
/// against the harness itself being broken (so a failure of the reconnect test
/// is attributable to the reconnect, not to setup).
#[tokio::test]
async fn subscription_delivers_without_reconnect() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let gateway = ConnectorAssistantGateway::connect(&cfg)
        .await
        .expect("gateway connects over uds");

    let mut rx = gateway.subscribe().await.expect("subscribe");
    let _id = timeout(
        Duration::from_secs(3),
        gateway.send_prompt_with_system_refinement("conv-1", "hi", ""),
    )
    .await
    .expect("send should ack")
    .expect("ack ok");

    let response = await_complete(&mut rx).await;
    assert_eq!(response, "hello");

    let _ = shutdown.send(());
}
