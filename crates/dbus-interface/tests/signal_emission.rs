//! Regression test for voice#85 (V-5): the `org.desktopAssistant.Voice` signals
//! StateChanged / TranscriptReady / SpeakingText are DECLARED on the interface
//! but were never emitted, so a client had to poll `GetState`. This test drives
//! the `run_signal_forwarder` and asserts each signal actually goes out.
//!
//! It uses a peer-to-peer zbus connection pair over an in-process socket, so the
//! assertion is hermetic — no session bus, no collision with a live adele-voice
//! daemon.

use std::time::Duration;

use adele_voice_core::domain::State;
use adele_voice_dbus_interface::{VoiceSignal, run_signal_forwarder};
use futures_util::StreamExt;
use tokio::sync::{mpsc, watch};
use zbus::object_server::SignalEmitter;
use zbus::{Guid, MessageStream, conn::Builder};

const PATH: &str = "/org/desktopAssistant/Voice";
const IFACE: &str = "org.desktopAssistant.Voice";

/// Build a connected p2p (server, client) zbus pair over an in-process socket.
async fn connection_pair() -> (zbus::Connection, zbus::Connection) {
    let guid = Guid::generate();
    let (server_sock, client_sock) = tokio::net::UnixStream::pair().unwrap();

    let server_fut = Builder::unix_stream(server_sock)
        .p2p()
        .server(guid)
        .unwrap()
        .build();
    let client_fut = Builder::unix_stream(client_sock).p2p().build();

    let (server, client) = tokio::join!(server_fut, client_fut);
    (server.expect("server conn"), client.expect("client conn"))
}

/// Collect the next signal (member name + first string arg) on `stream` for this
/// interface/path, within a timeout.
async fn next_signal(stream: &mut MessageStream) -> (String, String) {
    let deadline = tokio::time::sleep(Duration::from_secs(3));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => panic!("timed out waiting for a signal"),
            msg = stream.next() => {
                let msg = msg.expect("stream ended").expect("valid message");
                let header = msg.header();
                if header.message_type() != zbus::message::Type::Signal {
                    continue;
                }
                let member = header.member().map(|m| m.to_string()).unwrap_or_default();
                let iface = header.interface().map(|i| i.to_string()).unwrap_or_default();
                let path = header.path().map(|p| p.to_string()).unwrap_or_default();
                if iface != IFACE || path != PATH {
                    continue;
                }
                let arg: String = msg.body().deserialize().unwrap_or_default();
                return (member, arg);
            }
        }
    }
}

/// All three signals are emitted, with the right member name and payload, when
/// the corresponding pipeline event flows through the forwarder.
#[tokio::test]
async fn emits_state_transcript_and_speaking_signals() {
    let (server, client) = connection_pair().await;

    let (state_tx, state_rx) = watch::channel(State::Idle);
    let (signal_tx, signal_rx) = mpsc::channel::<VoiceSignal>(16);

    let emitter = SignalEmitter::new(&server, PATH).unwrap();
    tokio::spawn(run_signal_forwarder(emitter, state_rx, signal_rx));

    // Listen on the client side for incoming signals.
    let mut stream = MessageStream::from(client);

    // The forwarder emits the INITIAL state on start — consume it (Idle) so the
    // assertions below see the events we drive.
    let (member, arg) = next_signal(&mut stream).await;
    assert_eq!(member, "StateChanged");
    assert_eq!(arg, "Idle");

    // 1) State transition -> StateChanged
    state_tx.send(State::Listening).unwrap();
    let (member, arg) = next_signal(&mut stream).await;
    assert_eq!(
        member, "StateChanged",
        "a state change must emit StateChanged"
    );
    assert_eq!(arg, "Listening");

    // 2) Transcript -> TranscriptReady
    signal_tx
        .send(VoiceSignal::TranscriptReady("what's the weather".into()))
        .await
        .unwrap();
    let (member, arg) = next_signal(&mut stream).await;
    assert_eq!(member, "TranscriptReady");
    assert_eq!(arg, "what's the weather");

    // 3) Speaking -> SpeakingText
    signal_tx
        .send(VoiceSignal::SpeakingText("It's sunny.".into()))
        .await
        .unwrap();
    let (member, arg) = next_signal(&mut stream).await;
    assert_eq!(member, "SpeakingText");
    assert_eq!(arg, "It's sunny.");
}

/// The forwarder shuts down cleanly when the pipeline drops both sources (so the
/// daemon doesn't leak the task).
#[tokio::test]
async fn forwarder_stops_when_pipeline_is_gone() {
    let (server, _client) = connection_pair().await;

    let (state_tx, state_rx) = watch::channel(State::Idle);
    let (signal_tx, signal_rx) = mpsc::channel::<VoiceSignal>(16);

    let emitter = SignalEmitter::new(&server, PATH).unwrap();
    let handle = tokio::spawn(run_signal_forwarder(emitter, state_rx, signal_rx));

    drop(state_tx);
    drop(signal_tx);

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("forwarder must exit when both sources are gone")
        .expect("task should not panic");
}
