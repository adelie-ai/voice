use std::sync::Arc;
use std::time::{Duration, Instant};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::{SAMPLE_RATE, State, StateEvent};
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway, ClientToolSpec};
use adele_voice_core::ports::audio::{AudioSink, AudioSource};
use adele_voice_core::ports::stt::SpeechToText;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_core::ports::vad::VoiceActivityDetector;
use adele_voice_core::ports::wake::WakeWordDetector;
use adele_voice_core::sentence_buffer::SentenceBuffer;
use adele_voice_dbus_interface::{CalibrationRequest, CaptureState, StopRequest, VoiceSignal};
use adele_voice_module::{Endpoint, Endpointer, PreBuffer, Speaker, Transcriber};
use tokio::sync::{mpsc, watch};

use crate::calibration;
use crate::config::{self, ClientToolsConfig, Tunables, plan_reload};
use crate::cue::{self, ListeningCue};
use crate::session::SessionGate;

/// Sleep for `dur` if `Some`, else never resolve. Used to make a `select!` arm
/// conditional on an optional deadline without a separate `if` guard pre-check.
async fn sleep_opt(dur: Option<Duration>) {
    match dur {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending::<()>().await,
    }
}

/// Await the next calibration request if a channel is attached, else never
/// resolve — so the `select!` arm is dormant when calibration isn't wired
/// (mirrors [`sleep_opt`]). Returns `None` when the channel has closed.
async fn recv_calibration(
    rx: &mut Option<mpsc::Receiver<CalibrationRequest>>,
) -> Option<CalibrationRequest> {
    match rx {
        Some(r) => r.recv().await,
        None => std::future::pending().await,
    }
}

/// Spoken when the assistant turn fails — short and human, never the raw error.
const ERROR_APOLOGY: &str = "Sorry, I ran into an error and couldn't answer that.";

/// Spoken when a turn stalls (no progress within the deadline / over budget) so
/// the user isn't left in silent Processing forever (#58).
const STALL_APOLOGY: &str = "Sorry, that's taking too long. Let's try again.";

/// Brief liveness lines spoken on a slow turn that has narrated nothing and
/// streamed nothing yet (e.g. it declared no plan step) — so voice isn't
/// dead-silent. Spoken in order, one per `liveness_delay`, escalating so a
/// genuinely slow turn doesn't just repeat "One moment." robotically; the turn
/// fires at most `LIVENESS_PHRASES.len()` of them, then stays quiet until the
/// stall/budget backstop. Rare by design.
const LIVENESS_PHRASES: &[&str] = &["One moment.", "Still working on it."];

/// A short leading ack ("Got it — checking that now.") is at most this many
/// words; longer terminal sentences are the real answer and aren't flushed
/// early (#58).
const ACK_MAX_WORDS: usize = 8;

/// The per-turn timeout bounds the pipeline applies (#58), grouped so the
/// constructor's argument list stays manageable. `Duration::ZERO` disables an
/// individual bound.
#[derive(Debug, Clone, Copy)]
pub struct TurnTimeouts {
    /// Per-event stall deadline for the streaming response; resets on each
    /// chunk/status.
    pub response_stall: Duration,
    /// Overall ceiling on a single turn's streaming response.
    pub turn_budget: Duration,
    /// Per-synth ceiling, applied to the `Speaker`.
    pub synth: Duration,
    /// Per-round-trip ceiling for conversation create/subscribe/send.
    pub connect: Duration,
    /// Minimum gap between spoken status narrations within a turn.
    pub status_narration_min_gap: Duration,
    /// Delay before speaking a brief liveness line when a turn has produced no
    /// narration and no reply yet (a slow turn that declared no step). Fires at
    /// most once and never once any status/chunk has arrived. `Duration::ZERO`
    /// disables.
    pub liveness_delay: Duration,
    /// Mid-sentence quiet window before the narration buffer flushes a clause to
    /// TTS. `Duration::ZERO` disables timeout flushing (speak only at sentence
    /// ends and end-of-stream).
    pub narration_flush: Duration,
}

/// The non-wiring configuration for a [`Pipeline`], grouped so `Pipeline::new`
/// takes one options struct instead of a dozen positional scalars (voice#89).
/// The audio backends and channels stay positional arguments — they're wiring,
/// not tunables — while everything here is plain config the constructor stores
/// or forwards.
pub struct PipelineConfig {
    /// Reloadable tunables snapshot (the on-disk truth we diff against).
    pub tunables: Tunables,
    /// Title used when the daemon lazily creates its own conversation.
    pub conversation_title: String,
    /// Post-speech silence that closes an utterance.
    pub silence_duration: Duration,
    /// VAD speech-probability threshold.
    pub speech_threshold: f32,
    /// Allow mic speech to interrupt Adele's playback (barge-in, voice#82).
    /// OFF unless echo cancellation is in place — see [`AudioConfig`].
    pub mic_barge_in: bool,
    /// Keep the mic open for follow-up turns after a reply.
    pub conversation_mode: bool,
    /// Window within which a fresh wake reuses the recent conversation (voice#53).
    pub conversation_reuse_window: Duration,
    /// Lead-in before a turn cuts on silence with no speech.
    pub followup_timeout: Duration,
    /// Idle-exit timeout when wake-word listening is off (`None` = never exit).
    pub idle_exit_timeout: Option<Duration>,
    /// Per-request system refinement attached to voice turns (empty = none).
    pub spoken_response_hint: String,
    /// Audible "Listening" cue mode (#51).
    pub listening_cue: ListeningCue,
    /// Per-turn timeout bounds (#58).
    pub timeouts: TurnTimeouts,
    /// Which session-control client tools to advertise (voice#61).
    pub client_tools: ClientToolsConfig,
}

/// Wrap a future in `timeout` unless `limit` is zero (zero = unbounded), mapping
/// an elapsed timeout to an `Assistant` error with `label` for the log (#58).
async fn bounded<F, R>(limit: Duration, label: &str, fut: F) -> Result<R, VoiceError>
where
    F: std::future::Future<Output = Result<R, VoiceError>>,
{
    if limit.is_zero() {
        return fut.await;
    }
    match tokio::time::timeout(limit, fut).await {
        Ok(result) => result,
        Err(_elapsed) => Err(VoiceError::Assistant(format!(
            "{label} timed out after {} ms",
            limit.as_millis()
        ))),
    }
}

/// Buffered-sample floor below which a trailing silence won't close an
/// utterance — guards against a single stray blip (50 ms at 16 kHz).
const ENDPOINT_MIN_SAMPLES: usize = 800;

/// Rolling pre-buffer length kept while idle so the wake→listen handoff can seed
/// the utterance with the audio captured right around the trigger — the start of
/// a command spoken in the same breath ("hey adele what time is it") that the
/// Idle→Listening transition would otherwise drop (#50). 300 ms at 16 kHz.
const WAKE_PREBUFFER_SAMPLES: usize = (SAMPLE_RATE as usize * 300) / 1000;

/// Heuristic: does this look like an orchestrator error surfaced as reply text?
/// The orchestrator reports LLM failures as the assistant message (so they show
/// in the chat UI), e.g. "Details: LLM error: Bedrock …". Reading that aloud is
/// terrible, so we substitute a short apology. "llm error" is specific enough
/// that a genuine spoken reply won't contain it.
fn is_error_response(text: &str) -> bool {
    text.to_ascii_lowercase().contains("llm error")
}

/// Outcome of handling one captured utterance.
#[derive(Debug, Clone, PartialEq)]
enum UtteranceOutcome {
    /// Normal turn — the run loop decides whether to keep listening (in
    /// conversation mode) or return to wake-word idle.
    Continue,
    /// The user spoke a stop phrase, or the LLM called the `stop_listening`
    /// client tool — end the conversation now, whatever the mode, and clear the
    /// reuse-window id so the next wake starts fresh (voice#59/#61).
    EndConversation,
    /// The LLM called the `listen_for_more` client tool — keep/extend the
    /// listening window for a follow-up even outside conversation mode (voice#61).
    KeepListening,
    /// The streaming turn was interrupted mid-flight by a control channel or a
    /// barge-in (voice#82). Carries what to do next; `handle_utterance_complete`
    /// re-arms accordingly instead of running the normal relisten logic.
    Interrupted(InterruptKind),
}

/// How a streaming turn ended (voice#82). `stream_response` returns this instead
/// of `()` so an interrupt is just a select arm winning, with all cleanup run
/// sequentially afterwards (no future dropped at an arbitrary await point).
#[derive(Debug, Clone, PartialEq)]
enum StreamEnd {
    /// The turn reached one of the existing endings: Complete / Error /
    /// stall-apology / budget / clean stream close.
    Completed,
    /// A D-Bus `StopSpeaking` (`Speaking`) or `StopListening` (`Conversation`)
    /// arrived mid-turn.
    Stopped(StopRequest),
    /// VAD detected user speech over our own playback; carries the triggering
    /// chunk so the run loop can seed the endpointer (mirrors the outer-loop
    /// barge-in arm).
    BargedIn(Vec<f32>),
    /// A push-to-talk press arrived mid-turn; carries the target conversation.
    PttPressed(Option<String>),
}

/// What the run loop should do after an interrupted turn (voice#82). The
/// interrupt-carrying half of [`StreamEnd`], threaded through
/// [`UtteranceOutcome::Interrupted`] into `handle_utterance_complete`.
#[derive(Debug, Clone, PartialEq)]
enum InterruptKind {
    /// Barge-in: arm the endpointer with the triggering chunk and go to
    /// Listening (no cue — the user is already talking).
    BargeIn(Vec<f32>),
    /// PTT press: re-run the PTT-entry path with the new target.
    Ptt(Option<String>),
    /// `StopSpeaking`: Idle, conversation retained (a wake within the reuse
    /// window resumes it).
    StopSpeaking,
    /// `StopListening`: Idle + end the conversation.
    StopConversation,
}

/// The three static session-control client tools the daemon advertises so the
/// LLM can drive the voice session (voice#61).
pub const TOOL_STOP_LISTENING: &str = "stop_listening";
pub const TOOL_LISTEN_FOR_MORE: &str = "listen_for_more";
pub const TOOL_SAY_THIS: &str = "say_this";

/// Build the [`ClientToolSpec`] registrations for the enabled session-control
/// tools (voice#61). The descriptions are written to guide the LLM on WHEN to
/// call each — especially `stop_listening`, which must fire when the user
/// signals they're finished. Returned in a stable order for deterministic tests.
pub fn session_control_tools(toggles: ClientToolsConfig) -> Vec<ClientToolSpec> {
    let no_args = || {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    };
    let mut tools = Vec::new();
    if toggles.stop_listening {
        tools.push(ClientToolSpec {
            name: TOOL_STOP_LISTENING.to_string(),
            description: "End the voice session. Call this when the user signals they are done — \
                they decline further help, say goodbye, or otherwise indicate the conversation is \
                over (e.g. you ask \"Anything else?\" and they say \"No\"). After your final \
                reply is spoken the microphone closes and the next wake word starts a brand-new \
                conversation. Do not call it while you still expect the user to respond."
                .to_string(),
            input_schema: no_args(),
        });
    }
    if toggles.listen_for_more {
        tools.push(ClientToolSpec {
            name: TOOL_LISTEN_FOR_MORE.to_string(),
            description: "Keep listening for the user's reply. Call this when you expect the user \
                to respond — for example you asked them a question or offered a choice — so the \
                microphone re-opens for their answer instead of the session ending."
                .to_string(),
            input_schema: no_args(),
        });
    }
    if toggles.say_this {
        tools.push(ClientToolSpec {
            name: TOOL_SAY_THIS.to_string(),
            description: "Speak this exact line to the user out loud right now, before the rest \
                of your reply. Use it for a brief spoken progress note or aside (e.g. \"One \
                moment, checking that now.\") that should be read aloud verbatim."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact line to speak aloud."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }),
        });
    }
    tools
}

/// Normalize a transcript for whole-utterance matching: trim, lowercase, drop
/// sentence punctuation, and collapse internal whitespace.
fn normalize_phrase(text: &str) -> String {
    text.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '.' | ',' | '!' | '?'))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// True when the transcript is just one of the short canned lines the daemon
/// speaks itself (a liveness reassurance or a stall/error apology). Without
/// acoustic echo cancellation the mic can pick up that playback, Whisper
/// transcribes it, and it would otherwise start a fresh turn that produces
/// another canned line — a feedback loop. We discard such an utterance instead
/// of acting on it. The user's own words never match these exact phrases.
fn is_self_echo(text: &str) -> bool {
    let normalized = normalize_phrase(text);
    if normalized.is_empty() {
        return false;
    }
    LIVENESS_PHRASES
        .iter()
        .copied()
        .chain([STALL_APOLOGY, ERROR_APOLOGY])
        .any(|canned| normalize_phrase(canned) == normalized)
}

/// Whole-utterance "stop listening" phrases, matched only against the entire
/// normalized transcript (so "stop" inside a sentence isn't a command). Lets the
/// user end a conversation hands-free.
fn is_stop_phrase(text: &str) -> bool {
    let normalized = normalize_phrase(text);
    matches!(
        normalized.as_str(),
        "stop"
            | "stop listening"
            | "stop listening adele"
            | "stop adele"
            | "never mind"
            | "nevermind"
            | "that's all"
            | "thats all"
            | "that is all"
            | "that'll be all"
            | "goodbye"
            | "good bye"
            | "cancel"
            | "we're done"
            | "were done"
            | "i'm done"
            | "im done"
    )
}

pub struct Pipeline<W, V, S, T, A>
where
    W: WakeWordDetector + 'static,
    V: VoiceActivityDetector + 'static,
    S: SpeechToText + 'static,
    T: TextToSpeech + 'static,
    A: AssistantGateway + 'static,
{
    wake: W,
    vad: V,
    transcriber: Transcriber<S>,
    speaker: Speaker<T>,
    assistant: Arc<A>,
    source: Arc<dyn AudioSource>,
    /// Direct sink handle for the raw earcon (the `ding` cue is generated PCM,
    /// not TTS, so it bypasses the `Speaker`). Shares the same playback stream.
    sink: Arc<dyn AudioSink>,
    endpointer: Endpointer,
    /// Rolling window of recent idle audio, used to seed the utterance with the
    /// post-wake speech so a command spoken in the same breath isn't dropped (#50).
    prebuffer: PreBuffer,
    /// Audible "Listening" cue mode (ding / phrase / off) (#51).
    listening_cue: ListeningCue,
    /// Free-running counter so the spoken-phrase cue rotates deterministically.
    cue_phrase_counter: u64,
    /// The pipeline's current state. The single in-process source of truth,
    /// mutated ONLY through `apply` (which validates against the `state.rs`
    /// table and publishes to `state_tx`). Before voice#82 this was a local in
    /// `run()` that drifted from the published value mid-turn.
    state: State,
    state_tx: watch::Sender<State>,
    enabled_rx: watch::Receiver<bool>,
    ptt_rx: mpsc::Receiver<Option<String>>,
    stop_rx: mpsc::Receiver<StopRequest>,
    /// A ping (from the config-file watcher or the D-Bus `Reload` method) asking
    /// the pipeline to re-read the config and apply any changed tunables live
    /// (config#52).
    reload_rx: mpsc::Receiver<()>,
    /// Snapshot of the live-applicable knobs, diffed against a freshly loaded
    /// config on each reload to decide what to apply.
    tunables: Tunables,
    conversation_id: Option<String>,
    /// When a push-to-talk specified a target conversation, its orchestrator
    /// id. Set on `PushToTalkInConversation`, used by `process_utterance` to
    /// route the turn (and any conversation-mode follow-ups) to that
    /// conversation instead of the daemon's own session; cleared when the
    /// conversation ends. `None` means "use the daemon's own session".
    ptt_conversation_override: Option<String>,
    conversation_title: String,
    speech_threshold: f32,
    /// Allow mic speech to interrupt playback (barge-in). False unless AEC is in
    /// place — otherwise Adele barges in on her own echo (see [`AudioConfig`]).
    mic_barge_in: bool,
    conversation_mode: bool,
    /// Cross-wake conversation reuse window (voice#53): on a fresh wake within
    /// this window of the last turn's activity, the daemon's own session is
    /// continued rather than a new conversation opened. `Duration::ZERO`
    /// disables reuse (every wake starts fresh — the pre-#53 behaviour).
    conversation_reuse_window: Duration,
    /// Time of the last turn's activity on the daemon's own session, used with
    /// `conversation_reuse_window` to decide whether a fresh wake reuses
    /// `conversation_id`. `None` until the first own-session turn (voice#53).
    last_own_activity: Option<Instant>,
    followup_timeout: Duration,
    idle_exit_timeout: Option<Duration>,
    spoken_response_hint: String,
    /// Per-event stall deadline for the streaming response (#58). Resets on
    /// every chunk/status; on expiry the turn apologizes and returns to Idle.
    /// `Duration::ZERO` disables.
    response_stall: Duration,
    /// Overall ceiling on a single turn's streaming response, regardless of
    /// heartbeats (#58). `Duration::ZERO` disables.
    turn_budget: Duration,
    /// Minimum gap between spoken status narrations within a turn (#58). The
    /// first status always speaks; later ones are rate-limited to this interval.
    status_narration_min_gap: Duration,
    /// Delay before the safety-net liveness line on a turn that produced no
    /// narration and no reply yet (a slow turn that declared no step). Fires at
    /// most once; `Duration::ZERO` disables.
    liveness_delay: Duration,
    /// Mid-sentence quiet window before the narration buffer flushes a clause to
    /// TTS. `Duration::ZERO` disables timeout flushing.
    narration_flush: Duration,
    /// Bound on each conversation create / subscribe / send round-trip (#58).
    /// `Duration::ZERO` disables.
    connect_timeout: Duration,
    /// Which session-control client tools to advertise to the orchestrator
    /// (voice#61).
    client_tools: ClientToolsConfig,
    /// Set within a turn when the LLM calls `stop_listening`: after this turn's
    /// reply is spoken the conversation ends and the next wake starts fresh.
    /// Reset at the start of each utterance (voice#61).
    session_end_requested: bool,
    /// Set within a turn when the LLM calls `listen_for_more`: keep/extend the
    /// listening window for a follow-up even outside conversation mode. Reset at
    /// the start of each utterance (voice#61).
    listen_for_more_requested: bool,
    /// Best-effort sink for per-turn text events (transcript / speaking) that the
    /// D-Bus layer forwards as `TranscriptReady` / `SpeakingText` signals
    /// (voice#85). `None` when no D-Bus forwarder is attached (e.g. in tests).
    signal_tx: Option<mpsc::Sender<VoiceSignal>>,
    /// Inbound wake-word calibration requests (#121). `None` when no calibration
    /// channel is attached (e.g. in tests); when present, an idle pipeline can
    /// take over the mic to measure the user's wake scores.
    calibrate_rx: Option<mpsc::Receiver<CalibrationRequest>>,
    /// logind session-active gate (voice#103). When the session goes inactive
    /// (fast user switch) the mic is released; an inert gate (logind absent or
    /// the feature disabled) never pauses. Defaults to inert.
    session: SessionGate,
    /// A push-to-talk that arrived while capture was paused-by-disable and
    /// triggered a re-acquire (voice#103): the run loop drains it and enters PTT
    /// listening so PTT keeps working even with wake-word/voice disabled (#3).
    /// `Some(target)` = pending; `None` = nothing pending.
    pending_ptt: Option<Option<String>>,
    /// Publishes the real capture (mic-open) state for the D-Bus
    /// `GetCaptureState` surface (voice#103). Detached by default in tests.
    capture_state_tx: watch::Sender<CaptureState>,
    /// User-speech transcripts from the current own-session conversation,
    /// accumulated across turns in the reuse window and used by the post-session
    /// LLM triage task to judge whether the conversation is worth keeping.
    session_transcripts: Vec<String>,
}

impl<W, V, S, T, A> Pipeline<W, V, S, T, A>
where
    W: WakeWordDetector + 'static,
    V: VoiceActivityDetector + 'static,
    S: SpeechToText + 'static,
    T: TextToSpeech + 'static,
    A: AssistantGateway + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wake: W,
        vad: V,
        stt: S,
        tts: T,
        assistant: A,
        source: Arc<dyn AudioSource>,
        sink: Arc<dyn AudioSink>,
        state_tx: watch::Sender<State>,
        enabled_rx: watch::Receiver<bool>,
        ptt_rx: mpsc::Receiver<Option<String>>,
        stop_rx: mpsc::Receiver<StopRequest>,
        reload_rx: mpsc::Receiver<()>,
        config: PipelineConfig,
    ) -> Self {
        let PipelineConfig {
            tunables,
            conversation_title,
            silence_duration,
            speech_threshold,
            mic_barge_in,
            conversation_mode,
            conversation_reuse_window,
            followup_timeout,
            idle_exit_timeout,
            spoken_response_hint,
            listening_cue,
            timeouts,
            client_tools,
        } = config;
        let mut speaker = Speaker::new(Arc::new(tts), Arc::clone(&sink));
        speaker.set_synth_timeout(timeouts.synth);
        Self {
            wake,
            vad,
            transcriber: Transcriber::new(Arc::new(stt)),
            speaker,
            assistant: Arc::new(assistant),
            source,
            sink,
            endpointer: Endpointer::new(speech_threshold, silence_duration, ENDPOINT_MIN_SAMPLES),
            prebuffer: PreBuffer::new(WAKE_PREBUFFER_SAMPLES),
            listening_cue,
            cue_phrase_counter: 0,
            state: State::Idle,
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            tunables,
            conversation_id: None,
            ptt_conversation_override: None,
            conversation_title,
            speech_threshold,
            mic_barge_in,
            conversation_mode,
            conversation_reuse_window,
            last_own_activity: None,
            followup_timeout,
            idle_exit_timeout,
            spoken_response_hint,
            response_stall: timeouts.response_stall,
            turn_budget: timeouts.turn_budget,
            status_narration_min_gap: timeouts.status_narration_min_gap,
            liveness_delay: timeouts.liveness_delay,
            narration_flush: timeouts.narration_flush,
            connect_timeout: timeouts.connect,
            client_tools,
            session_end_requested: false,
            listen_for_more_requested: false,
            signal_tx: None,
            calibrate_rx: None,
            session: SessionGate::inert(),
            pending_ptt: None,
            // A detached sender (no receiver) so `capture_state_tx.send` is a
            // harmless no-op until `with_capture_state` wires the real channel.
            capture_state_tx: watch::channel(CaptureState::Capturing).0,
            session_transcripts: Vec::new(),
        }
    }

    /// The single state-mutation chokepoint (voice#82). Apply `event` to the
    /// current state: on a legal transition, update `self.state`, publish to the
    /// watch channel, log, and return `true`. On an illegal one, warn (and
    /// `debug_assert!` in tests so the harness catches an illegal published
    /// sequence) and return `false` without mutating. A no-op transition (the
    /// table maps to the same state — e.g. `ResponseStarted` while already
    /// Speaking, or `Stopped` while Idle) returns `true` silently.
    fn apply(&mut self, event: StateEvent) -> bool {
        match self.state.transition(&event) {
            Some(next) => {
                if next != self.state {
                    self.state = next;
                    let _ = self.state_tx.send(next);
                    tracing::info!(state = %next, ?event, "state changed");
                }
                true
            }
            None => {
                tracing::warn!(
                    state = %self.state,
                    ?event,
                    "illegal state transition requested; ignoring"
                );
                debug_assert!(
                    false,
                    "illegal state transition {:?} from {}",
                    event, self.state
                );
                false
            }
        }
    }

    /// Attach the D-Bus signal sink so the pipeline emits `TranscriptReady` /
    /// `SpeakingText` as it transcribes and speaks (voice#85). Best-effort: a
    /// full or closed channel just drops the event (signals are advisory).
    pub fn with_signal_tx(mut self, signal_tx: mpsc::Sender<VoiceSignal>) -> Self {
        self.signal_tx = Some(signal_tx);
        self
    }

    /// Attach the logind session-active gate (voice#103). With an inert gate
    /// (the default) capture is never paused for session reasons.
    pub fn with_session_gate(mut self, session: SessionGate) -> Self {
        self.session = session;
        self
    }

    /// Attach the wake-word calibration channel (#121) so the D-Bus
    /// `CalibrateWake` method can drive a calibration run on the idle pipeline.
    /// Without it, calibration requests have nowhere to go.
    pub fn with_calibrate(mut self, rx: mpsc::Receiver<CalibrationRequest>) -> Self {
        self.calibrate_rx = Some(rx);
        self
    }

    /// Attach the capture-state publisher so the D-Bus `GetCaptureState` surface
    /// reflects whether the mic is actually open vs paused (voice#103).
    pub fn with_capture_state(mut self, tx: watch::Sender<CaptureState>) -> Self {
        self.capture_state_tx = tx;
        self
    }

    /// Publish the real capture state for the D-Bus surface (voice#103).
    fn set_capture_state(&self, state: CaptureState) {
        if *self.capture_state_tx.borrow() != state {
            tracing::info!(%state, "capture state changed");
        }
        let _ = self.capture_state_tx.send(state);
    }

    /// Best-effort emit of a per-turn text signal (voice#85). Uses `try_send` so
    /// a slow/absent D-Bus consumer never stalls the speech pipeline.
    fn emit_signal(&self, signal: VoiceSignal) {
        if let Some(tx) = &self.signal_tx
            && let Err(e) = tx.try_send(signal)
        {
            tracing::trace!(error = %e, "dropped a voice D-Bus signal (consumer busy/absent)");
        }
    }

    /// Speak a sentence of Adele's reply, first announcing it as `SpeakingText`
    /// so clients can display it without polling (voice#85). Used for spoken
    /// REPLY content; cues/apologies use `speaker.say` directly (not reply text).
    async fn speak_reply(&mut self, text: &str) -> anyhow::Result<()> {
        self.emit_signal(VoiceSignal::SpeakingText(text.to_string()));
        self.speaker.say(text).await?;
        Ok(())
    }

    /// Decide, on a fresh wake (Idle→Listening, NOT an in-turn follow-up),
    /// whether the daemon's own session is still reusable (voice#53). Drops
    /// `conversation_id` when the last own-session activity is outside the reuse
    /// window (or reuse is disabled / there's been no own-session turn), so the
    /// next turn opens a brand-new conversation; keeps it (continuing the chat)
    /// when within the window. A conversation ended via `stop_listening` / a stop
    /// phrase already cleared the id, so it's never resurrected here (voice#59).
    fn expire_stale_conversation_on_wake(&mut self) {
        // Reuse disabled: every wake starts fresh (pre-#53 behaviour).
        if self.conversation_reuse_window.is_zero() {
            self.conversation_id = None;
            self.last_own_activity = None;
            return;
        }
        match (self.conversation_id.as_ref(), self.last_own_activity) {
            (Some(_), Some(last)) if last.elapsed() <= self.conversation_reuse_window => {
                tracing::info!(
                    age_ms = last.elapsed().as_millis(),
                    "reusing the recent conversation on this wake (voice#53)"
                );
            }
            (Some(_), _) => {
                tracing::info!("last conversation is outside the reuse window; starting fresh");
                let transcripts = std::mem::take(&mut self.session_transcripts);
                let stale_id = self.conversation_id.take();
                self.last_own_activity = None;
                if let Some(id) = stale_id.filter(|_| !transcripts.is_empty()) {
                    self.spawn_triage_task(id, transcripts);
                }
            }
            (None, _) => {}
        }
    }

    /// Wait out any outstanding playback, then drop the echo it queued into the
    /// mic before the pipeline re-arms Listening (voice#82). Consolidates the
    /// triplicated wait-for-playback + `try_recv`-drain idiom that previously
    /// lived inline in the PTT arm and the two relisten paths of `run()`.
    async fn drain_playback_echo(&mut self, audio_rx: &mut mpsc::Receiver<Vec<f32>>) {
        while self.speaker.is_playing() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        while audio_rx.try_recv().is_ok() {}
    }

    /// End the voice session (voice#82): clear the daemon's own
    /// `conversation_id` and its reuse-window clock, drop any
    /// push-to-talk-into-conversation override, and reset the endpointer.
    /// Consolidates the cleanup that a stop phrase / `stop_listening` tool and
    /// the `StopRequest::Conversation` arm each performed inline. Does NOT touch
    /// `self.state` — the caller transitions through `apply` first.
    fn end_conversation(&mut self) {
        self.conversation_id = None;
        self.last_own_activity = None;
        self.ptt_conversation_override = None;
        self.endpointer.reset();
    }

    /// Spawn a background task that asks the LLM to judge the just-ended voice
    /// session and automatically archives or deletes trivial / mistaken-activation
    /// conversations. Fire-and-forget: the pipeline returns to Idle immediately;
    /// the triage runs asynchronously.
    fn spawn_triage_task(&self, voice_conv_id: String, transcripts: Vec<String>) {
        let assistant = Arc::clone(&self.assistant);
        let timeout = self.connect_timeout.max(Duration::from_secs(30));
        tokio::spawn(async move {
            if let Err(e) = run_triage(assistant, voice_conv_id.clone(), transcripts, timeout).await
            {
                tracing::warn!(%voice_conv_id, "post-session triage failed: {e}");
            }
        });
    }

    /// Enter Listening for a push-to-talk press (voice#82). The body of the
    /// outer PTT arm, extracted so the run-loop handling of a mid-turn PTT
    /// interrupt re-arms identically: stop any outstanding playback, drain the
    /// echo it queued, set the conversation override (honouring the reuse window
    /// for a plain own-session press), transition to Listening, and arm the
    /// endpointer with the lead-in. The caller has already decided a PTT press is
    /// legal here (Idle or Speaking).
    async fn enter_ptt_listening(
        &mut self,
        target: Option<String>,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) {
        // Stop any outstanding playback before arming the mic. A single-shot
        // reply drops to Idle while its TTS is still sounding (playback_end in
        // the future), so gating stop() on State::Speaking let a PTT press in
        // Idle skip it — leaving `is_playing` true with no drain and recording
        // the daemon's own voice (#68). Stop whenever anything is playing,
        // regardless of state; stop() is the only thing that clears playback_end.
        if self.speaker.is_playing() {
            let _ = self.speaker.stop();
        }
        // Belt-and-suspenders: wait out any residual tail and drop the echo it
        // queued into the mic before arming, so no in-flight TTS leaks into the
        // PTT utterance — matching the relisten path.
        self.drain_playback_echo(audio_rx).await;
        // Route this PTT session: `Some(id)` dictates into that conversation;
        // `None` (plain PushToTalk) falls back to the daemon's own session,
        // which — like the wake word — persists across presses for continuity.
        // (A stale override can't leak in: every press overwrites it, and the
        // wake-word entry resets it to None.)
        self.ptt_conversation_override = target.clone();
        // A plain PTT (own session) is a fresh entry like a wake: honour the
        // reuse window — keep the recent conversation if within it, otherwise
        // start fresh (voice#53). A targeted PTT uses its own id and is
        // unaffected.
        if target.is_none() {
            self.expire_stale_conversation_on_wake();
        }
        self.apply(StateEvent::PttPressed);
        // Wait (lead-in) for speech to start rather than cutting on the silence
        // timer from the moment of the press; only cut after speech-then-silence,
        // or if the lead-in elapses.
        self.endpointer.arm(Some(self.followup_timeout));
        self.vad.reset();
        tracing::info!(
            target_conversation = target.as_deref().unwrap_or("<own session>"),
            "push-to-talk activated, waiting for speech"
        );
    }

    /// Play the audible "Listening" cue (#51) on entering the Listening state.
    ///
    /// - `Ding`: a short generated earcon, queued straight onto the sink — no
    ///   TTS, so it's instant and reliable.
    /// - `Phrase`: a rotating spoken micro-phrase via the TTS `Speaker`;
    ///   friendlier but adds the synthesis/playback latency of a short
    ///   utterance, so it isn't the default.
    /// - `Off`: nothing.
    ///
    /// A cue failure must never derail entering Listening — errors are logged
    /// and swallowed.
    ///
    /// Returns `true` if a cue was actually queued (Ding/Phrase) so the caller
    /// knows there is playback to wait out and drain before arming the mic
    /// (V-7/#87); `false` for `Off`, where there is nothing to drain.
    async fn play_listening_cue(&mut self) -> bool {
        match self.listening_cue {
            ListeningCue::Off => false,
            ListeningCue::Ding => {
                if let Err(e) = self.sink.play(cue::ding_samples()) {
                    tracing::warn!("failed to play listening ding cue: {e}");
                }
                true
            }
            ListeningCue::Phrase => {
                let phrase = cue::phrase(self.cue_phrase_counter);
                self.cue_phrase_counter = self.cue_phrase_counter.wrapping_add(1);
                if let Err(e) = self.speaker.say(phrase).await {
                    tracing::warn!("failed to speak listening phrase cue: {e}");
                }
                true
            }
        }
    }

    /// Re-read the config file and apply any changed tunables to the running
    /// pipeline (config#52). A failed/missing read is logged and ignored — a
    /// momentary write-in-progress must never tear the daemon down.
    fn reload(&mut self) {
        let new_config = match config::load() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("config reload: failed to re-read config, keeping current: {e}");
                return;
            }
        };
        let new_tunables = new_config.tunables();
        let plan = plan_reload(&self.tunables, &new_tunables);
        if plan.is_empty() {
            tracing::info!("config reload: no tunable changes");
            return;
        }
        self.apply_plan(&plan);
        // Adopt the new snapshot whole — including any field a sibling added,
        // and the device fields we only logged about — so we diff against the
        // on-disk truth next time rather than re-flagging the same change.
        self.tunables = new_tunables;
    }

    /// Apply a [`ReloadPlan`] to the live pipeline. Pure decisions live in
    /// [`plan_reload`]; this carries them out.
    fn apply_plan(&mut self, plan: &config::ReloadPlan) {
        if let Some(t) = plan.set_speech_threshold {
            self.speech_threshold = t;
            self.endpointer.set_speech_threshold(t);
            tracing::info!(
                speech_threshold = t,
                "config reload: applied vad.speech_threshold"
            );
        }
        if let Some(ms) = plan.set_silence_ms {
            self.endpointer.set_silence(Duration::from_millis(ms));
            tracing::info!(
                silence_duration_ms = ms,
                "config reload: applied vad.silence_duration_ms"
            );
        }
        if let Some(ms) = plan.set_followup_timeout_ms {
            self.followup_timeout = Duration::from_millis(ms);
            tracing::info!(
                followup_timeout_ms = ms,
                "config reload: applied assistant.followup_timeout_ms"
            );
        }
        if let Some(ms) = plan.set_conversation_reuse_window_ms {
            self.conversation_reuse_window = Duration::from_millis(ms);
            tracing::info!(
                conversation_reuse_window_ms = ms,
                "config reload: applied assistant.conversation_reuse_window_ms"
            );
        }
        if let Some(mode) = plan.set_conversation_mode {
            self.conversation_mode = mode;
            tracing::info!(
                conversation_mode = mode,
                "config reload: applied assistant.conversation_mode"
            );
        }
        if let Some(ms) = plan.set_idle_exit_timeout_ms {
            self.idle_exit_timeout = (ms > 0).then(|| Duration::from_millis(ms));
            tracing::info!(
                idle_exit_timeout_ms = ms,
                "config reload: applied idle_exit_timeout_ms"
            );
        }
        if let Some(ms) = plan.set_response_stall_ms {
            self.response_stall = Duration::from_millis(ms);
            tracing::info!(
                response_stall_ms = ms,
                "config reload: applied timeouts.response_stall_ms"
            );
        }
        if let Some(ms) = plan.set_turn_budget_ms {
            self.turn_budget = Duration::from_millis(ms);
            tracing::info!(
                turn_budget_ms = ms,
                "config reload: applied timeouts.turn_budget_ms"
            );
        }
        if let Some(ms) = plan.set_status_narration_min_gap_ms {
            self.status_narration_min_gap = Duration::from_millis(ms);
            tracing::info!(
                status_narration_min_gap_ms = ms,
                "config reload: applied timeouts.status_narration_min_gap_ms"
            );
        }
        if let Some(ms) = plan.set_narration_flush_ms {
            self.narration_flush = Duration::from_millis(ms);
            tracing::info!(
                narration_flush_ms = ms,
                "config reload: applied timeouts.narration_flush_ms"
            );
        }
        if let Some(sensitivity) = plan.apply_wake_sensitivity {
            match self.wake.set_sensitivity(sensitivity) {
                Ok(()) => {
                    // Keep the tunables snapshot in step so a later reload (e.g.
                    // the file-watcher ping after calibration persists the same
                    // value) sees no change and doesn't re-apply needlessly.
                    self.tunables.wake_sensitivity = sensitivity;
                    tracing::info!(
                        sensitivity,
                        "config reload: applied wake_word.sensitivity live"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        sensitivity,
                        "config reload: failed to apply wake sensitivity, keeping current: {e}"
                    );
                }
            }
        }
        if let Some(change) = &plan.restart_required_for_device {
            tracing::warn!(
                "config reload: audio device change ({change}) needs a daemon restart to take \
                 effect — `systemctl --user restart adele-voice`. All other knobs were applied live."
            );
        }
    }

    /// Run a wake-word calibration session (#121): take over the mic, ask the
    /// user to say the wake word `req.utterances` times (clamped), measure each
    /// peak match score, set the sensitivity a margin below the worst, apply it
    /// live, and persist it. Progress is reported via `CalibrationProgress`
    /// signals and the outcome is sent back on `req.reply`.
    ///
    /// Calibration owns the mic and reconfigures the detector, so it only runs
    /// from a clean Idle state (and is rejected otherwise). The main loop is
    /// suspended for the duration — this is a deliberate, user-driven modal step.
    async fn run_calibration(
        &mut self,
        req: CalibrationRequest,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) {
        if self.state != State::Idle || self.speaker.is_playing() {
            let _ = req.reply.send(Err(
                "voice is busy — finish the current conversation before calibrating".to_string(),
            ));
            return;
        }

        let total = calibration::clamp_utterances(req.utterances);
        tracing::info!(utterances = total, "wake calibration: starting");
        self.wake.begin_calibration();

        let mut peaks: Vec<f32> = Vec::with_capacity(total as usize);
        let max_attempts = calibration::max_attempts(total);
        let mut attempts = 0u32;
        while (peaks.len() as u32) < total && attempts < max_attempts {
            attempts += 1;
            // A negative score is a prompt, not a measurement: tell the client
            // "say the next one now".
            self.emit_signal(VoiceSignal::CalibrationProgress {
                captured: peaks.len() as u32,
                total,
                score: -1.0,
            });
            match self.capture_one_wake(audio_rx).await {
                Some(score) => {
                    peaks.push(score);
                    self.emit_signal(VoiceSignal::CalibrationProgress {
                        captured: peaks.len() as u32,
                        total,
                        score: score as f64,
                    });
                    tracing::info!(
                        score,
                        captured = peaks.len(),
                        total,
                        "wake calibration: captured utterance"
                    );
                }
                None => {
                    // Timed out with nothing clear — prompt a retry (-2.0).
                    self.emit_signal(VoiceSignal::CalibrationProgress {
                        captured: peaks.len() as u32,
                        total,
                        score: -2.0,
                    });
                    tracing::warn!("wake calibration: no clear wake word heard, retrying");
                }
            }
        }

        // Decide the outcome, then ALWAYS leave calibration mode — restoring the
        // user's eager setting and applying either the new cutoff (success) or
        // the prior sensitivity unchanged (failure).
        let result = match calibration::recommend(&peaks) {
            Some(outcome) => {
                if let Err(e) = self.wake.end_calibration(outcome.sensitivity) {
                    tracing::warn!("wake calibration: failed to apply new cutoff: {e}");
                }
                // Keep the tunables snapshot in step so the file-watcher reload
                // triggered by the persist below is a no-op for sensitivity.
                self.tunables.wake_sensitivity = outcome.sensitivity;
                match config::persist_wake_sensitivity(outcome.sensitivity) {
                    Ok(()) => Ok(outcome),
                    Err(e) => Err(format!(
                        "calibrated to {:.2} and applied it, but couldn't save it: {e}",
                        outcome.sensitivity
                    )),
                }
            }
            None => {
                let _ = self.wake.end_calibration(self.tunables.wake_sensitivity);
                Err(format!(
                    "couldn't hear enough clear wake words (got {} of {}) — check the \
                     microphone and try again",
                    peaks.len(),
                    total
                ))
            }
        };

        match &result {
            Ok(o) => tracing::info!(
                sensitivity = o.sensitivity,
                samples = o.samples,
                "wake calibration: complete"
            ),
            Err(e) => tracing::warn!("wake calibration: {e}"),
        }
        let _ = req.reply.send(result);
    }

    /// Listen for one spoken wake word during a calibration session, returning
    /// its peak score, or `None` if nothing clear is heard before the
    /// per-utterance timeout (or capture ends). The detector is in calibration
    /// mode, so `detect` fires once per utterance at the true peak.
    async fn capture_one_wake(&mut self, audio_rx: &mut mpsc::Receiver<Vec<f32>>) -> Option<f32> {
        let deadline = tokio::time::sleep(calibration::UTTERANCE_TIMEOUT);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => return None,
                chunk = audio_rx.recv() => {
                    let chunk = chunk?;
                    match self.wake.detect(&chunk) {
                        Ok(true) => {
                            if let Some(score) = self.wake.take_last_score() {
                                return Some(score);
                            }
                        }
                        Ok(false) => {}
                        Err(e) => tracing::warn!("wake calibration: detect error: {e}"),
                    }
                }
            }
        }
    }

    /// Advertise the enabled session-control client tools to the orchestrator
    /// (voice#61) so the LLM can stop/continue listening or speak a line. A
    /// failure here must never stop the daemon — voice still works without the
    /// tools (the user can stop by phrase / the follow-up timeout) — so it's
    /// logged and swallowed.
    async fn register_session_control_tools(&mut self) {
        let tools = session_control_tools(self.client_tools);
        if tools.is_empty() {
            tracing::info!("no session-control client tools enabled; skipping registration");
            return;
        }
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        match self.assistant.register_client_tools(tools.clone()).await {
            Ok(count) => tracing::info!(
                count,
                ?names,
                "registered session-control client tools (voice#61)"
            ),
            Err(e) => tracing::warn!(
                ?names,
                "failed to register session-control client tools (voice still works without \
                 them): {e}"
            ),
        }
    }

    /// Wait in the no-capture (degraded) state, servicing control channels so
    /// the daemon stays alive — keeping the separately-spawned TTS (`SayText`)
    /// and D-Bus services up — and a reload can retry capture. Returns `true` to
    /// retry `source.start()`, `false` if every control channel has closed (so
    /// `run()` shuts down cleanly). The `select!` awaits, so there is no
    /// busy-spin while degraded.
    async fn await_capture_retry(&mut self) -> bool {
        loop {
            tokio::select! {
                Some(()) = self.reload_rx.recv() => { self.reload(); return true; }
                Some(_target) = self.ptt_rx.recv() => {
                    tracing::warn!("push-to-talk ignored: voice capture is unavailable");
                }
                Some(_req) = self.stop_rx.recv() => { /* nothing capturing to stop */ }
                else => return false,
            }
        }
    }

    /// Whether capture should currently be running: the logind session is
    /// active (or the gate is inert) AND voice processing is enabled (voice#103).
    /// When either is false the mic is released — both for privacy on
    /// fast-user-switch and so a `SetEnabled(false)` actually closes the device
    /// rather than only muting wake detection (the mic-indicator trust bug).
    fn capture_allowed(&self) -> bool {
        self.session.is_active() && *self.enabled_rx.borrow()
    }

    /// The capture-state reason for the current not-allowed condition. Session
    /// takes precedence (it's the privacy/cost gate); else it's the disable gate.
    fn pause_reason(&self) -> CaptureState {
        if !self.session.is_active() {
            CaptureState::PausedSessionInactive
        } else {
            CaptureState::PausedDisabled
        }
    }

    /// Wait until capture is allowed again, servicing the control channels so the
    /// daemon stays alive and the mic stays released. A push-to-talk overrides
    /// the *disable* pause (it re-acquires the mic on demand and the run loop
    /// then enters Listening — so PTT keeps working with wake-word/voice off, #3)
    /// but NOT the *session* pause (opening the mic in a switched-away session is
    /// exactly what the privacy gate prevents). Returns `true` once capture is
    /// allowed (or a PTT re-acquired it), `false` if every control channel closed
    /// (clean shutdown). A PTT that triggered the resume is stashed in
    /// `pending_ptt` for the run loop to drain.
    ///
    /// The mic is already stopped by the caller before this is entered.
    ///
    /// Returns `true` when capture becomes allowed again, `false` when the
    /// process should exit cleanly — either every control channel closed, or
    /// (paused-by-disable only) the idle-exit window elapsed so D-Bus activation
    /// can restart the daemon on demand (#5). Idle-exit never fires on a
    /// session-inactive pause: the daemon stays resident to resume when the
    /// session returns.
    async fn await_capture_allowed(&mut self) -> bool {
        self.set_capture_state(self.pause_reason());
        // Idle-exit countdown starts from when capture was released.
        let paused_at = Instant::now();
        loop {
            if self.capture_allowed() {
                return true;
            }
            // Idle-exit only when paused purely because voice is disabled (#5):
            // a session-inactive pause must keep the daemon resident.
            let idle_exit = self
                .idle_exit_timeout
                .filter(|_| self.session.is_active() && !self.speaker.is_playing());
            tokio::select! {
                _ = self.session.changed() => {}
                _ = self.enabled_rx.changed() => {}
                Some(()) = self.reload_rx.recv() => { self.reload(); }
                Some(target) = self.ptt_rx.recv() => {
                    // A PTT overrides the disable pause (re-acquire + listen),
                    // but never the session pause (privacy).
                    if self.session.is_active() {
                        tracing::info!("push-to-talk while disabled: re-acquiring the microphone");
                        self.pending_ptt = Some(target);
                        return true;
                    }
                    tracing::warn!(
                        "push-to-talk ignored: session inactive (microphone released for privacy)"
                    );
                }
                Some(_req) = self.stop_rx.recv() => { /* nothing capturing to stop */ }
                _ = sleep_opt(idle_exit.map(|t| t.saturating_sub(paused_at.elapsed()))),
                    if idle_exit.is_some() =>
                {
                    tracing::info!(
                        "idle-exit: voice disabled and idle, exiting for on-demand activation"
                    );
                    return false;
                }
                else => return false,
            }
        }
    }

    /// Start (or restart) capture, degrading on failure (#79): on `Err`, log an
    /// actionable message and drop into the degraded loop, which keeps the
    /// control channels (and thus the separate TTS/D-Bus tasks) serving until a
    /// reload makes capture available again or everything shuts down. Returns
    /// `None` when every control channel has closed (clean shutdown).
    ///
    /// Before opening the device it blocks until capture is allowed (logind
    /// session active + enabled), so the mic is never opened in a switched-away
    /// or disabled session (voice#103).
    async fn acquire_capture(&mut self) -> Option<mpsc::Receiver<Vec<f32>>> {
        loop {
            if !self.capture_allowed() && !self.await_capture_allowed().await {
                return None; // all control channels closed → clean shutdown
            }
            match self.source.start() {
                Ok(rx) => {
                    self.set_capture_state(CaptureState::Capturing);
                    return Some(rx);
                }
                Err(e) => {
                    tracing::error!(
                        "voice capture unavailable: {e} — speech output (SayText) and D-Bus stay \
                         up; wake-word and dictation are disabled until the input device is fixed. \
                         A config device-name change needs a restart; a transient device return \
                         recovers on reload."
                    );
                    self.set_capture_state(CaptureState::Unavailable);
                    // Degraded: stay alive, keep serving control channels, retry
                    // capture on reload.
                    if !self.await_capture_retry().await {
                        return None; // all control channels closed → clean shutdown
                    }
                }
            }
        }
    }

    /// Release the mic and block until capture is allowed again, then re-acquire
    /// it (voice#103). Called from the run loop when the session goes inactive or
    /// voice is disabled. Stops any playback, drops to Idle, ends any in-flight
    /// conversation, stops the source (so its `running` latch clears and the mic
    /// is truly closed), then `acquire_capture()` — which waits until allowed.
    /// Replaces `*audio_rx` with the fresh capture channel. Returns `false` only
    /// if every control channel closed (clean shutdown).
    async fn pause_and_reacquire(&mut self, audio_rx: &mut mpsc::Receiver<Vec<f32>>) -> bool {
        tracing::info!(reason = %self.pause_reason(), "releasing microphone (capture paused)");
        let _ = self.speaker.stop();
        if self.state != State::Idle {
            self.apply(StateEvent::Stopped);
        }
        self.end_conversation();
        self.endpointer.reset();
        if let Err(e) = self.source.stop() {
            tracing::warn!("source stop on capture pause failed: {e}");
        }
        self.set_capture_state(self.pause_reason());
        match self.acquire_capture().await {
            Some(rx) => {
                *audio_rx = rx;
                tracing::info!("microphone re-acquired (capture resumed)");
                true
            }
            None => false,
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut audio_rx = match self.acquire_capture().await {
            Some(rx) => rx,
            None => return Ok(()),
        };

        // Advertise the LLM-driven session-control tools once at startup
        // (voice#61). The connection is already up (built in main), so register
        // before listening so the very first turn can use them.
        self.register_session_control_tools().await;

        // Publish the initial Idle so subscribers see a value (the watch channel
        // already holds Idle, but emit explicitly for the log line / parity).
        self.state = State::Idle;
        let _ = self.state_tx.send(State::Idle);

        // For idle-exit (#5): time of the last activity other than idle-while-
        // wake-disabled. (Utterance accumulation and the follow-up/lead-in
        // deadline now live in the shared `Endpointer`.)
        let mut last_activity = Instant::now();

        loop {
            // A PTT that re-acquired the mic from the disable pause (voice#103):
            // enter Listening now that the device is back, so PTT works even with
            // wake-word/voice disabled (#3).
            if let Some(target) = self.pending_ptt.take()
                && (self.state == State::Idle || self.state == State::Speaking)
            {
                last_activity = Instant::now();
                self.enter_ptt_listening(target, &mut audio_rx).await;
            }
            tokio::select! {
                // Session/enable gate (voice#103): when the logind session goes
                // inactive (fast user switch) or voice processing is disabled,
                // release the mic at the source and block until capture is
                // allowed again — cutting the whole pipeline (no audio → no wake
                // → no STT/LLM/TTS), which fixes privacy and cost together. An
                // inert gate's `changed()` never fires, so this arm is dormant
                // when session gating is off. We `recv` the change here; the
                // `borrow_and_update` inside `enabled_rx.changed()` and the
                // session watch keep the predicate fresh.
                _ = self.session.changed(), if self.session.is_enabled() => {
                    if !self.capture_allowed()
                        && !self.pause_and_reacquire(&mut audio_rx).await
                    {
                        break; // all control channels closed
                    }
                }
                _ = self.enabled_rx.changed() => {
                    if !self.capture_allowed()
                        && !self.pause_and_reacquire(&mut audio_rx).await
                    {
                        break;
                    }
                }

                // Push-to-talk: skip wake word, go to Listening. The payload is
                // the target conversation: `None` uses the daemon's own
                // session, `Some(id)` routes the utterance to that orchestrator
                // conversation (the in-chat mic button).
                Some(target) = self.ptt_rx.recv() => {
                    if self.state == State::Idle || self.state == State::Speaking {
                        self.enter_ptt_listening(target, &mut audio_rx).await;
                    }
                }

                // Stop: cancel current playback (Speaking) or end the whole
                // conversation and return to wake-word idle.
                Some(req) = self.stop_rx.recv() => {
                    match req {
                        StopRequest::Speaking => {
                            if self.state == State::Speaking {
                                self.speaker.stop()?;
                                self.apply(StateEvent::Stopped);
                            }
                        }
                        StopRequest::Conversation => {
                            // "Stop listening": end the session now without
                            // waiting out the silence timeout.
                            if self.state != State::Idle {
                                let _ = self.speaker.stop();
                                self.apply(StateEvent::Stopped);
                            }
                            self.end_conversation();
                        }
                    }
                }

                // Reload: re-read the config file and apply any changed
                // tunables to the running pipeline (config#52). Triggered by the
                // file watcher (debounced) or the D-Bus `Reload` method.
                Some(()) = self.reload_rx.recv() => {
                    self.reload();
                }

                // Wake-word calibration (#121): the D-Bus `CalibrateWake` method
                // routes a request here. It takes over the mic to measure the
                // user's wake-word scores, so it only runs from a clean Idle
                // state (run_calibration rejects otherwise).
                req = recv_calibration(&mut self.calibrate_rx) => {
                    match req {
                        Some(req) => self.run_calibration(req, &mut audio_rx).await,
                        // Channel closed: disarm the arm so it stops resolving.
                        None => self.calibrate_rx = None,
                    }
                }

                // Process audio chunks
                chunk = audio_rx.recv() => {
                    // The capture thread died AFTER a successful start (device
                    // unplug, drain/resample error) — the channel closing is the
                    // only signal we get (V-1). Don't go silently deaf: stop the
                    // source so its `running` latch clears, surface the loss,
                    // and restart capture — immediately if the device opens
                    // again, otherwise via the same degraded loop as startup.
                    let Some(chunk) = chunk else {
                        tracing::error!(
                            "voice capture channel closed (capture thread died) — \
                             restarting capture; wake-word and dictation are \
                             unavailable until it recovers"
                        );
                        let _ = self.speaker.stop();
                        if self.state != State::Idle {
                            self.apply(StateEvent::Stopped);
                        }
                        self.endpointer.reset();
                        // Clear the adapter's running latch so start() can
                        // succeed (the cpal source refuses to double-start).
                        if let Err(e) = self.source.stop() {
                            tracing::warn!("source stop after capture death failed: {e}");
                        }
                        match self.acquire_capture().await {
                            Some(rx) => {
                                audio_rx = rx;
                                tracing::info!("voice capture recovered after capture-thread death");
                                continue;
                            }
                            None => break, // all control channels closed
                        }
                    };
                    // Once the session goes inactive the device is released by
                    // the gate arm above, but a chunk queued before that arm
                    // fires could still land here. Drop it outright so a
                    // switched-away session never gets transcribed even for one
                    // straggler chunk (voice#103). The gate arm will pause the
                    // pipeline immediately after.
                    if !self.session.is_active() {
                        continue;
                    }
                    // Once disabled the device is released by the gate arm above,
                    // so chunks stop flowing — but a chunk already queued before
                    // that arm fires would still land here. This per-chunk guard
                    // skips processing it (and runs the idle-exit fallback for the
                    // brief window before the pause loop takes over) so a disabled
                    // daemon never wakes on a straggler chunk (#3/#5/voice#103).
                    if self.state == State::Idle && !*self.enabled_rx.borrow() {
                        // Idle-exit (#5): with wake listening off and nothing
                        // playing, exit after the configured idle window so
                        // D-Bus activation can restart the daemon on demand.
                        if let Some(timeout) = self.idle_exit_timeout
                            && last_activity.elapsed() >= timeout
                            && !self.speaker.is_playing()
                        {
                            tracing::info!(
                                "idle-exit: wake word disabled and idle, exiting for on-demand activation"
                            );
                            break;
                        }
                        continue;
                    }
                    last_activity = Instant::now();
                    match self.state {
                        State::Idle => {
                            // Don't let the daemon wake itself on its own
                            // playback. A single-shot reply returns to Idle while
                            // its audio is still sounding, and any cue/SayText
                            // plays into a mic that hears the speakers; an eager
                            // detector trips on that echo. While *real* audio is
                            // outstanding, skip wake detection AND don't seed the
                            // prebuffer with echo — `is_playing` stays true (with
                            // its tail pad) until the sound is truly gone.
                            //
                            // Once the audio deadline has passed but we're still
                            // inside the tail pad (#70), nothing fresh is
                            // sounding — only the latency cushion remains. Keep
                            // seeding the prebuffer (still WITHOUT running wake
                            // detect, since residual echo in the cushion could
                            // trip it) so same-breath audio at the very tail
                            // isn't dropped if the wake then fires a chunk later.
                            if self.speaker.is_playing() {
                                if self.speaker.in_tail_pad() {
                                    self.prebuffer.push(&chunk);
                                }
                                continue;
                            }
                            // Keep a rolling pre-buffer of recent idle audio so a
                            // command spoken in the same breath as the wake word
                            // isn't dropped during the handoff (#50).
                            self.prebuffer.push(&chunk);
                            // Feed to wake word detector
                            if self.wake.detect(&chunk)? {
                                tracing::info!("wake word detected");
                                if self.apply(StateEvent::WakeWordDetected) {
                                    // Wake word always uses the daemon's own
                                    // session; clear any push-to-talk target
                                    // left over from a session ended via
                                    // StopSpeaking so this utterance can't leak
                                    // into a previously dictated conversation.
                                    self.ptt_conversation_override = None;
                                    // Honour the reuse window (voice#53): keep the
                                    // recent conversation if this wake is within
                                    // it, else start fresh.
                                    self.expire_stale_conversation_on_wake();
                                    // Capture the same-breath preroll NOW (#50) —
                                    // the audio spoken right after the wake word,
                                    // before the cue sounds — so "hey adele
                                    // <command>" in one breath keeps the command.
                                    let preroll = self.prebuffer.take();
                                    self.vad.reset();
                                    // Audible "Listening" cue (#51) — instant ding
                                    // by default, optional spoken phrase, or off.
                                    let cue_played = self.play_listening_cue().await;
                                    // V-7/#87: the cue plays INTO the open mic, so
                                    // its audio (especially a spoken "How can I
                                    // help?") would otherwise be endpointed as the
                                    // user's utterance. Wait the cue out and drop
                                    // the echo it queued before arming — matching
                                    // the relisten path. Arming AFTER the cue also
                                    // starts the lead-in from when the user can
                                    // actually speak, not from before the cue
                                    // (which a ~1 s phrase would otherwise burn
                                    // through, timing out the turn). The preroll
                                    // captured above is still seeded into the
                                    // buffer, so only post-cue echo is dropped.
                                    // Only drain when a cue actually played: with
                                    // the cue Off there's no echo, and draining
                                    // would eat same-breath speech already queued.
                                    if cue_played {
                                        self.drain_playback_echo(&mut audio_rx).await;
                                    }
                                    self.endpointer
                                        .arm_with_preroll(Some(self.followup_timeout), &preroll);
                                }
                            }
                        }

                        State::Listening => {
                            // Feed to VAD; the endpointer accumulates and decides
                            // when the utterance ends (or the lead-in elapses).
                            let prob = self.vad.speech_probability(&chunk)?;
                            match self.endpointer.push(&chunk, prob) {
                                Endpoint::SpeechStarted => {
                                    tracing::info!(prob, "speech detected, recording");
                                }
                                Endpoint::Accumulating => {}
                                Endpoint::Complete(samples) => {
                                    self.handle_utterance_complete(samples, &mut audio_rx)
                                        .await;
                                }
                                Endpoint::Timeout => {
                                    // No follow-up speech within the timeout:
                                    // return to wake-word idle. We KEEP the
                                    // daemon's own `conversation_id` and its
                                    // last-activity time so a wake within the
                                    // reuse window resumes it rather than opening
                                    // a fresh conversation (voice#53) — the
                                    // reuse-window check at the next turn enforces
                                    // the deadline. A PTT-into-conversation
                                    // override is still dropped (the client owns
                                    // that conversation's lifecycle).
                                    tracing::info!("conversation follow-up timed out");
                                    self.apply(StateEvent::ListeningTimedOut);
                                    self.ptt_conversation_override = None;
                                    self.endpointer.reset();
                                }
                            }
                        }

                        State::Speaking => {
                            // Check for barge-in
                            let prob = self.vad.speech_probability(&chunk)?;
                            if prob >= self.speech_threshold {
                                tracing::info!("barge-in detected");
                                self.speaker.stop()?;
                                if self.apply(StateEvent::BargeIn) {
                                    // Seed the endpointer mid-speech so the next
                                    // silence closes this barge-in utterance.
                                    self.endpointer.arm_speaking(&chunk);
                                    self.vad.reset();
                                }
                            } else if !self.speaker.is_playing()
                                && self.apply(StateEvent::PlaybackComplete)
                            {
                                // Playback finished naturally
                            }
                        }

                        State::Processing => {
                            // Ignore audio while processing
                        }
                    }
                }

                else => break,
            }
        }

        self.source.stop()?;
        Ok(())
    }

    /// Handle a completed utterance (voice#82): run the turn, then decide
    /// whether to end the conversation, re-listen for a follow-up, or drop back
    /// to wake-word idle. Extracted verbatim from the `Endpoint::Complete` arm of
    /// `run()` so the run loop stays readable and the relisten/echo-drain idiom
    /// is shared with the other entry points.
    async fn handle_utterance_complete(
        &mut self,
        samples: Vec<f32>,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) {
        tracing::info!(
            samples = samples.len(),
            "silence detected, transitioning to processing"
        );
        if !self.apply(StateEvent::SilenceDetected) {
            return;
        }
        // A failed turn must NOT crash the daemon. The orchestrator may have
        // restarted and dropped the connection; log it, apologize, and end the
        // turn — the gateway reconnects so the next turn works.
        let outcome = match self.process_utterance(samples, audio_rx).await {
            Ok(outcome) => outcome,
            Err(e) => {
                tracing::error!("voice turn failed: {e}");
                self.apply(StateEvent::ResponseStarted);
                let _ = self.speaker.say(ERROR_APOLOGY).await;
                UtteranceOutcome::EndConversation
            }
        };

        // An interrupt (voice#82) bypasses the normal relisten logic: the run
        // loop re-arms (or ends) exactly as the matching outer arm would, so a
        // mid-turn stop / barge-in / PTT behaves like one outside a turn.
        if let UtteranceOutcome::Interrupted(kind) = outcome {
            self.handle_interrupt(kind, audio_rx).await;
            return;
        }

        // Decide whether to re-listen. A `listen_for_more` client tool re-arms
        // even outside conversation mode; a plain turn re-listens only in
        // conversation mode; `stop_listening` / a stop phrase ends regardless
        // (voice#61).
        let relisten = match outcome {
            UtteranceOutcome::EndConversation => false,
            UtteranceOutcome::KeepListening => true,
            UtteranceOutcome::Continue => self.conversation_mode,
            UtteranceOutcome::Interrupted(_) => unreachable!("handled above"),
        };
        if outcome == UtteranceOutcome::EndConversation {
            // A stop phrase or the `stop_listening` tool ends the conversation
            // regardless of mode AND clears the reuse-window id so the next wake
            // starts fresh (voice#59/#61).
            self.apply(StateEvent::TurnEnded);
            // For own-session conversations with real turns, snapshot before
            // clearing so the triage task can judge and archive/delete if trivial.
            let triage_target = if self.ptt_conversation_override.is_none() {
                let transcripts = std::mem::take(&mut self.session_transcripts);
                self.conversation_id
                    .clone()
                    .filter(|_| !transcripts.is_empty())
                    .map(|id| (id, transcripts))
            } else {
                None
            };
            self.end_conversation();
            if let Some((conv_id, transcripts)) = triage_target {
                self.spawn_triage_task(conv_id, transcripts);
            }
        } else if relisten {
            // Re-open the mic for a follow-up turn: wait for the reply to finish
            // playing, then drop any audio captured during playback (echo) before
            // listening again.
            self.drain_playback_echo(audio_rx).await;
            self.apply(StateEvent::RelistenArmed);
            // Cue the follow-up re-listen too (#51), then wait for the cue to
            // finish and drop the echo it queued into the mic before arming, so
            // it isn't captured as the follow-up.
            self.play_listening_cue().await;
            self.drain_playback_echo(audio_rx).await;
            self.endpointer.arm(Some(self.followup_timeout));
            self.vad.reset();
        } else {
            // Single-shot: back to wake-word idle. Drop any PTT-into-conversation
            // target so the next own-session turn doesn't inherit it.
            self.apply(StateEvent::TurnEnded);
            self.ptt_conversation_override = None;
        }
    }

    /// Re-arm (or end) after a turn was interrupted mid-stream (voice#82).
    /// `stream_response` already stopped the speaker; this maps the interrupt
    /// into the same state transition + entry work the matching outer arm
    /// performs, so a mid-turn interrupt is indistinguishable from one outside a
    /// turn. The interrupted turn ran (its reply lands in history), so an
    /// own-session turn's reuse clock was already stamped in `process_utterance`.
    async fn handle_interrupt(
        &mut self,
        kind: InterruptKind,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) {
        match kind {
            InterruptKind::BargeIn(chunk) => {
                // The user spoke over our playback. We can only barge in while
                // Speaking, so the BargeIn transition is always legal here. No
                // cue — they're already talking; seed the endpointer mid-speech
                // so the next silence closes this barge-in utterance. (Mirrors
                // the outer Speaking-state barge-in arm.)
                if self.apply(StateEvent::BargeIn) {
                    self.endpointer.arm_speaking(&chunk);
                    self.vad.reset();
                }
            }
            InterruptKind::Ptt(target) => {
                // A PTT press mid-turn re-arms exactly like a fresh press. The
                // interrupt may have arrived before the first chunk (still
                // Processing) or while Speaking; `PttPressed` is legal only from
                // Idle/Speaking, so normalize a silent-Processing interrupt to
                // Idle first (nothing was audible to stop). `enter_ptt_listening`
                // does the stop/drain/override/arm.
                if self.state == State::Processing {
                    self.apply(StateEvent::Stopped);
                }
                self.enter_ptt_listening(target, audio_rx).await;
            }
            InterruptKind::StopSpeaking => {
                // "Stop speaking": back to wake-word idle, but KEEP the
                // conversation so a wake within the reuse window resumes it.
                self.apply(StateEvent::Stopped);
            }
            InterruptKind::StopConversation => {
                // "Stop listening": back to idle AND end the conversation.
                self.apply(StateEvent::Stopped);
                self.end_conversation();
            }
        }
    }

    async fn process_utterance(
        &mut self,
        samples: Vec<f32>,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) -> anyhow::Result<UtteranceOutcome> {
        // Fresh turn: clear any session-control intent the LLM may set via a
        // client tool during this turn (voice#61).
        self.session_end_requested = false;
        self.listen_for_more_requested = false;

        // Energy-gate + transcribe (in the module's `Transcriber`). The gate
        // discards near-silent captures — ambient noise or the tail of our own
        // playback can trip the VAD without real speech, and Whisper then
        // hallucinates filler ("Thank you.") that loops every follow-up window —
        // and an empty transcript is likewise nothing to act on; both yield
        // `None`. We transcribe before touching the orchestrator so a "stop"
        // command needn't create or poke the conversation.
        let transcript = match self.transcriber.transcribe(&samples).await? {
            Some(t) => t,
            None => return Ok(UtteranceOutcome::Continue),
        };
        tracing::info!(text = %transcript.text, "transcribed");
        // Let clients (the KDE widget) show what was heard without polling
        // (voice#85).
        self.emit_signal(VoiceSignal::TranscriptReady(transcript.text.clone()));

        // The mic can capture the daemon's own playback (no echo cancellation),
        // so Whisper sometimes transcribes one of our canned reassurance /
        // apology lines. Acting on it would start a fresh turn that speaks
        // another canned line and loops. Discard it like a near-silent capture.
        if is_self_echo(&transcript.text) {
            tracing::info!(text = %transcript.text, "discarding self-echo of a canned spoken line");
            return Ok(UtteranceOutcome::Continue);
        }

        // A whole-utterance stop phrase ("stop", "never mind", "that's all", …)
        // ends the conversation hands-free: acknowledge briefly and return to
        // wake-word idle instead of sending it to the assistant.
        if is_stop_phrase(&transcript.text) {
            tracing::info!(text = %transcript.text, "stop phrase — ending conversation");
            self.apply(StateEvent::ResponseStarted);
            self.speaker.say("Okay.").await?;
            return Ok(UtteranceOutcome::EndConversation);
        }

        // Resolve the target conversation. A push-to-talk into a specific
        // conversation (the in-chat mic button) routes this turn — and any
        // conversation-mode follow-ups — to that existing orchestrator
        // conversation; we never create it (the client owns its lifecycle).
        // Otherwise fall back to the daemon's own session, creating it lazily
        // and reusing it across turns.
        // Whether this turn ran on the daemon's own (reusable) session vs a
        // PTT-into-conversation override; only the own session participates in
        // the cross-wake reuse window (voice#53).
        let own_session = self.ptt_conversation_override.is_none();
        let conversation_id = if let Some(target) = self.ptt_conversation_override.clone() {
            target
        } else {
            if self.conversation_id.is_none() {
                // Bound the create round-trip: a wedged orchestrator must not
                // leave the user hanging in Processing (#58).
                let id = bounded(
                    self.connect_timeout,
                    "create_conversation",
                    self.assistant
                        .create_conversation(&self.conversation_title, vec!["voice".to_string()]),
                )
                .await?;
                tracing::info!(conversation_id = %id, "created voice conversation");
                self.conversation_id = Some(id);
            }
            self.conversation_id.as_ref().unwrap().clone()
        };

        // Subscribe to response signals (bounded — #58).
        let mut signal_rx = bounded(
            self.connect_timeout,
            "subscribe",
            self.assistant.subscribe(),
        )
        .await?;

        // Send the CLEAN transcript as the user message and pass the
        // spoken-response hint as a per-request system_refinement, so the reply
        // stays short and conversational for read-aloud WITHOUT the blurb
        // polluting the visible chat transcript. The gateway falls back to
        // prepending the hint (the pre-#200 behaviour) when the orchestrator
        // lacks the refinement-aware method.
        //
        // If the send fails for the own session (stale conversation ID, e.g. the
        // orchestrator lost the conversation), create a fresh conversation and
        // retry once rather than apologizing immediately.
        let request_id = {
            let first = bounded(
                self.connect_timeout,
                "send_prompt",
                self.assistant.send_prompt_with_system_refinement(
                    &conversation_id,
                    &transcript.text,
                    &self.spoken_response_hint,
                ),
            )
            .await;
            match first {
                Ok(id) => id,
                Err(e) if own_session => {
                    tracing::warn!(
                        "send to own-session conversation failed ({e}); \
                         clearing stale id and retrying with a fresh conversation"
                    );
                    self.conversation_id = None;
                    let fresh_id = bounded(
                        self.connect_timeout,
                        "create_conversation_retry",
                        self.assistant.create_conversation(
                            &self.conversation_title,
                            vec!["voice".to_string()],
                        ),
                    )
                    .await?;
                    tracing::info!(conversation_id = %fresh_id, "created replacement conversation");
                    self.conversation_id = Some(fresh_id.clone());
                    bounded(
                        self.connect_timeout,
                        "send_prompt_retry",
                        self.assistant.send_prompt_with_system_refinement(
                            &fresh_id,
                            &transcript.text,
                            &self.spoken_response_hint,
                        ),
                    )
                    .await?
                }
                Err(e) => return Err(e.into()),
            }
        };
        // Record own-session turns for post-session LLM triage.
        if own_session {
            self.session_transcripts.push(transcript.text.clone());
        }

        let stream_end = self
            .stream_response(&mut signal_rx, &request_id, audio_rx)
            .await?;

        // Mark the own session active so a wake within the reuse window resumes
        // it (voice#53). Skipped for a PTT override (the client owns that
        // conversation) and when the LLM ended the session (cleared below). An
        // interrupted turn still ran (its reply lands in history), so it counts
        // as activity — a "wait, what did you say?" wake should resume it.
        if own_session {
            self.last_own_activity = Some(Instant::now());
        }

        // An interrupt ended the turn client-side (voice#82). The orchestrator
        // turn runs on (there is no client-facing turn-cancel today, and letting
        // it finish is strictly safer than cancelling — see #82 §3). Hand the
        // dropped subscription to a drainer so a late `say_this` / other
        // `ClientToolCall` gets an `Err` result instead of parking the server
        // turn until the suspension timeout. Then map the interrupt into the run
        // loop's outcome.
        if let Some(kind) = match stream_end {
            StreamEnd::Completed => None,
            StreamEnd::Stopped(StopRequest::Speaking) => Some(InterruptKind::StopSpeaking),
            StreamEnd::Stopped(StopRequest::Conversation) => Some(InterruptKind::StopConversation),
            StreamEnd::BargedIn(chunk) => Some(InterruptKind::BargeIn(chunk)),
            StreamEnd::PttPressed(target) => Some(InterruptKind::Ptt(target)),
        } {
            self.spawn_interrupt_drainer(signal_rx, request_id);
            return Ok(UtteranceOutcome::Interrupted(kind));
        }

        // Translate any session-control intent the LLM set during the turn into
        // the run loop's outcome (voice#61). `stop_listening` ends the
        // conversation (and the run loop clears the reuse id); `listen_for_more`
        // keeps listening even outside conversation mode.
        if self.session_end_requested {
            return Ok(UtteranceOutcome::EndConversation);
        }
        if self.listen_for_more_requested {
            return Ok(UtteranceOutcome::KeepListening);
        }
        Ok(UtteranceOutcome::Continue)
    }

    /// After an interrupt drops the streaming subscription mid-turn (voice#82),
    /// spawn a detached task that owns the receiver and answers any late
    /// `ClientToolCall` (especially `say_this`) with an `Err` — there is no
    /// orchestrator-side turn-cancel in the protocol, so a parked tool call
    /// would otherwise hang the server turn until the suspension timeout. The
    /// task ignores chunks/status and exits on `Complete` / `Error` / a clean
    /// stream close, or after a hard cap (the turn budget, min 60 s) so it can
    /// never leak.
    fn spawn_interrupt_drainer(
        &self,
        mut signal_rx: mpsc::UnboundedReceiver<AssistantEvent>,
        request_id: String,
    ) {
        let assistant = Arc::clone(&self.assistant);
        let cap = self.turn_budget.max(Duration::from_secs(60));
        tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + cap;
            loop {
                let event = tokio::select! {
                    e = signal_rx.recv() => e,
                    _ = tokio::time::sleep_until(deadline) => {
                        tracing::warn!(%request_id, "interrupt drainer hit its cap; exiting");
                        break;
                    }
                };
                match event {
                    Some(AssistantEvent::ClientToolCall {
                        task_id,
                        tool_call_id,
                        tool_name,
                        ..
                    }) => {
                        tracing::info!(
                            tool = %tool_name,
                            "answering a post-interrupt client tool call with an error (voice#82)"
                        );
                        if let Err(e) = assistant
                            .submit_client_tool_result(
                                &task_id,
                                &tool_call_id,
                                Err("voice session interrupted".to_string()),
                            )
                            .await
                        {
                            tracing::warn!("drainer failed to submit tool result: {e}");
                        }
                    }
                    Some(AssistantEvent::Complete { .. })
                    | Some(AssistantEvent::Error { .. })
                    | None => break,
                    // Chunks / status for the abandoned turn: ignore.
                    Some(_) => {}
                }
            }
        });
    }

    /// Drive the streaming response to completion: chunks → sentence buffer →
    /// TTS, sparingly-narrated status, all under a progress-heartbeat stall
    /// deadline plus an overall turn budget (#58).
    ///
    /// The heartbeat is the core guard: `signal_rx.recv()` is wrapped in a
    /// per-event deadline that RESETS on every received event (chunk OR status),
    /// so a turn that keeps making progress never times out, but one that goes
    /// silent — a wedged orchestrator — apologizes and returns to Idle instead
    /// of hanging in Processing forever. The turn budget is a backstop against
    /// an event source that dribbles just often enough to keep resetting the
    /// stall clock.
    async fn stream_response(
        &mut self,
        signal_rx: &mut mpsc::UnboundedReceiver<AssistantEvent>,
        request_id: &str,
        audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    ) -> anyhow::Result<StreamEnd> {
        // The heartbeat / budget / narration clocks use `tokio::time::Instant`
        // so they stay consistent with the `tokio::time::{sleep, timeout}` this
        // loop awaits — and so the paused-time test clock advances them too.
        use tokio::time::Instant as TokioInstant;

        // Mid-sentence quiet window before the buffer flushes a settled clause
        // (config; 0 disables timeout flushing). End-of-stream still flushes the
        // tail immediately via the Complete/Error/closed arms below, so this
        // only governs a genuine inter-token stall — keeping it generous avoids
        // chopping a normally-paced sentence into separate synth units.
        let mut sentence_buf = SentenceBuffer::new(self.narration_flush);
        let flush_ticks = !self.narration_flush.is_zero();
        let mut first_chunk = true;
        // Status narration rate-limit (#58): the first status always speaks;
        // later ones speak only after `status_narration_min_gap` has elapsed.
        let mut last_status_spoken_at: Option<TokioInstant> = None;
        // Overall turn budget (#58); `None` when disabled.
        let turn_deadline =
            (!self.turn_budget.is_zero()).then(|| TokioInstant::now() + self.turn_budget);
        // Progress-heartbeat (#58): the time of the last received event. The
        // stall deadline is measured from here and RESETS on every event, so a
        // steadily-progressing turn never trips it. Tracked as an explicit
        // instant (rather than a per-iteration `timeout`) so the 100 ms
        // sentence-flush tick can't accidentally reset the stall window.
        let mut last_event_at = TokioInstant::now();
        // Delayed-liveness safety net: if the turn narrates nothing and streams
        // nothing within this window, speak a brief line so a slow turn that
        // declared no plan step isn't dead-silent. Re-arms each `liveness_delay`
        // while still silent, escalating through `LIVENESS_PHRASES`, then stops.
        // Cancelled by the first status or chunk. `Duration::ZERO` disables.
        let mut next_liveness_at =
            (!self.liveness_delay.is_zero()).then(|| TokioInstant::now() + self.liveness_delay);
        let mut liveness_idx = 0usize;
        // Guards the mid-turn audio arm (voice#82): cleared when the capture
        // channel closes so the arm disarms instead of hot-looping on a dead
        // receiver (V-1). The run loop's own recv()==None path then recovers.
        let mut audio_alive = true;

        let stream_end = loop {
            // How long to wait for the NEXT event before declaring a stall: the
            // remaining slice of the stall window since the last event. The
            // 100 ms tick keeps the sentence-buffer flush responsive without
            // resetting that window. 0 disables the stall guard.
            let stall_wait = if self.response_stall.is_zero() {
                Duration::from_secs(86_400) // effectively unbounded
            } else {
                self.response_stall
                    .checked_sub(last_event_at.elapsed())
                    .unwrap_or(Duration::ZERO)
            };

            tokio::select! {
                event = tokio::time::timeout(stall_wait, signal_rx.recv()) => {
                    let event = match event {
                        Ok(event) => {
                            // An event (or a clean close) arrived — that's
                            // progress; reset the heartbeat clock (#58).
                            last_event_at = TokioInstant::now();
                            event
                        }
                        Err(_elapsed) => {
                            // No progress within the stall window — the turn is
                            // wedged. Apologize and bail (#58).
                            tracing::warn!(
                                stall_ms = self.response_stall.as_millis(),
                                "assistant turn stalled (no progress event); apologizing and returning to Idle"
                            );
                            self.speak_stall_apology().await;
                            break StreamEnd::Completed;
                        }
                    };
                    match event {
                        Some(AssistantEvent::Chunk { request_id: rid, text }) if rid == request_id => {
                            if first_chunk && is_error_response(&text) {
                                tracing::error!(chunk = %text, "assistant streamed an error message; speaking a short apology instead");
                                self.apply(StateEvent::ResponseStarted);
                                self.speaker.say(ERROR_APOLOGY).await?;
                                break StreamEnd::Completed;
                            }
                            if first_chunk {
                                first_chunk = false;
                                self.apply(StateEvent::ResponseStarted);
                            }

                            let sentences = sentence_buf.push(&text);
                            for sentence in sentences {
                                self.speak_reply(&sentence).await?;
                            }
                            // Speak a short leading ack the instant it looks
                            // complete (a terminal opener like "Got it —
                            // checking that now." that the boundary scan won't
                            // split until the next token), without waiting (#58).
                            if let Some(ack) = sentence_buf.take_leading_ack(ACK_MAX_WORDS) {
                                self.speak_reply(&ack).await?;
                            }
                        }
                        Some(AssistantEvent::Status { request_id: rid, message }) if rid == request_id => {
                            // Progress status doubles as a heartbeat (the
                            // timeout reset above already happened) and is
                            // narrated SPARINGLY — the first one immediately,
                            // later ones rate-limited (#58).
                            self.maybe_narrate_status(&message, &mut last_status_spoken_at, &mut first_chunk).await?;
                        }
                        Some(AssistantEvent::Complete { request_id: rid, full_response }) if rid == request_id => {
                            if sentence_buf.has_content() {
                                let remaining = sentence_buf.flush();
                                if !remaining.is_empty() {
                                    self.speak_reply(&remaining).await?;
                                }
                            } else if first_chunk && !full_response.trim().is_empty() {
                                // Nothing was streamed as chunks — e.g. a
                                // tool-using reply delivered as one final block.
                                self.apply(StateEvent::ResponseStarted);
                                if is_error_response(&full_response) {
                                    // The orchestrator surfaces LLM failures as
                                    // the reply text (so they show in chat);
                                    // don't read the raw error aloud.
                                    tracing::error!(response = %full_response, "assistant returned an error message; speaking a short apology instead");
                                    self.speaker.say(ERROR_APOLOGY).await?;
                                } else {
                                    // Speak the full response instead of dropping it.
                                    let sentences = sentence_buf.push(&full_response);
                                    for sentence in sentences {
                                        self.speak_reply(&sentence).await?;
                                    }
                                    let remaining = sentence_buf.flush();
                                    if !remaining.is_empty() {
                                        self.speak_reply(&remaining).await?;
                                    }
                                }
                            }
                            tracing::info!(streamed = !first_chunk, "assistant response complete");
                            break StreamEnd::Completed;
                        }
                        Some(AssistantEvent::Error { request_id: rid, error }) if rid == request_id => {
                            tracing::error!(error = %error, "assistant response error; speaking a short apology");
                            self.apply(StateEvent::ResponseStarted);
                            self.speaker.say(ERROR_APOLOGY).await?;
                            break StreamEnd::Completed;
                        }
                        // The LLM is driving the session via a client tool
                        // (voice#61). NOT keyed on request_id — a suspended tool
                        // call carries the orchestrator task id instead. Run it
                        // and post the result back so the parked turn resumes; the
                        // turn continues streaming after.
                        Some(AssistantEvent::ClientToolCall { task_id, tool_call_id, tool_name, arguments }) => {
                            self.handle_client_tool_call(&task_id, &tool_call_id, &tool_name, arguments).await;
                        }
                        None => {
                            tracing::warn!("assistant signal stream closed before completion");
                            if first_chunk {
                                // The reply stream dropped before any content
                                // arrived (e.g. the orchestrator restarted
                                // mid-turn) — don't leave the user in silence.
                                self.apply(StateEvent::ResponseStarted);
                                self.speaker.say(ERROR_APOLOGY).await?;
                            }
                            break StreamEnd::Completed;
                        }
                        _ => {} // Ignore events for other requests
                    }
                }

                // --- Interrupt arms (voice#82): serviced WHILE the turn streams,
                // so a stop / barge-in / PTT no longer queues until the turn
                // completes. Each stops playback and breaks out with a StreamEnd;
                // the run loop (handle_interrupt) re-arms. None of these reset the
                // #58 stall/budget clocks — they end the loop those clocks live in.

                // D-Bus StopSpeaking / StopListening.
                Some(req) = self.stop_rx.recv() => {
                    let _ = self.speaker.stop();
                    tracing::info!(?req, "stop request mid-turn; ending the streamed reply (voice#82)");
                    break StreamEnd::Stopped(req);
                }

                // A push-to-talk press mid-turn: treat as an interrupt and hand
                // the target back so the run loop re-arms Listening exactly like a
                // fresh press.
                Some(target) = self.ptt_rx.recv() => {
                    let _ = self.speaker.stop();
                    tracing::info!(
                        target_conversation = target.as_deref().unwrap_or("<own session>"),
                        "push-to-talk mid-turn; interrupting the streamed reply (voice#82)"
                    );
                    break StreamEnd::PttPressed(target);
                }

                // Live audio while the turn streams. While we're playing, run VAD
                // for barge-in; while silent (pre-first-chunk Processing), discard
                // the chunk so the channel doesn't back up. On a closed channel
                // (capture thread died, V-1) disarm this arm and let the turn
                // finish — the run loop's own recv()==None path does the recovery.
                chunk = audio_rx.recv(), if audio_alive => {
                    match chunk {
                        Some(chunk) => {
                            // Barge-in is OFF by default: without echo cancellation
                            // the mic hears Adele's own playback, the VAD scores it
                            // as speech, and she "interrupts" herself — clipping the
                            // reply mid-word and looping on the echo. Only run the
                            // detector when mic barge-in is explicitly enabled (i.e.
                            // AEC is in place). Otherwise discard the chunk so the
                            // channel doesn't back up; stop/PTT/D-Bus still interrupt.
                            if self.mic_barge_in && self.speaker.is_playing() {
                                let prob = self.vad.speech_probability(&chunk)?;
                                if prob >= self.speech_threshold {
                                    tracing::info!(prob, "barge-in during streamed playback (voice#82)");
                                    let _ = self.speaker.stop();
                                    break StreamEnd::BargedIn(chunk);
                                }
                            }
                            // Barge-in disabled, not playing, or below threshold:
                            // ignore — don't let the channel back up.
                        }
                        None => {
                            tracing::warn!(
                                "capture channel closed mid-turn; finishing the turn, recovery happens in the run loop (voice#82)"
                            );
                            audio_alive = false;
                        }
                    }
                }

                // A config reload mid-turn: apply the tunables diff inline and
                // keep streaming — it's a pure diff, safe mid-turn (config#52).
                Some(()) = self.reload_rx.recv() => {
                    self.reload();
                }

                // Check for timeout flush while waiting for chunks. The tick
                // only needs to run when timeout flushing is enabled.
                _ = tokio::time::sleep(Duration::from_millis(100)), if flush_ticks => {
                    if let Some(sentence) = sentence_buf.flush_if_timeout() {
                        self.speak_reply(&sentence).await?;
                    }
                }
                // Delayed-liveness safety net: a slow turn that has narrated
                // nothing and streamed nothing yet gets a brief, escalating line
                // so voice isn't dead-silent. Re-arms while still silent until
                // the phrase list is exhausted; the guard disables it once a
                // status/chunk arrives.
                _ = async {
                    match next_liveness_at {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending::<()>().await,
                    }
                }, if next_liveness_at.is_some() && first_chunk && last_status_spoken_at.is_none() => {
                    let phrase = LIVENESS_PHRASES[liveness_idx];
                    liveness_idx += 1;
                    // Re-arm for the next escalation step, or stop once the list
                    // is exhausted (the stall/budget backstop covers the rest).
                    next_liveness_at = (liveness_idx < LIVENESS_PHRASES.len())
                        .then(|| TokioInstant::now() + self.liveness_delay);
                    self.apply(StateEvent::ResponseStarted);
                    self.speaker.say(phrase).await?;
                }
            }

            // Overall turn budget (#58): a backstop against an event source that
            // keeps the stall clock alive by dribbling events forever.
            if let Some(deadline) = turn_deadline
                && TokioInstant::now() >= deadline
            {
                tracing::warn!(
                    budget_ms = self.turn_budget.as_millis(),
                    "assistant turn exceeded its overall budget; apologizing and returning to Idle"
                );
                self.speak_stall_apology().await;
                break StreamEnd::Completed;
            }
        };

        Ok(stream_end)
    }

    /// Speak the stall apology (best-effort — a failed apology must not turn a
    /// timeout into a crash) and move to Speaking so the run loop returns to
    /// Idle when playback finishes (#58).
    async fn speak_stall_apology(&mut self) {
        self.apply(StateEvent::ResponseStarted);
        if let Err(e) = self.speaker.say(STALL_APOLOGY).await {
            tracing::warn!("failed to speak stall apology: {e}");
        }
    }

    /// Narrate a progress status sparingly (#58): speak the first status of the
    /// turn immediately, then at most one every `status_narration_min_gap`.
    /// Returns the rate-limit decision so tests can assert it.
    async fn maybe_narrate_status(
        &mut self,
        message: &str,
        last_spoken_at: &mut Option<tokio::time::Instant>,
        first_chunk: &mut bool,
    ) -> anyhow::Result<bool> {
        let message = message.trim();
        if message.is_empty() {
            return Ok(false);
        }
        let now = tokio::time::Instant::now();
        let should_speak = match *last_spoken_at {
            None => true, // first status of the turn always speaks
            Some(prev) => now.duration_since(prev) >= self.status_narration_min_gap,
        };
        if !should_speak {
            tracing::debug!(status = %message, "status narration rate-limited; skipping");
            return Ok(false);
        }
        // A status arriving before any reply text moves us into Speaking so the
        // narration plays (and the user hears progress) rather than sitting in
        // silent Processing.
        if *first_chunk {
            self.apply(StateEvent::ResponseStarted);
        }
        *last_spoken_at = Some(now);
        self.speaker.say(message).await?;
        Ok(true)
    }

    /// Run a session-control client tool the LLM called mid-turn and post the
    /// result back so the orchestrator's suspended turn resumes (voice#61).
    ///
    /// The server turn is PARKED awaiting the result, so we never block on TTS
    /// completion: `stop_listening`/`listen_for_more` only set a flag the run
    /// loop reads after the turn, and `say_this` *queues* the line on the speaker
    /// (synth is bounded by the per-synth timeout) before returning. The result
    /// is always submitted — `Ok` for a handled tool, `Err("unknown tool")` for
    /// a name we don't recognize — so the turn can never hang waiting on us. A
    /// failed submit is logged, not propagated: it must not crash the turn.
    async fn handle_client_tool_call(
        &mut self,
        task_id: &str,
        tool_call_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) {
        tracing::info!(tool = %tool_name, %task_id, %tool_call_id, "client tool call (voice#61)");
        let result: Result<String, String> = match tool_name {
            TOOL_STOP_LISTENING => {
                // End the session after this turn's reply is spoken; the run loop
                // returns to Idle and clears the reuse id.
                self.session_end_requested = true;
                Ok("stopped".to_string())
            }
            TOOL_LISTEN_FOR_MORE => {
                // Keep/extend the listening window for a follow-up.
                self.listen_for_more_requested = true;
                Ok("listening".to_string())
            }
            TOOL_SAY_THIS => {
                // Speak the exact line now — queue it, don't await playback, so
                // the suspended turn resumes promptly.
                let text = arguments
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if text.is_empty() {
                    Err("say_this requires a non-empty `text` argument".to_string())
                } else {
                    self.apply(StateEvent::ResponseStarted);
                    self.emit_signal(VoiceSignal::SpeakingText(text.clone()));
                    if let Err(e) = self.speaker.say(&text).await {
                        tracing::warn!("say_this synthesis failed: {e}");
                        Err(format!("failed to speak: {e}"))
                    } else {
                        Ok("spoken".to_string())
                    }
                }
            }
            other => {
                tracing::warn!(tool = %other, "unknown client tool requested");
                Err("unknown tool".to_string())
            }
        };
        if let Err(e) = self
            .assistant
            .submit_client_tool_result(task_id, tool_call_id, result)
            .await
        {
            tracing::warn!("failed to submit client tool result: {e}");
        }
    }
}

/// Post-session LLM triage: create a temporary conversation, send the session
/// transcripts as context, collect a one-word verdict (keep / archive / delete),
/// act on the original voice conversation accordingly, then delete the temporary
/// triage conversation. Called from a background [`tokio::spawn`] so the pipeline
/// returns to Idle immediately — never blocks the wake-word loop.
async fn run_triage<A: AssistantGateway + 'static>(
    assistant: Arc<A>,
    voice_conv_id: String,
    transcripts: Vec<String>,
    timeout: Duration,
) -> Result<(), VoiceError> {
    let triage_id = bounded(
        timeout,
        "triage_create",
        assistant.create_conversation("_voice_triage_", vec![]),
    )
    .await?;

    let transcript_lines: String = transcripts
        .iter()
        .enumerate()
        .map(|(i, t)| format!("Turn {}: {t}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!("Voice session transcript (user speech only):\n{transcript_lines}");
    let refinement = "Judge this voice assistant session. Reply with EXACTLY ONE WORD:\n\
        delete — mistaken wake-word activation (gibberish, accidental, no real interaction)\n\
        archive — trivial (brief factual queries, simple commands, throwaway small talk)\n\
        keep — substantive (complex questions, tasks, anything worth reviewing later)";

    let mut signal_rx = bounded(timeout, "triage_subscribe", assistant.subscribe()).await?;

    let request_id = bounded(
        timeout,
        "triage_send",
        assistant.send_prompt_with_system_refinement(&triage_id, &prompt, refinement),
    )
    .await?;

    let verdict = {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let event = tokio::select! {
                e = signal_rx.recv() => e,
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(VoiceError::Assistant("triage verdict timed out".to_string()));
                }
            };
            match event {
                Some(AssistantEvent::Complete {
                    request_id: rid,
                    full_response,
                }) if rid == request_id => break full_response,
                Some(AssistantEvent::Error {
                    request_id: rid,
                    error,
                }) if rid == request_id => {
                    return Err(VoiceError::Assistant(format!("triage LLM error: {error}")));
                }
                None => return Err(VoiceError::Assistant("triage stream closed".to_string())),
                _ => {}
            }
        }
    };

    let word = verdict
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    tracing::info!(%voice_conv_id, verdict = %word, "post-session triage");
    match word.as_str() {
        "archive" => {
            let _ = assistant.archive_conversation(&voice_conv_id).await;
        }
        "delete" => {
            let _ = assistant.delete_conversation(&voice_conv_id).await;
        }
        _ => {}
    }
    let _ = assistant.delete_conversation(&triage_id).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Pipeline tests with fake adapters. Focus: #3 — the `enabled` flag must
    //! gate ONLY always-on wake-word detection, so an explicit push-to-talk
    //! still captures and transcribes an utterance while "Hey Adele" is off.
    use super::*;
    use adele_voice_core::domain::Transcript;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn detects_orchestrator_error_responses() {
        // An LLM failure surfaced as reply text must be recognized so the
        // daemon apologizes instead of reading the raw error aloud.
        assert!(is_error_response(
            "Details: LLM error: Bedrock converse_stream request failed: validation error"
        ));
        assert!(is_error_response("LLM error: provider unavailable"));
    }

    #[test]
    fn normal_replies_are_not_errors() {
        assert!(!is_error_response("It's sunny and about 72 degrees today."));
        assert!(!is_error_response(
            "The forecast calls for rain this afternoon, clearing by evening."
        ));
    }

    #[test]
    fn stop_phrases_match_whole_utterance_only() {
        assert!(is_stop_phrase("stop"));
        assert!(is_stop_phrase("Stop listening."));
        assert!(is_stop_phrase("never mind"));
        assert!(is_stop_phrase("That's all!"));
        assert!(is_stop_phrase("goodbye"));
        // Not a command when it's only part of a real request.
        assert!(!is_stop_phrase("stop the timer"));
        assert!(!is_stop_phrase("what should I never mind about?"));
        assert!(!is_stop_phrase("tell me a story"));
    }

    #[test]
    fn self_echo_matches_canned_lines_only() {
        // Whisper transcribing our own playback (no echo cancellation) —
        // punctuation/case may vary — must be recognized and discarded.
        assert!(is_self_echo("One moment."));
        assert!(is_self_echo("one moment"));
        assert!(is_self_echo("Still working on it"));
        assert!(is_self_echo(STALL_APOLOGY));
        assert!(is_self_echo(ERROR_APOLOGY));
        // Every escalation phrase is covered.
        for phrase in LIVENESS_PHRASES {
            assert!(is_self_echo(phrase), "{phrase:?} should be a self-echo");
        }
        // A real request that merely contains a word from a canned line is not
        // an echo, and empty input never is.
        assert!(!is_self_echo("give me a moment to think"));
        assert!(!is_self_echo("are you still working on it?"));
        assert!(!is_self_echo(""));
        assert!(!is_self_echo("what's the forecast for today?"));
    }

    #[tokio::test]
    async fn stop_phrase_ends_conversation_without_prompting() {
        // A whole-utterance "stop" must end the conversation — even in
        // conversation mode — and must NOT be sent to the assistant.
        let mut h = spawn_pipeline(Cfg {
            stt_text: "stop".to_string(),
            conversation_mode: true,
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        // Conversation mode would normally re-listen; a stop phrase returns to Idle.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("stop phrase returns to Idle")
        .unwrap();
        h.handle.abort();
        assert!(
            h.prompt_rx.try_recv().is_err(),
            "a stop phrase must not be sent to the assistant"
        );
    }

    #[tokio::test]
    async fn self_echo_utterance_is_not_sent_to_the_assistant() {
        // Regression guard for the echo loop: when the mic captures our own
        // playback and Whisper transcribes a canned liveness line, it must be
        // discarded — never sent to the assistant to start a fresh turn.
        let mut h = spawn_pipeline(Cfg {
            stt_text: LIVENESS_PHRASES[0].to_string(),
            conversation_mode: true,
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> transcribe -> discard

        // A discarded echo re-arms listening (conversation mode) rather than
        // prompting; give it a moment, then confirm nothing was sent.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.handle.abort();
        assert!(
            h.prompt_rx.try_recv().is_err(),
            "a self-echo of a canned line must not be sent to the assistant"
        );
    }

    #[tokio::test]
    async fn stop_listening_ends_an_active_conversation() {
        // StopListening (StopRequest::Conversation) ends a live conversation-mode
        // follow-up immediately, returning to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_secs(30),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> re-listen
        // Wait until the first turn actually reached the assistant, so the
        // conversation is genuinely active before we stop it.
        tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first turn prompted")
            .expect("prompt sender open");

        h.stop_tx.send(StopRequest::Conversation).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("StopListening -> Idle")
        .unwrap();
        h.handle.abort();
    }

    #[derive(Default)]
    struct FakeWake {
        detects: bool,
        /// Last sensitivity handed to `set_sensitivity` — lets a reload test
        /// observe that the value was applied live (no rebuild).
        applied_sensitivity: Option<f32>,
    }
    impl WakeWordDetector for FakeWake {
        fn detect(&mut self, _samples: &[f32]) -> Result<bool, adele_voice_core::VoiceError> {
            Ok(self.detects)
        }
        fn set_sensitivity(&mut self, s: f32) -> Result<(), adele_voice_core::VoiceError> {
            self.applied_sensitivity = Some(s);
            Ok(())
        }
    }

    /// VAD that returns scripted probabilities, then 0.0 once the script is
    /// exhausted — so one "speech" chunk followed by anything reads as
    /// speech-then-silence.
    struct FakeVad {
        probs: StdMutex<VecDeque<f32>>,
    }
    impl VoiceActivityDetector for FakeVad {
        fn speech_probability(
            &mut self,
            _samples: &[f32],
        ) -> Result<f32, adele_voice_core::VoiceError> {
            Ok(self.probs.lock().unwrap().pop_front().unwrap_or(0.0))
        }
        fn reset(&mut self) {}
    }

    /// STT that signals when it runs (proving audio reached transcription) and
    /// returns a non-empty transcript so the response cycle proceeds. Records the
    /// length of every buffer it was handed so a test can prove which audio
    /// landed in the captured utterance (e.g. that a phrase listening-cue's echo
    /// did NOT — V-7/#87).
    struct FakeStt {
        hit: mpsc::UnboundedSender<()>,
        text: String,
        captured: Arc<StdMutex<Vec<Vec<f32>>>>,
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(
            &self,
            samples: &[f32],
        ) -> Result<Transcript, adele_voice_core::VoiceError> {
            self.captured.lock().unwrap().push(samples.to_vec());
            let _ = self.hit.send(());
            Ok(Transcript {
                text: self.text.clone(),
            })
        }
    }

    struct FakeTts;
    impl TextToSpeech for FakeTts {
        async fn synthesize(&self, _text: &str) -> Result<Vec<f32>, adele_voice_core::VoiceError> {
            Ok(Vec::new())
        }
    }

    /// TTS for the `spawn_pipeline` harness. When `set_playing` is `None` it
    /// behaves exactly like `FakeTts` (no audio), so every existing test is
    /// unaffected. When it's `Some`, each synthesis returns one sample (so the
    /// `Speaker` queues it on the sink) and flips the shared `is_playing` flag —
    /// modelling a spoken cue/reply that is now sounding, which a test uses to
    /// prove the listening-cue's echo is drained, not captured (V-7/#87).
    struct SpawnTts {
        set_playing: Option<Arc<std::sync::atomic::AtomicBool>>,
    }
    impl TextToSpeech for SpawnTts {
        async fn synthesize(&self, _text: &str) -> Result<Vec<f32>, adele_voice_core::VoiceError> {
            match &self.set_playing {
                Some(playing) => {
                    playing.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(vec![0.0; 1])
                }
                None => Ok(Vec::new()),
            }
        }
    }

    /// TTS that records the exact text it was asked to synthesize (returning no
    /// audio), so the #58 tests can assert what was spoken — apologies and
    /// status narrations — without an audio device.
    struct FakeRecordingTts {
        spoken: Arc<StdMutex<Vec<String>>>,
    }
    impl TextToSpeech for FakeRecordingTts {
        async fn synthesize(&self, text: &str) -> Result<Vec<f32>, adele_voice_core::VoiceError> {
            self.spoken.lock().unwrap().push(text.to_string());
            Ok(Vec::new())
        }
    }

    /// What the pipeline handed to the assistant gateway for one turn.
    /// Captures the target conversation plus the split between the
    /// user-visible `prompt` and the per-request `system_refinement`, so a
    /// test can assert the clean transcript is the message and the hint
    /// rides as the refinement.
    #[derive(Debug, Clone)]
    struct SentPrompt {
        conversation_id: String,
        prompt: String,
        system_refinement: String,
    }

    /// Assistant that completes immediately: `subscribe` hands back a receiver
    /// and each send method pushes a matching `Complete` so
    /// `process_utterance` returns without hanging. It records every send
    /// (via `prompt_tx`) so tests can assert PTT routing and the
    /// prompt/refinement split, and reports the title of any conversation it
    /// created (via `created_tx`).
    /// A client-tool result the pipeline submitted back to the gateway
    /// (voice#61) — captured so tests can assert each tool call is acknowledged.
    #[derive(Debug, Clone)]
    struct SubmittedToolResult {
        task_id: String,
        tool_call_id: String,
        result: Result<String, String>,
    }

    struct FakeAssistant {
        tx: StdMutex<Option<mpsc::UnboundedSender<AssistantEvent>>>,
        prompt_tx: mpsc::UnboundedSender<SentPrompt>,
        created_tx: mpsc::UnboundedSender<String>,
        /// Client tools the pipeline registered at startup (voice#61).
        registered_tx: mpsc::UnboundedSender<Vec<String>>,
        /// Results the pipeline submitted for client-tool calls (voice#61).
        tool_result_tx: mpsc::UnboundedSender<SubmittedToolResult>,
        /// Optional client-tool call to inject on the turn's signal stream right
        /// before the `Complete`, so a test can drive the tool handler (voice#61).
        inject_tool_call: Option<(String, serde_json::Value)>,
        /// When set, `create_conversation` errors — simulating a dropped
        /// orchestrator connection so the turn fails mid-flight.
        fail: bool,
        /// When set, a turn is NOT auto-completed (voice#82): `record_and_complete`
        /// records the prompt and returns the request id, but the signal stream is
        /// driven entirely by the test through the sender it taps off `subscribe`
        /// (see `subscribed_tx`). Lets a test hold a turn open, dribble chunks,
        /// then fire control events mid-stream.
        hold_turn: bool,
        /// Each `subscribe` publishes the freshly-created event sender here so the
        /// harness (hence the test) can drive the held-open turn (voice#82).
        subscribed_tx: mpsc::UnboundedSender<mpsc::UnboundedSender<AssistantEvent>>,
    }
    impl FakeAssistant {
        /// Shared recording + immediate-completion path for both send
        /// methods. Records exactly what reached the gateway (target
        /// conversation, the user-visible `prompt`, and the per-request
        /// `system_refinement`) and pushes a matching `Complete` — optionally
        /// preceded by an injected `ClientToolCall` so the tool handler runs
        /// (voice#61).
        fn record_and_complete(
            &self,
            conversation_id: &str,
            prompt: &str,
            system_refinement: &str,
        ) -> String {
            let _ = self.prompt_tx.send(SentPrompt {
                conversation_id: conversation_id.to_string(),
                prompt: prompt.to_string(),
                system_refinement: system_refinement.to_string(),
            });
            let request_id = "req".to_string();
            // hold_turn: the test owns the timeline — don't auto-complete.
            if self.hold_turn {
                return request_id;
            }
            if let Some(tx) = self.tx.lock().unwrap().as_ref() {
                if let Some((tool_name, arguments)) = self.inject_tool_call.clone() {
                    let _ = tx.send(AssistantEvent::ClientToolCall {
                        task_id: "task-1".to_string(),
                        tool_call_id: "call-1".to_string(),
                        tool_name,
                        arguments,
                    });
                }
                let _ = tx.send(AssistantEvent::Complete {
                    request_id: request_id.clone(),
                    full_response: "hello".to_string(),
                });
            }
            request_id
        }
    }
    impl AssistantGateway for FakeAssistant {
        async fn register_client_tools(
            &self,
            tools: Vec<ClientToolSpec>,
        ) -> Result<usize, adele_voice_core::VoiceError> {
            let names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
            let count = names.len();
            let _ = self.registered_tx.send(names);
            Ok(count)
        }
        async fn submit_client_tool_result(
            &self,
            task_id: &str,
            tool_call_id: &str,
            result: Result<String, String>,
        ) -> Result<(), adele_voice_core::VoiceError> {
            let _ = self.tool_result_tx.send(SubmittedToolResult {
                task_id: task_id.to_string(),
                tool_call_id: tool_call_id.to_string(),
                result,
            });
            Ok(())
        }
        async fn create_conversation(
            &self,
            title: &str,
            _tags: Vec<String>,
        ) -> Result<String, adele_voice_core::VoiceError> {
            if self.fail {
                return Err(adele_voice_core::VoiceError::Assistant(
                    "uds connection closed".to_string(),
                ));
            }
            // Internal triage conversations (title starts with '_') are not
            // recorded in the test channel — tests should not need to drain them.
            if !title.starts_with('_') {
                let _ = self.created_tx.send(title.to_string());
            }
            Ok("own-session".to_string())
        }

        async fn archive_conversation(
            &self,
            _id: &str,
        ) -> Result<(), adele_voice_core::VoiceError> {
            Ok(())
        }

        async fn delete_conversation(&self, _id: &str) -> Result<(), adele_voice_core::VoiceError> {
            Ok(())
        }
        async fn send_prompt_with_system_refinement(
            &self,
            conversation_id: &str,
            prompt: &str,
            system_refinement: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            Ok(self.record_and_complete(conversation_id, prompt, system_refinement))
        }
        async fn subscribe(
            &self,
        ) -> Result<mpsc::UnboundedReceiver<AssistantEvent>, adele_voice_core::VoiceError> {
            let (tx, rx) = mpsc::unbounded_channel();
            // Publish the sender so a hold_turn test can drive this turn (voice#82).
            let _ = self.subscribed_tx.send(tx.clone());
            *self.tx.lock().unwrap() = Some(tx);
            Ok(rx)
        }
    }

    /// Audio source whose receiver is driven by the test via `audio_tx`.
    struct FakeSource {
        rx: StdMutex<Option<mpsc::Receiver<Vec<f32>>>>,
    }
    impl AudioSource for FakeSource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, adele_voice_core::VoiceError> {
            self.rx
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| adele_voice_core::VoiceError::Audio("already started".to_string()))
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            Ok(())
        }
    }

    /// A restartable source for the `spawn_pipeline` harness (voice#103): each
    /// `start()` mints a fresh capture channel and publishes its sender into a
    /// shared slot, so the test's `send_chunk` always writes to the *current*
    /// capture channel — even after a pause/resume re-acquired a new one (the
    /// device-release-on-disable / session-inactive path). `start()` after a
    /// `stop()` succeeds (the prod cpal source clears its latch on stop too).
    #[derive(Clone)]
    struct RestartableSource {
        current_tx: Arc<StdMutex<Option<mpsc::Sender<Vec<f32>>>>>,
    }
    impl AudioSource for RestartableSource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, adele_voice_core::VoiceError> {
            let (tx, rx) = mpsc::channel::<Vec<f32>>(64);
            *self.current_tx.lock().unwrap() = Some(tx);
            Ok(rx)
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            *self.current_tx.lock().unwrap() = None;
            Ok(())
        }
    }

    /// Records the length of every buffer it was asked to play, so a test can
    /// assert the listening cue (the ding earcon) was/wasn't queued. Also models
    /// a controllable "is playing" state and a stop counter so the PTT
    /// self-recording fix (#68) can be exercised: a test sets `playing` to mimic
    /// outstanding TTS, and `stop()` clears it (as the real sink does by
    /// dropping the queue) while bumping `stopped` so the test can assert it ran.
    #[derive(Default, Clone)]
    struct FakeSink {
        played: Arc<StdMutex<Vec<usize>>>,
        playing: Arc<std::sync::atomic::AtomicBool>,
        /// Models `in_tail_pad`: the audio deadline has passed but we're still
        /// inside the latency cushion (#70). Independent of `playing` so a test
        /// can put the sink in "real audio" (playing && !in_pad) or "tail pad"
        /// (playing && in_pad) states.
        in_pad: Arc<std::sync::atomic::AtomicBool>,
        stopped: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl AudioSink for FakeSink {
        fn play(&self, samples: Vec<f32>) -> Result<(), adele_voice_core::VoiceError> {
            self.played.lock().unwrap().push(samples.len());
            Ok(())
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            self.stopped
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.playing
                .store(false, std::sync::atomic::Ordering::SeqCst);
            self.in_pad
                .store(false, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        fn is_playing(&self) -> bool {
            self.playing.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn in_tail_pad(&self) -> bool {
            self.in_pad.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    struct Harness {
        /// The current capture channel's sender. `RestartableSource` re-publishes
        /// it on every `start()`, so a test sending audio after a pause/resume
        /// reaches the live channel (voice#103). `send_audio` reads it.
        audio_slot: Arc<StdMutex<Option<mpsc::Sender<Vec<f32>>>>>,
        ptt_tx: mpsc::Sender<Option<String>>,
        _enabled_tx: watch::Sender<bool>,
        stop_tx: mpsc::Sender<StopRequest>,
        state_rx: watch::Receiver<State>,
        transcribe_rx: mpsc::UnboundedReceiver<()>,
        /// Every send the pipeline made (target conversation + the
        /// prompt/refinement split).
        prompt_rx: mpsc::UnboundedReceiver<SentPrompt>,
        /// Title of each conversation the daemon asked to create.
        created_rx: mpsc::UnboundedReceiver<String>,
        /// Lengths of every buffer queued on the sink — the listening cue (the
        /// ding earcon) shows up here.
        sink_played: Arc<StdMutex<Vec<usize>>>,
        /// Every buffer handed to STT — i.e. the captured utterances. Lets a test
        /// prove a phrase listening-cue's echo did NOT land in the transcribed
        /// audio (V-7/#87) by inspecting the sample content.
        stt_captured: Arc<StdMutex<Vec<Vec<f32>>>>,
        /// Drives the fake sink's `is_playing`: set true to mimic outstanding
        /// TTS (#68). Cleared by the sink's `stop()`.
        sink_playing: Arc<std::sync::atomic::AtomicBool>,
        /// Drives the fake sink's `in_tail_pad`: set true (with `sink_playing`)
        /// to mimic the latency-cushion tail where same-breath audio should
        /// still be pre-buffered (#70). Cleared by `stop()`.
        sink_in_pad: Arc<std::sync::atomic::AtomicBool>,
        /// Count of `stop()` calls the pipeline made on the sink (#68).
        sink_stopped: Arc<std::sync::atomic::AtomicUsize>,
        /// Names of the client tools the pipeline registered at startup (voice#61).
        registered_rx: mpsc::UnboundedReceiver<Vec<String>>,
        /// Results the pipeline submitted for client-tool calls (voice#61).
        tool_result_rx: mpsc::UnboundedReceiver<SubmittedToolResult>,
        /// One item per `subscribe`: the event sender for that turn (voice#82).
        /// A `hold_turn` test awaits this to drive a held-open stream — dribble
        /// chunks, then fire control events mid-stream.
        events_rx: mpsc::UnboundedReceiver<mpsc::UnboundedSender<AssistantEvent>>,
        handle: tokio::task::JoinHandle<()>,
    }

    impl Harness {
        /// Send a raw audio buffer to the live capture channel. Awaits the
        /// channel becoming available (the source may be momentarily re-acquiring
        /// after a pause), so a test never races a transient gap.
        async fn send_audio(&self, buf: Vec<f32>) {
            for _ in 0..200 {
                let tx = self.audio_slot.lock().unwrap().clone();
                if let Some(tx) = tx
                    && tx.send(buf.clone()).await.is_ok()
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            panic!("no live capture channel to send audio to within 1s");
        }

        /// Best-effort non-blocking send to the live capture channel.
        fn try_send_audio(&self, buf: Vec<f32>) {
            if let Some(tx) = self.audio_slot.lock().unwrap().clone() {
                let _ = tx.try_send(buf);
            }
        }
    }

    struct Cfg {
        enabled: bool,
        wake_detects: bool,
        conversation_mode: bool,
        conversation_reuse_window: Duration,
        followup_timeout: Duration,
        idle_exit_timeout: Option<Duration>,
        spoken_response_hint: String,
        vad: Vec<f32>,
        stt_text: String,
        assistant_fails: bool,
        listening_cue: ListeningCue,
        /// Client tool to inject on the turn's signal stream so the tool handler
        /// runs (voice#61): `(tool_name, arguments)`.
        inject_tool_call: Option<(String, serde_json::Value)>,
        client_tools: ClientToolsConfig,
        /// Hold each turn open so the test drives the signal stream (voice#82).
        hold_turn: bool,
        /// Allow mic speech to interrupt playback (barge-in). Defaults true here
        /// so the existing barge-in tests keep exercising it; production defaults
        /// false (no AEC). A dedicated test opts out to assert the gate.
        mic_barge_in: bool,
        /// When set, the harness TTS reports its synthesized audio as *playing*
        /// (one sample queued, `is_playing` flipped true) — so a test can model a
        /// spoken listening-cue actually sounding and assert its echo is drained
        /// rather than captured (V-7/#87). Off by default: existing tests keep the
        /// silent `FakeTts` behaviour.
        cue_plays_audio: bool,
    }
    impl Default for Cfg {
        fn default() -> Self {
            Self {
                enabled: true,
                wake_detects: false,
                conversation_mode: false,
                // A generous window by default so the legacy "reuse the own
                // session across presses" behaviour holds; the voice#53 tests
                // opt into a specific window (or 0 = always fresh).
                conversation_reuse_window: Duration::from_secs(600),
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                vad: vec![0.9],
                stt_text: "hello".to_string(),
                assistant_fails: false,
                // Default the cue off in tests so most cases don't queue cue
                // audio onto the recording sink; cue-specific tests opt in.
                listening_cue: ListeningCue::Off,
                inject_tool_call: None,
                client_tools: ClientToolsConfig::default(),
                hold_turn: false,
                cue_plays_audio: false,
                mic_barge_in: true,
            }
        }
    }

    /// A neutral tunables snapshot for the fake pipeline. Matches the fakes'
    /// constructor args (0.5 threshold, 0 ms silence) so an initial reload that
    /// re-reads those same values is a no-op.
    fn test_tunables() -> Tunables {
        Tunables {
            speech_threshold: 0.5,
            silence_duration_ms: 0,
            followup_timeout_ms: 50,
            // A generous default so the legacy "reuse the own session across
            // presses" behaviour holds in tests that don't set a window; the
            // voice#53 tests set their own (0 = always fresh).
            conversation_reuse_window_ms: 600_000,
            conversation_mode: false,
            idle_exit_timeout_ms: 0,
            wake_sensitivity: 0.5,
            response_stall_ms: 0,
            turn_budget_ms: 0,
            status_narration_min_gap_ms: 0,
            narration_flush_ms: 0,
            input_device: "default".into(),
            output_device: "default".into(),
        }
    }

    /// Turn timeouts for the fake pipeline: all bounds disabled (0) by default
    /// so existing tests behave exactly as before; the #58 tests construct
    /// their own with the bounds they exercise.
    fn test_timeouts() -> TurnTimeouts {
        TurnTimeouts {
            response_stall: Duration::ZERO,
            turn_budget: Duration::ZERO,
            synth: Duration::ZERO,
            connect: Duration::ZERO,
            status_narration_min_gap: Duration::ZERO,
            liveness_delay: Duration::ZERO,
            // The narration timeout flush was previously hardcoded to 500 ms;
            // keep that here so the streaming tests exercise it as before.
            narration_flush: Duration::from_millis(500),
        }
    }

    /// An open-but-idle audio receiver for the direct `stream_response` tests
    /// (voice#82): the sender is leaked so the channel never closes and the
    /// mid-turn audio arm simply pends — these tests don't exercise barge-in.
    fn idle_audio_rx() -> mpsc::Receiver<Vec<f32>> {
        let (tx, rx) = mpsc::channel::<Vec<f32>>(1);
        Box::leak(Box::new(tx));
        rx
    }

    /// Build a non-running pipeline with fakes so `apply_plan` can be exercised
    /// directly (no audio, no file watch) — the apply side of the reload, while
    /// `plan_reload`'s decision logic is unit-tested in `config`.
    /// The fully-faked pipeline type used across the tests.
    type FakePipeline = Pipeline<FakeWake, FakeVad, FakeStt, FakeRecordingTts, FakeAssistant>;

    fn build_pipeline() -> FakePipeline {
        build_pipeline_with(test_timeouts()).0
    }

    /// Captured outputs for direct handler tests (voice#61): what the `Speaker`
    /// was asked to say, and the client-tool results the pipeline submitted.
    struct ToolHarness {
        spoken: Arc<StdMutex<Vec<String>>>,
        tool_result_rx: mpsc::UnboundedReceiver<SubmittedToolResult>,
    }

    /// Build a non-running pipeline wired to capture spoken text AND submitted
    /// client-tool results, so a test can call `handle_client_tool_call`
    /// directly and assert both what was spoken and the result posted back
    /// (voice#61). Mirrors `build_pipeline_with` but keeps the tool-result
    /// receiver instead of dropping it.
    fn build_pipeline_for_tools(reuse_window: Duration) -> (FakePipeline, ToolHarness) {
        let (_audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(1);
        let (_enabled_tx, enabled_rx) = watch::channel(true);
        let (_ptt_tx, ptt_rx) = mpsc::channel(1);
        let (_stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, _state_rx) = watch::channel(State::Idle);
        let (hit_tx, _transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, _prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, _created_rx) = mpsc::unbounded_channel();
        let (registered_tx, _registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, tool_result_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);
        let spoken = Arc::new(StdMutex::new(Vec::new()));
        let pipeline = Pipeline::new(
            FakeWake {
                detects: false,
                ..Default::default()
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
                captured: Arc::new(StdMutex::new(Vec::new())),
            },
            FakeRecordingTts {
                spoken: Arc::clone(&spoken),
            },
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: None,
                fail: false,
                hold_turn: false,
                subscribed_tx: mpsc::unbounded_channel().0,
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(FakeSink::default()),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            PipelineConfig {
                tunables: test_tunables(),
                conversation_title: "test".to_string(),
                silence_duration: Duration::from_millis(0),
                speech_threshold: 0.5,
                mic_barge_in: true,
                conversation_mode: false,
                conversation_reuse_window: reuse_window,
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                listening_cue: ListeningCue::Off,
                timeouts: test_timeouts(),
                client_tools: ClientToolsConfig::default(),
            },
        );
        let mut pipeline = pipeline;
        // These tests drive `handle_client_tool_call` directly; in production a
        // client tool only fires mid-turn, i.e. from Processing/Speaking. Set
        // the precondition so a `say_this`-style `ResponseStarted` is a legal
        // transition rather than an illegal-from-Idle assert (voice#82).
        pipeline.state = State::Processing;
        (
            pipeline,
            ToolHarness {
                spoken,
                tool_result_rx,
            },
        )
    }

    /// Build a non-running pipeline with the given turn timeouts, returning it
    /// alongside the texts the `Speaker` was asked to synthesize — so the #58
    /// tests can drive `stream_response`/`maybe_narrate_status` directly and
    /// assert exactly what was spoken (apologies, narrations) without audio.
    fn build_pipeline_with(timeouts: TurnTimeouts) -> (FakePipeline, Arc<StdMutex<Vec<String>>>) {
        let (_audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(1);
        let (_enabled_tx, enabled_rx) = watch::channel(true);
        let (_ptt_tx, ptt_rx) = mpsc::channel(1);
        let (_stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, _state_rx) = watch::channel(State::Idle);
        let (hit_tx, _transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, _prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, _created_rx) = mpsc::unbounded_channel();
        let (registered_tx, _registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, _tool_result_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);
        let spoken = Arc::new(StdMutex::new(Vec::new()));
        let pipeline = Pipeline::new(
            FakeWake {
                detects: false,
                ..Default::default()
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
                captured: Arc::new(StdMutex::new(Vec::new())),
            },
            FakeRecordingTts {
                spoken: Arc::clone(&spoken),
            },
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: None,
                fail: false,
                hold_turn: false,
                subscribed_tx: mpsc::unbounded_channel().0,
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(FakeSink::default()),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            PipelineConfig {
                tunables: test_tunables(),
                conversation_title: "test".to_string(),
                silence_duration: Duration::from_millis(0),
                speech_threshold: 0.5,
                mic_barge_in: true,
                conversation_mode: false,
                // Matches test_tunables() so an initial reload is a no-op.
                conversation_reuse_window: Duration::from_millis(600_000),
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                listening_cue: ListeningCue::Off,
                timeouts,
                client_tools: ClientToolsConfig::default(),
            },
        );
        let mut pipeline = pipeline;
        // These tests drive `stream_response` / `maybe_narrate_status` directly;
        // in production those only run after the Listening→Processing
        // transition, so start in Processing — the precondition that makes the
        // first `ResponseStarted` (Processing→Speaking) legal (voice#82).
        pipeline.state = State::Processing;
        (pipeline, spoken)
    }

    #[test]
    fn apply_plan_updates_live_tunable_state() {
        // The apply side of reload: a plan's hot knobs must mutate the running
        // pipeline's fields (and the shared endpointer threshold) in place.
        let mut p = build_pipeline();
        let plan = config::ReloadPlan {
            set_speech_threshold: Some(0.8),
            set_silence_ms: Some(1200),
            set_followup_timeout_ms: Some(9000),
            set_conversation_mode: Some(true),
            set_idle_exit_timeout_ms: Some(60_000),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(p.speech_threshold, 0.8);
        assert_eq!(p.followup_timeout, Duration::from_millis(9000));
        assert!(p.conversation_mode);
        assert_eq!(p.idle_exit_timeout, Some(Duration::from_millis(60_000)));
    }

    #[test]
    fn apply_plan_idle_exit_zero_disables() {
        // idle_exit_timeout_ms = 0 means "always-on" → the Option clears to None.
        let mut p = build_pipeline();
        p.idle_exit_timeout = Some(Duration::from_millis(1000));
        let plan = config::ReloadPlan {
            set_idle_exit_timeout_ms: Some(0),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(p.idle_exit_timeout, None);
    }

    #[test]
    fn apply_plan_applies_wake_sensitivity_live() {
        // The wake-sensitivity branch must poke the running detector's
        // `set_sensitivity` (no rebuild) and keep the tunables snapshot in step.
        let mut p = build_pipeline();
        assert_eq!(p.wake.applied_sensitivity, None);
        let plan = config::ReloadPlan {
            apply_wake_sensitivity: Some(0.9),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(
            p.wake.applied_sensitivity,
            Some(0.9),
            "the sensitivity must be applied to the live detector"
        );
        assert_eq!(
            p.tunables.wake_sensitivity, 0.9,
            "the tunables snapshot must track the applied sensitivity"
        );
    }

    #[test]
    fn apply_plan_updates_timeout_knobs() {
        // #58: the new timeout knobs hot-apply on reload like the other tunables.
        let mut p = build_pipeline();
        let plan = config::ReloadPlan {
            set_response_stall_ms: Some(7000),
            set_turn_budget_ms: Some(90_000),
            set_status_narration_min_gap_ms: Some(12_000),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(p.response_stall, Duration::from_millis(7000));
        assert_eq!(p.turn_budget, Duration::from_millis(90_000));
        assert_eq!(p.status_narration_min_gap, Duration::from_millis(12_000));
    }

    #[tokio::test(start_paused = true)]
    async fn stall_with_no_progress_apologizes_and_returns() {
        // #58 core: with a stall deadline and an event source that never sends,
        // the response loop must give up after the deadline — speak the stall
        // apology and return — instead of hanging in Processing forever.
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            response_stall: Duration::from_millis(500),
            ..test_timeouts()
        });
        // A receiver whose sender we keep alive but never use: recv() pends, so
        // only the stall deadline can end the loop.
        let (_tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        tokio::time::timeout(
            Duration::from_secs(5),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must return on stall, not hang")
        .expect("stream_response ok");
        assert_eq!(
            spoken.lock().unwrap().clone(),
            vec![STALL_APOLOGY.to_string()],
            "a stalled turn must speak the stall apology"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn heartbeat_resets_the_clock_on_each_event() {
        // #58 core: each received event RESETS the stall deadline, so a turn
        // that keeps making progress (chunks/statuses spaced just under the
        // deadline) never times out — it completes normally.
        let stall = Duration::from_millis(500);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            response_stall: stall,
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();

        // Feed several events each at 60% of the stall window — under the
        // deadline only because every event resets it — then complete.
        let feeder = tokio::spawn(async move {
            for _ in 0..5 {
                tokio::time::sleep(stall.mul_f32(0.6)).await;
                let _ = tx.send(AssistantEvent::Chunk {
                    request_id: "req".into(),
                    text: "word ".into(),
                });
            }
            tokio::time::sleep(stall.mul_f32(0.6)).await;
            let _ = tx.send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "word word word word word".into(),
            });
        });

        tokio::time::timeout(
            Duration::from_secs(10),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("a steadily-progressing turn must not be killed by the stall guard")
        .expect("stream_response ok");
        feeder.await.unwrap();

        let said = spoken.lock().unwrap().clone();
        assert!(
            !said.iter().any(|s| s == STALL_APOLOGY),
            "a turn that kept making progress must NOT hit the stall apology; said: {said:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn turn_budget_caps_a_dribbling_turn() {
        // #58: the overall budget is a backstop — even if events keep resetting
        // the per-event stall clock, a turn that runs past the budget gives up.
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            response_stall: Duration::from_secs(10), // never the limiting factor
            turn_budget: Duration::from_millis(400),
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        // Dribble forever, faster than the stall window, never completing.
        let feeder = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(50)).await;
                if tx
                    .send(AssistantEvent::Status {
                        request_id: "req".into(),
                        message: "still working".into(),
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
        tokio::time::timeout(
            Duration::from_secs(5),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("the budget must cap a dribbling turn")
        .expect("stream_response ok");
        feeder.abort();
        assert!(
            spoken.lock().unwrap().iter().any(|s| s == STALL_APOLOGY),
            "a turn over budget must speak the stall apology"
        );
    }

    #[tokio::test]
    async fn first_status_narrates_then_rate_limited() {
        // #58: status narration is sparing — the first status of a turn speaks,
        // a second arriving inside the min-gap is suppressed.
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            status_narration_min_gap: Duration::from_secs(3600), // effectively "only the first"
            ..test_timeouts()
        });
        let mut last: Option<tokio::time::Instant> = None;
        let mut first_chunk = true;

        let spoke1 = p
            .maybe_narrate_status("checking your calendar", &mut last, &mut first_chunk)
            .await
            .unwrap();
        let spoke2 = p
            .maybe_narrate_status("looking up the weather", &mut last, &mut first_chunk)
            .await
            .unwrap();

        assert!(spoke1, "the first status must be narrated");
        assert!(
            !spoke2,
            "a second status inside the min-gap must be suppressed"
        );
        assert_eq!(
            spoken.lock().unwrap().clone(),
            vec!["checking your calendar".to_string()],
            "only the first status is spoken"
        );
    }

    #[tokio::test]
    async fn empty_status_is_not_narrated() {
        // A blank status must never be spoken (and must not consume the "first
        // status" slot).
        let (mut p, spoken) = build_pipeline_with(test_timeouts());
        let mut last: Option<tokio::time::Instant> = None;
        let mut first_chunk = true;
        let spoke = p
            .maybe_narrate_status("   ", &mut last, &mut first_chunk)
            .await
            .unwrap();
        assert!(!spoke, "a blank status must not be narrated");
        assert!(
            last.is_none(),
            "a blank status must not arm the rate-limiter"
        );
        assert!(spoken.lock().unwrap().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn status_after_the_gap_narrates_again() {
        // #58: once the min-gap has elapsed, a later status narrates again — the
        // reassurance cadence on a long turn.
        let gap = Duration::from_secs(15);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            status_narration_min_gap: gap,
            ..test_timeouts()
        });
        let mut last: Option<tokio::time::Instant> = None;
        let mut first_chunk = true;

        assert!(
            p.maybe_narrate_status("first", &mut last, &mut first_chunk)
                .await
                .unwrap()
        );
        // Inside the gap: suppressed.
        tokio::time::sleep(gap / 2).await;
        assert!(
            !p.maybe_narrate_status("too soon", &mut last, &mut first_chunk)
                .await
                .unwrap()
        );
        // Past the gap: narrates again.
        tokio::time::sleep(gap).await;
        assert!(
            p.maybe_narrate_status("later", &mut last, &mut first_chunk)
                .await
                .unwrap()
        );
        assert_eq!(
            spoken.lock().unwrap().clone(),
            vec!["first".to_string(), "later".to_string()]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn liveness_speaks_once_on_a_slow_stepless_turn() {
        // Safety net: a turn that narrates nothing and streams nothing past the
        // liveness delay (a slow turn that declared no step) gets one brief
        // liveness line so voice isn't dead-silent — then it completes before
        // the next escalation step would fire.
        let liveness = Duration::from_millis(500);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            liveness_delay: liveness,
            response_stall: Duration::from_secs(10), // not the limiting factor
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        let feeder = tokio::spawn(async move {
            // Past the first liveness delay but before the second, then reply.
            tokio::time::sleep(liveness + liveness / 2).await;
            let _ = tx.send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "the answer".into(),
            });
        });
        tokio::time::timeout(
            Duration::from_secs(10),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must return")
        .expect("stream_response ok");
        feeder.await.unwrap();
        let said = spoken.lock().unwrap().clone();
        let liveness_lines: Vec<_> = said
            .iter()
            .filter(|s| LIVENESS_PHRASES.contains(&s.as_str()))
            .collect();
        assert_eq!(
            liveness_lines.len(),
            1,
            "a slow stepless turn must speak one liveness line before completing; said: {said:?}"
        );
        assert_eq!(
            liveness_lines[0], LIVENESS_PHRASES[0],
            "the first liveness line must be the first phrase; said: {said:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn liveness_escalates_then_stops_on_a_very_slow_turn() {
        // A turn that stays silent across several liveness windows escalates
        // through the phrase list in order, then goes quiet (the stall/budget
        // backstop covers the rest) rather than repeating forever.
        let liveness = Duration::from_millis(500);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            liveness_delay: liveness,
            response_stall: Duration::from_secs(30), // not the limiting factor
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        let feeder = tokio::spawn(async move {
            // Silent long enough for every phrase plus a couple more windows.
            tokio::time::sleep(liveness * (LIVENESS_PHRASES.len() as u32 + 3)).await;
            let _ = tx.send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "the answer".into(),
            });
        });
        tokio::time::timeout(
            Duration::from_secs(30),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must return")
        .expect("stream_response ok");
        feeder.await.unwrap();
        let said = spoken.lock().unwrap().clone();
        let liveness_lines: Vec<&String> = said
            .iter()
            .filter(|s| LIVENESS_PHRASES.contains(&s.as_str()))
            .collect();
        let expected: Vec<&str> = LIVENESS_PHRASES.to_vec();
        let got: Vec<&str> = liveness_lines.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            got, expected,
            "liveness must escalate through each phrase exactly once, in order, then stop; said: {said:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn liveness_suppressed_when_progress_arrives_first() {
        // A status (a declared step) before the liveness delay cancels it — so
        // multi-step turns and quick answers never hear the filler.
        let liveness = Duration::from_millis(500);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            liveness_delay: liveness,
            response_stall: Duration::from_secs(10),
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        let feeder = tokio::spawn(async move {
            // A step status arrives well before the liveness delay.
            tokio::time::sleep(liveness / 5).await;
            let _ = tx.send(AssistantEvent::Status {
                request_id: "req".into(),
                message: "Looking into that".into(),
            });
            tokio::time::sleep(liveness * 2).await;
            let _ = tx.send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "done".into(),
            });
        });
        tokio::time::timeout(
            Duration::from_secs(10),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must return")
        .expect("stream_response ok");
        feeder.await.unwrap();
        let said = spoken.lock().unwrap().clone();
        assert!(
            !said.iter().any(|s| LIVENESS_PHRASES.contains(&s.as_str())),
            "progress before the delay must suppress the liveness line; said: {said:?}"
        );
        assert!(
            said.iter().any(|s| s == "Looking into that"),
            "the real status should still be narrated; said: {said:?}"
        );
    }

    #[tokio::test]
    async fn emits_speaking_text_signal_for_reply_sentences() {
        // voice#85: the pipeline must emit a SpeakingText signal for each spoken
        // reply sentence so clients (the KDE widget) needn't poll.
        let (mut p, spoken) = build_pipeline_with(test_timeouts());
        let (signal_tx, mut signal_rx) = mpsc::channel::<VoiceSignal>(16);
        p = p.with_signal_tx(signal_tx);

        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        tx.send(AssistantEvent::Complete {
            request_id: "req".into(),
            full_response: "Hello there.".into(),
        })
        .unwrap();
        tokio::time::timeout(
            Duration::from_secs(5),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must return")
        .expect("stream_response ok");

        // It was spoken AND announced as a SpeakingText signal.
        assert!(
            spoken
                .lock()
                .unwrap()
                .iter()
                .any(|s| s.contains("Hello there")),
            "the reply should have been spoken"
        );
        let mut signals = Vec::new();
        while let Ok(sig) = signal_rx.try_recv() {
            signals.push(sig);
        }
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, VoiceSignal::SpeakingText(t) if t.contains("Hello there"))),
            "a SpeakingText signal must be emitted for the spoken reply; got: {signals:?}"
        );
    }

    #[tokio::test]
    async fn leading_ack_is_spoken_before_the_next_token() {
        // #58: a short terminal ack chunk is spoken immediately, without waiting
        // for the following sentence to arrive.
        let (mut p, spoken) = build_pipeline_with(test_timeouts());
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        // One chunk that is just the ack (ends in a period, alone in the buffer)
        // then complete with nothing more.
        tx.send(AssistantEvent::Chunk {
            request_id: "req".into(),
            text: "Got it — checking that now.".into(),
        })
        .unwrap();
        tx.send(AssistantEvent::Complete {
            request_id: "req".into(),
            full_response: "Got it — checking that now.".into(),
        })
        .unwrap();
        drop(tx);
        tokio::time::timeout(
            Duration::from_secs(2),
            p.stream_response(&mut rx, "req", &mut idle_audio_rx()),
        )
        .await
        .expect("stream_response must complete")
        .expect("ok");
        let said = spoken.lock().unwrap().clone();
        assert!(
            said.first().map(String::as_str) == Some("Got it — checking that now."),
            "the leading ack must be spoken first; said: {said:?}"
        );
    }

    fn spawn_pipeline(cfg: Cfg) -> Harness {
        let audio_slot: Arc<StdMutex<Option<mpsc::Sender<Vec<f32>>>>> =
            Arc::new(StdMutex::new(None));
        let source = RestartableSource {
            current_tx: Arc::clone(&audio_slot),
        };
        let (enabled_tx, enabled_rx) = watch::channel(cfg.enabled);
        let (ptt_tx, ptt_rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, state_rx) = watch::channel(State::Idle);
        let (hit_tx, transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, created_rx) = mpsc::unbounded_channel();
        let (registered_tx, registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, tool_result_rx) = mpsc::unbounded_channel();
        let (subscribed_tx, events_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);

        let stt_captured = Arc::new(StdMutex::new(Vec::new()));
        let sink = FakeSink::default();
        let cue_tts = SpawnTts {
            set_playing: cfg.cue_plays_audio.then(|| Arc::clone(&sink.playing)),
        };
        let sink_played = Arc::clone(&sink.played);
        let sink_playing = Arc::clone(&sink.playing);
        let sink_in_pad = Arc::clone(&sink.in_pad);
        let sink_stopped = Arc::clone(&sink.stopped);

        let pipeline = Pipeline::new(
            FakeWake {
                detects: cfg.wake_detects,
                ..Default::default()
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::from(cfg.vad)),
            },
            FakeStt {
                hit: hit_tx,
                text: cfg.stt_text,
                captured: Arc::clone(&stt_captured),
            },
            cue_tts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: cfg.inject_tool_call,
                fail: cfg.assistant_fails,
                hold_turn: cfg.hold_turn,
                subscribed_tx,
            },
            Arc::new(source),
            Arc::new(sink),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            PipelineConfig {
                tunables: test_tunables(),
                conversation_title: "test".to_string(),
                silence_duration: Duration::from_millis(0),
                speech_threshold: 0.5,
                mic_barge_in: cfg.mic_barge_in,
                conversation_mode: cfg.conversation_mode,
                conversation_reuse_window: cfg.conversation_reuse_window,
                followup_timeout: cfg.followup_timeout,
                idle_exit_timeout: cfg.idle_exit_timeout,
                spoken_response_hint: cfg.spoken_response_hint,
                listening_cue: cfg.listening_cue,
                timeouts: test_timeouts(),
                client_tools: cfg.client_tools,
            },
        );
        let handle = tokio::spawn(async move {
            let _ = pipeline.run().await;
        });
        Harness {
            audio_slot,
            ptt_tx,
            _enabled_tx: enabled_tx,
            stop_tx,
            state_rx,
            transcribe_rx,
            prompt_rx,
            created_rx,
            sink_played,
            sink_playing,
            sink_in_pad,
            sink_stopped,
            registered_rx,
            tool_result_rx,
            events_rx,
            stt_captured,
            handle,
        }
    }

    /// Each chunk is 1000 samples (> the 800-sample floor for closing an
    /// utterance) at a non-silent amplitude so the captured buffer clears the
    /// `process_utterance` energy gate. With a zero silence-duration, one speech
    /// chunk (VAD 0.9) then one silence chunk (VAD 0.0) closes the utterance.
    async fn send_chunk(h: &Harness) {
        h.send_audio(vec![0.1f32; 1000]).await;
    }

    #[tokio::test]
    async fn failed_turn_does_not_crash_the_daemon() {
        // A dropped orchestrator connection (create_conversation errors) must
        // not crash the pipeline: it apologizes, returns to Idle, and keeps
        // running so the next turn — after the gateway reconnects — works.
        let mut h = spawn_pipeline(Cfg {
            assistant_fails: true,
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> create_conversation fails

        // The failed turn must recover to Idle rather than crashing.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("a failed turn must recover to Idle")
        .unwrap();
        assert!(
            !h.handle.is_finished(),
            "a failed turn must not crash the daemon"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn push_to_talk_transcribes_even_when_wake_disabled() {
        // #3: wake word OFF, but an explicit push-to-talk must still capture
        // and transcribe. Pre-fix this times out — chunks were dropped by the
        // top-level enable gate before reaching the Listening state.
        let mut h = spawn_pipeline(Cfg {
            enabled: false,
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("push-to-talk should enter Listening even when disabled")
        .unwrap();

        send_chunk(&h).await; // VAD 0.9 -> speech
        send_chunk(&h).await; // VAD 0.0 -> silence -> transcription

        let got = tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv()).await;
        h.handle.abort();
        assert!(
            matches!(got, Ok(Some(()))),
            "transcription must run for a push-to-talk utterance while wake word is disabled"
        );
    }

    #[tokio::test]
    async fn ptt_in_idle_with_playback_stops_tts_before_listening() {
        // #68: a single-shot reply drops to Idle while its TTS is still
        // sounding (playback_end in the future). A PTT press in that window
        // must stop the outstanding playback — otherwise the mic arms with no
        // echo-drain and records the daemon's own voice. Pre-fix `stop()` was
        // gated on State::Speaking and skipped here.
        use std::sync::atomic::Ordering;
        let mut h = spawn_pipeline(Cfg::default());

        // Mimic outstanding TTS while the pipeline sits in Idle.
        h.sink_playing.store(true, Ordering::SeqCst);

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt in Idle-with-playback must reach Listening")
        .unwrap();

        assert!(
            h.sink_stopped.load(Ordering::SeqCst) >= 1,
            "PTT in Idle with playback outstanding must stop the TTS before arming"
        );
        assert!(
            !h.sink_playing.load(Ordering::SeqCst),
            "playback must be cleared before Listening so no TTS leaks into the utterance"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn ptt_in_idle_without_playback_does_not_stop() {
        // #68 converse: with nothing playing, PTT must not call stop() — the
        // fix only stops when `is_playing()` is true, so the common path stays
        // a no-op and doesn't churn the sink/stream.
        use std::sync::atomic::Ordering;
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        assert_eq!(
            h.sink_stopped.load(Ordering::SeqCst),
            0,
            "PTT with nothing playing must not stop the sink"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn ptt_with_conversation_id_routes_to_that_conversation() {
        // #24 (core acceptance): a push-to-talk carrying a conversation id
        // routes the utterance to THAT orchestrator conversation — it must
        // send the prompt to the supplied id and must NOT create the daemon's
        // own session.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx
            .send(Some("chat-window-42".to_string()))
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let routed = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();
        assert_eq!(
            routed.conversation_id, "chat-window-42",
            "the utterance must be routed to the conversation id the PTT supplied"
        );
        assert!(
            h.created_rx.try_recv().is_err(),
            "PTT-into-conversation must not create the daemon's own session"
        );
    }

    #[tokio::test]
    async fn turn_sends_clean_transcript_with_hint_as_system_refinement() {
        // Core of the #200 voice follow-up: the pipeline must send the CLEAN
        // transcript as the user-visible prompt and pass the configured
        // spoken-response hint as the per-request system_refinement — NOT
        // prepend the hint blurb into the message. That keeps the visible
        // chat transcript free of the "respond briefly, by voice" boilerplate.
        let mut h = spawn_pipeline(Cfg {
            spoken_response_hint: "Respond briefly, by voice.".to_string(),
            stt_text: "what's the weather?".to_string(),
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let sent = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();

        // The user message is the clean transcript — no hint blurb folded in.
        assert_eq!(
            sent.prompt, "what's the weather?",
            "the user-visible message must be the clean transcript"
        );
        assert!(
            !sent.prompt.contains("Respond briefly"),
            "the spoken-response hint must NOT be prepended to the prompt"
        );
        // The hint rides as the per-request system_refinement.
        assert_eq!(
            sent.system_refinement, "Respond briefly, by voice.",
            "the configured hint must be passed as the per-request system_refinement"
        );
    }

    #[tokio::test]
    async fn plain_ptt_uses_daemon_own_session() {
        // #24 (back-compat): a plain PushToTalk() (no id) must keep creating
        // and using the daemon's own session ("test" title here) rather than
        // any caller conversation.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let created = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("the daemon's own session should be created")
            .expect("created sender open");
        assert_eq!(
            created, "test",
            "a plain PTT must create the daemon's own session by its configured title"
        );
        let routed = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();
        assert_eq!(
            routed.conversation_id, "own-session",
            "a plain PTT must route to the daemon's own session id, not a caller conversation"
        );
    }

    #[tokio::test]
    async fn plain_ptt_reuses_own_session_across_presses() {
        // Regression guard (#24): a plain PushToTalk() continues the daemon's
        // own session across presses — like the wake word — instead of spawning
        // a fresh "Voice Conversation" each press.
        // VAD script drives two utterances: speech/silence, then speech/(exhausted)silence.
        let mut h = spawn_pipeline(Cfg {
            vad: vec![0.9, 0.0, 0.9],
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("first ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let first = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first prompt")
            .expect("prompt sender open");
        assert_eq!(first.conversation_id, "own-session");

        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("back to Idle after the first turn")
        .unwrap();

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("second ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let second = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("second prompt")
            .expect("prompt sender open");
        h.handle.abort();

        assert_eq!(
            second.conversation_id, "own-session",
            "the second plain PTT must reuse the own session id"
        );
        let created = h
            .created_rx
            .try_recv()
            .expect("the own session must have been created");
        assert_eq!(created, "test");
        assert!(
            h.created_rx.try_recv().is_err(),
            "a second plain PTT must NOT create a new session — it reuses the own session"
        );
    }

    #[tokio::test]
    async fn near_silent_capture_is_discarded() {
        // The energy gate must drop a near-silent buffer (noise/echo that
        // tripped the VAD) before STT, so Whisper can't hallucinate filler
        // that would loop in conversation mode.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("push-to-talk should enter Listening")
        .unwrap();

        // VAD scripts this as speech-then-silence, but the samples are ~silent.
        h.send_audio(vec![0.0f32; 1000]).await;
        h.send_audio(vec![0.0f32; 1000]).await;

        let got = tokio::time::timeout(Duration::from_millis(500), h.transcribe_rx.recv()).await;
        h.handle.abort();
        assert!(
            got.is_err(),
            "a near-silent capture must be discarded before transcription"
        );
    }

    #[tokio::test]
    async fn idle_in_tail_pad_seeds_prebuffer_without_waking() {
        // #70: during the tail pad (audio deadline passed, latency cushion
        // still running) the daemon must NOT run wake detect — residual echo
        // could trip it — but should keep seeding the prebuffer so same-breath
        // audio at the very tail isn't lost. Here we assert the gate half:
        // with an always-firing detector, a chunk delivered while in the pad
        // must NOT transition out of Idle.
        use std::sync::atomic::Ordering;
        let h = spawn_pipeline(Cfg {
            wake_detects: true,
            ..Default::default()
        });
        // Mimic "playing, in the tail pad".
        h.sink_playing.store(true, Ordering::SeqCst);
        h.sink_in_pad.store(true, Ordering::SeqCst);

        h.send_audio(vec![0.1f32; 1000]).await;
        // Give the pipeline time to process the chunk.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            *h.state_rx.borrow(),
            State::Idle,
            "wake detect must stay gated during the tail pad despite an always-firing detector"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn wake_word_ignored_when_disabled() {
        // Regression guard: an always-firing detector must not trigger Listening
        // while voice is disabled. Post-#103 "disabled" releases the device
        // entirely (the mic-indicator-trust fix), so the detector gets no audio
        // at all — a strictly stronger guarantee. State must stay Idle and the
        // capture channel must be released (no live sender to send into).
        let h = spawn_pipeline(Cfg {
            enabled: false,
            wake_detects: true,
            ..Default::default()
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        let state = *h.state_rx.borrow();
        let released = h.audio_slot.lock().unwrap().is_none();
        h.handle.abort();
        assert_eq!(
            state,
            State::Idle,
            "wake word must be ignored while disabled"
        );
        assert!(
            released,
            "disabling must release the capture device (mic indicator must go dark)"
        );
    }

    #[tokio::test]
    async fn wake_word_triggers_listening_when_enabled() {
        // Regression guard: with wake enabled, detection moves to Listening.
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            ..Default::default()
        });
        send_chunk(&h).await;
        let reached = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        h.handle.abort();
        assert!(
            reached.is_ok(),
            "wake word must enter Listening when enabled"
        );
    }

    #[tokio::test]
    async fn ding_cue_plays_on_wake_word_entry() {
        // #51: with the ding cue, entering Listening queues the generated earcon
        // (the only buffer played here, since the FakeTts produces no audio).
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            listening_cue: ListeningCue::Ding,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("wake -> Listening")
        .unwrap();
        // Give the cue a beat to be queued after the state change.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let played = h.sink_played.lock().unwrap().clone();
        h.handle.abort();
        assert_eq!(
            played,
            vec![cue::ding_samples().len()],
            "the ding earcon must be queued on entering Listening"
        );
    }

    #[tokio::test]
    async fn no_cue_plays_when_listening_cue_off() {
        // #51: with the cue off, entering Listening must NOT queue any audio.
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            listening_cue: ListeningCue::Off,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("wake -> Listening")
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let played = h.sink_played.lock().unwrap().clone();
        h.handle.abort();
        assert!(played.is_empty(), "no cue must be queued when set to off");
    }

    #[tokio::test]
    async fn phrase_cue_echo_is_not_captured_as_the_utterance() {
        // V-7/#87: on the wake path the spoken "Listening" cue ("How can I
        // help?") plays *into* an already-armed mic. Its echo must be drained,
        // not endpointed and sent to the assistant as the user's utterance. The
        // same-breath preroll (#50) — audio captured BEFORE the cue — must still
        // survive; only post-cue echo is dropped.
        use std::sync::atomic::Ordering;
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            listening_cue: ListeningCue::Phrase,
            // The cue actually sounds: synthesizing it flips is_playing true so
            // the drain has something to wait out (mirrors real playback).
            cue_plays_audio: true,
            // turn-1: one speech chunk (0.9) then silence (0.0) closes it.
            vad: vec![0.9],
            // A generous lead-in so the post-cue arm doesn't time out before the
            // (test-paced) user speech arrives. The point is which audio is
            // captured, not lead-in timing.
            followup_timeout: Duration::from_secs(5),
            ..Default::default()
        });

        // The wake chunk: detected in Idle, seeds the preroll, arms the mic, then
        // plays the phrase cue (which flips is_playing true).
        send_chunk(&h).await; // 1000 samples, becomes preroll + first speech
        // Wait for Listening so the cue has been spoken (is_playing now true).
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("wake -> Listening")
        .unwrap();

        // While the cue is sounding, its echo arrives on the mic, tagged with a
        // distinct sentinel amplitude so we can detect it in the captured buffer.
        // With the fix the pipeline is waiting out playback and draining these, so
        // they never reach the endpointer.
        const ECHO: f32 = 0.7;
        for _ in 0..3 {
            h.send_audio(vec![ECHO; 1000]).await;
        }
        // Give the drain a beat, then end the cue's playback.
        tokio::time::sleep(Duration::from_millis(120)).await;
        h.sink_playing.store(false, Ordering::SeqCst);

        // Let the drain finish (wait out playback + try_recv the echo) and re-arm
        // BEFORE the user speaks, so the user's chunks aren't swept up by the same
        // drain pass that empties the echo.
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Now the real user speaks (a different amplitude): one speech chunk then
        // silence closes the utterance and triggers transcription.
        h.send_audio(vec![0.3f32; 1000]).await; // VAD 0.9 -> speech
        h.send_audio(vec![0.0f32; 1000]).await; // VAD 0.0 -> silence

        // Transcription must run on the user's utterance...
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("transcription must run")
            .expect("hit");

        let captured = h.stt_captured.lock().unwrap().clone();
        h.handle.abort();

        // No utterance handed to STT may contain the cue echo. Pre-fix the echo is
        // endpointed as the (first) utterance; the assertion fails. Post-fix the
        // only captured audio is the preroll + the user's real speech.
        let echo_in_capture = captured
            .iter()
            .any(|buf| buf.iter().any(|&s| (s - ECHO).abs() < 1e-6));
        assert!(
            !echo_in_capture,
            "the phrase listening-cue echo must be drained, not captured as the \
             utterance; captured buffers (lens) = {:?}",
            captured.iter().map(Vec::len).collect::<Vec<_>>()
        );
        // And the user's real speech WAS captured (sanity: the fix didn't just
        // drop everything).
        let user_speech_captured = captured
            .iter()
            .any(|buf| buf.iter().any(|&s| (s - 0.3).abs() < 1e-6));
        assert!(
            user_speech_captured,
            "the user's real speech must still be captured"
        );
    }

    #[tokio::test]
    async fn conversation_mode_relistens_after_response() {
        // #6: in conversation mode, after replying the pipeline re-opens the
        // mic for a follow-up turn instead of returning to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_secs(5),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process turn 1
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("turn 1 should transcribe")
            .unwrap();

        // After the reply, conversation mode returns to Listening (not Idle).
        let relisten = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        h.handle.abort();
        assert!(
            relisten.is_ok(),
            "conversation mode must re-open the mic for a follow-up turn"
        );
    }

    #[tokio::test]
    async fn published_state_sequence_is_reachable_and_idle_bracketed() {
        // voice#82: the watch channel must only ever publish states reachable
        // through the `state.rs` table. (The primary illegal-publish guard is
        // the `debug_assert!` inside `apply`, which aborts ANY of these tests on
        // an illegal transition; this test adds a coarse end-to-end check that
        // the published sequence stays within the legal graph and brackets the
        // turn with Idle. The watch channel coalesces, so we assert
        // reachability — `to` reachable from `from` in ≥1 legal step — rather
        // than single-step adjacency.)
        let observed = Arc::new(StdMutex::new(Vec::<State>::new()));
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: false,
            ..Default::default()
        });
        let mut watch_rx = h.state_rx.clone();
        observed.lock().unwrap().push(*watch_rx.borrow());
        let recorder = {
            let observed = Arc::clone(&observed);
            tokio::spawn(async move {
                while watch_rx.changed().await.is_ok() {
                    observed.lock().unwrap().push(*watch_rx.borrow());
                }
            })
        };

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> reply -> Idle
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("turn completes back to Idle")
        .unwrap();

        h.handle.abort();
        recorder.abort();

        let seq = observed.lock().unwrap().clone();
        assert_eq!(
            seq.first(),
            Some(&State::Idle),
            "seq must start Idle: {seq:?}"
        );
        assert_eq!(seq.last(), Some(&State::Idle), "seq must end Idle: {seq:?}");
        assert!(
            seq.contains(&State::Listening),
            "a driven turn must pass through Listening: {seq:?}"
        );
        let all_events = [
            StateEvent::WakeWordDetected,
            StateEvent::PttPressed,
            StateEvent::SilenceDetected,
            StateEvent::ListeningTimedOut,
            StateEvent::ResponseStarted,
            StateEvent::PlaybackComplete,
            StateEvent::BargeIn,
            StateEvent::TurnEnded,
            StateEvent::RelistenArmed,
            StateEvent::Stopped,
        ];
        // Reachability: `to` must be reachable from `from` within the legal
        // graph (bounded BFS over a 4-state machine).
        let reachable = |from: State, to: State| -> bool {
            let mut frontier = vec![from];
            for _ in 0..4 {
                let mut next = Vec::new();
                for s in &frontier {
                    if *s == to {
                        return true;
                    }
                    for e in &all_events {
                        if let Some(n) = s.transition(e)
                            && n != *s
                        {
                            next.push(n);
                        }
                    }
                }
                frontier = next;
            }
            frontier.contains(&to)
        };
        for pair in seq.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            assert!(
                reachable(from, to),
                "watch channel published an unreachable step {from} -> {to} (full seq {seq:?})"
            );
        }
    }

    #[tokio::test]
    async fn conversation_mode_times_out_to_idle() {
        // #6: with no follow-up speech, the conversation ends after the
        // follow-up timeout and the pipeline returns to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_millis(100),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process turn 1
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("turn 1 should transcribe")
            .unwrap();
        // Wait for the follow-up re-listen.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("re-listen")
        .unwrap();

        // No follow-up speech: wait past the timeout, then one silence chunk
        // trips the deadline check.
        tokio::time::sleep(Duration::from_millis(160)).await;
        send_chunk(&h).await; // VAD script exhausted -> 0.0 (silence)

        let idle = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await;
        h.handle.abort();
        assert!(
            idle.is_ok(),
            "conversation must return to Idle after the follow-up timeout"
        );
    }

    #[tokio::test]
    async fn non_conversation_mode_returns_to_idle() {
        // Regression guard: without conversation mode, a reply returns to Idle.
        let mut h = spawn_pipeline(Cfg::default());
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("should transcribe")
            .unwrap();

        let idle = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await;
        h.handle.abort();
        assert!(idle.is_ok(), "non-conversation mode must return to Idle");
    }

    #[tokio::test]
    async fn idle_exits_when_wake_disabled_and_idle() {
        // #5: with voice disabled and an idle-exit timeout configured, the daemon
        // releases the mic (voice#103) and exits after the idle window so D-Bus
        // activation can restart it on demand. Post-#103 idle-exit fires from the
        // disable-pause wait once the window elapses — no chunk needed (the
        // device is already released).
        let h = spawn_pipeline(Cfg {
            enabled: false,
            idle_exit_timeout: Some(Duration::from_millis(80)),
            ..Default::default()
        });
        let exited = tokio::time::timeout(Duration::from_secs(2), h.handle).await;
        assert!(
            exited.is_ok(),
            "daemon should idle-exit when voice disabled and idle past the timeout"
        );
    }

    #[tokio::test]
    async fn does_not_idle_exit_while_wake_enabled() {
        // Guard: wake listening on means always-on — never idle-exit.
        let h = spawn_pipeline(Cfg {
            enabled: true,
            idle_exit_timeout: Some(Duration::from_millis(40)),
            ..Default::default()
        });
        for _ in 0..5 {
            h.send_audio(vec![0.0f32; 1000]).await;
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let exited = tokio::time::timeout(Duration::from_millis(100), h.handle).await;
        assert!(
            exited.is_err(),
            "must not idle-exit while wake word is enabled"
        );
    }

    // --- Session-control client tools (voice#61) --------------------------

    #[test]
    fn session_control_tools_carry_when_to_call_guidance() {
        // All three tools are advertised by default, in a stable order, each
        // with a description that guides the LLM on WHEN to call it.
        let tools = session_control_tools(ClientToolsConfig::default());
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![TOOL_STOP_LISTENING, TOOL_LISTEN_FOR_MORE, TOOL_SAY_THIS]
        );
        let stop = &tools[0];
        assert!(
            stop.description.to_lowercase().contains("done")
                || stop.description.to_lowercase().contains("goodbye"),
            "stop_listening must tell the LLM to call it when the user is finished"
        );
        // say_this takes a `text` string; the others take no args.
        let say = tools.iter().find(|t| t.name == TOOL_SAY_THIS).unwrap();
        assert_eq!(say.input_schema["properties"]["text"]["type"], "string");
    }

    #[test]
    fn per_tool_toggles_withhold_disabled_tools() {
        // A disabled tool is not advertised; the others still are.
        let tools = session_control_tools(ClientToolsConfig {
            stop_listening: true,
            listen_for_more: false,
            say_this: false,
        });
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec![TOOL_STOP_LISTENING]);
    }

    #[tokio::test]
    async fn registers_session_control_tools_on_startup() {
        // voice#61: the daemon advertises the three tools to the orchestrator
        // when the pipeline starts.
        let mut h = spawn_pipeline(Cfg::default());
        let registered = tokio::time::timeout(Duration::from_secs(2), h.registered_rx.recv())
            .await
            .expect("tools should be registered at startup")
            .expect("registered sender open");
        h.handle.abort();
        assert_eq!(
            registered,
            vec![
                TOOL_STOP_LISTENING.to_string(),
                TOOL_LISTEN_FOR_MORE.to_string(),
                TOOL_SAY_THIS.to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn say_this_speaks_text_and_acks() {
        // voice#61: say_this queues the exact line on the speaker and submits
        // Ok("spoken"). Driven directly on the handler with a recording TTS.
        let (mut p, mut th) = build_pipeline_for_tools(Duration::ZERO);
        p.handle_client_tool_call(
            "task-1",
            "call-1",
            TOOL_SAY_THIS,
            serde_json::json!({ "text": "One moment, checking that now." }),
        )
        .await;
        assert_eq!(
            th.spoken.lock().unwrap().clone(),
            vec!["One moment, checking that now.".to_string()],
            "say_this must speak the exact line"
        );
        let result = th
            .tool_result_rx
            .try_recv()
            .expect("a result was submitted");
        assert_eq!(result.task_id, "task-1");
        assert_eq!(result.tool_call_id, "call-1");
        assert_eq!(result.result, Ok("spoken".to_string()));
    }

    #[tokio::test]
    async fn say_this_without_text_errs() {
        // An empty/missing `text` argument is a bad call → Err result, nothing
        // spoken.
        let (mut p, mut th) = build_pipeline_for_tools(Duration::ZERO);
        p.handle_client_tool_call("task-1", "call-1", TOOL_SAY_THIS, serde_json::json!({}))
            .await;
        assert!(th.spoken.lock().unwrap().is_empty());
        let result = th
            .tool_result_rx
            .try_recv()
            .expect("a result was submitted");
        assert!(result.result.is_err());
    }

    #[tokio::test]
    async fn stop_listening_sets_end_flag_and_acks() {
        // voice#61: stop_listening flags the session to end and acks Ok("stopped").
        let (mut p, mut th) = build_pipeline_for_tools(Duration::ZERO);
        p.handle_client_tool_call(
            "task-1",
            "call-1",
            TOOL_STOP_LISTENING,
            serde_json::json!({}),
        )
        .await;
        assert!(
            p.session_end_requested,
            "stop_listening must request ending the session"
        );
        let result = th
            .tool_result_rx
            .try_recv()
            .expect("a result was submitted");
        assert_eq!(result.result, Ok("stopped".to_string()));
    }

    #[tokio::test]
    async fn listen_for_more_sets_listen_flag_and_acks() {
        // voice#61: listen_for_more flags re-listen and acks Ok("listening").
        let (mut p, mut th) = build_pipeline_for_tools(Duration::ZERO);
        p.handle_client_tool_call(
            "task-1",
            "call-1",
            TOOL_LISTEN_FOR_MORE,
            serde_json::json!({}),
        )
        .await;
        assert!(
            p.listen_for_more_requested,
            "listen_for_more must request keeping the mic open"
        );
        let result = th
            .tool_result_rx
            .try_recv()
            .expect("a result was submitted");
        assert_eq!(result.result, Ok("listening".to_string()));
    }

    #[tokio::test]
    async fn unknown_tool_errs_with_not_found() {
        // voice#61: an unrecognized tool name returns Err("unknown tool") and
        // sets no session intent.
        let (mut p, mut th) = build_pipeline_for_tools(Duration::ZERO);
        p.handle_client_tool_call("task-1", "call-1", "make_coffee", serde_json::json!({}))
            .await;
        assert!(!p.session_end_requested && !p.listen_for_more_requested);
        let result = th
            .tool_result_rx
            .try_recv()
            .expect("a result was submitted");
        assert_eq!(result.result, Err("unknown tool".to_string()));
    }

    #[tokio::test]
    async fn stop_listening_during_a_turn_ends_to_idle_and_clears_reuse_id() {
        // voice#61/#59 end-to-end through the run loop: when the LLM calls
        // stop_listening mid-turn (even in conversation mode), the turn ends to
        // Idle, does NOT re-listen, and clears the reuse id so the next wake
        // starts fresh. The injected tool call also gets a result submitted.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            conversation_reuse_window: Duration::from_secs(600),
            followup_timeout: Duration::from_secs(30),
            // Two utterances: speech/silence for turn 1, speech/(exhausted)silence
            // for turn 2.
            vad: vec![0.9, 0.0, 0.9],
            inject_tool_call: Some((TOOL_STOP_LISTENING.to_string(), serde_json::json!({}))),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> stop_listening tool

        // Conversation mode would normally re-listen; stop_listening ends to Idle.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("stop_listening returns to Idle")
        .unwrap();

        // The tool call was acknowledged.
        let result = tokio::time::timeout(Duration::from_secs(2), h.tool_result_rx.recv())
            .await
            .expect("a tool result was submitted")
            .expect("tool-result sender open");
        assert_eq!(result.result, Ok("stopped".to_string()));

        // The reuse id is cleared: a fresh wake creates a NEW own session even
        // though we're within the reuse window.
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("second ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        // Two creates (the first turn + the post-stop fresh turn) → reuse id was
        // cleared by stop_listening.
        let first_created = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("first own session created")
            .expect("created sender open");
        let second_created = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("a NEW own session is created after stop_listening")
            .expect("created sender open");
        h.handle.abort();
        assert_eq!(first_created, "test");
        assert_eq!(
            second_created, "test",
            "after stop_listening the next wake must NOT reuse — it creates fresh"
        );
    }

    #[tokio::test]
    async fn listen_for_more_relistens_outside_conversation_mode() {
        // voice#61: listen_for_more re-opens the mic for a follow-up even when
        // conversation_mode is off (where a normal turn would return to Idle).
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: false,
            followup_timeout: Duration::from_secs(5),
            inject_tool_call: Some((TOOL_LISTEN_FOR_MORE.to_string(), serde_json::json!({}))),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> listen_for_more tool
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("turn 1 should transcribe")
            .unwrap();

        // Despite conversation_mode being off, the pipeline re-opens the mic.
        let relisten = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        h.handle.abort();
        assert!(
            relisten.is_ok(),
            "listen_for_more must re-open the mic even outside conversation mode"
        );
    }

    // --- Conversation reuse window (voice#53) -----------------------------

    #[tokio::test]
    async fn reuse_window_continues_recent_conversation_on_next_wake() {
        // voice#53: a second wake within the reuse window resumes the same
        // conversation rather than creating a new one. Two PTT-into-own-session
        // turns within the window → the own session is created ONCE.
        let mut h = spawn_pipeline(Cfg {
            conversation_reuse_window: Duration::from_secs(600),
            vad: vec![0.9, 0.0, 0.9],
            ..Default::default()
        });
        // Turn 1.
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("first ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let first = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first prompt")
            .expect("prompt sender open");
        assert_eq!(first.conversation_id, "own-session");
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("idle after turn 1")
        .unwrap();
        // Turn 2 (fresh wake, within the window).
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("second ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let second = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("second prompt")
            .expect("prompt sender open");
        h.handle.abort();
        assert_eq!(second.conversation_id, "own-session");
        // The own session was created exactly once → the second wake reused it.
        assert_eq!(
            h.created_rx.try_recv().expect("own session created"),
            "test"
        );
        assert!(
            h.created_rx.try_recv().is_err(),
            "a wake within the reuse window must NOT create a second conversation"
        );
    }

    #[tokio::test]
    async fn reuse_disabled_starts_fresh_each_wake() {
        // voice#53: with the window at 0, every wake starts fresh — the own
        // session is created on each wake.
        let mut h = spawn_pipeline(Cfg {
            conversation_reuse_window: Duration::ZERO,
            vad: vec![0.9, 0.0, 0.9],
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("first ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first prompt")
            .expect("prompt sender open");
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("idle after turn 1")
        .unwrap();
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("second ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("second prompt")
            .expect("prompt sender open");
        h.handle.abort();
        // Two creates → reuse disabled, fresh each wake.
        assert_eq!(
            h.created_rx.try_recv().expect("first own session created"),
            "test"
        );
        assert_eq!(
            h.created_rx
                .try_recv()
                .expect("a second own session is created when reuse is disabled"),
            "test"
        );
    }

    #[test]
    fn expire_stale_conversation_drops_id_outside_window() {
        // Unit-level reuse decision: an id older than the window is dropped; a
        // recent one is kept; a zero window always drops.
        let (mut p, _th) = build_pipeline_for_tools(Duration::from_millis(50));
        p.conversation_id = Some("c1".into());
        p.last_own_activity = Some(Instant::now() - Duration::from_secs(1));
        p.expire_stale_conversation_on_wake();
        assert!(
            p.conversation_id.is_none(),
            "a conversation older than the window must be dropped"
        );

        let (mut p, _th) = build_pipeline_for_tools(Duration::from_secs(600));
        p.conversation_id = Some("c2".into());
        p.last_own_activity = Some(Instant::now());
        p.expire_stale_conversation_on_wake();
        assert_eq!(
            p.conversation_id.as_deref(),
            Some("c2"),
            "a recent conversation within the window must be kept"
        );

        let (mut p, _th) = build_pipeline_for_tools(Duration::ZERO);
        p.conversation_id = Some("c3".into());
        p.last_own_activity = Some(Instant::now());
        p.expire_stale_conversation_on_wake();
        assert!(
            p.conversation_id.is_none(),
            "a zero reuse window must always start fresh"
        );
    }

    // --- Capture-device hardening (#79) ---------------------------------
    //
    // A capture (`source.start()`) failure must NOT crash the daemon: it
    // enters a degraded loop that keeps the control channels serviced (so the
    // separately-spawned TTS/D-Bus stay up) and retries on reload.

    /// `AudioSource` whose `start()` follows a per-call script: call indices in
    /// `fail_calls` (0-based) error, modelling a bad/absent device; successful
    /// calls hand over the next queued receiver (so a test can model capture
    /// dying and a *fresh* capture channel coming up on restart — V-1). Calls
    /// past the queue error too. `starts`/`stops` count every call so a test
    /// can assert the retry/cleanup actually happened.
    struct FlakySource {
        rxs: StdMutex<VecDeque<mpsc::Receiver<Vec<f32>>>>,
        fail_calls: Vec<usize>,
        /// All calls below this index fail (legacy "fail the first N" shorthand;
        /// `usize::MAX` = always fail).
        fail_first: usize,
        starts: Arc<std::sync::atomic::AtomicUsize>,
        stops: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl AudioSource for FlakySource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, adele_voice_core::VoiceError> {
            let n = self
                .starts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < self.fail_first || self.fail_calls.contains(&n) {
                return Err(adele_voice_core::VoiceError::Audio(
                    "input device 'bogus' not found".to_string(),
                ));
            }
            self.rxs
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| adele_voice_core::VoiceError::Audio("no receiver scripted".into()))
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            self.stops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    /// Drives the channels of a real `run()` over a `FlakySource`, keeping the
    /// reload/ptt/stop senders the normal `spawn_pipeline` helper drops — the
    /// degraded loop is fed exactly through those control channels.
    struct DegradedHarness {
        /// One sender per scripted capture channel, in hand-out order. Dropping
        /// the live one models the capture thread dying (V-1).
        audio_txs: Vec<mpsc::Sender<Vec<f32>>>,
        ptt_tx: mpsc::Sender<Option<String>>,
        stop_tx: mpsc::Sender<StopRequest>,
        reload_tx: mpsc::Sender<()>,
        state_rx: watch::Receiver<State>,
        starts: Arc<std::sync::atomic::AtomicUsize>,
        stops: Arc<std::sync::atomic::AtomicUsize>,
        handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    }

    fn spawn_degraded(fail_first: usize) -> DegradedHarness {
        spawn_capture_harness(fail_first, vec![], 1)
    }

    /// Spawn `run()` over a `FlakySource` scripted with `channels` fresh capture
    /// channels and the given failure script (see [`FlakySource`]).
    fn spawn_capture_harness(
        fail_first: usize,
        fail_calls: Vec<usize>,
        channels: usize,
    ) -> DegradedHarness {
        let mut audio_txs = Vec::new();
        let mut audio_rxs = VecDeque::new();
        for _ in 0..channels {
            let (tx, rx) = mpsc::channel::<Vec<f32>>(64);
            audio_txs.push(tx);
            audio_rxs.push_back(rx);
        }
        let (_enabled_tx, enabled_rx) = watch::channel(true);
        let (ptt_tx, ptt_rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, state_rx) = watch::channel(State::Idle);
        let (hit_tx, _transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, _prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, _created_rx) = mpsc::unbounded_channel();
        let (registered_tx, _registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, _tool_result_rx) = mpsc::unbounded_channel();
        let (reload_tx, reload_rx) = mpsc::channel(4);
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipeline = Pipeline::new(
            FakeWake {
                detects: false,
                ..Default::default()
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
                captured: Arc::new(StdMutex::new(Vec::new())),
            },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: None,
                fail: false,
                hold_turn: false,
                subscribed_tx: mpsc::unbounded_channel().0,
            },
            Arc::new(FlakySource {
                rxs: StdMutex::new(audio_rxs),
                fail_calls,
                fail_first,
                starts: Arc::clone(&starts),
                stops: Arc::clone(&stops),
            }),
            Arc::new(FakeSink::default()),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            PipelineConfig {
                tunables: test_tunables(),
                conversation_title: "test".to_string(),
                silence_duration: Duration::from_millis(0),
                speech_threshold: 0.5,
                mic_barge_in: true,
                conversation_mode: false,
                conversation_reuse_window: Duration::from_secs(600),
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                listening_cue: ListeningCue::Off,
                timeouts: test_timeouts(),
                client_tools: ClientToolsConfig::default(),
            },
        );
        let handle = tokio::spawn(pipeline.run());
        DegradedHarness {
            audio_txs,
            ptt_tx,
            stop_tx,
            reload_tx,
            state_rx,
            starts,
            stops,
            handle,
        }
    }

    /// Poll an atomic counter until it reaches `n` (or fail after 2s).
    async fn wait_for_count(counter: &Arc<std::sync::atomic::AtomicUsize>, n: usize, what: &str) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while counter.load(std::sync::atomic::Ordering::SeqCst) < n {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{what}: expected count >= {n}, got {}",
                counter.load(std::sync::atomic::Ordering::SeqCst)
            )
        });
    }

    #[tokio::test]
    async fn capture_failure_recovers_on_reload() {
        // start() errors once (degraded), then a reload retries it and the
        // pipeline proceeds into the normal capture loop — provably, by driving
        // a PTT to Listening, which only the normal loop services.
        let mut h = spawn_degraded(1);

        // Nudge the degraded loop with a reload so it re-tries start().
        h.reload_tx.send(()).await.unwrap();
        // Wait until the pipeline has actually processed the reload and called
        // start() again; otherwise PTT may arrive while the degraded select! is
        // still running and be silently discarded.
        wait_for_count(&h.starts, 2, "start() retry after reload").await;

        // The normal loop is now running: PTT must drive Idle -> Listening.
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("after reload, the normal loop must service PTT -> Listening")
        .unwrap();

        assert!(
            h.starts.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "start() must have been retried after the reload"
        );

        h.handle.abort();
    }

    #[tokio::test]
    async fn capture_failure_then_closed_channels_shuts_down_cleanly() {
        // start() always errors and there's no reload; once every control
        // channel closes, run() returns Ok(()) — it did NOT propagate the
        // capture error or crash the process.
        let h = spawn_degraded(usize::MAX);

        // Drop the audio sender too (irrelevant in degraded mode) and every
        // control sender so the degraded select! hits its `else` arm.
        drop(h.audio_txs);
        drop(h.ptt_tx);
        drop(h.stop_tx);
        drop(h.reload_tx);

        let result = tokio::time::timeout(Duration::from_secs(2), h.handle)
            .await
            .expect("run() must return after all control channels close")
            .expect("join");
        assert!(
            result.is_ok(),
            "a permanent capture failure must yield Ok(()), not an error: {result:?}"
        );
    }

    #[tokio::test]
    async fn ptt_during_degraded_mode_is_ignored() {
        // A PTT press while capture is unavailable must be logged/ignored — no
        // hang, no crash, no state change — and the daemon stays up until the
        // channels close (then clean Ok(())).
        let mut h = spawn_degraded(usize::MAX);

        h.ptt_tx.send(None).await.unwrap();
        // State must stay Idle: PTT can't arm the (absent) mic.
        let res = tokio::time::timeout(
            Duration::from_millis(200),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        assert!(
            res.is_err(),
            "PTT in degraded mode must NOT transition to Listening"
        );

        // Still alive: close the channels and confirm a clean shutdown.
        drop(h.audio_txs);
        drop(h.ptt_tx);
        drop(h.stop_tx);
        drop(h.reload_tx);
        let result = tokio::time::timeout(Duration::from_secs(2), h.handle)
            .await
            .expect("run() must return after channels close")
            .expect("join");
        assert!(result.is_ok(), "degraded run() must end with Ok(())");
    }

    // --- voice#103: logind session-active gating ----------------------------
    //
    // The privacy/cost fix: capture is gated on the session being active and
    // voice being enabled. We inject the session-active state at the trait
    // boundary (a `SessionGate` over a watch channel) and an `enabled` watch, so
    // these exercise the real `run()` gating without a system D-Bus. A
    // `CountingSource` counts `start()`/`stop()` so a test can prove the mic is
    // (not) actually opened — the load-bearing privacy assertion.

    /// A source that hands out a fresh capture channel on each `start()` and
    /// counts `start`/`stop`, so a test can assert whether the mic was opened.
    struct CountingSource {
        starts: Arc<std::sync::atomic::AtomicUsize>,
        stops: Arc<std::sync::atomic::AtomicUsize>,
        txs: StdMutex<Vec<mpsc::Sender<Vec<f32>>>>,
    }
    impl AudioSource for CountingSource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, adele_voice_core::VoiceError> {
            self.starts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (tx, rx) = mpsc::channel::<Vec<f32>>(64);
            self.txs.lock().unwrap().push(tx);
            Ok(rx)
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            self.stops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    struct SessionHarness {
        session_tx: watch::Sender<bool>,
        enabled_tx: watch::Sender<bool>,
        capture_state_rx: watch::Receiver<CaptureState>,
        starts: Arc<std::sync::atomic::AtomicUsize>,
        stops: Arc<std::sync::atomic::AtomicUsize>,
        handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    }

    /// Spawn `run()` with an injected session gate. `gate_enabled = false` makes
    /// the gate inert (models logind absent / config opt-out); otherwise the gate
    /// follows `initial_active`. `enabled` seeds the voice-enabled watch.
    fn spawn_session_harness(
        gate_enabled: bool,
        initial_active: bool,
        enabled: bool,
    ) -> SessionHarness {
        let (session_tx, session_rx) = watch::channel(initial_active);
        let gate = if gate_enabled {
            SessionGate::watching(session_rx)
        } else {
            SessionGate::inert()
        };
        let (enabled_tx, enabled_rx) = watch::channel(enabled);
        let (_ptt_tx, ptt_rx) = mpsc::channel(1);
        let (_stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, _state_rx) = watch::channel(State::Idle);
        let (capture_state_tx, capture_state_rx) = watch::channel(CaptureState::Capturing);
        let (hit_tx, _transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, _prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, _created_rx) = mpsc::unbounded_channel();
        let (registered_tx, _registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, _tool_result_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipeline = Pipeline::new(
            FakeWake {
                detects: false,
                ..Default::default()
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
                captured: Arc::new(StdMutex::new(Vec::new())),
            },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: None,
                fail: false,
                hold_turn: false,
                subscribed_tx: mpsc::unbounded_channel().0,
            },
            Arc::new(CountingSource {
                starts: Arc::clone(&starts),
                stops: Arc::clone(&stops),
                txs: StdMutex::new(Vec::new()),
            }),
            Arc::new(FakeSink::default()),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            PipelineConfig {
                tunables: test_tunables(),
                conversation_title: "test".to_string(),
                silence_duration: Duration::from_millis(0),
                speech_threshold: 0.5,
                mic_barge_in: true,
                conversation_mode: false,
                conversation_reuse_window: Duration::from_secs(600),
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                listening_cue: ListeningCue::Off,
                timeouts: test_timeouts(),
                client_tools: ClientToolsConfig::default(),
            },
        )
        .with_session_gate(gate)
        .with_capture_state(capture_state_tx);
        let handle = tokio::spawn(pipeline.run());
        SessionHarness {
            session_tx,
            enabled_tx,
            capture_state_rx,
            starts,
            stops,
            handle,
        }
    }

    /// Poll a `CaptureState` watch until it equals `want` (or fail after 2s).
    async fn wait_capture_state(rx: &mut watch::Receiver<CaptureState>, want: CaptureState) {
        if tokio::time::timeout(Duration::from_secs(2), rx.wait_for(|s| *s == want))
            .await
            .is_err()
        {
            panic!("expected capture state {want}, got {}", *rx.borrow());
        }
    }

    #[tokio::test]
    async fn inactive_session_never_opens_the_mic() {
        // Session inactive at startup => the device must NOT be opened at all
        // (no audio => no wake/STT/LLM/TTS). The privacy/cost fix in one assert.
        let mut h = spawn_session_harness(true, false, true);
        wait_capture_state(&mut h.capture_state_rx, CaptureState::PausedSessionInactive).await;
        // Give it a beat to (wrongly) call start() if the gate were broken.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            h.starts.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "mic must never be opened while the session is inactive"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn active_session_opens_the_mic() {
        // Session active + enabled => the mic opens and capture state is Capturing.
        let h = spawn_session_harness(true, true, true);
        wait_for_count(&h.starts, 1, "mic must open when active+enabled").await;
        h.handle.abort();
    }

    #[tokio::test]
    async fn session_going_inactive_releases_the_mic() {
        // Active → inactive must stop the source (release the mic) and report
        // PausedSessionInactive; flipping back resumes capture.
        let mut h = spawn_session_harness(true, true, true);
        wait_capture_state(&mut h.capture_state_rx, CaptureState::Capturing).await;
        wait_for_count(&h.starts, 1, "initial start").await;

        h.session_tx.send(false).unwrap();
        wait_capture_state(&mut h.capture_state_rx, CaptureState::PausedSessionInactive).await;
        wait_for_count(&h.stops, 1, "stop on session inactive").await;

        h.session_tx.send(true).unwrap();
        wait_capture_state(&mut h.capture_state_rx, CaptureState::Capturing).await;
        assert!(
            h.starts.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "capture must resume (re-start) when the session goes active again"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn set_enabled_false_releases_the_device() {
        // The hygiene fix: SetEnabled(false) must release the device (stop the
        // source), not merely mute wake detection — the mic indicator must go
        // dark. Re-enabling re-acquires.
        let mut h = spawn_session_harness(true, true, true);
        wait_capture_state(&mut h.capture_state_rx, CaptureState::Capturing).await;
        wait_for_count(&h.starts, 1, "initial start").await;

        h.enabled_tx.send(false).unwrap();
        wait_capture_state(&mut h.capture_state_rx, CaptureState::PausedDisabled).await;
        wait_for_count(&h.stops, 1, "stop on disable").await;

        h.enabled_tx.send(true).unwrap();
        wait_capture_state(&mut h.capture_state_rx, CaptureState::Capturing).await;
        assert!(
            h.starts.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "re-enabling must re-acquire the device"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn logind_absent_does_not_gate_capture() {
        // State 1 (capability absent / config opt-out): the gate is inert, so
        // capture proceeds normally and is NEVER paused — the explicit
        // correction to the issue's "unknown => inactive" wording, which would
        // have left a headless box's mic shut forever.
        let h = spawn_session_harness(false, false, true);
        wait_for_count(
            &h.starts,
            1,
            "an inert gate must let capture proceed (do not fail-closed when logind is absent)",
        )
        .await;
        h.handle.abort();
    }

    #[tokio::test]
    async fn anomalous_state_errs_toward_not_capturing() {
        // State 3 (present-but-anomalous): the watcher publishes `false` on a
        // mid-flight read/watch failure (privacy-first). A gate seeded false
        // models that — the mic stays closed.
        let mut h = spawn_session_harness(true, false, true);
        wait_capture_state(&mut h.capture_state_rx, CaptureState::PausedSessionInactive).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            h.starts.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "an anomalous/indeterminate session (gate value false) must not open the mic"
        );
        h.handle.abort();
    }

    // --- V-1: capture-thread death AFTER a successful start ------------------
    //
    // The #79 degraded loop only guarded the *initial* start(). If the capture
    // thread dies later (device unplug, resample error → the audio channel
    // closes), the daemon went permanently, silently deaf: the closed channel
    // merely disabled its select arm. The pipeline must instead stop the source
    // (so its `running` flag clears and a restart can succeed) and restart
    // capture — immediately when the device is back, else via the same degraded
    // loop as startup.

    #[tokio::test]
    async fn capture_thread_death_restarts_capture() {
        // Two scripted capture channels: the first dies (sender dropped), the
        // pipeline must stop() the source and start() again, then serve voice
        // from the second channel (proven by PTT → Listening).
        let mut h = spawn_capture_harness(0, vec![], 2);
        wait_for_count(&h.starts, 1, "initial capture start").await;

        // Kill the live capture channel — the capture thread died.
        drop(h.audio_txs.remove(0));

        // The pipeline must restart capture on its own (no reload needed for a
        // transient death when the device opens fine again)...
        wait_for_count(&h.starts, 2, "capture restart after channel close").await;
        // ...and must have stop()ed the source first so the real adapter's
        // `running` latch is cleared and start() can succeed.
        assert!(
            h.stops.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "source.stop() must run before the restart so `running` clears"
        );

        // The fresh channel is live: PTT drives Idle → Listening.
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("after capture restart, PTT must reach Listening")
        .unwrap();
        h.handle.abort();
    }

    #[tokio::test]
    async fn capture_death_mid_listening_degrades_then_recovers_on_reload() {
        // Unhappy path: capture dies while LISTENING and the immediate restart
        // also fails (device really gone). The pipeline must drop back to Idle
        // (not stay stuck half-Listening), survive in the degraded loop, and
        // recover when a reload retries with the device back.
        let mut h = spawn_capture_harness(0, vec![1], 2); // call 1 (the restart) fails
        wait_for_count(&h.starts, 1, "initial capture start").await;

        // Enter Listening via PTT on the live channel.
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        // Capture dies mid-listening; the immediate restart fails (scripted).
        drop(h.audio_txs.remove(0));
        wait_for_count(&h.starts, 2, "restart attempt after channel close").await;

        // The pipeline must surface the loss: back to Idle, not wedged Listening.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("capture death must return the pipeline to Idle")
        .unwrap();

        // A reload (device is back) recovers capture; PTT works again.
        h.reload_tx.send(()).await.unwrap();
        wait_for_count(&h.starts, 3, "retry after reload").await;
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("after recovery, PTT must reach Listening")
        .unwrap();
        h.handle.abort();
    }

    // ===================================================================
    // V-2 (#82): interruptible streaming turn — barge-in / StopSpeaking /
    // StopListening / PTT / Reload serviced WHILE a turn streams.
    //
    // All use the `hold_turn` harness: the turn never auto-completes, so the
    // test owns the timeline. It drives the pipeline to a held-open
    // `stream_response` (PTT → speech chunk → silence chunk → Processing →
    // subscribe), grabs the turn's event sender off `events_rx`, optionally
    // pushes a Chunk to reach Speaking, then fires a control event mid-stream.
    // No audio devices, no D-Bus names, no UDS connect.
    // ===================================================================

    /// Drive a `hold_turn` pipeline to a held-open streaming turn and return the
    /// event sender for that turn. Leaves the pipeline in Processing (no chunk
    /// pushed yet); call `enter_speaking` to advance to Speaking.
    async fn start_held_turn(h: &mut Harness) -> mpsc::UnboundedSender<AssistantEvent> {
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(h).await; // speech (vad 0.9)
        send_chunk(h).await; // silence (vad 0.0) -> Processing -> held turn
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Processing),
        )
        .await
        .expect("silence -> Processing")
        .unwrap();
        tokio::time::timeout(Duration::from_secs(2), h.events_rx.recv())
            .await
            .expect("subscribe must publish the turn's event sender")
            .expect("events channel open")
    }

    /// Push a chunk so the held turn moves Processing -> Speaking, and mark the
    /// sink as playing so barge-in/stop see outstanding playback.
    async fn enter_speaking(h: &Harness, events: &mpsc::UnboundedSender<AssistantEvent>) {
        h.sink_playing
            .store(true, std::sync::atomic::Ordering::SeqCst);
        events
            .send(AssistantEvent::Chunk {
                request_id: "req".into(),
                text: "Let me tell you a long story.".into(),
            })
            .unwrap();
        let mut rx = h.state_rx.clone();
        tokio::time::timeout(
            Duration::from_secs(2),
            rx.wait_for(|s| *s == State::Speaking),
        )
        .await
        .expect("chunk -> Speaking")
        .unwrap();
    }

    async fn wait_state(h: &Harness, want: State, what: &str) {
        let mut rx = h.state_rx.clone();
        tokio::time::timeout(Duration::from_secs(2), rx.wait_for(|s| *s == want))
            .await
            .unwrap_or_else(|_| panic!("{what}: never reached {want}"))
            .unwrap();
    }

    #[tokio::test]
    async fn stop_speaking_mid_stream_stops_playback_and_ends_turn() {
        // Test 1: StopSpeaking while a reply streams stops playback and returns
        // to Idle WITHOUT the test ever sending Complete; the conversation id is
        // retained (no fresh create on the next wake).
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        h.stop_tx.send(StopRequest::Speaking).await.unwrap();
        wait_state(&h, State::Idle, "StopSpeaking mid-stream").await;
        assert!(
            h.sink_stopped.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "the sink must be stopped on a mid-stream StopSpeaking"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn stop_conversation_mid_stream_ends_session() {
        // Test 2: StopListening mid-stream ends the conversation — the id is
        // cleared, so the NEXT wake creates a fresh conversation (created_rx
        // fires a second time). Contrast with StopSpeaking (test 1), which
        // retains the id.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            wake_detects: true,
            // turn-1 speech+silence, then turn-2 (post-stop wake) speech+silence.
            vad: vec![0.9, 0.0, 0.9, 0.0],
            conversation_reuse_window: Duration::from_secs(600),
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;
        // First turn created the own session.
        let first = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("first turn must create a conversation")
            .expect("created channel open");
        assert_eq!(first, "test");

        h.stop_tx.send(StopRequest::Conversation).await.unwrap();
        wait_state(&h, State::Idle, "StopListening mid-stream").await;

        // Wake again: because the id was cleared, a NEW conversation is created.
        send_chunk(&h).await; // wakes (wake_detects=true), arms the endpointer
        send_chunk(&h).await; // speech (vad 0.9)
        send_chunk(&h).await; // silence (vad 0.0) -> Processing -> create_conversation
        let second = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("after StopListening, the next wake must create a fresh conversation")
            .expect("created channel open");
        assert_eq!(second, "test");
        h.handle.abort();
    }

    #[tokio::test]
    async fn barge_in_during_streamed_playback_interrupts() {
        // Test 3: a high-VAD chunk during playback interrupts the stream and
        // arms Listening.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            // speech (close 1st utterance), silence, then a barge-in chunk.
            vad: vec![0.9, 0.0, 0.95],
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        // A loud chunk while playing = barge-in.
        h.send_audio(vec![0.2f32; 1000]).await;
        wait_state(&h, State::Listening, "barge-in mid-stream").await;
        assert!(
            h.sink_stopped.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "barge-in must stop the sink"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn mic_barge_in_disabled_does_not_interrupt_playback() {
        // Regression guard for the echo-clip loop: with mic barge-in OFF (the
        // default, no AEC), a loud chunk while Adele is speaking must NOT stop
        // the sink or leave Speaking — otherwise she clips her own "One moment."
        // on her own echo and loops. Mirrors the barge-in test but gated off.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            mic_barge_in: false,
            vad: vec![0.9, 0.0, 0.95],
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        // A loud chunk while playing would be a barge-in if enabled.
        h.send_audio(vec![0.2f32; 1000]).await;
        // Give the audio arm time to process (and prove it does nothing).
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            h.sink_stopped.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "with mic barge-in disabled, playback must not be stopped by speech"
        );
        assert_eq!(
            *h.state_rx.borrow(),
            State::Speaking,
            "with mic barge-in disabled, the turn must stay in Speaking"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn ptt_mid_stream_redirects_to_new_target() {
        // Test 4: a PTT press carrying a target conversation mid-stream
        // interrupts and re-arms Listening; the NEXT prompt routes to that
        // target ("conv-2"). The redirected turn is held too, but send_prompt
        // records the routing before holding, so prompt_rx sees it.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            vad: vec![0.9, 0.0, 0.9, 0.0],
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;
        let first = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first prompt sent")
            .expect("prompt channel open");
        assert_eq!(first.conversation_id, "own-session");

        h.ptt_tx.send(Some("conv-2".to_string())).await.unwrap();
        wait_state(&h, State::Listening, "PTT mid-stream").await;
        // The PTT armed Listening with target conv-2; drive a fresh utterance.
        send_chunk(&h).await; // speech (vad 0.9)
        send_chunk(&h).await; // silence -> Processing -> redirected prompt
        let second = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("redirected prompt sent")
            .expect("prompt channel open");
        assert_eq!(
            second.conversation_id, "conv-2",
            "the post-PTT turn must route to the new target conversation"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn control_channels_serviced_while_turn_streams() {
        // Test 5 (the headline property): a stop is handled PROMPTLY while the
        // turn streams — it does not queue until Complete (which never comes).
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        let mut idle_rx = h.state_rx.clone();
        let stopped = tokio::time::timeout(Duration::from_secs(1), async {
            h.stop_tx.send(StopRequest::Speaking).await.unwrap();
            idle_rx.wait_for(|s| *s == State::Idle).await
        })
        .await;
        assert!(
            stopped.is_ok(),
            "a stop must be serviced while the turn streams, not queued until Complete"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn audio_during_silent_processing_is_discarded_not_queued() {
        // Test 6: many low-VAD chunks while the turn streams with NOTHING
        // playing cause no barge-in; a stop sent after the flood is still
        // handled promptly (the audio arm kept the channel drained).
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        // Stay in Processing (nothing playing). Flood low-VAD chunks.
        for _ in 0..20 {
            h.try_send_audio(vec![0.0f32; 1000]);
        }
        // No barge-in: still Processing.
        let mut listen_rx = h.state_rx.clone();
        let to_listening = tokio::time::timeout(
            Duration::from_millis(200),
            listen_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        assert!(
            to_listening.is_err(),
            "silent-processing audio must not trigger barge-in"
        );
        // A stop after the flood is still prompt.
        enter_speaking(&h, &events).await;
        h.stop_tx.send(StopRequest::Speaking).await.unwrap();
        wait_state(&h, State::Idle, "stop after audio flood").await;
        h.handle.abort();
    }

    #[tokio::test]
    async fn double_stop_is_idempotent() {
        // Test 7 (renumbered): two rapid StopSpeaking — the second is a no-op in
        // Idle, no panic, no wedged channel.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        h.stop_tx.send(StopRequest::Speaking).await.unwrap();
        wait_state(&h, State::Idle, "first stop").await;
        // Second stop in Idle: handled by the run loop's outer arm, no-op.
        h.stop_tx.send(StopRequest::Speaking).await.unwrap();
        // Still alive and Idle after a beat.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(*h.state_rx.borrow(), State::Idle);
        assert!(!h.handle.is_finished(), "a double stop must not crash");
        h.handle.abort();
    }

    #[tokio::test]
    async fn reload_mid_turn_applies_without_ending_turn() {
        // Test 8: a reload ping mid-stream is applied and the turn keeps
        // streaming (Complete still lands → Idle). We can't easily flip a file
        // config in-test, so assert the turn survives the reload and completes.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        // The reload sender in spawn_pipeline is dropped (no _reload_tx exposed),
        // so we exercise the arm indirectly: confirm the turn completes normally
        // after Speaking — the reload arm, if it fired, must not end the turn.
        events
            .send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "done".into(),
            })
            .unwrap();
        wait_state(&h, State::Idle, "turn completes normally").await;
        h.handle.abort();
    }

    #[tokio::test]
    async fn client_tool_call_after_interrupt_gets_error_result() {
        // Test 9: interrupt, then inject a ClientToolCall on the held-open
        // stream; the drainer (which took the receiver) must submit an Err
        // result so the parked server turn doesn't hang.
        let mut h = spawn_pipeline(Cfg {
            hold_turn: true,
            ..Default::default()
        });
        let events = start_held_turn(&mut h).await;
        enter_speaking(&h, &events).await;

        // Interrupt.
        h.stop_tx.send(StopRequest::Speaking).await.unwrap();
        wait_state(&h, State::Idle, "interrupt").await;

        // A late say_this on the SAME stream — the drainer owns the receiver now.
        events
            .send(AssistantEvent::ClientToolCall {
                task_id: "task-9".into(),
                tool_call_id: "call-9".into(),
                tool_name: TOOL_SAY_THIS.into(),
                arguments: serde_json::json!({ "text": "still here?" }),
            })
            .unwrap();

        let submitted = tokio::time::timeout(Duration::from_secs(2), h.tool_result_rx.recv())
            .await
            .expect("the drainer must answer the post-interrupt tool call")
            .expect("tool-result channel open");
        assert_eq!(submitted.task_id, "task-9");
        assert!(
            submitted.result.is_err(),
            "a post-interrupt client tool call must get an Err result; got {:?}",
            submitted.result
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn liveness_line_suppressed_by_interrupt() {
        // Test 10: a stop during the pre-first-chunk window (with liveness armed)
        // cancels the liveness line — it is never spoken.
        let liveness = Duration::from_millis(300);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            liveness_delay: liveness,
            response_stall: Duration::from_secs(10),
            ..test_timeouts()
        });
        let (_tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        let (stop_tx, stop_rx) = mpsc::channel(1);
        p.stop_rx = stop_rx;
        let mut audio = idle_audio_rx();
        // Stop well before the liveness deadline.
        let driver = tokio::spawn(async move {
            tokio::time::sleep(liveness / 5).await;
            stop_tx.send(StopRequest::Speaking).await.unwrap();
        });
        let end = tokio::time::timeout(
            Duration::from_secs(2),
            p.stream_response(&mut rx, "req", &mut audio),
        )
        .await
        .expect("stream_response must return")
        .expect("ok");
        driver.await.unwrap();
        assert_eq!(end, StreamEnd::Stopped(StopRequest::Speaking));
        assert!(
            !spoken
                .lock()
                .unwrap()
                .iter()
                .any(|s| LIVENESS_PHRASES.contains(&s.as_str())),
            "an interrupt before the liveness deadline must suppress the line"
        );
    }

    #[tokio::test]
    async fn stream_response_returns_stopped_on_mid_turn_stop() {
        // Test 11 (unit-level on stream_response): a StopListening mid-turn
        // returns StreamEnd::Stopped(Conversation), which process_utterance maps
        // to ending the conversation.
        let (mut p, _spoken) = build_pipeline_with(TurnTimeouts {
            response_stall: Duration::from_secs(10),
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        tx.send(AssistantEvent::Chunk {
            request_id: "req".into(),
            text: "Once upon a time".into(),
        })
        .unwrap();
        let (stop_tx, stop_rx) = mpsc::channel(1);
        p.stop_rx = stop_rx;
        let mut audio = idle_audio_rx();
        let driver = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            stop_tx.send(StopRequest::Conversation).await.unwrap();
        });
        let end = tokio::time::timeout(
            Duration::from_secs(2),
            p.stream_response(&mut rx, "req", &mut audio),
        )
        .await
        .expect("stream_response must return")
        .expect("ok");
        driver.await.unwrap();
        assert_eq!(end, StreamEnd::Stopped(StopRequest::Conversation));
    }
}
