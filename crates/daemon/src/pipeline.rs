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
use adele_voice_dbus_interface::StopRequest;
use adele_voice_module::{Endpoint, Endpointer, PreBuffer, Speaker, Transcriber};
use tokio::sync::{mpsc, watch};

use crate::config::{self, Tunables, plan_reload};
use crate::cue::{self, ListeningCue};

/// Builds a fresh wake detector at a given sensitivity. rustpotter bakes the
/// detection threshold in at construction, so changing it on reload (config#52)
/// means rebuilding the detector rather than poking a setter.
pub type WakeBuilder<W> = Box<dyn Fn(f32) -> Result<W, VoiceError> + Send>;

/// Spoken when the assistant turn fails — short and human, never the raw error.
const ERROR_APOLOGY: &str = "Sorry, I ran into an error and couldn't answer that.";

/// Spoken when a turn stalls (no progress within the deadline / over budget) so
/// the user isn't left in silent Processing forever (#58).
const STALL_APOLOGY: &str = "Sorry, that's taking too long. Let's try again.";

/// Brief liveness line spoken once on a slow turn that has narrated nothing and
/// streamed nothing yet (e.g. it declared no plan step) — so voice isn't
/// dead-silent. Rare by design, so it stays natural rather than repetitive.
const LIVENESS_PHRASE: &str = "One moment.";

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// The three static session-control client tools the daemon advertises so the
/// LLM can drive the voice session (voice#61).
pub const TOOL_STOP_LISTENING: &str = "stop_listening";
pub const TOOL_LISTEN_FOR_MORE: &str = "listen_for_more";
pub const TOOL_SAY_THIS: &str = "say_this";

/// Which session-control client tools to advertise (voice#61). Mirrors the
/// `[assistant.client_tools]` config toggles without the pipeline depending on
/// the daemon's config module.
#[derive(Debug, Clone, Copy)]
pub struct ClientToolToggles {
    pub stop_listening: bool,
    pub listen_for_more: bool,
    pub say_this: bool,
}

impl Default for ClientToolToggles {
    fn default() -> Self {
        Self {
            stop_listening: true,
            listen_for_more: true,
            say_this: true,
        }
    }
}

/// Build the [`ClientToolSpec`] registrations for the enabled session-control
/// tools (voice#61). The descriptions are written to guide the LLM on WHEN to
/// call each — especially `stop_listening`, which must fire when the user
/// signals they're finished. Returned in a stable order for deterministic tests.
pub fn session_control_tools(toggles: ClientToolToggles) -> Vec<ClientToolSpec> {
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

/// Whole-utterance "stop listening" phrases, matched only against the entire
/// normalized transcript (so "stop" inside a sentence isn't a command). Lets the
/// user end a conversation hands-free.
fn is_stop_phrase(text: &str) -> bool {
    let normalized = text
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '.' | ',' | '!' | '?'))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
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
    state_tx: watch::Sender<State>,
    enabled_rx: watch::Receiver<bool>,
    ptt_rx: mpsc::Receiver<Option<String>>,
    stop_rx: mpsc::Receiver<StopRequest>,
    /// A ping (from the config-file watcher or the D-Bus `Reload` method) asking
    /// the pipeline to re-read the config and apply any changed tunables live
    /// (config#52).
    reload_rx: mpsc::Receiver<()>,
    /// Rebuilds the wake detector when `wake_word.sensitivity` changes.
    wake_builder: WakeBuilder<W>,
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
    /// Bound on each conversation create / subscribe / send round-trip (#58).
    /// `Duration::ZERO` disables.
    connect_timeout: Duration,
    /// Which session-control client tools to advertise to the orchestrator
    /// (voice#61).
    client_tools: ClientToolToggles,
    /// Set within a turn when the LLM calls `stop_listening`: after this turn's
    /// reply is spoken the conversation ends and the next wake starts fresh.
    /// Reset at the start of each utterance (voice#61).
    session_end_requested: bool,
    /// Set within a turn when the LLM calls `listen_for_more`: keep/extend the
    /// listening window for a follow-up even outside conversation mode. Reset at
    /// the start of each utterance (voice#61).
    listen_for_more_requested: bool,
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
        wake_builder: WakeBuilder<W>,
        tunables: Tunables,
        conversation_title: String,
        silence_duration: Duration,
        speech_threshold: f32,
        conversation_mode: bool,
        conversation_reuse_window: Duration,
        followup_timeout: Duration,
        idle_exit_timeout: Option<Duration>,
        spoken_response_hint: String,
        listening_cue: ListeningCue,
        timeouts: TurnTimeouts,
        client_tools: ClientToolToggles,
    ) -> Self {
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
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            wake_builder,
            tunables,
            conversation_id: None,
            ptt_conversation_override: None,
            conversation_title,
            speech_threshold,
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
            connect_timeout: timeouts.connect,
            client_tools,
            session_end_requested: false,
            listen_for_more_requested: false,
        }
    }

    fn set_state(&self, state: State) {
        let _ = self.state_tx.send(state);
        tracing::info!(state = %state, "state changed");
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
                self.conversation_id = None;
                self.last_own_activity = None;
            }
            (None, _) => {}
        }
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
    async fn play_listening_cue(&mut self) {
        match self.listening_cue {
            ListeningCue::Off => {}
            ListeningCue::Ding => {
                if let Err(e) = self.sink.play(cue::ding_samples()) {
                    tracing::warn!("failed to play listening ding cue: {e}");
                }
            }
            ListeningCue::Phrase => {
                let phrase = cue::phrase(self.cue_phrase_counter);
                self.cue_phrase_counter = self.cue_phrase_counter.wrapping_add(1);
                if let Err(e) = self.speaker.say(phrase).await {
                    tracing::warn!("failed to speak listening phrase cue: {e}");
                }
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
        if let Some(sensitivity) = plan.rebuild_wake_sensitivity {
            match (self.wake_builder)(sensitivity) {
                Ok(wake) => {
                    self.wake = wake;
                    tracing::info!(
                        sensitivity,
                        "config reload: rebuilt wake detector for wake_word.sensitivity"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        sensitivity,
                        "config reload: failed to rebuild wake detector, keeping current: {e}"
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

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut audio_rx = self.source.start()?;

        // Advertise the LLM-driven session-control tools once at startup
        // (voice#61). The connection is already up (built in main), so register
        // before listening so the very first turn can use them.
        self.register_session_control_tools().await;

        let mut state = State::Idle;
        self.set_state(state);

        // For idle-exit (#5): time of the last activity other than idle-while-
        // wake-disabled. (Utterance accumulation and the follow-up/lead-in
        // deadline now live in the shared `Endpointer`.)
        let mut last_activity = Instant::now();

        loop {
            tokio::select! {
                // Push-to-talk: skip wake word, go to Listening. The payload is
                // the target conversation: `None` uses the daemon's own
                // session, `Some(id)` routes the utterance to that orchestrator
                // conversation (the in-chat mic button).
                Some(target) = self.ptt_rx.recv() => {
                    if state == State::Idle || state == State::Speaking {
                        // Stop any outstanding playback before arming the mic.
                        // A single-shot reply drops to Idle while its TTS is
                        // still sounding (playback_end in the future), so gating
                        // stop() on State::Speaking let a PTT press in Idle skip
                        // it — leaving `is_playing` true with no drain and
                        // recording the daemon's own voice (#68). Stop whenever
                        // anything is playing, regardless of state; stop() is the
                        // only thing that clears playback_end.
                        if self.speaker.is_playing() {
                            self.speaker.stop()?;
                        }
                        // Belt-and-suspenders: wait out any residual tail and
                        // drop the echo it queued into the mic before arming, so
                        // no in-flight TTS leaks into the PTT utterance — matching
                        // the relisten path.
                        while self.speaker.is_playing() {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                        while audio_rx.try_recv().is_ok() {}
                        // Route this PTT session: `Some(id)` dictates into that
                        // conversation; `None` (plain PushToTalk) falls back to
                        // the daemon's own session, which — like the wake word —
                        // persists across presses for continuity. (A stale
                        // override can't leak in: every press overwrites it, and
                        // the wake-word entry resets it to None.)
                        self.ptt_conversation_override = target.clone();
                        // A plain PTT (own session) is a fresh entry like a wake:
                        // honour the reuse window — keep the recent conversation if
                        // within it, otherwise start fresh (voice#53). A targeted
                        // PTT uses its own id and is unaffected.
                        if target.is_none() {
                            self.expire_stale_conversation_on_wake();
                        }
                        state = State::Listening;
                        self.set_state(state);
                        // Wait (lead-in) for speech to start rather than cutting
                        // on the silence timer from the moment of the press; only
                        // cut after speech-then-silence, or if the lead-in elapses.
                        self.endpointer.arm(Some(self.followup_timeout));
                        self.vad.reset();
                        tracing::info!(
                            target_conversation = target.as_deref().unwrap_or("<own session>"),
                            "push-to-talk activated, waiting for speech"
                        );
                    }
                }

                // Stop: cancel current playback (Speaking) or end the whole
                // conversation and return to wake-word idle.
                Some(req) = self.stop_rx.recv() => {
                    match req {
                        StopRequest::Speaking => {
                            if state == State::Speaking {
                                self.speaker.stop()?;
                                state = State::Idle;
                                self.set_state(state);
                            }
                        }
                        StopRequest::Conversation => {
                            // "Stop listening": end the session now without
                            // waiting out the silence timeout.
                            if state != State::Idle {
                                let _ = self.speaker.stop();
                                state = State::Idle;
                                self.set_state(state);
                            }
                            self.conversation_id = None;
                            self.last_own_activity = None;
                            self.ptt_conversation_override = None;
                            self.endpointer.reset();
                        }
                    }
                }

                // Reload: re-read the config file and apply any changed
                // tunables to the running pipeline (config#52). Triggered by the
                // file watcher (debounced) or the D-Bus `Reload` method.
                Some(()) = self.reload_rx.recv() => {
                    self.reload();
                }

                // Process audio chunks
                Some(chunk) = audio_rx.recv() => {
                    // `enabled` governs only always-on wake-word listening:
                    // push-to-talk (and SayText) must work even when "Hey
                    // Adele" is off, so the gate is scoped to the Idle state
                    // rather than the whole handler (#3).
                    if state == State::Idle && !*self.enabled_rx.borrow() {
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
                    match state {
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
                                if let Some(new_state) = state.transition(&StateEvent::WakeWordDetected) {
                                    state = new_state;
                                    self.set_state(state);
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
                                    // Seed the utterance with the post-wake audio
                                    // so "hey adele <command>" said in one breath
                                    // captures the command (#50). The lead-in still
                                    // applies and the VAD must still confirm speech.
                                    let preroll = self.prebuffer.take();
                                    self.endpointer
                                        .arm_with_preroll(Some(self.followup_timeout), &preroll);
                                    self.vad.reset();
                                    // Audible "Listening" cue (#51) — instant ding
                                    // by default, optional spoken phrase, or off.
                                    self.play_listening_cue().await;
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
                                    tracing::info!(
                                        samples = samples.len(),
                                        "silence detected, transitioning to processing"
                                    );
                                    if let Some(new_state) =
                                        state.transition(&StateEvent::SilenceDetected)
                                    {
                                        state = new_state;
                                        self.set_state(state);

                                        // A failed turn must NOT crash the
                                        // daemon. The orchestrator may have
                                        // restarted and dropped the connection;
                                        // log it, apologize, and end the turn —
                                        // the gateway reconnects so the next
                                        // turn works.
                                        let outcome = match self.process_utterance(samples).await {
                                            Ok(outcome) => outcome,
                                            Err(e) => {
                                                tracing::error!("voice turn failed: {e}");
                                                self.set_state(State::Speaking);
                                                let _ = self.speaker.say(ERROR_APOLOGY).await;
                                                UtteranceOutcome::EndConversation
                                            }
                                        };

                                        // Decide whether to re-listen. A
                                        // `listen_for_more` client tool re-arms
                                        // even outside conversation mode; a plain
                                        // turn re-listens only in conversation
                                        // mode; `stop_listening` / a stop phrase
                                        // ends regardless (voice#61).
                                        let relisten = match outcome {
                                            UtteranceOutcome::EndConversation => false,
                                            UtteranceOutcome::KeepListening => true,
                                            UtteranceOutcome::Continue => self.conversation_mode,
                                        };
                                        if outcome == UtteranceOutcome::EndConversation {
                                            // A stop phrase or the `stop_listening`
                                            // tool ends the conversation regardless
                                            // of mode AND clears the reuse-window id
                                            // so the next wake starts fresh
                                            // (voice#59/#61).
                                            state = State::Idle;
                                            self.set_state(state);
                                            self.conversation_id = None;
                                            self.last_own_activity = None;
                                            self.ptt_conversation_override = None;
                                            self.endpointer.reset();
                                        } else if relisten {
                                            // Re-open the mic for a follow-up turn:
                                            // wait for the reply to finish playing,
                                            // then drop any audio captured during
                                            // playback (echo) before listening again.
                                            while self.speaker.is_playing() {
                                                tokio::time::sleep(Duration::from_millis(50)).await;
                                            }
                                            while audio_rx.try_recv().is_ok() {}
                                            state = State::Listening;
                                            self.set_state(state);
                                            // Cue the follow-up re-listen too (#51),
                                            // then wait for the cue to finish and
                                            // drop the echo it queued into the mic
                                            // before arming, so it isn't captured as
                                            // the follow-up.
                                            self.play_listening_cue().await;
                                            while self.speaker.is_playing() {
                                                tokio::time::sleep(Duration::from_millis(50)).await;
                                            }
                                            while audio_rx.try_recv().is_ok() {}
                                            self.endpointer.arm(Some(self.followup_timeout));
                                            self.vad.reset();
                                        } else {
                                            // Single-shot: back to wake-word idle.
                                            // Drop any PTT-into-conversation target
                                            // so the next own-session turn doesn't
                                            // inherit it.
                                            state = State::Idle;
                                            self.set_state(state);
                                            self.ptt_conversation_override = None;
                                        }
                                    }
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
                                    state = State::Idle;
                                    self.set_state(state);
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
                                if let Some(new_state) = state.transition(&StateEvent::BargeIn) {
                                    state = new_state;
                                    self.set_state(state);
                                    // Seed the endpointer mid-speech so the next
                                    // silence closes this barge-in utterance.
                                    self.endpointer.arm_speaking(&chunk);
                                    self.vad.reset();
                                }
                            } else if !self.speaker.is_playing()
                                && let Some(new_state) =
                                    state.transition(&StateEvent::PlaybackComplete)
                            {
                                // Playback finished naturally
                                state = new_state;
                                self.set_state(state);
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

    async fn process_utterance(&mut self, samples: Vec<f32>) -> anyhow::Result<UtteranceOutcome> {
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

        // A whole-utterance stop phrase ("stop", "never mind", "that's all", …)
        // ends the conversation hands-free: acknowledge briefly and return to
        // wake-word idle instead of sending it to the assistant.
        if is_stop_phrase(&transcript.text) {
            tracing::info!(text = %transcript.text, "stop phrase — ending conversation");
            self.set_state(State::Speaking);
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
                    self.assistant.create_conversation(&self.conversation_title),
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
        let request_id = bounded(
            self.connect_timeout,
            "send_prompt",
            self.assistant.send_prompt_with_system_refinement(
                &conversation_id,
                &transcript.text,
                &self.spoken_response_hint,
            ),
        )
        .await?;

        self.stream_response(&mut signal_rx, &request_id).await?;

        // Mark the own session active so a wake within the reuse window resumes
        // it (voice#53). Skipped for a PTT override (the client owns that
        // conversation) and when the LLM ended the session (cleared below).
        if own_session {
            self.last_own_activity = Some(Instant::now());
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
    ) -> anyhow::Result<()> {
        // The heartbeat / budget / narration clocks use `tokio::time::Instant`
        // so they stay consistent with the `tokio::time::{sleep, timeout}` this
        // loop awaits — and so the paused-time test clock advances them too.
        use tokio::time::Instant as TokioInstant;

        let mut sentence_buf = SentenceBuffer::new(Duration::from_millis(500));
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
        // nothing within this window, speak one brief line so a slow turn that
        // declared no plan step isn't dead-silent. Cancelled by the first status
        // or chunk; fires at most once. `Duration::ZERO` disables.
        let liveness_deadline =
            (!self.liveness_delay.is_zero()).then(|| TokioInstant::now() + self.liveness_delay);
        let mut liveness_spoken = false;

        loop {
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
                            break;
                        }
                    };
                    match event {
                        Some(AssistantEvent::Chunk { request_id: rid, text }) if rid == request_id => {
                            if first_chunk && is_error_response(&text) {
                                tracing::error!(chunk = %text, "assistant streamed an error message; speaking a short apology instead");
                                self.set_state(State::Speaking);
                                self.speaker.say(ERROR_APOLOGY).await?;
                                break;
                            }
                            if first_chunk {
                                first_chunk = false;
                                self.set_state(State::Speaking);
                            }

                            let sentences = sentence_buf.push(&text);
                            for sentence in sentences {
                                self.speaker.say(&sentence).await?;
                            }
                            // Speak a short leading ack the instant it looks
                            // complete (a terminal opener like "Got it —
                            // checking that now." that the boundary scan won't
                            // split until the next token), without waiting (#58).
                            if let Some(ack) = sentence_buf.take_leading_ack(ACK_MAX_WORDS) {
                                self.speaker.say(&ack).await?;
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
                                    self.speaker.say(&remaining).await?;
                                }
                            } else if first_chunk && !full_response.trim().is_empty() {
                                // Nothing was streamed as chunks — e.g. a
                                // tool-using reply delivered as one final block.
                                self.set_state(State::Speaking);
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
                                        self.speaker.say(&sentence).await?;
                                    }
                                    let remaining = sentence_buf.flush();
                                    if !remaining.is_empty() {
                                        self.speaker.say(&remaining).await?;
                                    }
                                }
                            }
                            tracing::info!(streamed = !first_chunk, "assistant response complete");
                            break;
                        }
                        Some(AssistantEvent::Error { request_id: rid, error }) if rid == request_id => {
                            tracing::error!(error = %error, "assistant response error; speaking a short apology");
                            self.set_state(State::Speaking);
                            self.speaker.say(ERROR_APOLOGY).await?;
                            break;
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
                                self.set_state(State::Speaking);
                                self.speaker.say(ERROR_APOLOGY).await?;
                            }
                            break;
                        }
                        _ => {} // Ignore events for other requests
                    }
                }
                // Check for timeout flush while waiting for chunks
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if let Some(sentence) = sentence_buf.flush_if_timeout() {
                        self.speaker.say(&sentence).await?;
                    }
                }
                // Delayed-liveness safety net: a slow turn that has narrated
                // nothing and streamed nothing yet gets one brief line so voice
                // isn't dead-silent. The guard disables it once a status/chunk
                // arrives or it has already fired.
                _ = async {
                    match liveness_deadline {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending::<()>().await,
                    }
                }, if !liveness_spoken && first_chunk && last_status_spoken_at.is_none() => {
                    liveness_spoken = true;
                    self.set_state(State::Speaking);
                    self.speaker.say(LIVENESS_PHRASE).await?;
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
                break;
            }
        }

        Ok(())
    }

    /// Speak the stall apology (best-effort — a failed apology must not turn a
    /// timeout into a crash) and move to Speaking so the run loop returns to
    /// Idle when playback finishes (#58).
    async fn speak_stall_apology(&mut self) {
        self.set_state(State::Speaking);
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
            self.set_state(State::Speaking);
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
                    self.set_state(State::Speaking);
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

    struct FakeWake {
        detects: bool,
    }
    impl WakeWordDetector for FakeWake {
        fn detect(&mut self, _samples: &[f32]) -> Result<bool, adele_voice_core::VoiceError> {
            Ok(self.detects)
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
    /// returns a non-empty transcript so the response cycle proceeds.
    struct FakeStt {
        hit: mpsc::UnboundedSender<()>,
        text: String,
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(
            &self,
            _samples: &[f32],
        ) -> Result<Transcript, adele_voice_core::VoiceError> {
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
        ) -> Result<String, adele_voice_core::VoiceError> {
            if self.fail {
                return Err(adele_voice_core::VoiceError::Assistant(
                    "uds connection closed".to_string(),
                ));
            }
            let _ = self.created_tx.send(title.to_string());
            Ok("own-session".to_string())
        }
        async fn send_prompt(
            &self,
            conversation_id: &str,
            prompt: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            Ok(self.record_and_complete(conversation_id, prompt, ""))
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
        audio_tx: mpsc::Sender<Vec<f32>>,
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
        handle: tokio::task::JoinHandle<()>,
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
        client_tools: ClientToolToggles,
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
                client_tools: ClientToolToggles::default(),
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
        }
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
        let wake_builder: WakeBuilder<FakeWake> =
            Box::new(|_sensitivity| Ok(FakeWake { detects: false }));
        let spoken = Arc::new(StdMutex::new(Vec::new()));
        let pipeline = Pipeline::new(
            FakeWake { detects: false },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
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
            wake_builder,
            test_tunables(),
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            false,
            reuse_window,
            Duration::from_millis(50),
            None,
            String::new(),
            ListeningCue::Off,
            test_timeouts(),
            ClientToolToggles::default(),
        );
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
        let wake_builder: WakeBuilder<FakeWake> =
            Box::new(|_sensitivity| Ok(FakeWake { detects: false }));
        let spoken = Arc::new(StdMutex::new(Vec::new()));
        let pipeline = Pipeline::new(
            FakeWake { detects: false },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
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
            wake_builder,
            test_tunables(),
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            false,
            // Matches test_tunables() so an initial reload is a no-op.
            Duration::from_millis(600_000),
            Duration::from_millis(50),
            None,
            String::new(),
            ListeningCue::Off,
            timeouts,
            ClientToolToggles::default(),
        );
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
    fn apply_plan_rebuilds_wake_detector_on_sensitivity_change() {
        // The wake-sensitivity branch must invoke the builder; the builder here
        // flips `detects` to true so we can observe the swap took effect.
        let mut p = build_pipeline();
        // Replace the builder with one that yields a detector that always fires.
        p.wake_builder = Box::new(|_s| Ok(FakeWake { detects: true }));
        assert!(!p.wake.detect(&[0.0; 10]).unwrap());
        let plan = config::ReloadPlan {
            rebuild_wake_sensitivity: Some(0.9),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert!(
            p.wake.detect(&[0.0; 10]).unwrap(),
            "the wake detector must be rebuilt by the builder on a sensitivity change"
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
        tokio::time::timeout(Duration::from_secs(5), p.stream_response(&mut rx, "req"))
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

        tokio::time::timeout(Duration::from_secs(10), p.stream_response(&mut rx, "req"))
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
        tokio::time::timeout(Duration::from_secs(5), p.stream_response(&mut rx, "req"))
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
        // liveness line so voice isn't dead-silent — then it completes.
        let liveness = Duration::from_millis(500);
        let (mut p, spoken) = build_pipeline_with(TurnTimeouts {
            liveness_delay: liveness,
            response_stall: Duration::from_secs(10), // not the limiting factor
            ..test_timeouts()
        });
        let (tx, mut rx) = mpsc::unbounded_channel::<AssistantEvent>();
        let feeder = tokio::spawn(async move {
            // Nothing for well past the liveness delay, then the reply arrives.
            tokio::time::sleep(liveness * 2).await;
            let _ = tx.send(AssistantEvent::Complete {
                request_id: "req".into(),
                full_response: "the answer".into(),
            });
        });
        tokio::time::timeout(Duration::from_secs(10), p.stream_response(&mut rx, "req"))
            .await
            .expect("stream_response must return")
            .expect("stream_response ok");
        feeder.await.unwrap();
        let said = spoken.lock().unwrap().clone();
        assert_eq!(
            said.iter().filter(|s| *s == LIVENESS_PHRASE).count(),
            1,
            "a slow stepless turn must speak the liveness line exactly once; said: {said:?}"
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
        tokio::time::timeout(Duration::from_secs(10), p.stream_response(&mut rx, "req"))
            .await
            .expect("stream_response must return")
            .expect("stream_response ok");
        feeder.await.unwrap();
        let said = spoken.lock().unwrap().clone();
        assert!(
            !said.iter().any(|s| s == LIVENESS_PHRASE),
            "progress before the delay must suppress the liveness line; said: {said:?}"
        );
        assert!(
            said.iter().any(|s| s == "Looking into that"),
            "the real status should still be narrated; said: {said:?}"
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
        tokio::time::timeout(Duration::from_secs(2), p.stream_response(&mut rx, "req"))
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
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(64);
        let (enabled_tx, enabled_rx) = watch::channel(cfg.enabled);
        let (ptt_tx, ptt_rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, state_rx) = watch::channel(State::Idle);
        let (hit_tx, transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, created_rx) = mpsc::unbounded_channel();
        let (registered_tx, registered_rx) = mpsc::unbounded_channel();
        let (tool_result_tx, tool_result_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);

        let wake_detects = cfg.wake_detects;
        let wake_builder: WakeBuilder<FakeWake> = Box::new(move |_sensitivity| {
            Ok(FakeWake {
                detects: wake_detects,
            })
        });

        let sink = FakeSink::default();
        let sink_played = Arc::clone(&sink.played);
        let sink_playing = Arc::clone(&sink.playing);
        let sink_in_pad = Arc::clone(&sink.in_pad);
        let sink_stopped = Arc::clone(&sink.stopped);

        let pipeline = Pipeline::new(
            FakeWake {
                detects: cfg.wake_detects,
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::from(cfg.vad)),
            },
            FakeStt {
                hit: hit_tx,
                text: cfg.stt_text,
            },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                registered_tx,
                tool_result_tx,
                inject_tool_call: cfg.inject_tool_call,
                fail: cfg.assistant_fails,
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(sink),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            wake_builder,
            test_tunables(),
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            cfg.conversation_mode,
            cfg.conversation_reuse_window,
            cfg.followup_timeout,
            cfg.idle_exit_timeout,
            cfg.spoken_response_hint,
            cfg.listening_cue,
            test_timeouts(),
            cfg.client_tools,
        );
        let handle = tokio::spawn(async move {
            let _ = pipeline.run().await;
        });
        Harness {
            audio_tx,
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
            handle,
        }
    }

    /// Each chunk is 1000 samples (> the 800-sample floor for closing an
    /// utterance) at a non-silent amplitude so the captured buffer clears the
    /// `process_utterance` energy gate. With a zero silence-duration, one speech
    /// chunk (VAD 0.9) then one silence chunk (VAD 0.0) closes the utterance.
    async fn send_chunk(h: &Harness) {
        h.audio_tx.send(vec![0.1f32; 1000]).await.unwrap();
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
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();

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

        h.audio_tx.send(vec![0.1f32; 1000]).await.unwrap();
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
        // Regression guard: an always-firing detector must not trigger
        // Listening while wake-word listening is disabled.
        let h = spawn_pipeline(Cfg {
            enabled: false,
            wake_detects: true,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let state = *h.state_rx.borrow();
        h.handle.abort();
        assert_eq!(
            state,
            State::Idle,
            "wake word must be ignored while disabled"
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
        // #5: with wake listening off and an idle-exit timeout configured, the
        // daemon exits after the idle window so D-Bus activation can restart it.
        let h = spawn_pipeline(Cfg {
            enabled: false,
            idle_exit_timeout: Some(Duration::from_millis(80)),
            ..Default::default()
        });
        // Stay idle past the window, then one chunk trips the idle-exit check.
        tokio::time::sleep(Duration::from_millis(120)).await;
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
        let exited = tokio::time::timeout(Duration::from_secs(2), h.handle).await;
        assert!(
            exited.is_ok(),
            "daemon should idle-exit when wake disabled and idle past the timeout"
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
            h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
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
        let tools = session_control_tools(ClientToolToggles::default());
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
        let tools = session_control_tools(ClientToolToggles {
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
}
