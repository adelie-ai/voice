/// The voice pipeline state machine.
///
/// ```text
/// Idle ──[wake word]──→ Listening ──[silence]──→ Processing ──[first chunk]──→ Speaking
///  ↑                                                                              │
///  └──────────────────────[playback done]─────────────────────────────────────────┘
///                                                                                │
///                          Listening ←──[barge-in: speech during playback]────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Always-on wake word detection on mic input.
    Idle,
    /// VAD-guided recording; ends after silence threshold.
    Listening,
    /// STT transcription → send prompt to daemon.
    Processing,
    /// Streaming TTS playback of the response.
    Speaking,
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Listening => write!(f, "Listening"),
            Self::Processing => write!(f, "Processing"),
            Self::Speaking => write!(f, "Speaking"),
        }
    }
}

/// Events that trigger state transitions.
///
/// The table below is the single source of truth for legal transitions; the
/// pipeline funnels every state change through `State::transition` (via its
/// `apply` chokepoint), so a row here is the *only* way a transition can
/// happen. New events were added so the table covers every transition the
/// pipeline actually performs, not just the happy path (voice#82).
#[derive(Debug, Clone)]
pub enum StateEvent {
    /// Wake word detected — Idle → Listening.
    WakeWordDetected,
    /// Push-to-talk pressed — Idle → Listening, or Speaking → Listening
    /// (interrupting playback to re-arm the mic).
    PttPressed,
    /// Silence after speech — Listening → Processing.
    SilenceDetected,
    /// Listening lead-in / follow-up window elapsed with no (further) speech —
    /// Listening → Idle.
    ListeningTimedOut,
    /// First response chunk (or status/narration) — Processing → Speaking.
    /// A no-op while already Speaking.
    ResponseStarted,
    /// All audio playback finished — Speaking → Idle.
    PlaybackComplete,
    /// User spoke during playback — Speaking → Listening.
    BargeIn,
    /// A turn finished (reply done, single-shot, turn failed, or a stall
    /// apology) — Processing → Idle or Speaking → Idle.
    TurnEnded,
    /// A conversation-mode (or `listen_for_more`) follow-up re-arms the mic —
    /// Speaking → Listening or Processing → Listening.
    RelistenArmed,
    /// An explicit stop (StopSpeaking / StopListening / a stop phrase / capture
    /// death) returns the pipeline to wake-word idle — any non-Idle → Idle.
    Stopped,
}

impl State {
    /// Apply an event to the current state, returning the new state.
    ///
    /// Returns `Some(state)` on a legal transition (including a no-op that
    /// stays in the current state, e.g. `ResponseStarted` while already
    /// Speaking, or `Stopped` while already Idle). Returns `None` for an
    /// illegal transition.
    pub fn transition(self, event: &StateEvent) -> Option<State> {
        use State::*;
        use StateEvent::*;
        match (self, event) {
            (Idle, WakeWordDetected) => Some(Listening),
            (Idle, PttPressed) => Some(Listening),
            (Speaking, PttPressed) => Some(Listening),
            (Listening, SilenceDetected) => Some(Processing),
            (Listening, ListeningTimedOut) => Some(Idle),
            (Processing, ResponseStarted) => Some(Speaking),
            // First-chunk / status / narration while already Speaking is a
            // legal no-op — the pipeline publishes Speaking from several arms.
            (Speaking, ResponseStarted) => Some(Speaking),
            (Speaking, PlaybackComplete) => Some(Idle),
            (Speaking, BargeIn) => Some(Listening),
            (Processing, TurnEnded) => Some(Idle),
            (Speaking, TurnEnded) => Some(Idle),
            (Speaking, RelistenArmed) => Some(Listening),
            (Processing, RelistenArmed) => Some(Listening),
            // `Stopped` returns any non-Idle state to Idle, and is a no-op when
            // already Idle (a double-stop is idempotent).
            (Idle, Stopped) => Some(Idle),
            (Listening, Stopped) => Some(Idle),
            (Processing, Stopped) => Some(Idle),
            (Speaking, Stopped) => Some(Idle),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cycle() {
        let state = State::Idle;
        let state = state.transition(&StateEvent::WakeWordDetected).unwrap();
        assert_eq!(state, State::Listening);
        let state = state.transition(&StateEvent::SilenceDetected).unwrap();
        assert_eq!(state, State::Processing);
        let state = state.transition(&StateEvent::ResponseStarted).unwrap();
        assert_eq!(state, State::Speaking);
        let state = state.transition(&StateEvent::PlaybackComplete).unwrap();
        assert_eq!(state, State::Idle);
    }

    #[test]
    fn barge_in() {
        let state = State::Speaking;
        let state = state.transition(&StateEvent::BargeIn).unwrap();
        assert_eq!(state, State::Listening);
    }

    #[test]
    fn invalid_transition_returns_none() {
        assert!(
            State::Idle
                .transition(&StateEvent::SilenceDetected)
                .is_none()
        );
        assert!(
            State::Listening
                .transition(&StateEvent::WakeWordDetected)
                .is_none()
        );
        assert!(
            State::Processing
                .transition(&StateEvent::PlaybackComplete)
                .is_none()
        );
    }

    #[test]
    fn display() {
        assert_eq!(State::Idle.to_string(), "Idle");
        assert_eq!(State::Listening.to_string(), "Listening");
        assert_eq!(State::Processing.to_string(), "Processing");
        assert_eq!(State::Speaking.to_string(), "Speaking");
    }

    // ---- Extended table (voice#82): every legal pair, then illegal pairs. ----

    #[test]
    fn ptt_pressed_from_idle_or_speaking() {
        assert_eq!(
            State::Idle.transition(&StateEvent::PttPressed),
            Some(State::Listening)
        );
        assert_eq!(
            State::Speaking.transition(&StateEvent::PttPressed),
            Some(State::Listening)
        );
        // PTT from Listening/Processing is not a transition the pipeline makes.
        assert!(
            State::Listening
                .transition(&StateEvent::PttPressed)
                .is_none()
        );
        assert!(
            State::Processing
                .transition(&StateEvent::PttPressed)
                .is_none()
        );
    }

    #[test]
    fn listening_timed_out() {
        assert_eq!(
            State::Listening.transition(&StateEvent::ListeningTimedOut),
            Some(State::Idle)
        );
        assert!(
            State::Idle
                .transition(&StateEvent::ListeningTimedOut)
                .is_none()
        );
        assert!(
            State::Speaking
                .transition(&StateEvent::ListeningTimedOut)
                .is_none()
        );
    }

    #[test]
    fn response_started_is_noop_while_speaking() {
        assert_eq!(
            State::Processing.transition(&StateEvent::ResponseStarted),
            Some(State::Speaking)
        );
        // Publishing Speaking again from Speaking is a legal no-op.
        assert_eq!(
            State::Speaking.transition(&StateEvent::ResponseStarted),
            Some(State::Speaking)
        );
        // But not from Idle/Listening.
        assert!(
            State::Idle
                .transition(&StateEvent::ResponseStarted)
                .is_none()
        );
        assert!(
            State::Listening
                .transition(&StateEvent::ResponseStarted)
                .is_none()
        );
    }

    #[test]
    fn turn_ended_from_processing_or_speaking() {
        assert_eq!(
            State::Processing.transition(&StateEvent::TurnEnded),
            Some(State::Idle)
        );
        assert_eq!(
            State::Speaking.transition(&StateEvent::TurnEnded),
            Some(State::Idle)
        );
        assert!(State::Idle.transition(&StateEvent::TurnEnded).is_none());
        assert!(
            State::Listening
                .transition(&StateEvent::TurnEnded)
                .is_none()
        );
    }

    #[test]
    fn relisten_armed_from_speaking_or_processing() {
        assert_eq!(
            State::Speaking.transition(&StateEvent::RelistenArmed),
            Some(State::Listening)
        );
        assert_eq!(
            State::Processing.transition(&StateEvent::RelistenArmed),
            Some(State::Listening)
        );
        assert!(State::Idle.transition(&StateEvent::RelistenArmed).is_none());
        assert!(
            State::Listening
                .transition(&StateEvent::RelistenArmed)
                .is_none()
        );
    }

    #[test]
    fn stopped_from_any_non_idle_and_idempotent_in_idle() {
        for from in [State::Listening, State::Processing, State::Speaking] {
            assert_eq!(
                from.transition(&StateEvent::Stopped),
                Some(State::Idle),
                "Stopped from {from} should go Idle"
            );
        }
        // Idempotent: a second stop while already Idle is a no-op, not an error.
        assert_eq!(
            State::Idle.transition(&StateEvent::Stopped),
            Some(State::Idle)
        );
    }

    #[test]
    fn barge_in_only_from_speaking() {
        assert_eq!(
            State::Speaking.transition(&StateEvent::BargeIn),
            Some(State::Listening)
        );
        assert!(State::Idle.transition(&StateEvent::BargeIn).is_none());
        assert!(State::Listening.transition(&StateEvent::BargeIn).is_none());
        assert!(State::Processing.transition(&StateEvent::BargeIn).is_none());
    }
}
