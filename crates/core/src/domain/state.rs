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
#[derive(Debug, Clone)]
pub enum StateEvent {
    /// Wake word detected — transition Idle → Listening.
    WakeWordDetected,
    /// Silence after speech — transition Listening → Processing.
    SilenceDetected,
    /// First response chunk received — transition Processing → Speaking.
    ResponseStarted,
    /// All audio playback finished — transition Speaking → Idle.
    PlaybackComplete,
    /// User spoke during playback — transition Speaking → Listening.
    BargeIn,
}

impl State {
    /// Apply an event to the current state, returning the new state.
    /// Returns `None` if the transition is invalid.
    pub fn transition(self, event: &StateEvent) -> Option<State> {
        match (self, event) {
            (State::Idle, StateEvent::WakeWordDetected) => Some(State::Listening),
            (State::Listening, StateEvent::SilenceDetected) => Some(State::Processing),
            (State::Processing, StateEvent::ResponseStarted) => Some(State::Speaking),
            (State::Speaking, StateEvent::PlaybackComplete) => Some(State::Idle),
            (State::Speaking, StateEvent::BargeIn) => Some(State::Listening),
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
}
