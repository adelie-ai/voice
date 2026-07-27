//! Capture-stream health: how a dying input stream becomes a signal the
//! pipeline can act on.
//!
//! cpal reports a running stream's problems through an error callback. The
//! callback runs on the backend's audio thread and, on ALSA, fires in a tight
//! loop while a stream is failing, so it must stay cheap and must not block:
//! it classifies the error, records it here, and leaves the acting to the
//! capture (drain) thread, which owns the stream and the audio sender.
//!
//! The pipeline's only signal that capture has died is the audio channel
//! closing, so "capture is dead" has to end the drain thread. Two conditions
//! do that:
//!
//! - a **fatal** [`cpal::StreamError`] — the device went away or the stream was
//!   invalidated, so no audio can ever arrive on it again;
//! - a **stall** — nothing was captured for [`CAPTURE_STALL_TIMEOUT`]. A live
//!   capture stream delivers frames continuously (silence is zero-valued
//!   samples, not an absence of data), so a gap that long means the device or
//!   the sound server behind it has stopped, whether or not it said so. This is
//!   the backstop for the errors that cannot be classified: ALSA reports even a
//!   USB unplug as a backend-specific error.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Longest capture may deliver nothing before the stream is presumed dead.
/// Long enough that only a stopped device trips it, short enough that the user
/// is not left talking to a deaf assistant.
pub(crate) const CAPTURE_STALL_TIMEOUT: Duration = Duration::from_secs(5);

/// Shortest a capture stream may live before its death propagates to the
/// pipeline. Death travels by the capture thread ending, so a device that opens
/// cleanly and then fails immediately would otherwise spin the pipeline's
/// open/die/open loop as fast as the OS allows. Holding the dead thread until
/// the stream has existed this long caps the retry rate; a stream that ran
/// longer is unaffected.
const MIN_CAPTURE_LIFETIME: Duration = Duration::from_secs(1);

/// How the capture thread must treat a [`cpal::StreamError`].
#[derive(Clone, Copy, Debug)]
enum StreamFault {
    /// The stream can never produce audio again; capture has to be torn down
    /// and the device re-opened.
    Fatal,
    /// A glitch the backend recovers from on its own; audio keeps flowing.
    Transient,
}

/// Classify a stream error.
///
/// Only the two variants that state the stream is finished are treated as
/// death. An xrun is a dropped-samples glitch the backend re-prepares from, and
/// a backend-specific error is unclassifiable by construction — ALSA reports a
/// USB unplug that way, but also transient read failures — so it is counted and
/// left to the stall watchdog, which sees the difference: a dead device stops
/// delivering, a recovered one does not.
fn classify_stream_error(err: &cpal::StreamError) -> StreamFault {
    match err {
        cpal::StreamError::DeviceNotAvailable | cpal::StreamError::StreamInvalidated => {
            StreamFault::Fatal
        }
        cpal::StreamError::BufferUnderrun | cpal::StreamError::BackendSpecific { .. } => {
            StreamFault::Transient
        }
    }
}

/// The cpal input-stream error callback: classify, record, and report once.
pub(crate) fn on_stream_error(health: &CaptureHealth, err: &cpal::StreamError) {
    match classify_stream_error(err) {
        StreamFault::Fatal => {
            if health.mark_fatal() {
                tracing::error!("input stream error (fatal): {err}");
            }
        }
        StreamFault::Transient => {
            if health.record_transient() {
                tracing::warn!(
                    "input stream error (recoverable): {err} — repeats are counted, not logged"
                );
            }
        }
    }
}

/// Health of one capture stream, shared between the cpal error callback (which
/// only records) and the capture thread (which logs and acts). Cloning shares
/// the same state; a fresh stream gets a fresh [`CaptureHealth`].
#[derive(Clone)]
pub(crate) struct CaptureHealth(Arc<CaptureHealthInner>);

struct CaptureHealthInner {
    /// Set once the stream is known to be finished.
    fatal: AtomicBool,
    /// Recoverable errors since the capture thread last reported them.
    pending: AtomicUsize,
    /// Whether a recoverable error has already been reported, so an error storm
    /// is counted rather than written to the journal line by line.
    reported: AtomicBool,
}

impl CaptureHealth {
    pub(crate) fn new() -> Self {
        Self(Arc::new(CaptureHealthInner {
            fatal: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            reported: AtomicBool::new(false),
        }))
    }

    /// Record that the stream is finished. Returns `true` for the first caller
    /// only, so an error storm is reported once.
    fn mark_fatal(&self) -> bool {
        !self.0.fatal.swap(true, Ordering::SeqCst)
    }

    /// Whether the stream is known to be finished.
    pub(crate) fn is_fatal(&self) -> bool {
        self.0.fatal.load(Ordering::SeqCst)
    }

    /// Record a recoverable error. Returns `true` for the first one only, so a
    /// storm is counted rather than logged line by line.
    fn record_transient(&self) -> bool {
        self.0.pending.fetch_add(1, Ordering::Relaxed);
        !self.0.reported.swap(true, Ordering::SeqCst)
    }

    /// Read and clear the recoverable-error count, so the capture thread's
    /// periodic summary does not double-count.
    pub(crate) fn take_transient(&self) -> usize {
        self.0.pending.swap(0, Ordering::Relaxed)
    }

    /// Whether the drain loop should keep draining. `running` is the `stop()`
    /// latch; a dead stream ends the loop even while that latch is still set,
    /// because nothing else clears it.
    pub(crate) fn keep_draining(&self, running: bool) -> bool {
        running && !self.is_fatal()
    }
}

/// Watches for capture going quiet: the backstop for a stream that stops
/// delivering without reporting a fatal error. The clock is injected so the
/// decision is pure.
pub(crate) struct StallWatchdog {
    last_data: Instant,
    timeout: Duration,
}

impl StallWatchdog {
    pub(crate) fn new(now: Instant, timeout: Duration) -> Self {
        Self {
            last_data: now,
            timeout,
        }
    }

    /// Note that audio was captured. Called after a drain tick that read
    /// samples, so a pipeline that blocked the sender for a long turn restarts
    /// the clock from when it finished, not from when the samples arrived.
    pub(crate) fn saw_data(&mut self, now: Instant) {
        self.last_data = now;
    }

    /// Whether capture has been quiet long enough to presume the device is gone.
    pub(crate) fn is_stalled(&self, now: Instant) -> bool {
        now.duration_since(self.last_data) >= self.timeout
    }
}

/// How long a capture thread that has presumed its stream dead must wait before
/// letting the pipeline re-open the device (see [`MIN_CAPTURE_LIFETIME`]).
/// `lifetime` is how long the stream ran.
pub(crate) fn restart_backoff(lifetime: Duration) -> Duration {
    MIN_CAPTURE_LIFETIME.saturating_sub(lifetime)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend_error(description: &str) -> cpal::StreamError {
        cpal::StreamError::BackendSpecific {
            err: cpal::BackendSpecificError {
                description: description.to_string(),
            },
        }
    }

    // --- #140: a fatal stream error must end capture, not just log ---

    #[test]
    fn fatal_device_loss_marks_capture_dead() {
        // A USB mic unplugged mid-session. The stream can never produce audio
        // again, so capture must be torn down for the pipeline to re-open it.
        let health = CaptureHealth::new();
        on_stream_error(&health, &cpal::StreamError::DeviceNotAvailable);
        assert!(
            health.is_fatal(),
            "losing the device must be recorded as capture death"
        );
    }

    #[test]
    fn stream_invalidation_marks_capture_dead() {
        // A sound server (PipeWire/Pulse) restarting invalidates the stream: it
        // has to be rebuilt, so the current one is finished.
        let health = CaptureHealth::new();
        on_stream_error(&health, &cpal::StreamError::StreamInvalidated);
        assert!(
            health.is_fatal(),
            "an invalidated stream must be recorded as capture death"
        );
    }

    #[test]
    fn drain_loop_stops_on_fatal_error_while_running_is_still_latched() {
        // The regression itself: `running` stays true until stop() or the
        // thread's own exit guard clears it, so the drain loop has to notice the
        // dead stream on its own. Without this the thread loops forever, never
        // drops the audio sender, and the pipeline's capture-death recovery
        // never runs — the daemon is deaf while still reporting Capturing.
        let health = CaptureHealth::new();
        assert!(
            health.keep_draining(true),
            "healthy capture keeps draining while running"
        );
        on_stream_error(&health, &cpal::StreamError::DeviceNotAvailable);
        assert!(
            !health.keep_draining(true),
            "a dead stream must end the drain loop even with `running` latched true"
        );
    }

    #[test]
    fn repeated_fatal_errors_leave_capture_dead() {
        // ALSA keeps calling the error callback after the device is gone; the
        // storm must not undo or re-report the death, and a fatal error is not
        // a recoverable one.
        let health = CaptureHealth::new();
        for _ in 0..100 {
            on_stream_error(&health, &cpal::StreamError::DeviceNotAvailable);
        }
        assert!(health.is_fatal(), "capture stays dead");
        assert_eq!(
            health.take_transient(),
            0,
            "a fatal error is not counted as a recoverable one"
        );
    }

    #[test]
    fn explicit_stop_still_ends_the_drain_loop() {
        // The pre-existing exit condition must survive: stop() clears `running`.
        let health = CaptureHealth::new();
        assert!(!health.keep_draining(false), "stop() must end the loop");
    }

    // --- recoverable errors must not cost us the microphone ---

    #[test]
    fn buffer_underrun_does_not_kill_capture() {
        // An xrun is a dropped-samples glitch: ALSA re-prepares the PCM and
        // capture continues. Tearing down here would drop the mic on a hiccup.
        let health = CaptureHealth::new();
        on_stream_error(&health, &cpal::StreamError::BufferUnderrun);
        assert!(!health.is_fatal(), "an xrun is not capture death");
        assert!(
            health.keep_draining(true),
            "the drain loop keeps going through an xrun"
        );
        assert_eq!(health.take_transient(), 1, "but it is counted");
    }

    #[test]
    fn backend_specific_error_is_counted_not_fatal() {
        // ALSA reports everything it cannot map — including a USB unplug — this
        // way, so a single one cannot be trusted to mean death. The stall
        // watchdog is what catches the ones that do.
        let health = CaptureHealth::new();
        on_stream_error(
            &health,
            &backend_error("ALSA function 'snd_pcm_readi' failed"),
        );
        assert!(
            !health.is_fatal(),
            "an unclassifiable backend error must not tear capture down by itself"
        );
        assert_eq!(health.take_transient(), 1);
    }

    #[test]
    fn transient_error_storm_is_reported_once_and_tallied() {
        // On ALSA a failing stream calls the error callback in a tight loop;
        // a line per error floods the journal.
        let health = CaptureHealth::new();
        assert!(health.record_transient(), "the first error reports");
        for _ in 0..999 {
            assert!(
                !health.record_transient(),
                "the storm after it is counted, not reported"
            );
        }
        assert_eq!(health.take_transient(), 1000, "every error is counted");
        assert_eq!(
            health.take_transient(),
            0,
            "draining resets so summaries do not double-count"
        );
    }

    #[test]
    fn error_reporting_reopens_once_the_tally_is_drained() {
        // Flood control must not blind the operator for the life of a stream
        // that runs for days: a different failure hours later still reaches the
        // journal, at most once per summary window.
        let health = CaptureHealth::new();
        assert!(health.record_transient(), "the first error reports");
        assert!(!health.record_transient(), "an immediate repeat does not");
        assert_eq!(health.take_transient(), 2, "both are counted");
        assert!(
            health.record_transient(),
            "the next window reports its first error again"
        );
    }

    #[test]
    fn healthy_capture_reports_no_faults() {
        let health = CaptureHealth::new();
        assert!(!health.is_fatal());
        assert_eq!(health.take_transient(), 0);
        assert!(health.keep_draining(true));
    }

    #[test]
    fn capture_health_survives_concurrent_errors_from_the_audio_thread() {
        // The error callback runs on the backend's audio thread while the
        // capture thread reads the tally: no error may be lost and exactly one
        // caller may claim the first report.
        let health = CaptureHealth::new();
        let mut workers = Vec::new();
        for _ in 0..4 {
            let h = health.clone();
            workers.push(std::thread::spawn(move || {
                (0..250).filter(|_| h.record_transient()).count()
            }));
        }
        let firsts: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();
        assert_eq!(firsts, 1, "exactly one caller reports the first error");
        assert_eq!(
            health.take_transient(),
            1000,
            "no error is lost across threads"
        );
    }

    // --- the stall backstop: silence that nobody reported ---

    #[test]
    fn silent_capture_is_presumed_dead_at_the_stall_timeout() {
        // Covers what classification cannot: an unplug ALSA reported as a
        // backend-specific error, or a sound server that wedges without saying
        // anything at all.
        let start = Instant::now();
        let watchdog = StallWatchdog::new(start, CAPTURE_STALL_TIMEOUT);
        assert!(
            watchdog.is_stalled(start + CAPTURE_STALL_TIMEOUT),
            "no audio for the whole window means capture is dead"
        );
    }

    #[test]
    fn stall_watchdog_tolerates_a_gap_just_under_the_timeout() {
        let start = Instant::now();
        let watchdog = StallWatchdog::new(start, CAPTURE_STALL_TIMEOUT);
        let just_under = start + CAPTURE_STALL_TIMEOUT - Duration::from_millis(1);
        assert!(
            !watchdog.is_stalled(just_under),
            "a gap shorter than the timeout is not death"
        );
    }

    #[test]
    fn flowing_capture_is_never_presumed_dead() {
        // A long assistant turn blocks the capture thread's sender for longer
        // than the timeout. The clock restarts from when audio was last seen —
        // measured after the send returns — so a busy pipeline is not mistaken
        // for a dead microphone.
        let start = Instant::now();
        let mut watchdog = StallWatchdog::new(start, CAPTURE_STALL_TIMEOUT);
        let after_long_turn = start + Duration::from_secs(30);
        watchdog.saw_data(after_long_turn);
        assert!(
            !watchdog.is_stalled(after_long_turn + Duration::from_millis(20)),
            "audio just seen means capture is alive"
        );
        assert!(
            watchdog.is_stalled(after_long_turn + CAPTURE_STALL_TIMEOUT),
            "and the clock still runs from that moment"
        );
    }

    #[test]
    fn stall_watchdog_honours_the_timeout_it_was_given() {
        // The timeout is a parameter, not a constant baked into the check.
        let start = Instant::now();
        let watchdog = StallWatchdog::new(start, Duration::from_millis(200));
        assert!(!watchdog.is_stalled(start + Duration::from_millis(199)));
        assert!(watchdog.is_stalled(start + Duration::from_millis(200)));
    }

    // --- a device that fails instantly must not spin the restart loop ---

    #[test]
    fn immediate_stream_death_is_rate_limited_before_restart() {
        // Death reaches the pipeline when the capture thread ends, so a stream
        // that opens and fails at once would otherwise spin the pipeline's
        // open/die/open loop as fast as the OS allows.
        let wait = restart_backoff(Duration::from_millis(5));
        assert!(
            wait >= Duration::from_millis(500) && wait <= Duration::from_secs(2),
            "an instantly dead stream must throttle the next open, got {wait:?}"
        );
    }

    #[test]
    fn long_lived_stream_death_propagates_without_delay() {
        assert_eq!(
            restart_backoff(Duration::from_secs(600)),
            Duration::ZERO,
            "a stream that ran for minutes must be re-opened immediately"
        );
    }
}
