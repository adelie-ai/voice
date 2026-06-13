//! logind session-active gating (voice#103).
//!
//! On a systemd `--user` service, fast user switching makes the logind session
//! go *inactive* while `user@` services (us) keep running — so the daemon kept
//! the mic open and transcribed whoever was near it for the whole switched-away
//! period (privacy + cloud-pipeline cost + device contention). The fix is to gate
//! capture on the logind `Session.Active` property and stop at the source when
//! the session isn't active.
//!
//! ## Three-state capability model (NOT the issue's naive "unknown ⇒ inactive")
//!
//! The issue's literal wording — "if session state can't be determined, treat as
//! inactive" — is wrong for headless/containerized/non-systemd hosts: it would
//! leave the mic permanently shut on a box that has no logind at all (the
//! capability-based-degradation rule: a feature you cannot evaluate is OFF, not
//! fail-closed). So we distinguish:
//!
//! 1. **logind capability ABSENT** (no `org.freedesktop.login1` on the system
//!    bus, no resolvable session, headless/container/non-systemd) → the gate is
//!    **inert**: capture proceeds normally (prior behavior). Logged once.
//! 2. **present, Active known** → use it: stop on inactive, resume on active.
//! 3. **present but anomalous** (we resolved a session, but a later property
//!    read/watch fails mid-flight) → THIS is where privacy-first conservatism
//!    belongs: prefer *not* capturing and warn loudly. The watcher publishes
//!    `false` (paused) on such a failure rather than silently assuming active.
//!
//! `[wake_word] pause_on_session_inactive` (default `true`) gates the whole
//! feature; with it `false`, or with logind absent, the gate is inert regardless.

use std::sync::Arc;

use tokio::sync::watch;
use zbus::Connection;

/// An injectable view of "is this logind session active?" used by the pipeline
/// to decide whether the mic may be open.
///
/// Two shapes:
/// - [`SessionGate::inert`] — the gate is OFF (logind capability absent, or the
///   `pause_on_session_inactive` opt-out is set). [`is_active`](Self::is_active)
///   is always `true` and [`changed`](Self::changed) never fires; capture is
///   never paused for session reasons.
/// - [`SessionGate::watching`] — backed by a `watch` channel a logind watcher
///   task drives. `true` = active, `false` = inactive/anomalous (pause).
///
/// Unit tests construct the watching variant directly from a channel they own,
/// so the run-loop gating is testable without a real system bus.
#[derive(Clone)]
pub struct SessionGate {
    inner: GateInner,
}

#[derive(Clone)]
enum GateInner {
    /// Gate disabled — always active, never changes.
    Inert,
    /// Gate live — follows the watched session-active value.
    Watching(watch::Receiver<bool>),
}

impl SessionGate {
    /// A gate that never pauses capture (capability absent or feature disabled).
    pub fn inert() -> Self {
        Self {
            inner: GateInner::Inert,
        }
    }

    /// A gate backed by a session-active watch channel.
    pub fn watching(rx: watch::Receiver<bool>) -> Self {
        Self {
            inner: GateInner::Watching(rx),
        }
    }

    /// Whether the gate is live (i.e. capable of pausing capture). `false` for an
    /// inert gate. Surfaced so an operator / health report can see *why* gating
    /// is on or off.
    pub fn is_enabled(&self) -> bool {
        matches!(self.inner, GateInner::Watching(_))
    }

    /// Whether the session is currently active (mic may be open). Always `true`
    /// for an inert gate.
    pub fn is_active(&self) -> bool {
        match &self.inner {
            GateInner::Inert => true,
            GateInner::Watching(rx) => *rx.borrow(),
        }
    }

    /// Await the next change in session-active state. For an inert gate this
    /// never resolves (so a `select!` arm using it is effectively disabled),
    /// which is exactly the "gate off" behavior we want.
    pub async fn changed(&mut self) {
        match &mut self.inner {
            GateInner::Inert => std::future::pending::<()>().await,
            GateInner::Watching(rx) => {
                // A closed channel (watcher task gone) shouldn't busy-spin this
                // arm; treat it as "no more changes" and park forever. The last
                // observed value stays readable via `is_active`.
                if rx.changed().await.is_err() {
                    std::future::pending::<()>().await
                }
            }
        }
    }
}

/// Resolve the logind capability and, if present and enabled, spawn a watcher
/// task that tracks this process's session `Active` property and publishes it to
/// a `watch` channel. Returns the [`SessionGate`] the pipeline gates on.
///
/// `enabled` is the `pause_on_session_inactive` config knob. With it `false`,
/// the gate is inert regardless of logind presence.
///
/// Capability detection is fully degrading: any failure to reach the system bus,
/// the `login1` manager, or resolve a session is treated as **absent** (state 1)
/// — log once and return an inert gate. It must never abort startup or disable
/// other functionality.
pub async fn spawn_session_gate(enabled: bool) -> SessionGate {
    if !enabled {
        tracing::info!(
            "session gating disabled by config (pause_on_session_inactive=false); the mic will \
             NOT be released on fast-user-switch / session-inactive"
        );
        return SessionGate::inert();
    }

    match resolve_session().await {
        Ok((conn, session_path)) => {
            let initial = read_active(&conn, &session_path).await.unwrap_or_else(|e| {
                // Resolved the session but couldn't read Active right now — state
                // 3 (anomalous). Start paused (conservative) and let the watcher
                // recover.
                tracing::warn!(
                    "logind session resolved ({session_path}) but initial Active read failed: {e} \
                     — starting paused (privacy-first); will resume when Active is readable"
                );
                false
            });
            let (tx, rx) = watch::channel(initial);
            tracing::info!(
                session = %session_path,
                active = initial,
                "logind session gating ENABLED; capture follows Session.Active"
            );
            tokio::spawn(watch_active(conn, session_path, Arc::new(tx)));
            SessionGate::watching(rx)
        }
        Err(e) => {
            // State 1: capability absent. Do NOT fail-close — a headless /
            // container / non-systemd host has no logind, and pausing the mic
            // there would mean it never opens. Degrade to the prior behavior.
            tracing::info!(
                "logind session gating UNAVAILABLE ({e}); capture is NOT session-gated on this \
                 host (headless/container/non-systemd?) — proceeding as before"
            );
            SessionGate::inert()
        }
    }
}

/// Connect to the system bus, find the login1 manager, and resolve this
/// process's session object path. Errors model "capability absent" (state 1).
async fn resolve_session() -> Result<(Connection, String), String> {
    let conn = Connection::system()
        .await
        .map_err(|e| format!("system bus unavailable: {e}"))?;

    // Prefer an explicit $XDG_SESSION_ID (cheap, no PID round-trip); fall back to
    // GetSessionByPID for our own PID.
    let session_path = if let Some(id) = std::env::var_os("XDG_SESSION_ID") {
        let id = id.to_string_lossy().into_owned();
        get_session(&conn, &id).await?
    } else {
        let pid = std::process::id();
        get_session_by_pid(&conn, pid).await?
    };
    Ok((conn, session_path))
}

/// `org.freedesktop.login1.Manager.GetSession(id) -> objpath`.
async fn get_session(conn: &Connection, id: &str) -> Result<String, String> {
    call_manager_for_path(conn, "GetSession", &(id)).await
}

/// `org.freedesktop.login1.Manager.GetSessionByPID(pid) -> objpath`.
async fn get_session_by_pid(conn: &Connection, pid: u32) -> Result<String, String> {
    call_manager_for_path(conn, "GetSessionByPID", &(pid)).await
}

async fn call_manager_for_path<B>(
    conn: &Connection,
    method: &str,
    body: &B,
) -> Result<String, String>
where
    B: serde::Serialize + zbus::zvariant::DynamicType,
{
    let reply = conn
        .call_method(
            Some("org.freedesktop.login1"),
            "/org/freedesktop/login1",
            Some("org.freedesktop.login1.Manager"),
            method,
            body,
        )
        .await
        .map_err(|e| format!("{method} failed: {e}"))?;
    let path: zbus::zvariant::OwnedObjectPath = reply
        .body()
        .deserialize()
        .map_err(|e| format!("{method} reply decode failed: {e}"))?;
    Ok(path.as_str().to_owned())
}

/// Read the `Active` property of a session object.
async fn read_active(conn: &Connection, session_path: &str) -> Result<bool, String> {
    let reply = conn
        .call_method(
            Some("org.freedesktop.login1"),
            session_path,
            Some("org.freedesktop.DBus.Properties"),
            "Get",
            &("org.freedesktop.login1.Session", "Active"),
        )
        .await
        .map_err(|e| format!("Get(Active) failed: {e}"))?;
    let value: zbus::zvariant::OwnedValue = reply
        .body()
        .deserialize()
        .map_err(|e| format!("Active reply decode failed: {e}"))?;
    bool::try_from(value).map_err(|e| format!("Active is not a bool: {e}"))
}

/// Build a `PropertiesChanged` proxy bound to the session object.
async fn build_properties_proxy(
    conn: &Connection,
    session_path: &str,
) -> Result<zbus::fdo::PropertiesProxy<'static>, zbus::Error> {
    zbus::fdo::PropertiesProxy::builder(conn)
        .destination("org.freedesktop.login1")?
        .path(session_path.to_owned())?
        .build()
        .await
}

/// Watch the session's `Active` property via `PropertiesChanged` and republish
/// every change to `tx`. On a watch/stream failure (state 3, anomalous) publish
/// `false` (pause, privacy-first) and warn; keep trying to recover by re-reading.
async fn watch_active(conn: Connection, session_path: String, tx: Arc<watch::Sender<bool>>) {
    use futures_util::StreamExt;

    let props = match build_properties_proxy(&conn, &session_path).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                "session watcher: failed to build PropertiesProxy for {session_path}: {e} — \
                 pausing capture (privacy-first); no further session updates"
            );
            let _ = tx.send(false);
            return;
        }
    };

    let mut changes = match props.receive_properties_changed().await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                "session watcher: failed to subscribe to PropertiesChanged on {session_path}: {e} \
                 — pausing capture (privacy-first)"
            );
            let _ = tx.send(false);
            return;
        }
    };

    while let Some(change) = changes.next().await {
        let args = match change.args() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("session watcher: bad PropertiesChanged signal: {e}");
                continue;
            }
        };
        // `Active` may arrive in the changed map or only as an invalidated name;
        // on either, re-read the authoritative value rather than guessing.
        let mentions_active = args.changed_properties().contains_key("Active")
            || args.invalidated_properties().contains(&"Active");
        if !mentions_active {
            continue;
        }
        match read_active(&conn, &session_path).await {
            Ok(active) => {
                if *tx.borrow() != active {
                    tracing::info!(active, "logind Session.Active changed");
                }
                if tx.send(active).is_err() {
                    return; // pipeline gone
                }
            }
            Err(e) => {
                // Anomalous (state 3): we had a working session but can't read
                // its state. Err toward not-capturing and warn loudly.
                tracing::warn!(
                    "session watcher: Active re-read failed: {e} — pausing capture (privacy-first)"
                );
                let _ = tx.send(false);
            }
        }
    }

    // The signal stream ended (bus dropped). Don't silently leave the mic open on
    // a stale "active": pause and warn.
    tracing::warn!(
        "session watcher: PropertiesChanged stream ended for {session_path} — pausing capture \
         (privacy-first); session updates have stopped"
    );
    let _ = tx.send(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inert_gate_is_always_active_and_disabled() {
        let gate = SessionGate::inert();
        assert!(gate.is_active(), "inert gate must report active");
        assert!(!gate.is_enabled(), "inert gate is not an enabled gate");
    }

    #[test]
    fn watching_gate_reflects_channel() {
        let (tx, rx) = watch::channel(false);
        let gate = SessionGate::watching(rx);
        assert!(gate.is_enabled());
        assert!(!gate.is_active(), "false channel => inactive");
        tx.send(true).unwrap();
        assert!(gate.is_active(), "true channel => active");
    }

    #[tokio::test]
    async fn changed_never_fires_for_inert() {
        let mut gate = SessionGate::inert();
        let r = tokio::time::timeout(std::time::Duration::from_millis(100), gate.changed()).await;
        assert!(r.is_err(), "inert gate.changed() must never resolve");
    }

    #[tokio::test]
    async fn disabled_by_config_yields_inert_gate() {
        // pause_on_session_inactive=false => inert, no bus access at all.
        let gate = spawn_session_gate(false).await;
        assert!(!gate.is_enabled());
        assert!(gate.is_active());
    }
}
