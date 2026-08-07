//! Runs `adele-voice --telemetry-probe` as a real subprocess, so a test can
//! check where a genuine `adelie_telemetry::init` call routes its output.
//!
//! This can't be checked in-process: the console layer writes straight to
//! the OS-level stderr file descriptor, and `init` may only be claimed once
//! per process, so a second in-process call (as a different test would make)
//! can't rebuild it with a different writer. An integration test target is
//! also what reliably gets `CARGO_BIN_EXE_adele-voice` set by Cargo; the
//! `adele-voice` package has no `[lib]` target, so a unit test compiled as
//! the bin's own `--test` harness does not trigger the plain bin artifact to
//! be built alongside it (voice#158).

fn run_telemetry_probe(rust_log: Option<&str>) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_adele-voice");
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("--telemetry-probe");
    match rust_log {
        Some(value) => {
            cmd.env("RUST_LOG", value);
        }
        None => {
            cmd.env_remove("RUST_LOG");
        }
    }
    cmd.output()
        .expect("adele-voice --telemetry-probe must run")
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    // D1: the MCP stdio transport frames JSON-RPC on stdout, so a stray log
    // line there would corrupt the protocol stream.
    let output = run_telemetry_probe(None);
    assert!(
        output.status.success(),
        "the probe must exit cleanly: {output:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout.is_empty(),
        "no log output may reach stdout: got {stdout:?}"
    );
    assert!(
        stderr.contains("probe: info line"),
        "the info line must reach stderr: {stderr:?}"
    );
}

#[test]
fn ort_stays_at_warn_when_rust_log_is_unset() {
    // The default filter is `info,ort=warn`; onnxruntime's allocator and
    // arena chatter (emitted at INFO under the `ort` target) must stay
    // suppressed unless an operator asks for it.
    let output = run_telemetry_probe(None);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("probe: info line"),
        "a normal info line must still appear: {stderr:?}"
    );
    assert!(
        !stderr.contains("probe: ort info line"),
        "an ort-target info line must be suppressed by the default filter: {stderr:?}"
    );
}
