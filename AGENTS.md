# Agent Instructions

## Dependency Security Policy

After adding any new packages, **scan for CVEs before building**.

Build scripts (e.g. `build.rs` in Rust, install scripts in npm) execute at build time and are a potential attack vector. Scanning the updated lockfile *before* running a build catches malicious or vulnerable transitive dependencies before any build-time code can execute.

### Workflow

1. Add the dependency (e.g. `cargo add <crate>`) — this updates the lockfile but does not build.
2. Scan immediately using `cve-mcp scan_packages` — parse the updated lockfile and pass all (name, version, ecosystem) tuples to the tool.
3. Review findings. Investigate any Critical or High severity issues before proceeding.
4. Only build once the scan is clean (or findings are understood and accepted).

This applies regardless of ecosystem (Cargo, npm, PyPI, etc.).

## Cross-project engineering standards

These apply to every repo under `github.com/adelie-ai`. They're embedded in each repo's `AGENTS.md` (not centralized) so a contributor working in a single repo has them in hand. Operator-specific preferences and machine-specific deploy recipes are intentionally not here.

### Don't break `main`
- `main` is the release: at any commit it must build, test, and run.
- Merge a green change as soon as it's independently shippable — additive, behavior-preserving, or behind a default that preserves the old path. Don't hold green work hostage to a coordinated release.
- Co-dependent changes land together; name the interlock ("blocked-by #X" / "must merge with #Y") so it's visible without reading the diff.
- "Green" is more than CI: review passed, tests cover the new behavior (not just "no panic"), warnings clean, security pass done, change stands on its own. With no active CI in these repos, "green" rests on local `cargo test` + `fmt` + `clippy --all-targets` + `cargo audit`, run by the author (via `just check` where the repo provides it).
- When in doubt, hold. A half-coupled "fix-forward" merge breaks `main` for everyone.

### Tests are spec-driven (TDD)
- Every change carries a Testing section: acceptance criteria as testable assertions, each criterion a named test whose name is legible from test output.
- Write failing tests first, in their own commit before the implementation commit — that commit is the spec.
- Cover all new code: every branch, error path, edge case. Gaps are a review finding.
- Assert the desired outcome, not just that a call returned `Ok`.
- Enumerate unhappy paths deliberately: empty/missing input, boundary/max, concurrent/racy, authorization/tenant boundaries, partial reads/writes/dropped streams, malformed input. A test list with none of these is testing wishes.

### Warnings are failures
- Compiler warnings, clippy lints, formatter diffs, and advisories all count — fix the root cause. If a lint truly doesn't apply, suppress at the narrowest scope with a one-line justification; never crate-wide.
- This repo enforces it **mechanically** via a `[lints]` table denying `rust.warnings` and `clippy.all`, so `cargo build`/`test`/`clippy` hard-fail on a warning — it isn't left to reviewer attention.
- Never `--no-verify` past hooks. If a hook is genuinely broken, fix it in its own commit and explain why.
- Don't `#[ignore]` a test you broke; fix it, or open a tracking issue and reference it from the attribute.
- Pre-existing warnings in a file you touch are yours to address (in-change or a small follow-up) — don't pile new code on an ignored signal.

### Security review before requesting review
- Read your own diff adversarially: untrusted input crossing trust boundaries (network, IPC, D-Bus, MCP tool args), secrets in logs, missing auth checks, panic-on-input, unparameterized SQL/shell.
- Scan dependencies whenever the lockfile changed (`cargo audit` or the `cve-mcp` server) — and scan BEFORE the first build, because build scripts execute attacker-controlled code at build time.
- High/critical CVEs are hard blockers: patch in the same change, prove the path unreachable and document why, or file a tracked follow-up referenced in the change. Never ship past one silently; never pin around an advisory without a comment or tracking issue.

### Maintainability / cognitive load
- Keep each change small enough to land independently with a clear deliverable.
- Don't introduce a new abstraction until ~3 call sites prove the pattern; when one new type unifies several needs, justify the unification explicitly.
- Reuse existing traits and patterns rather than inventing parallel ones; extend an existing crate over adding one unless the seam is obvious.


### Capability-based degradation
- Every reliance on an optional OS/desktop service (logind, screen-lock, KDE/Plasma, PipeWire specifics, any session- or system-bus D-Bus interface) must be capability-detected and degrade gracefully — never a hard dependency that errors or hangs when absent. The product may run headless, in containers, on other DEs, or as a system service.
- Distinguish "is the capability present?" from "did my call succeed?" Three states: absent → disable that feature, log once, fall back to prior behavior; present-and-known → use it; present-but-anomalous → stay conservative / last-known-state and warn. Scope any privacy/safety fail-safe to the last two — a fail-safe correct on the desktop can be pathological headless (e.g. "treat unknown session as inactive" ⇒ mic never opens).
- Detect each optional dependency independently; absence of one never disables the others or aborts startup. Surface the detected capability so an operator sees *why* a feature is on or off.

### GitHub issue / PR / board hygiene
- Self-assign an issue when you start it (or comment to claim it) so parallel work doesn't collide; move the board card to In Progress.
- Link the PR to the issue: `Closes #N` to auto-close, `Refs #N` when it only partially addresses it.
- Keep the board in sync with reality (In Review on open, Done on merge); if you can't move the card, comment the intended status.
- On multi-session work, leave a short status comment before stopping — what landed, what's next, what's blocked — so state is reconstructable without git log.

### Worktrees
- Do code work in a git worktree on its own branch off `origin/main`, never the primary checkout, so concurrent sessions don't collide. Convention: `~/Projects/adelie-ai/.worktrees/<repo>/issue-N-slug/`, branch mirroring the slug.
- Run independent tasks in parallel worktrees, but check first for shared files / shared `Cargo.toml` dep edits / shared migration ordinals — if they overlap, serialize. Brief each parallel agent on its scope ("own crate X, don't touch Y").
