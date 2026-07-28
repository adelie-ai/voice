# Agent Instructions

Shared standards live in [AGENTS.base.md](AGENTS.base.md), which is generated. This file holds the rules specific to this repo.

## Overrides and additions to the shared base

Everything in [AGENTS.base.md](AGENTS.base.md) applies to this repo. This section
records only the points where this repo deliberately differs from the base, or adds a
rule the base does not have.

### 3.1 The gate for this repo (addition)

The `adelie-ai` repos have no CI. The gate is local and the author runs it: `just check`.
Run `just install-hooks` once per clone to put the same gate on pre-push. Warnings are
denied mechanically by the workspace `[lints]` table, which every member crate inherits
with `[lints] workspace = true`, so `cargo build`, `cargo test`, and `cargo clippy` each
hard-fail on a warning.

### 4.3 Branch and pull request - merge when green (override, weaker than the base)

The base opens a pull request and waits for the user. In these repos the merge is delegated:
merge your own pull request as soon as it is green and independently shippable. Green here
means more than a clean build. The gate above passed, the tests cover the new behavior and
not only the absence of a panic, the security pass is done, and the change stands on its own.
Assign `dspadea` with `gh pr edit --add-assignee` and verify it; a review request from the
same account no-ops without an error, so never report a pull request as review-requested.
When in doubt, hold.

### 4.4 Worktrees - the group convention (addition)

Put the worktree at `.worktrees/<repo>/issue-N-slug/` under the group directory, on a branch
that mirrors the slug. Before you run tasks in parallel worktrees, look for shared files,
shared `Cargo.toml` dependency edits, and shared migration ordinals. Serialize the work where
they overlap, and tell each parallel agent the scope it owns.

### 6.1 Dependencies - the group's scan workflow (addition)

Base rule 6.1 sets the policy, including that a high or critical advisory blocks the change.
This group runs it with its own tooling:

1. Add the dependency (`cargo add <crate>`). This writes the lockfile but does not build.
2. Scan the updated lockfile with the `cve-mcp` server's `scan_packages` tool, or with
   `cargo audit`. Pass every (name, version, ecosystem) tuple.
3. Build only after the scan is clean, or after you have accepted the findings in writing.

### 9.1 Tracker for this project

GitHub Issues on `github.com/adelie-ai/voice`, together with the shared `adelie-ai` project
board. Manage entries with the `gh` CLI (`gh issue create`, `gh issue list`, `gh issue edit`,
`gh pr create`). The board states in use are In Progress, In Review, and Done.

### Capability-based degradation (addition)

Every reliance on an optional operating-system or desktop service - logind, the screen lock,
KDE and Plasma, PipeWire specifics, any session-bus or system-bus D-Bus interface - must be
capability-detected, and must degrade cleanly when the service is absent. Never make one a
hard dependency that errors or hangs. The product can run headless, in a container, on
another desktop environment, or as a system service.

Distinguish "is the capability present?" from "did my call succeed?". There are three states.
Absent: disable that feature, log once, and fall back to the prior behavior. Present and
known: use it. Present but anomalous: stay conservative, or hold the last known state, and
warn. Scope any privacy or safety fail-safe to the last two states only. A fail-safe that is
correct on the desktop can be pathological headless. "Treat an unknown session as inactive"
means the microphone never opens.

Detect each optional dependency on its own. The absence of one never disables the others and
never aborts startup. Surface the detected capability, so an operator can see why a feature
is on or off.
