set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# --- Local verification ("local CI") ---
# Run locally instead of GitHub Actions. `install-hooks` wires `check` into a
# git pre-push hook so it runs automatically before every push.
check: fmt-check lint build test
fmt-check:
    cargo fmt --all --check
fmt:
    cargo fmt --all
lint:
    cargo clippy --workspace --all-targets -- -D warnings
build:
    cargo build --workspace
test:
    cargo test --workspace
test-integration:
    cargo test --workspace -- --ignored
premerge:
    git fetch origin
    git rebase origin/main
    just check
install-hooks:
    git config core.hooksPath .githooks
    @echo "pre-push hook active — bypass once with: git push --no-verify"
