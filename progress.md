# Copilot official CLI transport

## Current state

- The `copilot` provider keeps native transport as the default and uses the
  official CLI only when explicitly selected.
- Each Jcode session/fork owns one managed ACP child and reuses it across
  normal turns with session-local model, cwd, and tool policy.
- Cross-process `session/load` `ResourceNotFound` is treated as a typed stale
  upstream session. The same user turn creates a fresh ACP session, applies the
  current model/cwd/system policy, and sends only the current user prompt once.
- No prior user, assistant, tool, or resource history is replayed. Legacy
  safe/unsafe session markers are accepted only to extract the old upstream ID
  and are replaced by the raw fresh ACP session ID.
- Replacement provider session IDs are persisted immediately by the real Agent
  stream path and restored by `Agent::new_with_session`.
- Every official provider request carries caller-owned typed content blocks.
  Soft interrupts, tool continuations, and images are taken only from the
  request-local message delta; absent context is rejected instead of inferred
  from prior assistant/user history. Recalled memory remains separate additive
  context.
- ACP I/O health is visible to the provider before the next turn. Between-turn
  child death starts a fresh worker; mid-turn child death returns one error and
  never automatically resends that prompt.
- Worker lifecycle is linearized as `Healthy -> Cancelling -> Closed` with
  generation-owned cancellation. Dropping a queued turn cannot cancel the
  active turn, while every active/queued turn reaches one explicit terminal
  outcome.
- Setup and completion commits are fenced by cancellation, lifecycle, and
  generation checks. A ready ACP response wins over simultaneous child EOF;
  cancellation still has priority and late events/permissions cannot cross
  generations.
- Existing cwd isolation, permission allowlist, internal-tool auth-test,
  official auth ownership, model-switch, and fork isolation behavior remains.

## Evidence

- The real headless Agent regression now creates and saves its initial upstream
  ID through the first turn, triggers typed stale recovery in a second Agent,
  fails that prompt before normal turn-final persistence, and restores the
  replacement ID in a third Agent. No provider ID is injected by the test.
- Red-capability mutation: suppressing only replacement-ID persistence made
  `headless_agent_persists_stale_replacement_across_restarts` fail with the
  on-disk ID still `fake-copilot-session` instead of
  `recovered-copilot-session`; restoring the save path made it pass.
- `cargo test -p jcode-provider-copilot-runtime --lib` passes 44 tests.
- `cargo test -p jcode-provider-copilot-runtime --test official_cli` passes 32
  integration tests.
- `cargo test -p jcode-app-core --no-default-features provider_session_tests`
  passes 3 tests.
- `cargo test -p jcode-app-core --no-default-features run_turn_streaming_mpsc`
  passes 3 tests; the exact soft-interrupt request-context regression passes 1
  additional targeted test.
- `cargo test -p jcode --no-default-features auth_test` passes 39 tests, and
  the Unix-only auth fixture is gated out of Windows test builds.
- `cargo test -p jcode-provider-core` passes 126 tests.
- Stale integration coverage asserts current system/current prompt exactly
  once, zero old history/resource for normal, side-effect, legacy, unmarked,
  and no-assistant histories, current model application, and replacement ID
  persistence across a real Agent restore.
- Lifecycle coverage asserts three normal turns use one child/session,
  between-turn death recovers before the next prompt, mid-turn crash errors
  once with no replay, cancelled late text/tool events do not leak, and a hung
  child exits when its provider is dropped.
- Fork/cwd tests use two real provider forks and assert no cwd/model/session
  cross-contamination.
- `cargo check -p jcode --no-default-features` passes.
- Final external review passed after targeted red/green coverage for exact
  per-request soft-interrupt text/images, queued-turn cancellation, post-await
  commit fencing, missing-context rejection, and response-before-EOF ordering.
- Repository-prescribed `jcode self-dev --build` completed its build and
  published immutable version `4a4979372-dirty-a8755071dafd` using the existing
  Rust 1.94.1 toolchain. `current`, `shared-server`, and the launcher resolve to
  that binary; forced server reload reported `handoff_ready=true`. The installed
  binary reports `jcode v0.81.23-dev (4a4979372, dirty)`.
- Serial fresh-login-shell smoke against raw `/opt/homebrew/bin/copilot` and the
  installed launcher passes: official auth-test returns `AUTH_TEST_OK` for both
  provider and read-only tool smoke, ordinary text returns `ACP_TEXT_OK`, and
  two fresh-process restores return `ACP_RESTART_OK` and `ACP_REOPEN_OK` for
  local session `session_hare_1788313019872_0efe92da4b2f927e`. No ACP child or
  orphan process remains.
- GitHub Copilot CLI 1.0.83-1 successfully loaded the real upstream session
  across these process restarts, so the real smoke did not naturally enter the
  typed `ResourceNotFound` branch; deterministic fake-ACP coverage exercises
  that branch and records load IDs changing from the stale ID to the persisted
  replacement ID.
- `cargo check -p jcode --no-default-features`, `cargo fmt --all -- --check`,
  and `git diff --check` pass.

## Next

- Push the reviewed branch and await upstream pull-request review.
