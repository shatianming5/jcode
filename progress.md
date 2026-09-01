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
- ACP I/O health is visible to the provider before the next turn. Between-turn
  child death starts a fresh worker; mid-turn child death returns one error and
  never automatically resends that prompt.
- Cancel stops event forwarding before notification, drains briefly, and then
  terminates the exact managed worker/child. Provider drop also requests
  immediate worker shutdown.
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
- `cargo test -p jcode-provider-copilot-runtime --lib` passes 39 tests.
- `cargo test -p jcode-provider-copilot-runtime --test official_cli` passes 24
  integration tests.
- `cargo test -p jcode-app-core --no-default-features provider_session_tests`
  passes 3 tests.
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
- `cargo build --profile selfdev -p jcode --no-default-features --bin jcode`
  passes without installing or repointing any jcode channel.
- Serial new-login-shell smoke against raw `/opt/homebrew/bin/copilot` and the
  uninstalled selfdev binary passes: official auth-test reports all four
  credential/refresh/provider/tool steps successful, ordinary text returns
  `ACP_TEXT_OK`, and two fresh-process resumes return `ACP_RESTART_OK` and
  `ACP_REOPEN_OK` with a persisted provider session ID and zero ACP orphans.
- GitHub Copilot CLI 1.0.83-1 successfully loaded the real upstream session
  across these process restarts, so the real smoke did not naturally enter the
  typed `ResourceNotFound` branch; deterministic fake-ACP coverage exercises
  that branch and records load IDs changing from the stale ID to the persisted
  replacement ID.
- The rejected `0.81.21-dev-e1cc8bcbe` build was removed from `current` and
  `shared-server`; both temporarily point to rollback build
  `0.81.19-dev-34e6ebeba` until the fixed build is installed.

## Next

- Keep installation deferred: active `current` and `shared-server` channels
  remain on the rollback build until the coordinator explicitly promotes this
  commit.
