# Copilot official CLI transport

## Current state

- Implemented explicit `native|official-cli` selection while preserving the
  `copilot` provider identity and native default.
- The official path uses GitHub Copilot CLI ACP v1 over managed stdio, skips
  ACP authentication, and lets the raw child inherit the parent environment.
- Review fixes now pass each Agent request's working directory into the
  provider, child process, and ACP session/new or session/load without shared
  cwd state.
- Official CLI tool availability and permission replies are derived from the
  filtered Jcode tool list; unknown kinds reject, AllowAlways is never chosen,
  and builtin GitHub MCP tools are disabled.
- Same-provider official Copilot model switches retain that local Agent's ACP
  session ID; opaque ACP sessions reject rewind before local history changes.
- Auth-test uses an ACP capability handshake in official mode, and usage reports
  remote quota as CLI-managed/unsupported without loading native credentials.
- Final review fixes make `MultiProvider::fork` call the Copilot runtime's own
  fork so model state stays session-local, validate official internal tool smoke
  from ACP status plus exact final text, and reject `auth-test --login` before
  Jcode's native device/token flow.
- Terminal review fixes also make Copilot fork-local premium mode, reasoning
  effort, and OnePerSession turn counting; expose structured provider-tool
  lifecycle updates; and scope official CLI configuration-notice stripping to
  auth-test only so ordinary assistant text is untouched.

## Evidence

- Red-before-fix:
  `view_only_profile_rejects_execute_permission_by_kind` selected AllowOnce on
  the old implementation.
- Red-before-final-fix:
  `copilot_forks_do_not_share_mutable_model_state` showed an A-side model switch
  changing B from `claude-sonnet-4.6` to `gpt-5-mini`.
- `cargo test -p jcode-provider-copilot --lib`
- `cargo test -p jcode-provider-copilot-runtime --lib`
- `cargo test -p jcode-provider-copilot-runtime --test official_cli` (14 passed,
  including real `MultiProvider`/fake-ACP dual-Agent resume isolation)
- `cargo test -p jcode-base --no-default-features
  copilot_forks_do_not_share_mutable_model_state`
- `cargo test -p jcode-provider-copilot-runtime --lib
  native_copilot_fork_keeps_model_state_session_local`
- Official auth-test command fixtures: full default smoke passes with exact
  `AUTH_TEST_OK` and ACP internal read-only tool status; official `--login`
  refuses before its login-runner spy and leaves no pending/token file; native
  `--login` still invokes the existing runner.
- Real new-login-shell `jcode auth-test --provider copilot` completed provider
  smoke and read-only ACP-internal tool smoke with exact `AUTH_TEST_OK` outputs
  and exit 0.
- Real official `auth-test --login` returned the CLI-managed-auth refusal before
  any native login flow; isolated pending-login and saved-token paths remained
  absent.
- Official CLI `Info: Disabled tools: ...` configuration notifications are
  stripped only by auth-test when they are independent notification fragments;
  a same-chunk `AUTH_TEST_OK` body is retained, while an ordinary real run
  returning `Info: Disabled tools: legitimate` preserves that text verbatim.
- Strict internal-tool smoke requires one read/search tool ID to enter
  pending/in-progress and then completed. Completed-only, failed, incomplete,
  and execute-kind fixtures are rejected; the real read-only `Cargo.toml` smoke
  passes this lifecycle check.
- Final auth-test cleanup buffers text across chunks and removes only complete,
  exact configuration-notice lines. A retained `ERROR` before `AUTH_TEST_OK`
  fails strict output validation, while split notification/marker chunks and
  legitimate `Info:` text preserve order and content.
- Configuration-line candidates are never deleted at chunk arrival. They remain
  buffered until an explicit newline or EOF; a later trailing space/body keeps
  the whole line and fails strict output validation, while an exact
  newline-free notification may be removed only at EOF.
- The provider now forwards ACP text chunks byte-for-byte with no auth-test flag
  or chunk-boundary newline synthesis. End-to-end provider-event tests cover
  continued candidate lines, exact EOF removal, and newline-delimited marker
  success. Internal tool smoke validates the tool lifecycle first, then performs
  a separate no-tool marker confirmation through the same line cleaner.
- Live diagnosis confirmed official Copilot ACP sessions are process-local:
  `session/load` returns typed `ResourceNotFound` in a new process with both the
  same and a different cwd, while the original process reports the session is
  already loaded.
- Each forked Jcode session now owns one persistent ACP worker/child across
  normal turns. Persisted provider IDs carry proven-safe/unsafe history state;
  typed stale IDs are replaced once instead of retried forever.
- Safe text-only history is attached to a replacement ACP session as an
  embedded READ-ONLY resource and only the current user prompt is sent as text.
  Legacy/unmarked or side-effect history is never replayed: Jcode creates and
  persists a replacement ID, returns one explicit resend boundary, then the
  next turn continues without another stale load.
- Fixture coverage includes one-child multi-turn/model switching, provider
  restart, child crash then recovery, safe resource recovery, unmarked history
  refusal, current-prompt-once, fork/cwd isolation, cancellation, and cleanup.
- Real isolated-server reproduction completed `TURN1_OK`, naturally ended the
  per-run ACP child lifecycle, then resumed the same local session with
  `TURN2_OK` and `TURN3_OK` without a repeated NotFound loop.
- Provider-tool validation now enforces monotonic per-ID transitions:
  Pending→Completed and InProgress→Completed pass; Completed-only,
  Pending→Completed→InProgress, failed, incomplete, and wrong-kind sequences
  fail. Completed must be the final observed state.
- `cargo test -p jcode-provider-copilot-runtime --test official_cli` now passes
  15 tests; fork settings and per-fork first-turn tests pass; full
  `cargo check -p jcode --no-default-features` passes.
- Final new-login-shell smokes: official auth-test exit 0, ordinary text exactly
  `OK`, legitimate `Info:`-prefixed reply preserved, and no orphan ACP child.
- `cargo test -p jcode-app-core --no-default-features provider_session_tests`
- `cargo test -p jcode-base --no-default-features
  official_copilot_usage_is_cli_managed_without_native_credentials`
- `cargo test -p jcode --no-default-features
  official_copilot_auth_probe_uses_acp_without_native_token_sources`
- `cargo check -p jcode-provider-doctor`
- `cargo check -p jcode --no-default-features`
- Targeted executable build: `cargo build -p jcode --no-default-features --bin jcode`
- New login shell against an isolated long-lived server started in repo cwd A:
  text response exactly `OK`; NDJSON resolved provider `Copilot` and connection
  `official-cli ACP (claude-sonnet-4.6)`.
- From a different cwd B, `--tools view` read B's `official_cli.rs` and returned
  its known cwd regression-test name; the same profile returned `DENIED` for a
  forced shell/write attempt and the requested target file remained absent.
- Full/default tools completed a safe read-only turn from B and returned
  `FULL_READ_OK`.
- Empty tools launch the official CLI with a bare `--available-tools`; the real
  CLI emitted no unknown-tool warning and the completed text was exactly `OK`.
- `jcode run --resume ... --model gpt-5-mini` currently restores the persisted
  local model instead of applying a model switch, so model-switch continuation
  is covered by the fake ACP integration and two-Agent isolation tests rather
  than claimed as a CLI E2E.
- No orphan `copilot --acp --stdio` process remained after smoke shutdown.
- Installed atomically at
  `~/.jcode/builds/versions/0.81.19-dev-34e6ebeba/jcode`; `current`,
  `shared-server`, and `~/.local/bin/jcode` resolve to it. The v0.81.4 binary
  and previous `0.81.17-dev-d63fd1631` dev build remain available for rollback.
- Fresh login shell reports `jcode v0.81.19-dev (34e6ebeba)`.
- `verify.sh --scan .` still reports the repository's pre-existing fake
  long-form credential fixtures; the change diff contains no credential value.

## Next

- Refresh draft PR #1 with terminal-review evidence. Upstream PR creation remains
  unnecessary after the known `CreatePullRequest`/404 permission blocker.
