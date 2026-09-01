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

## Evidence

- Red-before-fix:
  `view_only_profile_rejects_execute_permission_by_kind` selected AllowOnce on
  the old implementation.
- `cargo test -p jcode-provider-copilot --lib`
- `cargo test -p jcode-provider-copilot-runtime --lib`
- `cargo test -p jcode-provider-copilot-runtime --test official_cli` (13 passed)
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
  `~/.jcode/builds/versions/0.81.9-dev-c091fff9d/jcode`; `current`,
  `shared-server`, and `~/.local/bin/jcode` resolve to it. The v0.81.4 binary
  and previous `0.81.7-dev-5067d2f5a` dev build remain available for rollback.
- Fresh login shell reports `jcode v0.81.9-dev (c091fff9d)`.
- `verify.sh --scan .` still reports the repository's pre-existing fake
  long-form credential fixtures; the change diff contains no credential value.

## Next

- Push the final empty-tool/installation evidence commit and refresh draft PR
  `shatianming5/jcode#1`.
- Upstream draft PR creation previously returned the known
  `CreatePullRequest`/404 permission blocker; do not retry it indefinitely.
