# Copilot official CLI transport

## Current state

- Implemented explicit `native|official-cli` selection while preserving the
  `copilot` provider identity and native default.
- The official path uses GitHub Copilot CLI ACP v1 over managed stdio, skips
  ACP authentication, and lets the raw child inherit the parent environment.
- Session creation/loading, model selection, text/thought/tool status events,
  usage, stderr errors, permission decisions, cancellation, and child cleanup
  are covered by fake-process contracts.

## Evidence

- `cargo test -p jcode-provider-copilot --lib`
- `cargo test -p jcode-provider-copilot-runtime --lib`
- `cargo test -p jcode-provider-copilot-runtime --test official_cli`
- `cargo check -p jcode-provider-doctor`
- `cargo check -p jcode --no-default-features`
- Targeted executable build: `cargo build -p jcode --no-default-features --bin jcode`
- New login shell text smoke: provider `Copilot`, transport `official-cli ACP`,
  model `claude-sonnet-4.6`, exit 0, response `OK`.
- New login shell coding smoke: official CLI read `progress.md`, emitted ACP
  tool status, and returned `# Copilot official CLI transport`.
- Installed at
  `~/.jcode/builds/versions/0.81.5-dev-copilot-official-acp.1/jcode`; stable and
  `~/.local/bin/jcode` resolve to that binary, with v0.81.4 retained.
- `verify.sh --scan .` still reports the repository's pre-existing fake
  long-form credential fixtures; the change diff contains no credential value.

## Next

- Fork branch is pushed. GitHub rejected upstream draft PR creation for
  `shatianming5` with `CreatePullRequest` permission errors (REST returned 404),
  so no PR was opened; the implementation itself has no remaining blocker.
