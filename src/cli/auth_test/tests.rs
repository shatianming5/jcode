use super::*;
use std::os::unix::fs::PermissionsExt;

struct SavedEnv {
    values: Vec<(String, Option<String>)>,
}

impl SavedEnv {
    fn capture(keys: &[&str]) -> Self {
        Self {
            values: keys
                .iter()
                .map(|key| (key.to_string(), std::env::var(key).ok()))
                .collect(),
        }
    }
}

impl Drop for SavedEnv {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            match value {
                Some(value) => crate::env::set_var(key, value),
                None => crate::env::remove_var(key),
            }
        }
    }
}

fn write_fake_official_cli(path: &std::path::Path) {
    std::fs::write(
        path,
        r#"#!/usr/bin/python3
import json
import os
import sys

def log(value):
    path = os.environ.get("JCODE_FAKE_AUTH_TEST_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")

log({"args": sys.argv[1:]})
for line in sys.stdin:
    request = json.loads(line)
    log(request)
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": 1,
            "agentCapabilities": {"loadSession": True},
            "agentInfo": {"name": "Copilot", "version": "test"},
            "authMethods": [{"id": "copilot-login", "name": "CLI-managed"}],
        }
    elif method == "session/new":
        result = {
            "sessionId": "auth-test-session",
            "models": {
                "currentModelId": "claude-sonnet-4.6",
                "availableModels": [
                    {"modelId": "claude-sonnet-4.6", "name": "Claude Sonnet 4.6"}
                ],
            },
        }
    elif method == "session/load":
        result = {
            "models": {
                "currentModelId": "claude-sonnet-4.6",
                "availableModels": [
                    {"modelId": "claude-sonnet-4.6", "name": "Claude Sonnet 4.6"}
                ],
            }
        }
    elif method == "session/set_model":
        result = {}
    elif method == "session/prompt":
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "auth-test-session",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "Info: Disabled tools: bash, edit, write",
                    },
                },
            },
        }), flush=True)
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "auth-test-session",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "auth-tool",
                    "title": "Reading Cargo.toml",
                    "kind": "read",
                    "status": "pending",
                },
            },
        }), flush=True)
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "auth-test-session",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "auth-tool",
                    "status": "completed",
                },
            },
        }), flush=True)
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "auth-test-session",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "AUTH_TEST_OK"},
                },
            },
        }), flush=True)
        result = {
            "stopReason": "end_turn",
            "usage": {"totalTokens": 2, "inputTokens": 1, "outputTokens": 1},
        }
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"#,
    )
    .unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn official_copilot_auth_probe_uses_acp_without_native_token_sources() {
    let _guard = crate::storage::lock_test_env();
    let _saved = SavedEnv::capture(&[
        "JCODE_COPILOT_TRANSPORT",
        "JCODE_COPILOT_CLI_PATH",
        "JCODE_HOME",
        "HOME",
        "PATH",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ]);
    let temp = tempfile::tempdir().unwrap();
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let marker = temp.path().join("native-loader-invoked");
    let fake_gh = bin_dir.join("gh");
    std::fs::write(
        &fake_gh,
        format!(
            "#!/bin/sh\nprintf invoked >> '{}'\nexit 1\n",
            marker.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake_gh, std::fs::Permissions::from_mode(0o755)).unwrap();

    let fake_cli = bin_dir.join("copilot");
    write_fake_official_cli(&fake_cli);

    crate::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
    crate::env::set_var("JCODE_COPILOT_CLI_PATH", &fake_cli);
    crate::env::set_var("JCODE_HOME", temp.path().join("jcode-home"));
    crate::env::set_var("HOME", temp.path().join("home"));
    crate::env::set_var("PATH", &bin_dir);
    crate::env::remove_var("COPILOT_GITHUB_TOKEN");
    crate::env::remove_var("GH_TOKEN");
    crate::env::remove_var("GITHUB_TOKEN");

    let mut report = AuthTestProviderReport::new(AuthTestTarget::Copilot);
    probe_copilot_auth(&mut report).await;

    assert!(report.success, "{:?}", report.steps);
    assert!(report.steps.iter().any(|step| step.name == "refresh_probe"
        && step.ok
        && step.detail.contains("ACP capability handshake succeeded")));
    assert!(report.steps.iter().any(|step| {
        step.detail
            .contains("native GitHub token sources were not inspected")
    }));
    assert!(
        !marker.exists(),
        "official-cli auth probe invoked the native GitHub token loader"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn official_copilot_full_auth_test_uses_internal_read_only_tool_evidence() {
    let _guard = crate::storage::lock_test_env();
    let _saved = SavedEnv::capture(&[
        "JCODE_COPILOT_TRANSPORT",
        "JCODE_COPILOT_CLI_PATH",
        "JCODE_HOME",
        "JCODE_NON_INTERACTIVE",
        "JCODE_FAKE_AUTH_TEST_LOG",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ]);
    let temp = tempfile::tempdir().unwrap();
    let fake_cli = temp.path().join("copilot");
    let log = temp.path().join("auth-test.jsonl");
    write_fake_official_cli(&fake_cli);
    crate::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
    crate::env::set_var("JCODE_COPILOT_CLI_PATH", &fake_cli);
    crate::env::set_var("JCODE_HOME", temp.path().join("jcode-home"));
    crate::env::set_var("JCODE_NON_INTERACTIVE", "1");
    crate::env::set_var("JCODE_FAKE_AUTH_TEST_LOG", &log);
    crate::env::remove_var("COPILOT_GITHUB_TOKEN");
    crate::env::remove_var("GH_TOKEN");
    crate::env::remove_var("GITHUB_TOKEN");
    crate::cli::startup::register_external_provider_runtimes();

    let internal_output = run_provider_tool_smoke_for_choice(
        &super::super::provider_init::ProviderChoice::Copilot,
        Some("claude-sonnet-4.6"),
        DEFAULT_AUTH_TEST_TOOL_PROMPT,
    )
    .await
    .expect("official internal tool smoke");
    assert_eq!(internal_output, "AUTH_TEST_OK");

    run_auth_test_command(
        &super::super::provider_init::ProviderChoice::Copilot,
        Some("claude-sonnet-4.6"),
        false,
        false,
        false,
        false,
        None,
        true,
        None,
    )
    .await
    .expect("full official Copilot auth-test should pass");

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(requests.contains("--available-tools=view"), "{requests}");
    assert!(
        requests.matches("\"method\": \"session/prompt\"").count() >= 3,
        "{requests}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn official_copilot_auth_test_login_rejects_before_native_login_runner() {
    let _guard = crate::storage::lock_test_env();
    let _saved = SavedEnv::capture(&[
        "JCODE_COPILOT_TRANSPORT",
        "JCODE_COPILOT_CLI_PATH",
        "JCODE_HOME",
    ]);
    let temp = tempfile::tempdir().unwrap();
    crate::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
    crate::env::set_var("JCODE_COPILOT_CLI_PATH", "/usr/bin/false");
    crate::env::set_var("JCODE_HOME", temp.path().join("jcode-home"));
    let login_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spy = std::sync::Arc::clone(&login_called);
    let mut report = AuthTestProviderReport::new(AuthTestTarget::Copilot);

    run_auth_test_login_with(AuthTestTarget::Copilot, &mut report, move |_| {
        spy.store(true, std::sync::atomic::Ordering::SeqCst);
        async { Ok(()) }
    })
    .await;

    assert!(!login_called.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!report.success);
    assert!(report.steps.iter().any(|step| {
        step.name == "login"
            && !step.ok
            && step
                .detail
                .contains("Official Copilot CLI manages authentication")
    }));
    let jcode_home = temp.path().join("jcode-home");
    assert!(!jcode_home.join("pending-login/copilot.json").exists());
    assert!(
        !jcode_home
            .join("external/.config/github-copilot/hosts.json")
            .exists()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn native_copilot_auth_test_login_still_invokes_existing_login_runner() {
    let _guard = crate::storage::lock_test_env();
    let _saved = SavedEnv::capture(&["JCODE_COPILOT_TRANSPORT"]);
    crate::env::set_var("JCODE_COPILOT_TRANSPORT", "native");
    let login_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spy = std::sync::Arc::clone(&login_called);
    let mut report = AuthTestProviderReport::new(AuthTestTarget::Copilot);

    run_auth_test_login_with(AuthTestTarget::Copilot, &mut report, move |choice| {
        assert_eq!(choice, super::super::provider_init::ProviderChoice::Copilot);
        spy.store(true, std::sync::atomic::Ordering::SeqCst);
        async { Ok(()) }
    })
    .await;

    assert!(login_called.load(std::sync::atomic::Ordering::SeqCst));
    assert!(report.success);
    assert!(
        report
            .steps
            .iter()
            .any(|step| step.name == "login" && step.ok)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn official_copilot_full_auth_test_login_command_refuses_but_still_probes_cli() {
    let _guard = crate::storage::lock_test_env();
    let _saved = SavedEnv::capture(&[
        "JCODE_COPILOT_TRANSPORT",
        "JCODE_COPILOT_CLI_PATH",
        "JCODE_HOME",
        "JCODE_NON_INTERACTIVE",
        "JCODE_FAKE_AUTH_TEST_LOG",
    ]);
    let temp = tempfile::tempdir().unwrap();
    let fake_cli = temp.path().join("copilot");
    let log = temp.path().join("login-refusal.jsonl");
    write_fake_official_cli(&fake_cli);
    let jcode_home = temp.path().join("jcode-home");
    crate::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
    crate::env::set_var("JCODE_COPILOT_CLI_PATH", &fake_cli);
    crate::env::set_var("JCODE_HOME", &jcode_home);
    crate::env::set_var("JCODE_NON_INTERACTIVE", "1");
    crate::env::set_var("JCODE_FAKE_AUTH_TEST_LOG", &log);
    crate::cli::startup::register_external_provider_runtimes();

    let error = run_auth_test_command(
        &super::super::provider_init::ProviderChoice::Copilot,
        Some("claude-sonnet-4.6"),
        true,
        false,
        true,
        true,
        None,
        true,
        None,
    )
    .await
    .expect_err("official --login must be rejected");

    assert!(error.to_string().contains("One or more auth tests failed"));
    let requests = std::fs::read_to_string(log).unwrap();
    assert!(
        requests.contains("\"method\": \"initialize\""),
        "{requests}"
    );
    assert!(!jcode_home.join("pending-login/copilot.json").exists());
    assert!(
        !jcode_home
            .join("external/.config/github-copilot/hosts.json")
            .exists()
    );
}
