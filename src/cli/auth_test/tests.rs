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
    std::fs::write(
        &fake_cli,
        r#"#!/usr/bin/python3
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": True},
                "agentInfo": {"name": "Copilot", "version": "test"},
                "authMethods": [{"id": "copilot-login", "name": "CLI-managed"}],
            },
        }
        print(json.dumps(response), flush=True)
"#,
    )
    .unwrap();
    std::fs::set_permissions(&fake_cli, std::fs::Permissions::from_mode(0o755)).unwrap();

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
