use futures::StreamExt;
use jcode_message_types::{Message, StreamEvent, ToolDefinition};
use jcode_provider_copilot_runtime::{CopilotApiProvider, CopilotOfficialCliProcess};
use jcode_provider_core::Provider;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

fn fake_process(log: &Path) -> CopilotOfficialCliProcess {
    let mut env = BTreeMap::new();
    env.insert(
        "JCODE_FAKE_COPILOT_ACP_LOG".to_string(),
        log.display().to_string(),
    );
    CopilotOfficialCliProcess::with_command(env!("CARGO_BIN_EXE_jcode-fake-copilot-acp").into())
        .with_env(env)
}

fn one_tool() -> ToolDefinition {
    ToolDefinition {
        name: "view".to_string(),
        description: "Read a file".to_string(),
        input_schema: json!({"type":"object"}),
    }
}

#[test]
fn official_transport_requires_an_explicit_cli_path() {
    let _guard = jcode_base::storage::lock_test_env();
    let previous_transport = std::env::var_os("JCODE_COPILOT_TRANSPORT");
    let previous_path = std::env::var_os("JCODE_COPILOT_CLI_PATH");
    unsafe {
        std::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
        std::env::remove_var("JCODE_COPILOT_CLI_PATH");
    }
    let error = match CopilotApiProvider::new() {
        Ok(_) => panic!("official-cli without a raw CLI path should fail"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("JCODE_COPILOT_CLI_PATH"));
    unsafe {
        match previous_transport {
            Some(value) => std::env::set_var("JCODE_COPILOT_TRANSPORT", value),
            None => std::env::remove_var("JCODE_COPILOT_TRANSPORT"),
        }
        match previous_path {
            Some(value) => std::env::set_var("JCODE_COPILOT_CLI_PATH", value),
            None => std::env::remove_var("JCODE_COPILOT_CLI_PATH"),
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn fake_official_cli_covers_command_env_handshake_stream_usage_and_permissions() {
    let _guard = jcode_base::storage::lock_test_env();
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("official-acp.jsonl");
    unsafe {
        std::env::set_var("JCODE_FAKE_COPILOT_PARENT_SENTINEL", "inherited");
        std::env::set_var("COPILOT_ALLOW_ALL", "true");
    }
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    provider.prefetch_models().await.unwrap();
    assert_eq!(
        provider.available_models_display(),
        ["claude-sonnet-4.6", "gpt-5-mini"]
    );
    provider.set_model("gpt-5-mini").unwrap();

    let mut stream = provider
        .complete(
            &[Message::user("Reply exactly OK")],
            &[one_tool()],
            "outer-system",
            None,
        )
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    assert!(events.iter().any(
        |event| matches!(event, StreamEvent::ConnectionType { connection } if connection.contains("official-cli"))
    ));
    assert!(
        events.iter().any(
            |event| matches!(event, StreamEvent::SessionId(id) if id == "fake-copilot-session")
        )
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::ThinkingDelta(text) if text == "thinking"))
    );
    assert!(events.iter().any(
        |event| matches!(event, StreamEvent::StatusDetail { detail } if detail == "Viewing Cargo.toml")
    ));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::TokenUsage {
            input_tokens: Some(10),
            output_tokens: Some(2),
            cache_read_input_tokens: Some(3),
            cache_creation_input_tokens: Some(4),
        }
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
    );

    let requests = std::fs::read_to_string(&log).unwrap();
    assert!(requests.contains("\"--acp\""));
    assert!(requests.contains("\"--stdio\""));
    assert!(requests.contains("\"--no-auto-update\""));
    assert!(!requests.contains("\"--allow-all\""));
    assert!(requests.contains("\"sentinel\":\"inherited\""));
    assert!(requests.contains("\"allow_all\":null"));
    assert!(requests.contains("\"method\":\"initialize\""));
    assert!(!requests.contains("\"method\":\"authenticate\""));
    assert!(requests.contains("\"method\":\"session/new\""));
    assert!(requests.contains("\"method\":\"session/set_model\""));
    assert!(requests.contains("\"optionId\":\"once\""));
    assert!(!requests.contains("\"optionId\":\"always\",\"outcome\""));
    assert!(!requests.contains("copilot_internal"));
    assert!(!requests.contains("COPILOT_GITHUB_TOKEN"));
    unsafe {
        std::env::remove_var("JCODE_FAKE_COPILOT_PARENT_SENTINEL");
        std::env::remove_var("COPILOT_ALLOW_ALL");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn resume_uses_session_load_without_replaying_or_flattening_history() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("resume.jsonl");
    let provider = CopilotApiProvider::with_official_process(fake_process(&log));
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(
            &[
                Message::user("old prompt"),
                Message::assistant_text("old answer"),
                Message::user("new prompt"),
            ],
            &[one_tool()],
            "outer-system",
            Some("existing-session"),
        )
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta(delta) = event.unwrap() {
            text.push_str(&delta);
        }
    }
    assert_eq!(text, "OK");

    let requests = std::fs::read_to_string(&log).unwrap();
    assert!(requests.contains("\"method\":\"session/load\""));
    assert!(requests.contains("\"sessionId\":\"existing-session\""));
    assert!(requests.contains("new prompt"));
    assert!(!requests.contains("old answer"));
}

#[tokio::test(flavor = "current_thread")]
async fn disconnected_history_is_rejected_instead_of_flattened() {
    let temp = tempfile::tempdir().unwrap();
    let provider =
        CopilotApiProvider::with_official_process(fake_process(&temp.path().join("history.jsonl")));
    provider.complete_init_without_tier_detection();
    let error = match provider
        .complete(
            &[
                Message::user("old prompt"),
                Message::assistant_text("old answer"),
                Message::user("new prompt"),
            ],
            &[one_tool()],
            "",
            None,
        )
        .await
    {
        Ok(_) => panic!("disconnected history should fail"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("cannot replay disconnected history"));
}

#[tokio::test(flavor = "current_thread")]
async fn official_failure_surfaces_stderr_without_native_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("failure.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_FAIL".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let error = provider.complete_simple("fail", "").await.unwrap_err();
    let detail = format!("{error:#}");
    assert!(detail.contains("fake official-cli failure"), "{detail}");
    assert!(
        detail.contains("official Copilot CLI request failed"),
        "{detail}"
    );
    assert!(!detail.contains("token exchange"), "{detail}");

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(!requests.contains("\"method\":\"authenticate\""));
}

#[tokio::test(flavor = "current_thread")]
async fn no_tool_profile_cancels_permission_requests() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("permission-cancel.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(&[Message::user("Reply exactly OK")], &[], "", None)
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(requests.contains("\"outcome\":\"cancelled\""));
    assert!(!requests.contains("\"optionId\":\"always\",\"outcome\""));
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_stream_cancels_prompt_and_terminates_official_cli() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("cancel.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_HANG".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(&[Message::user("wait")], &[one_tool()], "", None)
        .await
        .unwrap();
    let session = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("session setup timed out")
        .expect("stream closed before session setup")
        .unwrap();
    assert!(matches!(session, StreamEvent::ConnectionType { .. }));
    let session = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("session setup timed out")
        .expect("stream closed before session setup")
        .unwrap();
    assert!(matches!(session, StreamEvent::SessionId(_)));
    drop(stream);

    let mut cancelled = false;
    for _ in 0..40 {
        let requests = std::fs::read_to_string(&log).unwrap_or_default();
        if requests.contains("\"method\":\"session/cancel\"") {
            cancelled = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(cancelled, "stream drop did not send session/cancel");
}
