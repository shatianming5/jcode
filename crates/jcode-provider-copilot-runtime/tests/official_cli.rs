use futures::StreamExt;
use jcode_app_core::{agent::Agent, session::Session, tool::Registry};
use jcode_base::provider::{MultiProvider, external};
use jcode_message_types::{Message, StreamEvent, ToolDefinition};
use jcode_provider_copilot_runtime::{CopilotApiProvider, CopilotOfficialCliProcess};
use jcode_provider_core::{Provider, ProviderRequestContext};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

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

fn named_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: format!("{name} fixture"),
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
    let provider = CopilotApiProvider::with_official_process(process.clone());
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
            |event| matches!(event, StreamEvent::SessionId(id) if id == "jcode-copilot-acp-v1:safe:fake-copilot-session")
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
    assert!(requests.contains("\"--disable-builtin-mcps\""));
    assert!(requests.contains("\"--available-tools=view\""));
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
    let provider = CopilotApiProvider::with_official_process(process.clone());
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
async fn ordinary_run_preserves_legitimate_info_prefixed_assistant_text() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("legitimate-info-prefix.jsonl");
    let mut process = fake_process(&log);
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_REPLY".into(),
        "Info: Disabled tools: legitimate".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(
            &[Message::user("Return the requested literal text")],
            &[],
            "",
            None,
        )
        .await
        .unwrap();
    let mut output = String::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta(text) = event.unwrap() {
            output.push_str(&text);
        }
    }

    assert_eq!(output, "Info: Disabled tools: legitimate");
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
    assert!(requests.contains("\"--available-tools\""));
    assert!(!requests.contains("\"--available-tools="));
    assert!(requests.contains("\"optionId\":\"reject\""));
    assert!(!requests.contains("\"optionId\":\"always\",\"outcome\""));
}

#[tokio::test(flavor = "current_thread")]
async fn view_only_profile_rejects_execute_permission_by_kind() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("permission-view-only.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_PERMISSION_KIND".into(),
        "execute".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(
            &[Message::user("Try a shell command")],
            &[named_tool("view")],
            "",
            None,
        )
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(
        requests.contains("\"optionId\":\"reject\",\"outcome\":\"selected\""),
        "{requests}"
    );
    assert!(
        !requests.contains("\"optionId\":\"once\",\"outcome\":\"selected\""),
        "{requests}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn view_only_profile_rejects_write_permission_by_kind() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("permission-write-view-only.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_PERMISSION_KIND".into(),
        "edit".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(
            &[Message::user("Try a write")],
            &[named_tool("view")],
            "",
            None,
        )
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(
        requests.contains("\"optionId\":\"reject\",\"outcome\":\"selected\""),
        "{requests}"
    );
    assert!(
        requests.contains("\"--available-tools=view\""),
        "{requests}"
    );
    assert!(!requests.contains("available-tools=bash"), "{requests}");
    assert!(!requests.contains("available-tools=create"), "{requests}");
    assert!(!requests.contains("available-tools=edit"), "{requests}");
}

#[tokio::test(flavor = "current_thread")]
async fn unmapped_permission_kind_is_rejected_by_default() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("permission-unknown.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_PERMISSION_KIND".into(),
        "other".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let mut stream = provider
        .complete(
            &[Message::user("Try an unknown tool")],
            &[named_tool("view"), named_tool("bash"), named_tool("write")],
            "",
            None,
        )
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(
        requests.contains("\"optionId\":\"reject\",\"outcome\":\"selected\""),
        "{requests}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn full_profile_keeps_read_available_and_selects_allow_once_by_option_kind() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("permission-full-read.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_PERMISSION".into(), "1".into());
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_PERMISSION_KIND".into(),
        "read".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let tools = [named_tool("view"), named_tool("bash"), named_tool("write")];

    let mut stream = provider
        .complete(&[Message::user("Read a file")], &tools, "", None)
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let requests = std::fs::read_to_string(log).unwrap();
    assert!(
        requests.contains("\"optionId\":\"once\",\"outcome\":\"selected\""),
        "{requests}"
    );
    assert!(
        !requests.contains("\"optionId\":\"always\",\"outcome\":\"selected\""),
        "{requests}"
    );
    assert!(
        requests.contains("\"--available-tools=bash,create,view\""),
        "{requests}"
    );
}

async fn complete_in_dir(
    provider: &CopilotApiProvider,
    working_dir: &Path,
    messages: &[Message],
    resume_session_id: Option<&str>,
) -> Vec<StreamEvent> {
    let request_context = ProviderRequestContext::new(Some(working_dir.to_path_buf()));
    let mut stream = provider
        .complete_split_with_context(
            messages,
            &[one_tool()],
            "",
            "",
            resume_session_id,
            &request_context,
        )
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    events
}

#[tokio::test(flavor = "current_thread")]
async fn request_context_sets_child_and_acp_cwd_without_cross_session_leakage() {
    let temp = tempfile::tempdir().unwrap();
    let first_dir = temp.path().join("session-one");
    let second_dir = temp.path().join("session-two");
    std::fs::create_dir_all(&first_dir).unwrap();
    std::fs::create_dir_all(&second_dir).unwrap();
    let first_dir = std::fs::canonicalize(first_dir).unwrap();
    let second_dir = std::fs::canonicalize(second_dir).unwrap();
    assert_ne!(std::env::current_dir().unwrap(), first_dir);
    assert_ne!(std::env::current_dir().unwrap(), second_dir);
    let log_dir = temp.path().join("cwd-logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let mut process = CopilotOfficialCliProcess::with_command(
        env!("CARGO_BIN_EXE_jcode-fake-copilot-acp").into(),
    );
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_LOG_DIR".to_string(),
        log_dir.display().to_string(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let first_messages = [Message::user("first")];
    let second_messages = [Message::user("second")];

    let (first, second) = tokio::join!(
        complete_in_dir(&provider, &first_dir, &first_messages, None),
        complete_in_dir(&provider, &second_dir, &second_messages, None)
    );
    assert!(
        first
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
    );
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
    );

    for working_dir in [&first_dir, &second_dir] {
        let log = log_dir.join(format!(
            "{}.jsonl",
            working_dir.file_name().unwrap().to_string_lossy()
        ));
        let requests = std::fs::read_to_string(log).unwrap();
        let encoded = serde_json::to_string(working_dir).unwrap();
        let needle = format!("\"cwd\":{encoded}");
        assert!(
            requests.matches(&needle).count() >= 2,
            "child cwd and ACP cwd did not both use {} in {requests}",
            working_dir.display()
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn established_session_reuses_one_child_and_applies_model_switch() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("model-switch.jsonl");
    let provider = CopilotApiProvider::with_official_process(fake_process(&log));
    provider.complete_init_without_tier_detection();

    let first = complete_in_dir(
        &provider,
        &working_dir,
        &[Message::user("first turn")],
        None,
    )
    .await;
    let session_id = first
        .iter()
        .find_map(|event| match event {
            StreamEvent::SessionId(id) => Some(id.clone()),
            _ => None,
        })
        .expect("first turn session id");

    provider.set_model("gpt-5-mini").unwrap();
    let second = complete_in_dir(
        &provider,
        &working_dir,
        &[
            Message::user("first turn"),
            Message::assistant_text("OK"),
            Message::user("second turn"),
        ],
        Some(&session_id),
    )
    .await;
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
    );
    let third = complete_in_dir(
        &provider,
        &working_dir,
        &[
            Message::user("first turn"),
            Message::assistant_text("OK"),
            Message::user("second turn"),
            Message::assistant_text("OK"),
            Message::user("third turn"),
        ],
        Some(&session_id),
    )
    .await;
    assert!(
        third
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
    );

    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"process\"").count(), 1, "{requests}");
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/prompt\"").count(), 3);
    assert!(
        !requests.contains("\"method\":\"session/load\""),
        "{requests}"
    );
    assert!(
        requests.contains("\"sessionId\":\"fake-copilot-session\""),
        "{requests}"
    );
    assert!(
        requests.contains("\"method\":\"session/set_model\""),
        "{requests}"
    );
    assert_eq!(
        requests.matches("\"method\":\"session/set_model\"").count(),
        1
    );
    assert!(
        requests.contains("\"modelId\":\"gpt-5-mini\""),
        "{requests}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stale_cross_process_session_load_recovers_once_with_replaced_session_id() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("stale-session.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process.clone());
    provider.complete_init_without_tier_detection();

    let first = complete_in_dir(
        &provider,
        &working_dir,
        &[Message::user("first turn")],
        None,
    )
    .await;
    let stale_session_id = first
        .iter()
        .find_map(|event| match event {
            StreamEvent::SessionId(id) => Some(id.clone()),
            _ => None,
        })
        .expect("first turn session id");
    assert_eq!(
        stale_session_id,
        "jcode-copilot-acp-v1:safe:fake-copilot-session"
    );
    drop(provider);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let second = complete_in_dir(
        &provider,
        &working_dir,
        &[
            Message::user("first turn"),
            Message::assistant_text("OK"),
            Message::user("second turn"),
        ],
        Some(&stale_session_id),
    )
    .await;
    let recovered_session_id = second
        .iter()
        .find_map(|event| match event {
            StreamEvent::SessionId(id) => Some(id.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("recovered session id missing from {second:?}"));
    assert_eq!(
        recovered_session_id,
        "jcode-copilot-acp-v1:safe:recovered-copilot-session"
    );
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );

    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 2);
    assert_eq!(requests.matches("second turn").count(), 1);
    let recovered_prompt = requests
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| {
            value.get("method").and_then(serde_json::Value::as_str) == Some("session/prompt")
                && value.to_string().contains("second turn")
        })
        .expect("recovered prompt request");
    let prompt = recovered_prompt["params"]["prompt"].as_array().unwrap();
    assert_eq!(prompt.len(), 2, "{recovered_prompt}");
    assert_eq!(prompt[1]["type"], "text");
    assert_eq!(prompt[1]["text"], "second turn");
    assert_eq!(
        prompt[0]["type"], "resource",
        "history must be an embedded read-only resource: {recovered_prompt}"
    );
    assert!(
        prompt[0]
            .to_string()
            .contains("READ-ONLY historical transcript"),
        "{recovered_prompt}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn crashed_child_restarts_and_recovers_safe_history() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("crashed-child.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND".into(), "1".into());
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_EXIT_AFTER_PROMPT".into(),
        "1".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let first = complete_in_dir(
        &provider,
        &working_dir,
        &[Message::user("first turn")],
        None,
    )
    .await;
    let persisted = first
        .iter()
        .rev()
        .find_map(|event| match event {
            StreamEvent::SessionId(id) => Some(id.clone()),
            _ => None,
        })
        .expect("persisted session id");
    std::thread::sleep(std::time::Duration::from_millis(50));

    let messages = [
        Message::user("first turn"),
        Message::assistant_text("OK"),
        Message::user("second turn"),
    ];
    let request_context = ProviderRequestContext::new(Some(working_dir.clone()));
    let mut crashed_stream = provider
        .complete_split_with_context(
            &messages,
            &[one_tool()],
            "",
            "",
            Some(&persisted),
            &request_context,
        )
        .await
        .unwrap();
    let mut crash_error = None;
    while let Some(event) = crashed_stream.next().await {
        if let Err(error) = event {
            crash_error = Some(error.to_string());
        }
    }
    assert!(
        crash_error
            .as_deref()
            .is_some_and(|error| error.contains("official Copilot CLI request failed")),
        "{crash_error:?}"
    );
    std::thread::sleep(std::time::Duration::from_millis(50));

    let second = complete_in_dir(&provider, &working_dir, &messages, Some(&persisted)).await;
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );
    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"process\"").count(), 2, "{requests}");
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn unmarked_stale_history_replaces_id_once_without_replaying_current_prompt() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("unsafe-stale-session.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let request_context = ProviderRequestContext::new(Some(working_dir.clone()));
    let messages = [
        Message::user("old instruction"),
        Message::assistant_text("old answer"),
        Message::user("current prompt"),
    ];
    let mut stream = provider
        .complete_split_with_context(
            &messages,
            &[one_tool()],
            "",
            "",
            Some("fake-copilot-session"),
            &request_context,
        )
        .await
        .unwrap();
    let mut replacement = None;
    let mut boundary_error = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::SessionId(id)) => replacement = Some(id),
            Ok(_) => {}
            Err(error) => boundary_error = Some(error.to_string()),
        }
    }
    let replacement = replacement.expect("replacement session id");
    assert_eq!(
        replacement,
        "jcode-copilot-acp-v1:unsafe:recovered-copilot-session"
    );
    assert!(
        boundary_error
            .as_deref()
            .is_some_and(|error| error.contains("resend the current message once")),
        "{boundary_error:?}"
    );
    let requests = std::fs::read_to_string(&log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 1);
    assert!(
        !requests.contains("\"method\":\"session/prompt\""),
        "unsafe stale turn must not be consumed: {requests}"
    );
    assert!(!requests.contains("current prompt"), "{requests}");

    provider.set_model("gpt-5-mini").unwrap();
    let retry = complete_in_dir(&provider, &working_dir, &messages, Some(&replacement)).await;
    assert!(
        retry
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );
    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("current prompt").count(), 1);
    assert!(
        requests.contains("\"modelId\":\"gpt-5-mini\""),
        "{requests}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn multiprovider_forks_isolate_two_agents_and_resumed_acp_model_state() {
    let _guard = jcode_base::storage::lock_test_env();
    let saved = [
        (
            "JCODE_COPILOT_TRANSPORT",
            std::env::var("JCODE_COPILOT_TRANSPORT").ok(),
        ),
        (
            "JCODE_COPILOT_CLI_PATH",
            std::env::var("JCODE_COPILOT_CLI_PATH").ok(),
        ),
        ("JCODE_HOME", std::env::var("JCODE_HOME").ok()),
        (
            "JCODE_ACTIVE_PROVIDER",
            std::env::var("JCODE_ACTIVE_PROVIDER").ok(),
        ),
    ];
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("agent-workdir");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("multiprovider-forks.jsonl");
    let process = fake_process(&log);
    external::register_external_provider(external::COPILOT_RUNTIME, move || {
        let provider = CopilotApiProvider::with_official_process(process.clone());
        provider.complete_init_without_tier_detection();
        Arc::new(provider) as Arc<dyn Provider>
    });
    unsafe {
        std::env::set_var("JCODE_COPILOT_TRANSPORT", "official-cli");
        std::env::set_var(
            "JCODE_COPILOT_CLI_PATH",
            env!("CARGO_BIN_EXE_jcode-fake-copilot-acp"),
        );
        std::env::set_var("JCODE_HOME", temp.path().join("jcode-home"));
        std::env::set_var("JCODE_ACTIVE_PROVIDER", "copilot");
    }

    let template = MultiProvider::new_fast();
    template
        .set_model("copilot:claude-sonnet-4.6")
        .expect("select initial Copilot model");
    let first_provider = template.fork();
    let second_provider = template.fork();
    let first_registry = Registry::new(Arc::clone(&first_provider)).await;
    let second_registry = Registry::new(Arc::clone(&second_provider)).await;
    let mut first_session = Session::create_with_id("agent-a".to_string(), None, None);
    first_session.model = Some("claude-sonnet-4.6".to_string());
    first_session.working_dir = Some(working_dir.display().to_string());
    let mut second_session = Session::create_with_id("agent-b".to_string(), None, None);
    second_session.model = Some("claude-sonnet-4.6".to_string());
    second_session.working_dir = Some(working_dir.display().to_string());
    let mut first = Agent::new_with_session(first_provider, first_registry, first_session, None);
    let mut second =
        Agent::new_with_session(second_provider, second_registry, second_session, None);

    second.run_once_capture("first B turn").await.unwrap();
    first.set_model("copilot:gpt-5-mini").unwrap();
    assert_eq!(first.provider_model(), "gpt-5-mini");
    assert_eq!(second.provider_model(), "claude-sonnet-4.6");
    assert_eq!(
        template.fork().model(),
        "claude-sonnet-4.6",
        "the same fork seam used by headless/restored sessions must remain isolated"
    );
    second.run_once_capture("second B turn").await.unwrap();

    let requests = std::fs::read_to_string(&log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 0);
    assert!(
        !requests.contains("\"method\":\"session/set_model\""),
        "agent B inherited agent A's model switch: {requests}"
    );

    unsafe {
        for (key, value) in saved {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
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
