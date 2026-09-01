use futures::StreamExt;
use jcode_app_core::{agent::Agent, session::Session, tool::Registry};
use jcode_base::provider::{MultiProvider, external};
use jcode_message_types::{ContentBlock, Message, Role, StreamEvent, ToolDefinition};
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
    provider: &dyn Provider,
    working_dir: &Path,
    messages: &[Message],
    resume_session_id: Option<&str>,
) -> Vec<StreamEvent> {
    complete_in_dir_with_system(provider, working_dir, messages, "", resume_session_id).await
}

async fn complete_in_dir_with_system(
    provider: &dyn Provider,
    working_dir: &Path,
    messages: &[Message],
    system: &str,
    resume_session_id: Option<&str>,
) -> Vec<StreamEvent> {
    let request_context = ProviderRequestContext::new(Some(working_dir.to_path_buf()));
    let mut stream = provider
        .complete_split_with_context(
            messages,
            &[one_tool()],
            system,
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
    let first_provider = provider.fork();
    let second_provider = provider.fork();
    let first_messages = [Message::user("first")];
    let second_messages = [Message::user("second")];

    let (first, second) = tokio::join!(
        complete_in_dir(first_provider.as_ref(), &first_dir, &first_messages, None),
        complete_in_dir(
            second_provider.as_ref(),
            &second_dir,
            &second_messages,
            None
        )
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
    assert_eq!(stale_session_id, "fake-copilot-session");
    drop(provider);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    provider.set_model("gpt-5-mini").unwrap();

    let second = complete_in_dir_with_system(
        &provider,
        &working_dir,
        &[
            Message::user("old user history"),
            Message::assistant_text("old assistant history"),
            Message::user("current prompt"),
        ],
        "outer-system",
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
    assert_eq!(recovered_session_id, "recovered-copilot-session");
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );

    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 2);
    assert!(
        requests.contains("\"method\":\"session/set_model\"")
            && requests.contains("\"modelId\":\"gpt-5-mini\""),
        "{requests}"
    );
    assert_eq!(requests.matches("current prompt").count(), 1);
    assert!(!requests.contains("old user history"), "{requests}");
    assert!(!requests.contains("old assistant history"), "{requests}");
    assert!(!requests.contains("\"type\":\"resource\""), "{requests}");
    let recovered_prompt = requests
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| {
            value.get("method").and_then(serde_json::Value::as_str) == Some("session/prompt")
                && value.to_string().contains("current prompt")
        })
        .expect("recovered prompt request");
    let prompt = recovered_prompt["params"]["prompt"].as_array().unwrap();
    assert_eq!(prompt.len(), 1, "{recovered_prompt}");
    assert_eq!(prompt[0]["type"], "text");
    assert!(prompt[0]["text"].as_str().unwrap().contains("outer-system"));
    assert!(
        prompt[0]["text"]
            .as_str()
            .unwrap()
            .contains("current prompt")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stale_replacement_session_creation_failure_is_reported_to_the_turn() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("stale-new-failure.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND".into(), "1".into());
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_FAIL_STALE_NEW".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let request_context = ProviderRequestContext::new(Some(working_dir))
        .with_current_user_prompt(Some("current prompt".to_string()));
    let mut stream = provider
        .complete_split_with_context(
            &[Message::user("current prompt")],
            &[one_tool()],
            "current system",
            "",
            Some("stale-session"),
            &request_context,
        )
        .await
        .unwrap();

    let mut error = None;
    while let Some(event) = stream.next().await {
        if let Err(stream_error) = event {
            error = Some(stream_error.to_string());
        }
    }
    assert!(
        error
            .as_deref()
            .is_some_and(|message| message.contains("session/new")
                && message.contains("fake stale replacement failure")),
        "{error:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn child_death_between_turns_starts_fresh_without_failing_the_next_turn() {
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

    let second = complete_in_dir_with_system(
        &provider,
        &working_dir,
        &[
            Message::user("old turn"),
            Message::assistant_text("OK"),
            Message::user("current turn after child death"),
        ],
        "CURRENT_SYSTEM",
        Some(&persisted),
    )
    .await;
    assert!(
        second
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );
    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("\"process\"").count(), 2, "{requests}");
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 2);
    assert_eq!(
        requests.matches("current turn after child death").count(),
        1
    );
    assert!(!requests.contains("old turn"), "{requests}");
}

#[tokio::test(flavor = "current_thread")]
async fn legacy_and_unmarked_stale_history_continue_current_prompt_without_replay() {
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
            "CURRENT_SYSTEM",
            "",
            Some("jcode-copilot-acp-v1:unsafe:fake-copilot-session"),
            &request_context,
        )
        .await
        .unwrap();
    let mut replacement = None;
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::SessionId(id) => replacement = Some(id),
            StreamEvent::TextDelta(delta) => text.push_str(&delta),
            _ => {}
        }
    }
    assert_eq!(replacement.as_deref(), Some("recovered-copilot-session"));
    assert_eq!(text, "OK");
    let requests = std::fs::read_to_string(&log).unwrap();
    assert_eq!(requests.matches("\"method\":\"session/load\"").count(), 1);
    assert_eq!(requests.matches("\"method\":\"session/new\"").count(), 1);
    assert_eq!(requests.matches("current prompt").count(), 1);
    assert!(!requests.contains("old instruction"), "{requests}");
    assert!(!requests.contains("old answer"), "{requests}");
    assert!(!requests.contains("\"type\":\"resource\""), "{requests}");
}

#[tokio::test(flavor = "current_thread")]
async fn stale_recovery_never_replays_side_effect_or_no_assistant_history() {
    let temp = tempfile::tempdir().unwrap();
    for (name, messages) in [
        (
            "side-effect",
            vec![
                Message::user("OLD_SIDE_EFFECT_REQUEST"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "old-tool".to_string(),
                        name: "bash".to_string(),
                        input: json!({"command":"touch must-not-run"}),
                        thought_signature: None,
                    }],
                    timestamp: None,
                    tool_duration_ms: None,
                },
                Message::user("CURRENT_SIDE_EFFECT_CASE"),
            ],
        ),
        (
            "no-assistant",
            vec![
                Message::user("OLD_FAILED_USER_PROMPT"),
                Message::user("CURRENT_NO_ASSISTANT_CASE"),
            ],
        ),
    ] {
        let working_dir = temp.path().join(name);
        std::fs::create_dir_all(&working_dir).unwrap();
        let log = temp.path().join(format!("{name}.jsonl"));
        let mut process = fake_process(&log);
        process.env.insert(
            "JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND_ID".into(),
            "fake-copilot-session".into(),
        );
        let provider = CopilotApiProvider::with_official_process(process);
        provider.complete_init_without_tier_detection();
        let current = messages
            .last()
            .and_then(|message| match &message.content[0] {
                ContentBlock::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap();
        let request_context = ProviderRequestContext::new(Some(working_dir))
            .with_current_user_prompt(Some(current.clone()));
        let mut stream = provider
            .complete_split_with_context(
                &messages,
                &[one_tool()],
                "CURRENT_SYSTEM",
                "",
                Some("fake-copilot-session"),
                &request_context,
            )
            .await
            .unwrap();
        while let Some(event) = stream.next().await {
            event.unwrap();
        }

        let requests = std::fs::read_to_string(log).unwrap();
        assert_eq!(requests.matches(&current).count(), 1, "{requests}");
        assert!(requests.contains("CURRENT_SYSTEM"), "{requests}");
        assert!(!requests.contains("OLD_SIDE_EFFECT_REQUEST"), "{requests}");
        assert!(!requests.contains("touch must-not-run"), "{requests}");
        assert!(!requests.contains("OLD_FAILED_USER_PROMPT"), "{requests}");
        assert!(!requests.contains("\"type\":\"resource\""), "{requests}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn headless_agent_persists_stale_replacement_across_restarts() {
    let _guard = jcode_base::storage::lock_test_env();
    let saved_home = std::env::var_os("JCODE_HOME");
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("jcode-home");
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    unsafe {
        std::env::set_var("JCODE_HOME", &home);
    }
    let log = temp.path().join("agent-stale-persistence.jsonl");
    let mut process = fake_process(&log);
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND_ID".into(),
        "fake-copilot-session".into(),
    );
    let mut session = Session::create_with_id("agent-stale-persistence".to_string(), None, None);
    session.working_dir = Some(working_dir.display().to_string());
    session.save().unwrap();

    let provider = Arc::new(CopilotApiProvider::with_official_process(process.clone()));
    provider.complete_init_without_tier_detection();
    let provider_dyn: Arc<dyn Provider> = provider;
    let registry = Registry::new(Arc::clone(&provider_dyn)).await;
    let mut agent = Agent::new_with_session(provider_dyn, registry, session, None);
    agent.set_system_prompt("OLD_AGENT_SYSTEM");
    assert_eq!(
        agent.run_once_capture("OLD_AGENT_PROMPT").await.unwrap(),
        "OK"
    );
    let session_id = agent.session_id().to_string();
    let initially_persisted = Session::load(&session_id).unwrap();
    assert_eq!(
        initially_persisted.provider_session_id.as_deref(),
        Some("fake-copilot-session")
    );
    drop(agent);

    let mut failing_process = process.clone();
    failing_process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_FAIL".into(), "1".into());
    let provider = Arc::new(CopilotApiProvider::with_official_process(failing_process));
    provider.complete_init_without_tier_detection();
    let provider_dyn: Arc<dyn Provider> = provider;
    let registry = Registry::new(Arc::clone(&provider_dyn)).await;
    let mut agent = Agent::new_with_session(provider_dyn, registry, initially_persisted, None);
    agent.set_system_prompt("CURRENT_AGENT_SYSTEM");

    let error = agent
        .run_once_capture("CURRENT_AGENT_PROMPT")
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("fake official-cli failure"),
        "{error:#}"
    );
    let persisted = Session::load(&session_id).unwrap();
    assert_eq!(
        persisted.provider_session_id.as_deref(),
        Some("recovered-copilot-session")
    );
    assert_eq!(
        persisted
            .messages
            .iter()
            .filter(|message| message.content_preview() == "CURRENT_AGENT_PROMPT")
            .count(),
        1
    );
    drop(agent);

    let restored_provider = Arc::new(CopilotApiProvider::with_official_process(process));
    restored_provider.complete_init_without_tier_detection();
    let restored_provider_dyn: Arc<dyn Provider> = restored_provider;
    let restored_registry = Registry::new(Arc::clone(&restored_provider_dyn)).await;
    let mut restored =
        Agent::new_with_session(restored_provider_dyn, restored_registry, persisted, None);
    restored.set_system_prompt("CURRENT_AGENT_SYSTEM");
    assert_eq!(
        restored
            .run_once_capture("POST_RESTORE_PROMPT")
            .await
            .unwrap(),
        "OK"
    );

    let final_persisted = Session::load(&session_id).unwrap();
    assert_eq!(
        final_persisted.provider_session_id.as_deref(),
        Some("recovered-copilot-session")
    );

    let requests = std::fs::read_to_string(log).unwrap();
    let load_ids = requests
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["method"] == "session/load")
        .filter_map(|value| value["params"]["sessionId"].as_str().map(str::to_string))
        .collect::<Vec<_>>();
    assert_eq!(
        load_ids,
        ["fake-copilot-session", "recovered-copilot-session"]
    );
    let prompts = requests
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["method"] == "session/prompt")
        .map(|value| {
            value["params"]["prompt"][0]["text"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(prompts.len(), 3, "{requests}");
    assert_eq!(prompts[1].matches("CURRENT_AGENT_SYSTEM").count(), 1);
    assert_eq!(prompts[1].matches("CURRENT_AGENT_PROMPT").count(), 1);
    assert!(!prompts[1].contains("OLD_AGENT_SYSTEM"), "{prompts:?}");
    assert!(!prompts[1].contains("OLD_AGENT_PROMPT"), "{prompts:?}");
    assert_eq!(prompts[2].matches("POST_RESTORE_PROMPT").count(), 1);
    assert!(!prompts[2].contains("OLD_AGENT_PROMPT"), "{prompts:?}");
    assert!(!requests.contains("\"type\":\"resource\""), "{requests}");

    unsafe {
        match saved_home {
            Some(value) => std::env::set_var("JCODE_HOME", value),
            None => std::env::remove_var("JCODE_HOME"),
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn mid_turn_child_crash_errors_once_and_next_user_turn_starts_fresh() {
    let temp = tempfile::tempdir().unwrap();
    let working_dir = temp.path().join("session");
    std::fs::create_dir_all(&working_dir).unwrap();
    let log = temp.path().join("mid-turn-crash.jsonl");
    let mut process = fake_process(&log);
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND_ID".into(),
        "fake-copilot-session".into(),
    );
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_CRASH_PROMPT_MATCH".into(),
        "CRASH_CURRENT_PROMPT".into(),
    );
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();

    let first = complete_in_dir(
        &provider,
        &working_dir,
        &[Message::user("FIRST_COMPLETED_PROMPT")],
        None,
    )
    .await;
    let persisted = first
        .iter()
        .find_map(|event| match event {
            StreamEvent::SessionId(id) => Some(id.clone()),
            _ => None,
        })
        .unwrap();

    let request_context = ProviderRequestContext::new(Some(working_dir.clone()))
        .with_current_user_prompt(Some("CRASH_CURRENT_PROMPT".to_string()));
    let mut crashed = provider
        .complete_split_with_context(
            &[
                Message::user("FIRST_COMPLETED_PROMPT"),
                Message::assistant_text("OK"),
                Message::user("CRASH_CURRENT_PROMPT"),
            ],
            &[one_tool()],
            "CURRENT_SYSTEM",
            "",
            Some(&persisted),
            &request_context,
        )
        .await
        .unwrap();
    let mut errors = Vec::new();
    while let Some(event) = crashed.next().await {
        if let Err(error) = event {
            errors.push(error.to_string());
        }
    }
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("official Copilot CLI request failed"),
        "{errors:?}"
    );

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let recovered = complete_in_dir_with_system(
        &provider,
        &working_dir,
        &[
            Message::user("FIRST_COMPLETED_PROMPT"),
            Message::assistant_text("OK"),
            Message::user("CRASH_CURRENT_PROMPT"),
            Message::user("AFTER_CRASH_NEW_USER_TURN"),
        ],
        "CURRENT_SYSTEM",
        Some(&persisted),
    )
    .await;
    assert!(
        recovered
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );

    let requests = std::fs::read_to_string(log).unwrap();
    assert_eq!(requests.matches("CRASH_CURRENT_PROMPT").count(), 1);
    assert_eq!(requests.matches("AFTER_CRASH_NEW_USER_TURN").count(), 1);
    let recovered_prompt = requests
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| {
            value["method"] == "session/prompt"
                && value.to_string().contains("AFTER_CRASH_NEW_USER_TURN")
        })
        .unwrap();
    assert!(
        !recovered_prompt
            .to_string()
            .contains("CRASH_CURRENT_PROMPT")
    );
    assert!(
        !recovered_prompt
            .to_string()
            .contains("FIRST_COMPLETED_PROMPT")
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
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_HANG_PROMPT_MATCH".into(),
        "wait".into(),
    );
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_LATE_AFTER_CANCEL".into(),
        "1".into(),
    );
    process.env.insert(
        "JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND_ID".into(),
        "fake-copilot-session".into(),
    );
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
    let session_id = match session {
        StreamEvent::SessionId(id) => id,
        event => panic!("expected session id, got {event:?}"),
    };
    drop(stream);

    let mut cancelled_and_exited = false;
    for _ in 0..40 {
        let requests = std::fs::read_to_string(&log).unwrap_or_default();
        if requests.contains("\"method\":\"session/cancel\"")
            && requests.contains("\"process_exit\":\"cancelled\"")
        {
            cancelled_and_exited = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(
        cancelled_and_exited,
        "stream drop did not cancel and terminate the exact ACP child"
    );

    let next = complete_in_dir(
        &provider,
        temp.path(),
        &[Message::user("next turn")],
        Some(&session_id),
    )
    .await;
    assert!(
        next.iter()
            .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "OK"))
    );
    assert!(
        !next.iter().any(
            |event| matches!(event, StreamEvent::TextDelta(text) if text == "LATE_CANCELLED_TEXT")
        ),
        "{next:?}"
    );
    assert!(
        !next.iter().any(
            |event| matches!(event, StreamEvent::StatusDetail { detail } if detail == "LATE_CANCELLED_TOOL")
        ),
        "{next:?}"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn dropping_provider_terminates_a_hung_official_cli_child() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("provider-drop.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_HANG".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let mut stream = provider
        .complete(
            &[Message::user("hang until provider drop")],
            &[one_tool()],
            "",
            None,
        )
        .await
        .unwrap();
    for _ in 0..2 {
        tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await
            .expect("session setup timed out")
            .expect("stream closed before session setup")
            .unwrap();
    }
    let child_pid = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|value| value["process"]["pid"].as_u64())
        .expect("fake child pid");

    drop(provider);
    let mut exited = false;
    for _ in 0..40 {
        if !std::process::Command::new("kill")
            .args(["-0", &child_pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
        {
            exited = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(exited, "hung ACP child {child_pid} survived provider drop");
    drop(stream);
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn cancel_terminates_a_child_that_ignores_the_notification() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("ignored-cancel.jsonl");
    let mut process = fake_process(&log);
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_HANG".into(), "1".into());
    process
        .env
        .insert("JCODE_FAKE_COPILOT_ACP_IGNORE_CANCEL".into(), "1".into());
    let provider = CopilotApiProvider::with_official_process(process);
    provider.complete_init_without_tier_detection();
    let mut stream = provider
        .complete(
            &[Message::user("hang until cancel")],
            &[one_tool()],
            "",
            None,
        )
        .await
        .unwrap();
    for _ in 0..2 {
        tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await
            .expect("session setup timed out")
            .expect("stream closed before session setup")
            .unwrap();
    }
    let child_pid = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|value| value["process"]["pid"].as_u64())
        .expect("fake child pid");

    drop(stream);
    let mut exited = false;
    for _ in 0..60 {
        if !std::process::Command::new("kill")
            .args(["-0", &child_pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
        {
            exited = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(
        exited,
        "ACP child {child_pid} survived the bounded cancel deadline"
    );
}
