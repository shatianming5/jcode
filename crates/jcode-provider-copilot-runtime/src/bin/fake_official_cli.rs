use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};

fn append_log(value: &Value) {
    let path = if let Ok(directory) = std::env::var("JCODE_FAKE_COPILOT_ACP_LOG_DIR") {
        let name = std::env::current_dir()
            .expect("fake cwd")
            .file_name()
            .expect("fake cwd name")
            .to_string_lossy()
            .to_string();
        std::path::PathBuf::from(directory).join(format!("{name}.jsonl"))
    } else {
        std::env::var("JCODE_FAKE_COPILOT_ACP_LOG")
            .expect("fake log path")
            .into()
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open fake log");
    writeln!(file, "{value}").expect("write fake log");
}

fn send(value: Value) {
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    writeln!(stdout, "{value}").expect("write response");
    stdout.flush().expect("flush response");
}

fn response(id: Value, result: Value) {
    send(json!({"jsonrpc":"2.0", "id":id, "result":result}));
}

fn maybe_sleep_ms(name: &str) {
    let Some(milliseconds) = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return;
    };
    std::thread::sleep(std::time::Duration::from_millis(milliseconds));
}

fn main() {
    append_log(&json!({
        "process": {
            "pid": std::process::id(),
            "args": std::env::args().skip(1).collect::<Vec<_>>(),
            "cwd": std::env::current_dir().expect("fake cwd"),
            "sentinel": std::env::var("JCODE_FAKE_COPILOT_PARENT_SENTINEL").ok(),
            "allow_all": std::env::var("COPILOT_ALLOW_ALL").ok(),
        }
    }));

    let stdin = std::io::stdin();
    let mut lines = BufReader::new(stdin.lock()).lines();
    let mut stale_load_seen = false;
    while let Some(line) = lines.next() {
        let line = line.expect("read request");
        let value: Value = serde_json::from_str(&line).expect("valid JSON-RPC request");
        append_log(&value);
        let Some(method) = value.get("method").and_then(Value::as_str) else {
            continue;
        };
        let id = value.get("id").cloned().unwrap_or(Value::Null);
        match method {
            "initialize" => response(
                id,
                json!({
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": true},
                    "agentInfo": {"name":"Copilot", "version":"1.0.83-0"},
                    "authMethods": [{
                        "id":"copilot-login",
                        "name":"Log in with Copilot CLI"
                    }]
                }),
            ),
            "session/new" => {
                maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_NEW_MS");
                if stale_load_seen {
                    maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_STALE_NEW_MS");
                }
                if stale_load_seen
                    && std::env::var_os("JCODE_FAKE_COPILOT_ACP_FAIL_STALE_NEW").is_some()
                {
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32000, "message":"fake stale replacement failure"}
                    }));
                    continue;
                }
                response(
                    id,
                    json!({
                        "sessionId": if stale_load_seen {
                            "recovered-copilot-session"
                        } else {
                            "fake-copilot-session"
                        },
                        "models": {
                            "currentModelId":"claude-sonnet-4.6",
                            "availableModels":[
                                {"modelId":"claude-sonnet-4.6", "name":"Claude Sonnet 4.6"},
                                {"modelId":"gpt-5-mini", "name":"GPT-5 mini"}
                            ]
                        }
                    }),
                );
            }
            "session/load" => {
                maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_LOAD_MS");
                let requested_session_id =
                    value["params"]["sessionId"].as_str().unwrap_or_default();
                let configured_stale_id =
                    std::env::var("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND_ID").ok();
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_LOAD_NOT_FOUND").is_some()
                    || configured_stale_id.as_deref() == Some(requested_session_id)
                {
                    stale_load_seen = true;
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{
                            "code":-32002,
                            "message":format!(
                                "Resource not found: Session {requested_session_id} not found"
                            )
                        }
                    }));
                    continue;
                }
                send(json!({
                    "jsonrpc":"2.0",
                    "method":"session/update",
                    "params":{
                        "sessionId":"fake-copilot-session",
                        "update":{
                            "sessionUpdate":"agent_message_chunk",
                            "content":{"type":"text", "text":"REPLAYED_HISTORY"}
                        }
                    }
                }));
                response(
                    id,
                    json!({
                        "models": {
                            "currentModelId":"claude-sonnet-4.6",
                            "availableModels":[
                                {"modelId":"claude-sonnet-4.6", "name":"Claude Sonnet 4.6"},
                                {"modelId":"gpt-5-mini", "name":"GPT-5 mini"}
                            ]
                        }
                    }),
                );
            }
            "session/set_model" => {
                maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_SET_MODEL_MS");
                response(id, json!({}));
            }
            "session/prompt" => {
                let prompt_json = value["params"]["prompt"].to_string();
                maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_PROMPT_MS");
                let hang_match = std::env::var("JCODE_FAKE_COPILOT_ACP_HANG_PROMPT_MATCH").ok();
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_HANG").is_some()
                    || hang_match
                        .as_deref()
                        .is_some_and(|needle| prompt_json.contains(needle))
                {
                    continue;
                }
                if std::env::var("JCODE_FAKE_COPILOT_ACP_CRASH_PROMPT_MATCH")
                    .ok()
                    .as_deref()
                    .is_some_and(|needle| prompt_json.contains(needle))
                {
                    return;
                }
                let fail_match = std::env::var("JCODE_FAKE_COPILOT_ACP_FAIL_PROMPT_MATCH").ok();
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_FAIL").is_some()
                    || fail_match
                        .as_deref()
                        .is_some_and(|needle| prompt_json.contains(needle))
                {
                    eprintln!("official Copilot CLI request failed");
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32000, "message":"fake official-cli failure"}
                    }));
                    continue;
                }
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_PERMISSION").is_some() {
                    let kind = std::env::var("JCODE_FAKE_COPILOT_ACP_PERMISSION_KIND")
                        .unwrap_or_else(|_| "read".to_string());
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":900,
                        "method":"session/request_permission",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "toolCall":{
                                "toolCallId":"tool-1",
                                "title":"Permission fixture",
                                "kind":kind
                            },
                            "options":[
                                {"optionId":"always", "name":"Always allow", "kind":"allow_always"},
                                {"optionId":"reject", "name":"Reject once", "kind":"reject_once"},
                                {"optionId":"once", "name":"Allow once", "kind":"allow_once"}
                            ]
                        }
                    }));
                    let permission = lines
                        .next()
                        .expect("permission response")
                        .expect("read permission response");
                    append_log(
                        &serde_json::from_str(&permission).expect("valid permission response"),
                    );
                }
                if let Ok(notice) = std::env::var("JCODE_FAKE_COPILOT_ACP_CONFIG_NOTICE") {
                    send(json!({
                        "jsonrpc":"2.0",
                        "method":"session/update",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "update":{
                                "sessionUpdate":"agent_message_chunk",
                                "content":{"type":"text", "text":notice}
                            }
                        }
                    }));
                }
                send(json!({
                    "jsonrpc":"2.0",
                    "method":"session/update",
                    "params":{
                        "sessionId":"fake-copilot-session",
                        "update":{
                            "sessionUpdate":"agent_thought_chunk",
                            "content":{"type":"text", "text":"thinking"}
                        }
                    }
                }));
                let reply = std::env::var("JCODE_FAKE_COPILOT_ACP_REPLY")
                    .unwrap_or_else(|_| "OK".to_string());
                send(json!({
                    "jsonrpc":"2.0",
                    "method":"session/update",
                    "params":{
                        "sessionId":"fake-copilot-session",
                        "update":{
                            "sessionUpdate":"tool_call",
                            "toolCallId":"tool-1",
                            "title":"Viewing Cargo.toml",
                            "kind":"read",
                            "status":"pending"
                        }
                    }
                }));
                send(json!({
                    "jsonrpc":"2.0",
                    "method":"session/update",
                    "params":{
                        "sessionId":"fake-copilot-session",
                        "update":{
                            "sessionUpdate":"tool_call_update",
                            "toolCallId":"tool-1",
                            "status":"completed"
                        }
                    }
                }));
                send(json!({
                    "jsonrpc":"2.0",
                    "method":"session/update",
                    "params":{
                        "sessionId":"fake-copilot-session",
                        "update":{
                            "sessionUpdate":"agent_message_chunk",
                            "content":{"type":"text", "text":reply}
                        }
                    }
                }));
                response(
                    id,
                    json!({
                        "stopReason":"end_turn",
                        "usage":{
                            "totalTokens":12,
                            "inputTokens":10,
                            "outputTokens":2,
                            "cachedReadTokens":3,
                            "cachedWriteTokens":4
                        }
                    }),
                );
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_EXIT_AFTER_PROMPT").is_some() {
                    return;
                }
            }
            "session/cancel" => {
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_LATE_PERMISSION_AFTER_CANCEL").is_some()
                {
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":901,
                        "method":"session/request_permission",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "toolCall":{
                                "toolCallId":"late-permission",
                                "title":"Late cancelled permission",
                                "kind":"read"
                            },
                            "options":[
                                {"optionId":"reject", "name":"Reject once", "kind":"reject_once"},
                                {"optionId":"once", "name":"Allow once", "kind":"allow_once"}
                            ]
                        }
                    }));
                    loop {
                        let permission = lines
                            .next()
                            .expect("late permission response")
                            .expect("read late permission response");
                        let permission: Value = serde_json::from_str(&permission)
                            .expect("valid late permission response");
                        append_log(&permission);
                        if permission["id"] == 901 && permission.get("method").is_none() {
                            break;
                        }
                    }
                }
                maybe_sleep_ms("JCODE_FAKE_COPILOT_ACP_DELAY_CANCEL_MS");
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_IGNORE_CANCEL").is_some() {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    continue;
                }
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_LATE_AFTER_CANCEL").is_some() {
                    send(json!({
                        "jsonrpc":"2.0",
                        "method":"session/update",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "update":{
                                "sessionUpdate":"agent_message_chunk",
                                "content":{"type":"text", "text":"LATE_CANCELLED_TEXT"}
                            }
                        }
                    }));
                    send(json!({
                        "jsonrpc":"2.0",
                        "method":"session/update",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "update":{
                                "sessionUpdate":"tool_call",
                                "toolCallId":"late-tool",
                                "title":"LATE_CANCELLED_TOOL",
                                "kind":"execute",
                                "status":"pending"
                            }
                        }
                    }));
                }
                append_log(&json!({"process_exit":"cancelled"}));
                break;
            }
            other => panic!("unexpected ACP method: {other}"),
        }
    }
}
