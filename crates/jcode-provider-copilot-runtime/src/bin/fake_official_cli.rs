use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};

fn append_log(value: &Value) {
    let path = std::env::var("JCODE_FAKE_COPILOT_ACP_LOG").expect("fake log path");
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

fn main() {
    append_log(&json!({
        "process": {
            "args": std::env::args().skip(1).collect::<Vec<_>>(),
            "sentinel": std::env::var("JCODE_FAKE_COPILOT_PARENT_SENTINEL").ok(),
            "allow_all": std::env::var("COPILOT_ALLOW_ALL").ok(),
        }
    }));

    let stdin = std::io::stdin();
    let mut lines = BufReader::new(stdin.lock()).lines();
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
            "session/new" => response(
                id,
                json!({
                    "sessionId":"fake-copilot-session",
                    "models": {
                        "currentModelId":"claude-sonnet-4.6",
                        "availableModels":[
                            {"modelId":"claude-sonnet-4.6", "name":"Claude Sonnet 4.6"},
                            {"modelId":"gpt-5-mini", "name":"GPT-5 mini"}
                        ]
                    }
                }),
            ),
            "session/load" => {
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
            "session/set_model" => response(id, json!({})),
            "session/prompt" => {
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_HANG").is_some() {
                    continue;
                }
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_FAIL").is_some() {
                    eprintln!("official Copilot CLI request failed");
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32000, "message":"fake official-cli failure"}
                    }));
                    continue;
                }
                if std::env::var_os("JCODE_FAKE_COPILOT_ACP_PERMISSION").is_some() {
                    send(json!({
                        "jsonrpc":"2.0",
                        "id":900,
                        "method":"session/request_permission",
                        "params":{
                            "sessionId":"fake-copilot-session",
                            "toolCall":{"toolCallId":"tool-1", "title":"Run a command"},
                            "options":[
                                {"optionId":"always", "name":"Always allow", "kind":"allow_always"},
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
                            "content":{"type":"text", "text":"OK"}
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
            }
            "session/cancel" => break,
            other => panic!("unexpected ACP method: {other}"),
        }
    }
}
