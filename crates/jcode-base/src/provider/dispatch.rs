use super::*;

#[derive(Clone, Copy)]
pub(super) enum CompletionMode<'a> {
    Unified {
        system: &'a str,
    },
    Split {
        system_static: &'a str,
        system_dynamic: &'a str,
    },
}

impl CompletionMode<'_> {
    pub(super) fn log_suffix(self) -> &'static str {
        match self {
            CompletionMode::Unified { .. } => "",
            CompletionMode::Split { .. } => " (split)",
        }
    }

    pub(super) fn switch_log_prefix(self) -> &'static str {
        match self {
            CompletionMode::Unified { .. } => "Auto-fallback",
            CompletionMode::Split { .. } => "Auto-fallback (split)",
        }
    }
}

async fn complete_split_for_request(
    provider: &dyn Provider,
    messages: &[Message],
    tools: &[ToolDefinition],
    system_static: &str,
    system_dynamic: &str,
    resume_session_id: Option<&str>,
    request_context: Option<&ProviderRequestContext>,
) -> Result<EventStream> {
    if let Some(request_context) = request_context {
        provider
            .complete_split_with_context(
                messages,
                tools,
                system_static,
                system_dynamic,
                resume_session_id,
                request_context,
            )
            .await
    } else {
        provider
            .complete_split(
                messages,
                tools,
                system_static,
                system_dynamic,
                resume_session_id,
            )
            .await
    }
}

impl MultiProvider {
    pub(super) fn estimate_request_input(
        messages: &[Message],
        tools: &[ToolDefinition],
        mode: CompletionMode<'_>,
    ) -> (usize, usize) {
        let mut chars = serde_json::to_string(messages)
            .map(|value| value.len())
            .unwrap_or(0)
            + serde_json::to_string(tools)
                .map(|value| value.len())
                .unwrap_or(0);
        match mode {
            CompletionMode::Unified { system } => {
                chars += system.len();
            }
            CompletionMode::Split {
                system_static,
                system_dynamic,
            } => {
                chars += system_static.len() + system_dynamic.len();
            }
        }
        let tokens = chars / 4;
        (chars, tokens)
    }

    pub(super) async fn complete_on_provider(
        &self,
        provider: ActiveProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
        resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        self.reconcile_auth_if_provider_missing(provider);
        match provider {
            ActiveProvider::Claude => {
                if let Some(anthropic) = self.anthropic_provider() {
                    anthropic
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else if let Some(claude) = self.claude_provider() {
                    claude
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "Claude credentials not available. Run `claude` to log in."
                    ))
                }
            }
            ActiveProvider::OpenAI => {
                if let Some(openai) = self.openai_provider() {
                    openai
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "OpenAI credentials not available. Run `jcode login --provider openai` to log in."
                    ))
                }
            }
            ActiveProvider::Copilot => {
                let copilot = self
                    .copilot_api
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(copilot) = copilot {
                    copilot
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(copilot::unavailable_message()))
                }
            }
            ActiveProvider::Antigravity => {
                let antigravity = self.antigravity_provider();
                if let Some(antigravity) = antigravity {
                    antigravity
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "Antigravity is not available. Run `jcode login --provider antigravity`."
                    ))
                }
            }
            ActiveProvider::Gemini => {
                let gemini = self
                    .gemini
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(gemini) = gemini {
                    gemini
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "Gemini is not available. Run `jcode login --provider gemini`."
                    ))
                }
            }
            ActiveProvider::Cursor => {
                let cursor = self
                    .cursor
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(cursor) = cursor {
                    cursor
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "Cursor is not available. Run `jcode login --provider cursor`."
                    ))
                }
            }
            ActiveProvider::Bedrock => {
                if let Some(bedrock) = self.bedrock_provider() {
                    bedrock
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "AWS Bedrock is not available. Configure AWS credentials and region, or set AWS_PROFILE/AWS_REGION."
                    ))
                }
            }
            ActiveProvider::OpenRouter => {
                let openrouter = self.active_openrouter_execution_provider();
                if let Some(openrouter) = openrouter {
                    openrouter
                        .complete(messages, tools, system, resume_session_id)
                        .await
                } else {
                    Err(anyhow::anyhow!(
                        "OpenRouter credentials not available. Set OPENROUTER_API_KEY environment variable."
                    ))
                }
            }
        }
    }

    pub(super) async fn complete_split_on_provider(
        &self,
        provider: ActiveProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_static: &str,
        system_dynamic: &str,
        resume_session_id: Option<&str>,
        request_context: Option<&ProviderRequestContext>,
    ) -> Result<EventStream> {
        self.reconcile_auth_if_provider_missing(provider);
        match provider {
            ActiveProvider::Claude => {
                if let Some(anthropic) = self.anthropic_provider() {
                    complete_split_for_request(
                        anthropic.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else if let Some(claude) = self.claude_provider() {
                    complete_split_for_request(
                        claude.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "Claude credentials not available. Run `claude` to log in."
                    ))
                }
            }
            ActiveProvider::OpenAI => {
                if let Some(openai) = self.openai_provider() {
                    complete_split_for_request(
                        openai.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "OpenAI credentials not available. Run `jcode login --provider openai` to log in."
                    ))
                }
            }
            ActiveProvider::Copilot => {
                let copilot = self
                    .copilot_api
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(copilot) = copilot {
                    complete_split_for_request(
                        copilot.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(copilot::unavailable_message()))
                }
            }
            ActiveProvider::Antigravity => {
                let antigravity = self.antigravity_provider();
                if let Some(antigravity) = antigravity {
                    complete_split_for_request(
                        antigravity.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "Antigravity is not available. Run `jcode login --provider antigravity`."
                    ))
                }
            }
            ActiveProvider::Gemini => {
                let gemini = self
                    .gemini
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(gemini) = gemini {
                    complete_split_for_request(
                        gemini.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "Gemini is not available. Run `jcode login --provider gemini`."
                    ))
                }
            }
            ActiveProvider::Cursor => {
                let cursor = self
                    .cursor
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Some(cursor) = cursor {
                    complete_split_for_request(
                        cursor.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "Cursor is not available. Run `jcode login --provider cursor`."
                    ))
                }
            }
            ActiveProvider::Bedrock => {
                if let Some(bedrock) = self.bedrock_provider() {
                    complete_split_for_request(
                        bedrock.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "AWS Bedrock is not available. Configure AWS credentials and region, or set AWS_PROFILE/AWS_REGION."
                    ))
                }
            }
            ActiveProvider::OpenRouter => {
                let openrouter = self.active_openrouter_execution_provider();
                if let Some(openrouter) = openrouter {
                    complete_split_for_request(
                        openrouter.as_ref(),
                        messages,
                        tools,
                        system_static,
                        system_dynamic,
                        resume_session_id,
                        request_context,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!(
                        "OpenRouter credentials not available. Set OPENROUTER_API_KEY environment variable."
                    ))
                }
            }
        }
    }
}
