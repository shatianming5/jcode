//! GitHub Copilot provider runtime with an explicit native API or official CLI
//! ACP transport. It lives outside `jcode-base` so provider edits compile only
//! this crate plus a binary relink instead of rebuilding the base -> app-core
//! -> tui spine. The binary's composition root registers
//! [`CopilotApiProvider`] with `jcode_base::provider::external` at startup.

use acp::Agent as _;
use agent_client_protocol as acp;
use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::Utc;
use futures::Stream;
use jcode_base::auth::copilot as copilot_auth;
use jcode_message_types::{
    ContentBlock, Message as ChatMessage, ProviderToolKind, ProviderToolStatus, Role, StreamEvent,
    ToolDefinition, messages_with_dynamic_system_context,
};
#[cfg(test)]
use jcode_provider_copilot::max_token_parameter_for_model as copilot_max_token_parameter_for_model;
use jcode_provider_copilot::{
    COPILOT_API_VERSION, CopilotTransport, PersistedCatalog,
    add_max_token_parameter as add_copilot_max_token_parameter,
    build_messages as build_copilot_messages, build_tools as build_copilot_tools,
    official_cli_path_from_env,
};
use jcode_provider_copilot::{DEFAULT_MODEL, FALLBACK_MODELS};
pub use jcode_provider_core::PremiumMode;
use jcode_provider_core::{EventStream, Provider, ProviderRequestContext};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use uuid::Uuid;

const ACP_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const ACP_PROMPT_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const ACP_CANCEL_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);
const STDERR_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogSource {
    None,
    Cached,
    Live,
}

#[derive(Clone, Debug)]
pub struct CopilotOfficialCliProcess {
    pub command: PathBuf,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

impl CopilotOfficialCliProcess {
    pub fn from_env() -> Result<Self> {
        let command = official_cli_path_from_env().map_err(anyhow::Error::msg)?;
        Ok(Self::with_command(command))
    }

    pub fn with_command(command: PathBuf) -> Self {
        Self {
            command,
            args: vec![
                "--acp".to_string(),
                "--stdio".to_string(),
                "--no-auto-update".to_string(),
                "--log-level".to_string(),
                "none".to_string(),
                "--no-custom-instructions".to_string(),
                "--disable-builtin-mcps".to_string(),
            ],
            env: BTreeMap::new(),
        }
    }

    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CopilotOfficialToolPolicy {
    available_tools: BTreeSet<&'static str>,
    allow_read: bool,
    allow_search: bool,
    allow_write: bool,
    allow_execute: bool,
    allow_fetch: bool,
}

impl CopilotOfficialToolPolicy {
    fn from_jcode_tools(tools: &[ToolDefinition]) -> Self {
        let mut policy = Self::default();
        for tool in tools {
            match tool.name.as_str() {
                "read" | "view" | "read_file" | "file_read" => {
                    policy.available_tools.insert("view");
                    policy.allow_read = true;
                }
                "agentgrep" | "grep" | "rg" | "file_grep" => {
                    policy.available_tools.insert("grep");
                    policy.available_tools.insert("glob");
                    policy.allow_read = true;
                    policy.allow_search = true;
                }
                "ls" | "glob" => {
                    policy.available_tools.insert("glob");
                    policy.allow_read = true;
                    policy.allow_search = true;
                }
                "bash" | "shell" | "shell_exec" => {
                    policy.available_tools.insert("bash");
                    policy.allow_execute = true;
                }
                "write" | "write_file" | "file_write" => {
                    policy.available_tools.insert("create");
                    policy.allow_write = true;
                }
                "edit" | "multiedit" | "edit_file" | "file_edit" => {
                    policy.available_tools.insert("edit");
                    policy.allow_write = true;
                }
                "patch" | "apply_patch" => {}
                "webfetch" | "web_fetch" => {
                    policy.available_tools.insert("web_fetch");
                    policy.allow_fetch = true;
                }
                "websearch" | "web_search" => {}
                _ => {}
            }
        }
        policy
    }

    fn configure_command(&self, command: &mut Command) {
        if self.available_tools.is_empty() {
            command.arg("--available-tools");
        } else {
            command.arg(format!(
                "--available-tools={}",
                self.available_tools
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }

    fn allows(&self, kind: Option<acp::ToolKind>) -> bool {
        match kind {
            Some(acp::ToolKind::Read) => self.allow_read,
            Some(acp::ToolKind::Search) => self.allow_search,
            Some(acp::ToolKind::Edit | acp::ToolKind::Delete | acp::ToolKind::Move) => {
                self.allow_write
            }
            Some(acp::ToolKind::Execute) => self.allow_execute,
            Some(acp::ToolKind::Fetch) => self.allow_fetch,
            _ => false,
        }
    }
}

#[derive(Clone)]
enum CopilotBackend {
    Native {
        client: reqwest::Client,
        github_token: String,
        bearer_token: Arc<tokio::sync::RwLock<Option<copilot_auth::CopilotApiToken>>>,
        session_id: String,
        machine_id: String,
    },
    OfficialCli {
        process: CopilotOfficialCliProcess,
    },
}

impl CopilotBackend {
    fn transport(&self) -> CopilotTransport {
        match self {
            Self::Native { .. } => CopilotTransport::Native,
            Self::OfficialCli { .. } => CopilotTransport::OfficialCli,
        }
    }
}

/// Copilot provider with native API and official CLI ACP backends.
pub struct CopilotApiProvider {
    backend: CopilotBackend,
    model: Arc<RwLock<String>>,
    fetched_models: Arc<RwLock<Vec<String>>>,
    catalog_source: Arc<RwLock<CatalogSource>>,
    init_ready: Arc<tokio::sync::Notify>,
    init_done: Arc<std::sync::atomic::AtomicBool>,
    premium_mode: Arc<std::sync::atomic::AtomicU8>,
    user_turn_count: Arc<std::sync::atomic::AtomicU64>,
    reasoning_effort: Arc<RwLock<Option<String>>>,
    official_runtime: Arc<Mutex<Option<OfficialRuntimeHandle>>>,
    created_at: std::time::Instant,
}

/// Reasoning efforts supported by Copilot's claude-sonnet-5 route,
/// per live `/models` capabilities (issue #558).
const SONNET5_EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

fn copilot_model_supports_reasoning_effort(model: &str) -> bool {
    model == "claude-sonnet-5"
}

fn copilot_model_uses_responses_api(model: &str) -> bool {
    model.trim().to_ascii_lowercase().starts_with("gpt-5.6")
}

fn copilot_api_path(uses_responses_api: bool) -> &'static str {
    if uses_responses_api {
        "responses"
    } else {
        "chat/completions"
    }
}

impl CopilotApiProvider {
    #[cfg(test)]
    fn max_token_parameter_for_model(model: &str) -> &'static str {
        copilot_max_token_parameter_for_model(model)
    }

    fn add_max_token_parameter(body: &mut Value, model: &str, max_tokens: u32) {
        add_copilot_max_token_parameter(body, model, max_tokens);
    }

    fn current_reasoning_effort(&self) -> Option<String> {
        self.reasoning_effort
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Add top-level `reasoning_effort` when set and the model supports it.
    fn add_reasoning_effort_parameter(&self, body: &mut Value, model: &str) {
        if !copilot_model_supports_reasoning_effort(model) {
            return;
        }
        if let Some(effort) = self.current_reasoning_effort() {
            body["reasoning_effort"] = json!(effort);
        }
    }

    fn persisted_catalog_path() -> Result<std::path::PathBuf> {
        Ok(jcode_base::storage::app_config_dir()?.join("copilot_models_cache.json"))
    }

    fn load_persisted_catalog() -> Option<PersistedCatalog> {
        let path = Self::persisted_catalog_path().ok()?;
        jcode_base::storage::read_json(&path)
            .ok()
            .filter(|catalog: &PersistedCatalog| !catalog.models.is_empty())
    }

    fn persist_catalog(models: &[String]) {
        if models.is_empty() {
            return;
        }
        let Ok(path) = Self::persisted_catalog_path() else {
            return;
        };
        let payload = PersistedCatalog {
            models: models.to_vec(),
            fetched_at_rfc3339: Utc::now().to_rfc3339(),
        };
        if let Err(error) = jcode_base::storage::write_json(&path, &payload) {
            jcode_base::logging::warn(&format!(
                "Failed to persist Copilot model catalog {}: {}",
                path.display(),
                error
            ));
        }
    }

    fn seed_cached_catalog(&self) {
        if let Some(catalog) = Self::load_persisted_catalog() {
            if let Ok(mut models) = self.fetched_models.try_write() {
                *models = catalog.models;
            }
            if let Ok(mut source) = self.catalog_source.try_write() {
                *source = CatalogSource::Cached;
            }
        }
    }

    fn model_catalog_detail_impl(&self) -> String {
        match self
            .catalog_source
            .try_read()
            .map(|g| *g)
            .unwrap_or(CatalogSource::None)
        {
            CatalogSource::Live => String::new(),
            CatalogSource::Cached => "cached live catalog".to_string(),
            CatalogSource::None => "catalog still loading".to_string(),
        }
    }

    pub fn new() -> Result<Self> {
        match CopilotTransport::from_env().map_err(anyhow::Error::msg)? {
            CopilotTransport::Native => {
                let github_token = copilot_auth::load_github_token()?;
                Ok(Self::new_with_token(github_token))
            }
            CopilotTransport::OfficialCli => Ok(Self::with_official_process(
                CopilotOfficialCliProcess::from_env()?,
            )),
        }
    }

    fn common(backend: CopilotBackend) -> Self {
        let model =
            std::env::var("JCODE_COPILOT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let provider = Self {
            backend,
            model: Arc::new(RwLock::new(model)),
            fetched_models: Arc::new(RwLock::new(Vec::new())),
            catalog_source: Arc::new(RwLock::new(CatalogSource::None)),
            init_ready: Arc::new(tokio::sync::Notify::new()),
            init_done: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            premium_mode: Arc::new(std::sync::atomic::AtomicU8::new(Self::env_premium_mode())),
            user_turn_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            reasoning_effort: Arc::new(RwLock::new(None)),
            official_runtime: Arc::new(Mutex::new(None)),
            created_at: std::time::Instant::now(),
        };
        if matches!(&provider.backend, CopilotBackend::Native { .. }) {
            provider.seed_cached_catalog();
        }
        provider
    }

    pub fn has_credentials() -> bool {
        match CopilotTransport::from_env() {
            Ok(CopilotTransport::OfficialCli) => true,
            Ok(CopilotTransport::Native) => copilot_auth::has_copilot_credentials(),
            Err(_) => false,
        }
    }

    fn env_premium_mode() -> u8 {
        match std::env::var("JCODE_COPILOT_PREMIUM").ok().as_deref() {
            Some("0") => PremiumMode::Zero as u8,
            Some("1") => PremiumMode::OnePerSession as u8,
            _ => PremiumMode::Normal as u8,
        }
    }

    pub fn new_with_token(github_token: String) -> Self {
        Self::common(CopilotBackend::Native {
            client: jcode_provider_core::shared_http_client(),
            github_token,
            bearer_token: Arc::new(tokio::sync::RwLock::new(None)),
            session_id: Uuid::new_v4().to_string(),
            machine_id: Self::get_or_create_machine_id(),
        })
    }

    pub fn with_official_process(process: CopilotOfficialCliProcess) -> Self {
        Self::common(CopilotBackend::OfficialCli { process })
    }

    fn startup_prefetch_grace_ms() -> u64 {
        std::env::var("JCODE_COPILOT_PREFETCH_STARTUP_GRACE_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2000)
    }

    fn get_or_create_machine_id() -> String {
        let machine_id_path = dirs::home_dir()
            .unwrap_or_default()
            .join(".jcode")
            .join("machine_id");
        if let Ok(id) = std::fs::read_to_string(&machine_id_path) {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return id;
            }
        }
        let id = Uuid::new_v4().to_string().replace('-', "");
        let _ = std::fs::create_dir_all(machine_id_path.parent().unwrap_or(&machine_id_path));
        let _ = std::fs::write(&machine_id_path, &id);
        id
    }

    fn is_user_initiated_raw(messages: &[ChatMessage]) -> bool {
        for msg in messages.iter().rev() {
            if msg.role != Role::User {
                return true;
            }
            let has_tool_result = msg
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolResult { .. }));
            if has_tool_result {
                return false;
            }
            let is_text_only = msg
                .content
                .iter()
                .all(|block| matches!(block, ContentBlock::Text { .. }));
            if !is_text_only || msg.content.is_empty() {
                return true;
            }
            let is_system_reminder = msg.content.iter().any(|block| {
                if let ContentBlock::Text { text, .. } = block {
                    text.contains("<system-reminder>")
                } else {
                    false
                }
            });
            if is_system_reminder {
                continue;
            }
            return true;
        }
        true
    }

    fn is_user_initiated(&self, messages: &[ChatMessage]) -> bool {
        let raw = Self::is_user_initiated_raw(messages);
        if !raw {
            return false;
        }
        let mode = self.premium_mode.load(std::sync::atomic::Ordering::Relaxed);
        match mode {
            2 => false,
            1 => {
                let count = self
                    .user_turn_count
                    .load(std::sync::atomic::Ordering::Relaxed);
                count == 0
            }
            _ => true,
        }
    }

    pub fn set_premium_mode(&self, mode: PremiumMode) {
        self.premium_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
        if mode != PremiumMode::Normal {
            jcode_base::logging::info(&format!("Copilot premium mode set to {:?}", mode));
        }
    }

    pub fn get_premium_mode(&self) -> PremiumMode {
        match self.premium_mode.load(std::sync::atomic::Ordering::Relaxed) {
            1 => PremiumMode::OnePerSession,
            2 => PremiumMode::Zero,
            _ => PremiumMode::Normal,
        }
    }

    fn fork_for_session(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            model: Arc::new(RwLock::new(self.model())),
            fetched_models: self.fetched_models.clone(),
            catalog_source: self.catalog_source.clone(),
            init_ready: self.init_ready.clone(),
            init_done: self.init_done.clone(),
            premium_mode: Arc::new(std::sync::atomic::AtomicU8::new(
                self.premium_mode.load(std::sync::atomic::Ordering::Relaxed),
            )),
            user_turn_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            reasoning_effort: Arc::new(RwLock::new(self.current_reasoning_effort())),
            official_runtime: Arc::new(Mutex::new(None)),
            created_at: self.created_at,
        }
    }

    /// Detect the user's Copilot tier and set the best default model.
    /// Call this after construction. Fetches a bearer token and queries /models.
    /// If JCODE_COPILOT_MODEL is set, this is a no-op (user override).
    pub async fn detect_tier_and_set_default(&self) {
        if matches!(self.backend.transport(), CopilotTransport::OfficialCli) {
            if let Err(error) = self.probe_official_cli().await {
                jcode_base::logging::warn(&format!(
                    "Official Copilot CLI ACP handshake failed: {error:#}"
                ));
            }
            self.mark_init_done();
            return;
        }
        let CopilotBackend::Native { client, .. } = &self.backend else {
            unreachable!("official transport returned above");
        };

        let detect_start = std::time::Instant::now();
        if std::env::var("JCODE_COPILOT_MODEL").is_ok() {
            jcode_base::logging::info(
                "Copilot model overridden via JCODE_COPILOT_MODEL, skipping tier detection",
            );
            self.mark_init_done();
            return;
        }

        let bearer_start = std::time::Instant::now();
        let bearer = match self.get_bearer_token().await {
            Ok(t) => t,
            Err(e) => {
                jcode_base::logging::info(&format!(
                    "Copilot tier detection: failed to get bearer token after {}ms: {}",
                    bearer_start.elapsed().as_millis(),
                    e
                ));
                self.mark_init_done();
                return;
            }
        };

        let fetch_start = std::time::Instant::now();
        match copilot_auth::fetch_available_models(client, &bearer).await {
            Ok(models) => {
                let picker_models: Vec<String> = models
                    .iter()
                    .filter(|m| m.model_picker_enabled)
                    .map(|m| m.id.clone())
                    .collect();
                let all_ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
                let default = copilot_auth::choose_default_model(&models);
                jcode_base::logging::info(&format!(
                    "Copilot tier detection: bearer={}ms, fetch_models={}ms, total={}ms, {} total, {} picker-enabled, default -> {}. Picker: [{}]. All: [{}]",
                    bearer_start.elapsed().as_millis(),
                    fetch_start.elapsed().as_millis(),
                    detect_start.elapsed().as_millis(),
                    all_ids.len(),
                    picker_models.len(),
                    default,
                    picker_models.join(", "),
                    all_ids.join(", ")
                ));
                if let Ok(mut m) = self.model.try_write() {
                    *m = default;
                }
                let display_models = if picker_models.is_empty() {
                    all_ids
                } else {
                    picker_models
                };
                if let Ok(mut fm) = self.fetched_models.try_write() {
                    *fm = display_models;
                }
                if let Ok(mut source) = self.catalog_source.try_write() {
                    *source = CatalogSource::Live;
                }
                Self::persist_catalog(
                    &self
                        .fetched_models
                        .try_read()
                        .map(|models| models.clone())
                        .unwrap_or_default(),
                );
            }
            Err(e) => {
                jcode_base::logging::info(&format!(
                    "Copilot tier detection: bearer={}ms, fetch_models={}ms, total={}ms, failed to fetch models: {}",
                    bearer_start.elapsed().as_millis(),
                    fetch_start.elapsed().as_millis(),
                    detect_start.elapsed().as_millis(),
                    e
                ));
            }
        }
        self.mark_init_done();
    }

    pub async fn probe_official_cli(&self) -> Result<()> {
        let CopilotBackend::OfficialCli { process } = &self.backend else {
            bail!("Copilot transport is not official-cli");
        };
        let process = process.clone();
        run_on_acp_thread_with_process(
            process,
            |connection, _forward_updates, _event_tx, _io_health| {
                Box::pin(async move {
                    initialize_official_cli(&connection).await?;
                    Ok(())
                })
            },
        )
        .await
    }

    async fn discover_official_models(&self) -> Result<DiscoveredModels> {
        let CopilotBackend::OfficialCli { process } = &self.backend else {
            bail!("Copilot transport is not official-cli");
        };
        let process = process.clone();
        run_on_acp_thread_with_process(
            process,
            |connection, _forward_updates, _event_tx, _io_health| {
                Box::pin(async move {
                    initialize_official_cli(&connection).await?;
                    let cwd =
                        std::env::current_dir().context("Failed to determine working directory")?;
                    let response = timeout_acp_request(
                        "session/new",
                        connection
                            .new_session(acp::NewSessionRequest::new(cwd).mcp_servers(Vec::new())),
                    )
                    .await?;
                    Ok(models_from_session_state(response.models.as_ref()))
                })
            },
        )
        .await
    }

    fn update_official_models(&self, discovered: DiscoveredModels) {
        if !discovered.available.is_empty() {
            *self
                .fetched_models
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = discovered.available;
            *self
                .catalog_source
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = CatalogSource::Live;
        }
        if std::env::var_os("JCODE_COPILOT_MODEL").is_none()
            && let Some(current) = discovered.current.filter(|model| !model.trim().is_empty())
        {
            *self
                .model
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = current;
        }
    }

    fn mark_init_done(&self) {
        self.init_done
            .store(true, std::sync::atomic::Ordering::Release);
        self.init_ready.notify_waiters();
        jcode_base::bus::Bus::global().publish_models_updated();
    }

    pub fn complete_init_without_tier_detection(&self) {
        self.mark_init_done();
    }

    async fn wait_for_init(&self) {
        if self.init_done.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let notified = self.init_ready.notified();
        if self.init_done.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        notified.await;
    }

    async fn complete_official(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        system: &str,
        resume_session_id: Option<&str>,
        working_dir: PathBuf,
        current_user_prompt: Option<&str>,
    ) -> Result<EventStream> {
        self.wait_for_init().await;
        let CopilotBackend::OfficialCli { process } = &self.backend else {
            bail!("Copilot transport is not official-cli");
        };
        let resume_session_id = resume_session_id.map(parse_official_session_id);
        let prompt = build_official_prompt(messages, system, resume_session_id.is_some())?;
        let fresh_prompt = build_stale_fresh_prompt(messages, system, current_user_prompt)?;
        let selected_model = self.model();
        let tool_policy = CopilotOfficialToolPolicy::from_jcode_tools(tools);
        let (tx, rx) = mpsc::channel(128);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let config = OfficialRuntimeConfig {
            working_dir,
            tool_policy,
        };
        let command_tx = {
            let mut runtime = self
                .official_runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let replace = runtime
                .as_ref()
                .is_some_and(|worker| worker.config != config || !worker.is_healthy());
            if replace {
                runtime.take();
            }
            if runtime.is_none() {
                *runtime = Some(start_official_runtime(process.clone(), config.clone())?);
            }
            runtime
                .as_ref()
                .map(|worker| worker.tx.clone())
                .context("Official Copilot CLI runtime was unavailable")?
        };
        command_tx
            .send(OfficialRuntimeCommand::Turn(OfficialTurnCommand {
                selected_model,
                resume_session_id,
                prompt,
                fresh_prompt,
                tx,
                cancel_rx,
            }))
            .map_err(|_| anyhow!("Official Copilot CLI runtime stopped before the turn started"))?;

        Ok(Box::pin(CopilotOfficialEventStream {
            inner: ReceiverStream::new(rx),
            cancel: Some(cancel_tx),
        }))
    }

    /// Get a valid Copilot bearer token, refreshing if expired
    async fn get_bearer_token(&self) -> Result<String> {
        let CopilotBackend::Native {
            client,
            github_token,
            bearer_token,
            ..
        } = &self.backend
        else {
            bail!("Native Copilot bearer tokens are unavailable on official-cli transport");
        };
        {
            let guard = bearer_token.read().await;
            if let Some(ref token) = *guard
                && !token.is_expired()
            {
                return Ok(token.token.clone());
            }
        }
        // Need to refresh
        // Need to refresh
        let new_token = copilot_auth::exchange_github_token(client, github_token).await?;
        let token_str = new_token.token.clone();
        *bearer_token.write().await = Some(new_token);
        Ok(token_str)
    }

    /// Check if an error indicates token expiration
    fn is_auth_error(status: reqwest::StatusCode) -> bool {
        status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN
    }

    /// Build OpenAI-compatible messages array from our message format.
    fn build_messages(system: &str, messages: &[ChatMessage]) -> Vec<Value> {
        build_copilot_messages(system, messages)
    }

    /// Build OpenAI-compatible tools array.
    fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
        build_copilot_tools(tools)
    }

    /// Send a streaming request to Copilot API with retry logic
    async fn stream_request(
        &self,
        body: Value,
        uses_responses_api: bool,
        is_user_initiated: bool,
        tx: mpsc::Sender<Result<StreamEvent>>,
    ) {
        use jcode_message_types::ConnectionPhase;

        self.wait_for_init().await;
        let CopilotBackend::Native {
            client,
            bearer_token,
            session_id,
            machine_id,
            ..
        } = &self.backend
        else {
            let _ = tx
                .send(Err(anyhow!(
                    "Native Copilot request attempted on official-cli transport"
                )))
                .await;
            return;
        };
        let client = client.clone();
        let bearer_token = Arc::clone(bearer_token);
        let session_id = session_id.clone();
        let machine_id = machine_id.clone();
        let model = self
            .model
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let initiator = if is_user_initiated { "user" } else { "agent" };

        const MAX_RETRIES: u32 = 3;
        const RETRY_BASE_DELAY_MS: u64 = 1000;
        let mut last_error: Option<anyhow::Error> = None;
        let mut attempted_auth_refresh = false;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = jcode_provider_core::attempt_tracker::retry_backoff_delay(
                    attempt,
                    RETRY_BASE_DELAY_MS,
                );
                jcode_base::logging::info(&format!(
                    "Retrying Copilot API request (attempt {}/{}) after {}ms",
                    attempt + 1,
                    MAX_RETRIES,
                    delay.as_millis()
                ));
                let _ = tx
                    .send(Ok(StreamEvent::ConnectionPhase {
                        phase: ConnectionPhase::Retrying {
                            attempt: attempt + 1,
                            max: MAX_RETRIES,
                        },
                    }))
                    .await;
                tokio::time::sleep(delay).await;
            }

            jcode_base::logging::info(&format!(
                "Copilot request: X-Initiator={} model={}",
                initiator, model
            ));

            let bearer_token_value = match self.get_bearer_token().await {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            let request_id = Uuid::new_v4().to_string();

            // Retries use a fresh unpooled client: the fault that broke
            // attempt N (e.g. TLS BadRecordMac from a corrupting middlebox)
            // may also have poisoned other idle pooled connections opened
            // through the same path, so reusing the shared pool can fail
            // identically. A fresh client guarantees a new TCP+TLS connection.
            let attempt_client = if attempt == 0 {
                client.clone()
            } else {
                jcode_provider_core::fresh_transport_client()
            };

            let resp = attempt_client
                .post(format!(
                    "{}/{}",
                    copilot_auth::COPILOT_API_BASE,
                    copilot_api_path(uses_responses_api)
                ))
                .header("Authorization", format!("Bearer {}", bearer_token_value))
                .header("Editor-Version", copilot_auth::EDITOR_VERSION)
                .header("Editor-Plugin-Version", copilot_auth::EDITOR_PLUGIN_VERSION)
                .header(
                    "Copilot-Integration-Id",
                    copilot_auth::COPILOT_INTEGRATION_ID,
                )
                .header("Content-Type", "application/json")
                .header("X-Initiator", initiator)
                .header("X-Request-Id", &request_id)
                .header("Openai-Intent", "conversation-panel")
                .header("Openai-Organization", "github-copilot")
                .header("X-GitHub-Api-Version", COPILOT_API_VERSION)
                .header("Vscode-Sessionid", &session_id)
                .header("Vscode-Machineid", &machine_id)
                .json(&body)
                .send()
                .await;

            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    // Full anyhow chain ({:#}) so a `.context(...)`-wrapped
                    // transport cause (e.g. TLS BadRecordMac) is visible to the
                    // retry classifier.
                    let error_str = format!("{e:#}").to_lowercase();
                    if is_retryable_error(&error_str) && attempt + 1 < MAX_RETRIES {
                        jcode_base::logging::info(&format!(
                            "Transient Copilot error, will retry: {}",
                            e
                        ));
                        last_error = Some(anyhow::anyhow!("Copilot API request failed: {}", e));
                        continue;
                    }
                    let _ = tx
                        .send(Err(anyhow::anyhow!("Copilot API request failed: {}", e)))
                        .await;
                    return;
                }
            };

            let status = resp.status();

            // On auth error, invalidate token and retry once
            if Self::is_auth_error(status) && !attempted_auth_refresh {
                attempted_auth_refresh = true;
                *bearer_token.write().await = None;
                jcode_base::logging::info("Copilot bearer token expired, refreshing...");
                last_error = Some(anyhow::anyhow!("Copilot auth error (HTTP {})", status));
                continue;
            }

            if !status.is_success() {
                let body_text = jcode_base::util::http_error_body(resp, "HTTP error").await;
                let error_str =
                    format!("Copilot API error (HTTP {}): {}", status, body_text).to_lowercase();
                if is_retryable_error(&error_str) && attempt + 1 < MAX_RETRIES {
                    jcode_base::logging::info(&format!(
                        "Retryable Copilot HTTP error: {}",
                        error_str
                    ));
                    last_error = Some(anyhow::anyhow!(
                        "Copilot API error (HTTP {}): {}",
                        status,
                        body_text
                    ));
                    continue;
                }
                let _ = tx
                    .send(Err(anyhow::anyhow!(
                        "Copilot API error (HTTP {}): {}",
                        status,
                        body_text
                    )))
                    .await;
                return;
            }

            // Send connection type event
            let _ = tx
                .send(Ok(StreamEvent::ConnectionType {
                    connection: format!("copilot-api ({})", model),
                }))
                .await;

            // Track whether this attempt streams replay-visible output so a
            // mid-stream transport fault can roll the partial output back on
            // the consumer before the retry replays the response from the top.
            let (attempt_tx, attempt_guard) =
                jcode_provider_core::attempt_tracker::track_attempt_output(tx.clone());

            // Process SSE stream - returns Err on timeout/stream errors
            let stream_result = if uses_responses_api {
                self.process_responses_sse_stream(resp, attempt_tx).await
            } else {
                self.process_sse_stream(resp, attempt_tx).await
            };
            match stream_result {
                Ok(()) => {
                    let _ = attempt_guard.finish().await;
                    return;
                }
                Err(e) => {
                    let saw_output = attempt_guard.finish().await;
                    // Full anyhow chain ({:#}) so a `.context(...)`-wrapped
                    // transport cause (e.g. TLS BadRecordMac) is visible to the
                    // retry classifier.
                    let error_str = format!("{e:#}").to_lowercase();
                    if is_retryable_error(&error_str) && attempt + 1 < MAX_RETRIES {
                        if saw_output {
                            // Partial output already reached the consumer; tell
                            // it to discard the partial attempt so the retried
                            // response replays cleanly instead of duplicating.
                            jcode_base::logging::warn(&format!(
                                "Copilot stream failed after partial output (attempt {}/{}); rolling back partial attempt and retrying: {}",
                                attempt + 1,
                                MAX_RETRIES,
                                e
                            ));
                            let _ = tx
                                .send(Ok(StreamEvent::RetryRollback {
                                    attempt: attempt + 2,
                                    max: MAX_RETRIES,
                                }))
                                .await;
                        } else {
                            jcode_base::logging::info(&format!(
                                "Copilot stream failed (attempt {}/{}), will retry: {}",
                                attempt + 1,
                                MAX_RETRIES,
                                e
                            ));
                        }
                        last_error = Some(e);
                        continue;
                    }
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
        }

        // All retries exhausted
        if let Some(e) = last_error {
            let _ = tx
                .send(Err(anyhow::anyhow!(
                    "Copilot: failed after {} retries: {}",
                    MAX_RETRIES,
                    e
                )))
                .await;
        }
    }

    async fn process_responses_sse_stream(
        &self,
        resp: reqwest::Response,
        tx: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<()> {
        use futures::StreamExt;

        let timeout = jcode_base::provider::stream_idle_timeout();
        let mut stream =
            jcode_provider_openai::stream::OpenAIResponsesStream::new(resp.bytes_stream());
        let mut input_tokens = 0;
        let mut output_tokens = 0;

        loop {
            let event = match tokio::time::timeout(timeout, stream.next()).await {
                Ok(Some(event)) => event?,
                Ok(None) => break,
                Err(_) => anyhow::bail!(
                    "Stream read timeout: no data received for {} seconds",
                    timeout.as_secs()
                ),
            };
            if let StreamEvent::TokenUsage {
                input_tokens: input,
                output_tokens: output,
                ..
            } = &event
            {
                input_tokens = input.unwrap_or(0);
                output_tokens = output.unwrap_or(0);
            }
            tx.send(Ok(event))
                .await
                .map_err(|_| anyhow::anyhow!("Stream receiver dropped"))?;
        }

        jcode_base::copilot_usage::record_request(input_tokens, output_tokens, true);
        Ok(())
    }

    async fn process_sse_stream(
        &self,
        resp: reqwest::Response,
        tx: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<()> {
        use futures::StreamExt;

        // Idle timeout between streamed chunks. Configurable via
        // `[provider] stream_idle_timeout_secs` / `JCODE_STREAM_IDLE_TIMEOUT_SECS`
        // so slow reasoning models don't trip a premature timeout (issue #434).
        let sse_chunk_timeout = jcode_base::provider::stream_idle_timeout();

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_args = String::new();
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut saw_any_data = false;

        loop {
            let chunk = match tokio::time::timeout(sse_chunk_timeout, stream.next()).await {
                Ok(Some(Ok(c))) => c,
                Ok(Some(Err(e))) => {
                    anyhow::bail!("Stream error: {}", e);
                }
                Ok(None) => break, // stream ended normally
                Err(_) => {
                    jcode_base::logging::warn(&format!(
                        "Copilot SSE stream timed out (no data for {}s, saw_data={})",
                        sse_chunk_timeout.as_secs(),
                        saw_any_data
                    ));
                    anyhow::bail!(
                        "Stream read timeout: no data received for {} seconds",
                        sse_chunk_timeout.as_secs()
                    );
                }
            };
            saw_any_data = true;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end_matches('\r').to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = jcode_base::util::sse_data_line(&line) {
                    if data.trim() == "[DONE]" {
                        // Send usage info before done
                        if input_tokens > 0 || output_tokens > 0 {
                            let _ = tx
                                .send(Ok(StreamEvent::TokenUsage {
                                    input_tokens: Some(input_tokens),
                                    output_tokens: Some(output_tokens),
                                    cache_creation_input_tokens: None,
                                    cache_read_input_tokens: None,
                                }))
                                .await;
                        }
                        jcode_base::copilot_usage::record_request(
                            input_tokens,
                            output_tokens,
                            true,
                        );
                        let _ = tx
                            .send(Ok(StreamEvent::MessageEnd { stop_reason: None }))
                            .await;
                        return Ok(());
                    }

                    let parsed: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    // Extract usage if present
                    if let Some(usage) = parsed.get("usage") {
                        input_tokens = usage
                            .get("prompt_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        output_tokens = usage
                            .get("completion_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                    }

                    // Process choices
                    if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array()) {
                        for choice in choices {
                            let delta = match choice.get("delta") {
                                Some(d) => d,
                                None => continue,
                            };

                            // Text content
                            if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                && !content.is_empty()
                            {
                                let _ = tx
                                    .send(Ok(StreamEvent::TextDelta(content.to_string())))
                                    .await;
                            }

                            // Tool calls
                            if let Some(tool_calls) =
                                delta.get("tool_calls").and_then(|t| t.as_array())
                            {
                                for tc in tool_calls {
                                    // New tool call start
                                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                        // Flush previous tool call if any
                                        if !current_tool_id.is_empty() {
                                            let _ = tx.send(Ok(StreamEvent::ToolUseEnd)).await;
                                        }
                                        current_tool_id = id.to_string();
                                        current_tool_name = tc
                                            .get("function")
                                            .and_then(|f| f.get("name"))
                                            .and_then(|n| n.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        current_tool_args.clear();

                                        let _ = tx
                                            .send(Ok(StreamEvent::ToolUseStart {
                                                id: current_tool_id.clone(),
                                                name: current_tool_name.clone(),
                                            }))
                                            .await;
                                    }

                                    // Accumulate arguments
                                    if let Some(args) = tc
                                        .get("function")
                                        .and_then(|f| f.get("arguments"))
                                        .and_then(|a| a.as_str())
                                    {
                                        current_tool_args.push_str(args);
                                        let _ = tx
                                            .send(Ok(StreamEvent::ToolInputDelta(args.to_string())))
                                            .await;
                                    }
                                }
                            }

                            // Finish reason
                            if let Some(finish) =
                                choice.get("finish_reason").and_then(|f| f.as_str())
                            {
                                // Flush last tool call
                                if !current_tool_id.is_empty() {
                                    let _ = tx.send(Ok(StreamEvent::ToolUseEnd)).await;
                                    current_tool_id.clear();
                                    current_tool_name.clear();
                                    current_tool_args.clear();
                                }

                                let stop_reason = match finish {
                                    "stop" => "end_turn",
                                    "tool_calls" => "tool_use",
                                    "length" => "max_tokens",
                                    other => other,
                                };
                                let _ = tx
                                    .send(Ok(StreamEvent::MessageEnd {
                                        stop_reason: Some(stop_reason.to_string()),
                                    }))
                                    .await;
                            }
                        }
                    }
                }
            }
        }

        // Stream ended without [DONE]
        let _ = tx
            .send(Ok(StreamEvent::MessageEnd { stop_reason: None }))
            .await;
        Ok(())
    }
}

struct CopilotOfficialEventStream {
    inner: ReceiverStream<Result<StreamEvent>>,
    cancel: Option<oneshot::Sender<()>>,
}

impl Stream for CopilotOfficialEventStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for CopilotOfficialEventStream {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OfficialRuntimeConfig {
    working_dir: PathBuf,
    tool_policy: CopilotOfficialToolPolicy,
}

struct OfficialTurnCommand {
    selected_model: String,
    resume_session_id: Option<String>,
    prompt: String,
    fresh_prompt: String,
    tx: mpsc::Sender<Result<StreamEvent>>,
    cancel_rx: oneshot::Receiver<()>,
}

enum OfficialRuntimeCommand {
    Turn(OfficialTurnCommand),
    Shutdown,
}

struct OfficialRuntimeHandle {
    config: OfficialRuntimeConfig,
    tx: mpsc::UnboundedSender<OfficialRuntimeCommand>,
    thread: Option<std::thread::JoinHandle<()>>,
    io_closed: Arc<AtomicBool>,
    shutdown: OfficialShutdown,
}

impl OfficialRuntimeHandle {
    fn is_healthy(&self) -> bool {
        !self.io_closed.load(Ordering::Acquire)
            && !self
                .thread
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
    }
}

impl Drop for OfficialRuntimeHandle {
    fn drop(&mut self) {
        self.shutdown.request();
        let _ = self.tx.send(OfficialRuntimeCommand::Shutdown);
        self.thread.take();
    }
}

#[derive(Clone)]
struct OfficialShutdown {
    requested: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl OfficialShutdown {
    fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn request(&self) {
        self.requested.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    async fn requested(&self) {
        let notified = self.notify.notified();
        if self.is_requested() {
            return;
        }
        notified.await;
    }
}

#[derive(Default, Debug)]
struct DiscoveredModels {
    current: Option<String>,
    available: Vec<String>,
}

fn models_from_session_state(state: Option<&acp::SessionModelState>) -> DiscoveredModels {
    let Some(state) = state else {
        return DiscoveredModels::default();
    };
    let current = Some(state.current_model_id.0.to_string());
    let available = state
        .available_models
        .iter()
        .map(|model| model.model_id.0.to_string())
        .collect();
    DiscoveredModels { current, available }
}

fn build_official_prompt(messages: &[ChatMessage], system: &str, resumed: bool) -> Result<String> {
    if !resumed
        && messages
            .iter()
            .any(|message| message.role == Role::Assistant)
    {
        bail!(
            "Official Copilot CLI ACP cannot replay disconnected history; resume the provider session instead"
        );
    }

    let start = if resumed {
        messages
            .iter()
            .rposition(|message| message.role == Role::Assistant)
            .map_or(0, |index| index + 1)
    } else {
        0
    };

    let mut sections = Vec::new();
    if !resumed && !system.trim().is_empty() {
        sections.push(format!("<system>\n{}\n</system>", system.trim()));
    }
    for message in &messages[start..] {
        if message.role != Role::User {
            continue;
        }
        for block in &message.content {
            match block {
                ContentBlock::Text { text, .. } if !text.trim().is_empty() => {
                    sections.push(text.clone());
                }
                ContentBlock::ToolResult { content, .. } if !content.trim().is_empty() => {
                    sections.push(content.clone());
                }
                ContentBlock::Image { .. } => {
                    bail!("Image input is not supported by the official-cli ACP transport");
                }
                _ => {}
            }
        }
    }
    if sections.is_empty() {
        bail!("No user prompt found for official Copilot CLI request");
    }
    Ok(sections.join("\n\n"))
}

fn build_stale_fresh_prompt(
    messages: &[ChatMessage],
    system: &str,
    current_user_prompt: Option<&str>,
) -> Result<String> {
    let mut sections = Vec::new();
    if !system.trim().is_empty() {
        sections.push(format!("<system>\n{}\n</system>", system.trim()));
    }

    if let Some(prompt) = current_user_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        sections.push(prompt.to_string());
    } else {
        let message = messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .context("No current user prompt found for official Copilot CLI stale recovery")?;
        for block in &message.content {
            match block {
                ContentBlock::Text { text, .. } if !text.trim().is_empty() => {
                    sections.push(text.clone());
                }
                ContentBlock::ToolResult { content, .. } if !content.trim().is_empty() => {
                    sections.push(content.clone());
                }
                ContentBlock::Image { .. } => {
                    bail!("Image input is not supported by the official-cli ACP transport");
                }
                _ => {}
            }
        }
    }
    if sections.is_empty() {
        bail!("No current user prompt found for official Copilot CLI stale recovery");
    }
    Ok(sections.join("\n\n"))
}

fn parse_official_session_id(value: &str) -> String {
    const LEGACY_PREFIX: &str = "jcode-copilot-acp-v1:";
    value
        .strip_prefix(LEGACY_PREFIX)
        .and_then(|encoded| encoded.split_once(':').map(|(_, upstream_id)| upstream_id))
        .unwrap_or(value)
        .to_string()
}

fn start_official_runtime(
    process: CopilotOfficialCliProcess,
    config: OfficialRuntimeConfig,
) -> Result<OfficialRuntimeHandle> {
    let (tx, rx) = mpsc::unbounded_channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let io_closed = Arc::new(AtomicBool::new(false));
    let io_closed_notify = Arc::new(tokio::sync::Notify::new());
    let shutdown = OfficialShutdown::new();
    let thread = std::thread::Builder::new()
        .name("jcode-copilot-official-acp".to_string())
        .spawn({
            let config = config.clone();
            let io_closed = Arc::clone(&io_closed);
            let io_closed_notify = Arc::clone(&io_closed_notify);
            let shutdown = shutdown.clone();
            move || {
                let _ = run_official_runtime_thread(
                    process,
                    config,
                    rx,
                    ready_tx,
                    io_closed,
                    io_closed_notify,
                    shutdown,
                );
            }
        })
        .context("Failed to start official Copilot CLI ACP runtime thread")?;
    match ready_rx.recv_timeout(ACP_REQUEST_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(anyhow!(error)),
        Err(_) => {
            return Err(anyhow!(
                "Official Copilot CLI ACP runtime did not initialize within {}s",
                ACP_REQUEST_TIMEOUT.as_secs()
            ));
        }
    }
    Ok(OfficialRuntimeHandle {
        config,
        tx,
        thread: Some(thread),
        io_closed,
        shutdown,
    })
}

struct LiveOfficialSession {
    id: acp::SessionId,
    models: Option<acp::SessionModelState>,
}

fn run_official_runtime_thread(
    process: CopilotOfficialCliProcess,
    config: OfficialRuntimeConfig,
    mut commands: mpsc::UnboundedReceiver<OfficialRuntimeCommand>,
    ready: std::sync::mpsc::Sender<std::result::Result<(), String>>,
    io_closed: Arc<AtomicBool>,
    io_closed_notify: Arc<tokio::sync::Notify>,
    shutdown: OfficialShutdown,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to build official Copilot CLI ACP Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, async move {
        with_official_connection(
            process,
            config.tool_policy,
            Some(config.working_dir.clone()),
            mpsc::channel(1).0,
            io_closed,
            io_closed_notify,
            async move |connection,
                        forward_updates,
                        shared_event_tx,
                        io_health| {
                let initialized = match initialize_official_cli(&connection).await {
                    Ok(initialized) => initialized,
                    Err(error) => {
                        let _ = ready.send(Err(format!("{error:#}")));
                        return Err(error);
                    }
                };
                let _ = ready.send(Ok(()));
                let mut live_session: Option<LiveOfficialSession> = None;
                loop {
                    let io_closed = io_health.closed();
                    let shutdown_requested = shutdown.requested();
                    tokio::pin!(io_closed);
                    tokio::pin!(shutdown_requested);
                    if io_health.is_closed() || shutdown.is_requested() {
                        return Ok(());
                    }
                    let command = tokio::select! {
                        command = commands.recv() => command,
                        _ = &mut io_closed => return Ok(()),
                        _ = &mut shutdown_requested => return Ok(()),
                    };
                    let Some(command) = command else {
                        return Ok(());
                    };
                    let OfficialRuntimeCommand::Turn(turn) = command else {
                        return Ok(());
                    };
                    *shared_event_tx
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = turn.tx.clone();
                    turn.tx
                        .send(Ok(StreamEvent::ConnectionType {
                            connection: format!("official-cli ACP ({})", turn.selected_model),
                        }))
                        .await
                        .map_err(|_| anyhow!("Official Copilot CLI stream consumer closed"))?;

                    let mut recovered_stale = false;
                    if live_session.is_none() {
                        match open_official_session(
                            &connection,
                            &initialized,
                            &config.working_dir,
                            turn.resume_session_id.as_deref(),
                        )
                        .await
                        {
                            Ok(session) => live_session = Some(session),
                            Err(OpenOfficialSessionError::Stale) => recovered_stale = true,
                            Err(OpenOfficialSessionError::Other(error)) => {
                                let _ = turn.tx.send(Err(error)).await;
                                *shared_event_tx
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                    mpsc::channel(1).0;
                                return Ok(());
                            }
                        }
                        if recovered_stale {
                            recovered_stale = true;
                            let response = match timeout_acp_request(
                                "session/new",
                                connection.new_session(
                                    acp::NewSessionRequest::new(config.working_dir.clone())
                                        .mcp_servers(Vec::new()),
                                ),
                            )
                            .await
                            {
                                Ok(response) => response,
                                Err(error) => {
                                    let _ = turn.tx.send(Err(error)).await;
                                    *shared_event_tx
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                        mpsc::channel(1).0;
                                    return Ok(());
                                }
                            };
                            let new_session = LiveOfficialSession {
                                id: response.session_id,
                                models: response.models,
                            };
                            live_session = Some(new_session);
                            turn.tx
                                .send(Ok(StreamEvent::StatusDetail {
                                    detail: "Prior upstream context unavailable; continued fresh"
                                        .to_string(),
                                }))
                                .await
                                .ok();
                        }
                    }

                    let session = live_session.as_mut().context(
                        "Official Copilot CLI runtime did not have an active session",
                    )?;
                    turn.tx
                        .send(Ok(StreamEvent::SessionId(session.id.0.to_string())))
                        .await
                        .ok();

                    let current_model = session
                        .models
                        .as_ref()
                        .map(|models| models.current_model_id.0.as_ref());
                    if current_model != Some(turn.selected_model.as_str()) {
                        if let Err(error) = timeout_acp_request(
                            "session/set_model",
                            connection.set_session_model(acp::SetSessionModelRequest::new(
                                session.id.clone(),
                                turn.selected_model.clone(),
                            )),
                        )
                        .await
                        {
                            let _ = turn.tx.send(Err(error)).await;
                            *shared_event_tx
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                mpsc::channel(1).0;
                            return Ok(());
                        }
                        if let Some(models) = session.models.as_mut() {
                            models.current_model_id = turn.selected_model.clone().into();
                        }
                    }

                    forward_updates.store(true, Ordering::Release);
                    let prompt = if recovered_stale {
                        turn.fresh_prompt
                    } else {
                        turn.prompt
                    };
                    let content = vec![acp::ContentBlock::Text(acp::TextContent::new(prompt))];
                    let prompt_request = acp::PromptRequest::new(session.id.clone(), content);
                    let mut cancel_rx = turn.cancel_rx;
                    let prompt = connection.prompt(prompt_request);
                    tokio::pin!(prompt);
                    let prompt_timeout = tokio::time::sleep(ACP_PROMPT_TIMEOUT);
                    let shutdown_requested = shutdown.requested();
                    tokio::pin!(prompt_timeout);
                    tokio::pin!(shutdown_requested);
                    let response_result = tokio::select! {
                        response = &mut prompt => response
                            .map_err(|error| anyhow!("Official Copilot CLI ACP session/prompt failed: {error}")),
                        _ = &mut prompt_timeout => Err(anyhow!(
                            "Official Copilot CLI ACP session/prompt timed out after {}s",
                            ACP_PROMPT_TIMEOUT.as_secs()
                        )),
                        _ = &mut cancel_rx => {
                            forward_updates.store(false, Ordering::Release);
                            *shared_event_tx
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                mpsc::channel(1).0;
                            let _ = tokio::time::timeout(
                                ACP_CANCEL_DRAIN_TIMEOUT,
                                connection.cancel(acp::CancelNotification::new(session.id.clone())),
                            )
                            .await;
                            let _ = tokio::time::timeout(ACP_CANCEL_DRAIN_TIMEOUT, &mut prompt).await;
                            return Ok(());
                        }
                        _ = &mut shutdown_requested => {
                            forward_updates.store(false, Ordering::Release);
                            *shared_event_tx
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                mpsc::channel(1).0;
                            return Ok(());
                        }
                    };
                    let response = match response_result {
                        Ok(response) => response,
                        Err(error) => {
                            let _ = turn
                                .tx
                                .send(Err(error.context(
                                    "official Copilot CLI request failed",
                                )))
                                .await;
                            *shared_event_tx
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                mpsc::channel(1).0;
                            return Ok(());
                        }
                    };

                    if let Some(usage) = response.usage {
                        turn.tx
                            .send(Ok(StreamEvent::TokenUsage {
                                input_tokens: Some(usage.input_tokens),
                                output_tokens: Some(usage.output_tokens),
                                cache_read_input_tokens: usage.cached_read_tokens,
                                cache_creation_input_tokens: usage.cached_write_tokens,
                            }))
                            .await
                            .ok();
                    }
                    turn.tx
                        .send(Ok(StreamEvent::SessionId(session.id.0.to_string())))
                        .await
                        .ok();
                    turn.tx
                        .send(Ok(StreamEvent::MessageEnd {
                            stop_reason: Some(acp_stop_reason(response.stop_reason).to_string()),
                        }))
                        .await
                        .ok();
                    forward_updates.store(false, Ordering::Release);
                    *shared_event_tx
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = mpsc::channel(1).0;
                }
            },
        )
        .await
    })
}

enum OpenOfficialSessionError {
    Stale,
    Other(anyhow::Error),
}

async fn open_official_session(
    connection: &acp::ClientSideConnection,
    initialized: &acp::InitializeResponse,
    working_dir: &std::path::Path,
    resume_session_id: Option<&str>,
) -> std::result::Result<LiveOfficialSession, OpenOfficialSessionError> {
    let Some(resume_session_id) = resume_session_id else {
        let response = timeout_acp_request(
            "session/new",
            connection.new_session(
                acp::NewSessionRequest::new(working_dir.to_path_buf()).mcp_servers(Vec::new()),
            ),
        )
        .await
        .map_err(OpenOfficialSessionError::Other)?;
        return Ok(LiveOfficialSession {
            id: response.session_id,
            models: response.models,
        });
    };
    if !initialized.agent_capabilities.load_session {
        return Err(OpenOfficialSessionError::Other(anyhow!(
            "Official Copilot CLI does not advertise ACP session/load support"
        )));
    }
    let response = tokio::time::timeout(
        ACP_REQUEST_TIMEOUT,
        connection.load_session(
            acp::LoadSessionRequest::new(resume_session_id.to_string(), working_dir.to_path_buf())
                .mcp_servers(Vec::new()),
        ),
    )
    .await
    .map_err(|_| {
        OpenOfficialSessionError::Other(anyhow!(
            "Official Copilot CLI ACP session/load timed out after {}s",
            ACP_REQUEST_TIMEOUT.as_secs()
        ))
    })?;
    match response {
        Ok(response) => Ok(LiveOfficialSession {
            id: acp::SessionId::new(resume_session_id.to_string()),
            models: response.models,
        }),
        Err(error) if error.code == acp::ErrorCode::ResourceNotFound => {
            Err(OpenOfficialSessionError::Stale)
        }
        Err(error) => Err(OpenOfficialSessionError::Other(anyhow!(
            "Official Copilot CLI ACP session/load failed: {error}"
        ))),
    }
}

fn acp_stop_reason(reason: acp::StopReason) -> &'static str {
    match reason {
        acp::StopReason::EndTurn => "end_turn",
        acp::StopReason::MaxTokens => "max_tokens",
        acp::StopReason::MaxTurnRequests => "max_turn_requests",
        acp::StopReason::Refusal => "refusal",
        acp::StopReason::Cancelled => "cancelled",
        _ => "unknown",
    }
}

async fn initialize_official_cli(
    connection: &acp::ClientSideConnection,
) -> Result<acp::InitializeResponse> {
    let initialize = acp::InitializeRequest::new(acp::ProtocolVersion::V1)
        .client_info(acp::Implementation::new("jcode", env!("CARGO_PKG_VERSION")).title("Jcode"));
    let response = timeout_acp_request("initialize", connection.initialize(initialize)).await?;
    if response.protocol_version != acp::ProtocolVersion::V1 {
        bail!(
            "Official Copilot CLI negotiated unsupported ACP protocol version {:?}",
            response.protocol_version
        );
    }
    // Authentication remains entirely owned by the official CLI. In
    // particular, do not call ACP authenticate even when the CLI advertises
    // its interactive login method.
    Ok(response)
}

async fn timeout_acp_request<T>(
    name: &'static str,
    future: impl std::future::Future<Output = acp::Result<T>>,
) -> Result<T> {
    tokio::time::timeout(ACP_REQUEST_TIMEOUT, future)
        .await
        .map_err(|_| {
            anyhow!(
                "Official Copilot CLI ACP {name} timed out after {}s",
                ACP_REQUEST_TIMEOUT.as_secs()
            )
        })?
        .map_err(|error| anyhow!("Official Copilot CLI ACP {name} failed: {error}"))
}

type LocalConnectionFuture<T> = Pin<Box<dyn std::future::Future<Output = Result<T>> + 'static>>;
type SharedOfficialEventTx = Arc<Mutex<mpsc::Sender<Result<StreamEvent>>>>;

#[derive(Clone)]
struct OfficialIoHealth {
    closed: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl OfficialIoHealth {
    fn new() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn from_shared(closed: Arc<AtomicBool>, notify: Arc<tokio::sync::Notify>) -> Self {
        Self { closed, notify }
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn mark_closed(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    async fn closed(&self) {
        let notified = self.notify.notified();
        if self.is_closed() {
            return;
        }
        notified.await;
    }
}

async fn run_on_acp_thread_with_process<T: Send + 'static>(
    process: CopilotOfficialCliProcess,
    operation: impl FnOnce(
        acp::ClientSideConnection,
        Arc<AtomicBool>,
        SharedOfficialEventTx,
        OfficialIoHealth,
    ) -> LocalConnectionFuture<T>
    + Send
    + 'static,
) -> Result<T> {
    let (result_tx, result_rx) = oneshot::channel();
    std::thread::Builder::new()
        .name("jcode-copilot-official-acp-probe".to_string())
        .spawn(move || {
            let result = (|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let local = tokio::task::LocalSet::new();
                local.block_on(&runtime, async move {
                    let io_health = OfficialIoHealth::new();
                    with_official_connection(
                        process,
                        CopilotOfficialToolPolicy::default(),
                        None,
                        mpsc::channel(1).0,
                        Arc::clone(&io_health.closed),
                        Arc::clone(&io_health.notify),
                        operation,
                    )
                    .await
                })
            })();
            let _ = result_tx.send(result);
        })
        .context("Failed to start official Copilot CLI ACP probe thread")?;
    result_rx
        .await
        .context("Official Copilot CLI ACP probe thread exited without a result")?
}

async fn with_official_connection<T, F, Fut>(
    process: CopilotOfficialCliProcess,
    tool_policy: CopilotOfficialToolPolicy,
    working_dir: Option<PathBuf>,
    event_tx: mpsc::Sender<Result<StreamEvent>>,
    io_closed: Arc<AtomicBool>,
    io_closed_notify: Arc<tokio::sync::Notify>,
    operation: F,
) -> Result<T>
where
    F: FnOnce(
        acp::ClientSideConnection,
        Arc<AtomicBool>,
        SharedOfficialEventTx,
        OfficialIoHealth,
    ) -> Fut,
    Fut: std::future::Future<Output = Result<T>> + 'static,
{
    let mut command = Command::new(&process.command);
    command
        .args(&process.args)
        .envs(&process.env)
        // The user's interactive Copilot wrapper may set this globally. Jcode
        // answers ACP permission requests itself and must not inherit an
        // environment-level approve-all override.
        .env_remove("COPILOT_ALLOW_ALL")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    tool_policy.configure_command(&mut command);
    if let Some(working_dir) = working_dir.as_deref() {
        command.current_dir(working_dir);
    }
    let mut child = command.spawn().with_context(|| {
        format!(
            "Failed to launch official Copilot CLI at '{}'",
            process.command.display()
        )
    })?;
    let stdin = child
        .stdin
        .take()
        .context("Official Copilot CLI stdin was unavailable")?;
    let stdout = child
        .stdout
        .take()
        .context("Official Copilot CLI stdout was unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("Official Copilot CLI stderr was unavailable")?;
    let stderr_capture = Arc::new(std::sync::Mutex::new(String::new()));
    let mut stderr_task =
        tokio::task::spawn_local(capture_stderr(stderr, Arc::clone(&stderr_capture)));

    let forward_updates = Arc::new(AtomicBool::new(false));
    let shared_event_tx = Arc::new(Mutex::new(event_tx));
    let io_health = OfficialIoHealth::from_shared(io_closed, io_closed_notify);
    let client = CopilotAcpClient {
        tx: Arc::clone(&shared_event_tx),
        forward_updates: Arc::clone(&forward_updates),
        tool_policy,
    };
    let (connection, io) =
        acp::ClientSideConnection::new(client, stdin.compat_write(), stdout.compat(), |future| {
            tokio::task::spawn_local(future);
        });
    let io_task = tokio::task::spawn_local({
        let io_health = io_health.clone();
        async move {
            let result = io.await;
            io_health.mark_closed();
            result
        }
    });
    let result = operation(connection, forward_updates, shared_event_tx, io_health).await;
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
    io_task.abort();
    let _ = tokio::time::timeout(Duration::from_millis(100), &mut stderr_task).await;
    stderr_task.abort();
    let stderr = stderr_capture
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .trim()
        .to_string();
    result.map_err(|error| {
        if stderr.is_empty() {
            error
        } else {
            error.context(format!("Official Copilot CLI stderr: {stderr}"))
        }
    })
}

async fn capture_stderr(
    mut stderr: tokio::process::ChildStderr,
    capture: Arc<std::sync::Mutex<String>>,
) {
    let mut buffer = [0_u8; 4096];
    loop {
        let Ok(read) = stderr.read(&mut buffer).await else {
            return;
        };
        if read == 0 {
            return;
        }
        let mut output = capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if output.len() < STDERR_LIMIT {
            let remaining = STDERR_LIMIT - output.len();
            output.push_str(&String::from_utf8_lossy(&buffer[..read.min(remaining)]));
        }
    }
}

struct CopilotAcpClient {
    tx: SharedOfficialEventTx,
    forward_updates: Arc<AtomicBool>,
    tool_policy: CopilotOfficialToolPolicy,
}

#[async_trait(?Send)]
impl acp::Client for CopilotAcpClient {
    async fn request_permission(
        &self,
        request: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        let selected = if self.tool_policy.allows(request.tool_call.fields.kind) {
            request
                .options
                .iter()
                .find(|option| option.kind == acp::PermissionOptionKind::AllowOnce)
        } else {
            request
                .options
                .iter()
                .find(|option| option.kind == acp::PermissionOptionKind::RejectOnce)
        };
        let outcome = match selected {
            Some(option) => acp::RequestPermissionOutcome::Selected(
                acp::SelectedPermissionOutcome::new(option.option_id.clone()),
            ),
            None => acp::RequestPermissionOutcome::Cancelled,
        };
        Ok(acp::RequestPermissionResponse::new(outcome))
    }

    async fn session_notification(
        &self,
        notification: acp::SessionNotification,
    ) -> acp::Result<()> {
        if !self.forward_updates.load(Ordering::Acquire) {
            return Ok(());
        }
        let events: Vec<StreamEvent> = match notification.update {
            acp::SessionUpdate::AgentMessageChunk(chunk) => text_from_acp_content(chunk.content)
                .map(StreamEvent::TextDelta)
                .into_iter()
                .collect(),
            acp::SessionUpdate::AgentThoughtChunk(chunk) => text_from_acp_content(chunk.content)
                .map(StreamEvent::ThinkingDelta)
                .into_iter()
                .collect(),
            acp::SessionUpdate::ToolCall(call) => {
                let title = call.title;
                let kind = provider_tool_kind(call.kind);
                vec![
                    StreamEvent::ProviderToolUpdate {
                        id: call.tool_call_id.0.to_string(),
                        kind: Some(kind),
                        status: Some(provider_tool_status(call.status)),
                        title: Some(title.clone()),
                    },
                    StreamEvent::StatusDetail { detail: title },
                ]
            }
            acp::SessionUpdate::ToolCallUpdate(update) => {
                let fields = update.fields;
                let mut events = vec![StreamEvent::ProviderToolUpdate {
                    id: update.tool_call_id.0.to_string(),
                    kind: fields.kind.map(provider_tool_kind),
                    status: fields.status.map(provider_tool_status),
                    title: fields.title.clone(),
                }];
                if let Some(detail) = fields.title {
                    events.push(StreamEvent::StatusDetail { detail });
                }
                events
            }
            _ => Vec::new(),
        };
        let tx = self
            .tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for event in events {
            let _ = tx.send(Ok(event)).await;
        }
        Ok(())
    }
}

fn provider_tool_kind(kind: acp::ToolKind) -> ProviderToolKind {
    match kind {
        acp::ToolKind::Read => ProviderToolKind::Read,
        acp::ToolKind::Search => ProviderToolKind::Search,
        acp::ToolKind::Edit => ProviderToolKind::Edit,
        acp::ToolKind::Delete => ProviderToolKind::Delete,
        acp::ToolKind::Move => ProviderToolKind::Move,
        acp::ToolKind::Execute => ProviderToolKind::Execute,
        acp::ToolKind::Fetch => ProviderToolKind::Fetch,
        acp::ToolKind::Think => ProviderToolKind::Think,
        acp::ToolKind::SwitchMode => ProviderToolKind::SwitchMode,
        _ => ProviderToolKind::Other,
    }
}

fn provider_tool_status(status: acp::ToolCallStatus) -> ProviderToolStatus {
    match status {
        acp::ToolCallStatus::Pending => ProviderToolStatus::Pending,
        acp::ToolCallStatus::InProgress => ProviderToolStatus::InProgress,
        acp::ToolCallStatus::Completed => ProviderToolStatus::Completed,
        acp::ToolCallStatus::Failed => ProviderToolStatus::Failed,
        _ => ProviderToolStatus::Other,
    }
}

fn text_from_acp_content(content: acp::ContentBlock) -> Option<String> {
    match content {
        acp::ContentBlock::Text(text) => Some(text.text),
        _ => None,
    }
}

fn is_retryable_error(error_str: &str) -> bool {
    jcode_provider_core::is_transient_transport_error(error_str)
        || error_str.contains("500 internal server error")
        || error_str.contains("502 bad gateway")
        || error_str.contains("503 service unavailable")
        || error_str.contains("504 gateway timeout")
        || error_str.contains("overloaded")
        || error_str.contains("429 too many requests")
        || error_str.contains("rate limit")
        || error_str.contains("rate_limit")
        || error_str.contains("stream error")
        || error_str.contains("stream read timeout")
}

#[async_trait]
impl Provider for CopilotApiProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        system: &str,
        resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        self.wait_for_init().await;

        if matches!(&self.backend, CopilotBackend::OfficialCli { .. }) {
            let working_dir =
                std::env::current_dir().context("Failed to determine working directory")?;
            return self
                .complete_official(
                    messages,
                    tools,
                    system,
                    resume_session_id,
                    working_dir,
                    None,
                )
                .await;
        }

        self.get_bearer_token().await.map_err(|e| {
            jcode_base::logging::warn(&format!("Copilot bearer token acquisition failed: {}", e,));
            e
        })?;

        let is_user_initiated = self.is_user_initiated(messages);
        if is_user_initiated {
            self.user_turn_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let model_for_fingerprint = self.model();
        let uses_responses_api = copilot_model_uses_responses_api(&model_for_fingerprint);
        let (canonical_payload, fingerprint_input, system_value, built_tools) =
            if uses_responses_api {
                let input = jcode_provider_openai::build_responses_input(messages);
                let tools = jcode_provider_openai::build_tools(tools);
                let mut payload = json!({
                    "model": &model_for_fingerprint,
                    "input": &input,
                    "stream": true,
                    "max_output_tokens": 32_768u32,
                });
                if !system.is_empty() {
                    payload["instructions"] = json!(system);
                }
                if !tools.is_empty() {
                    payload["tools"] = json!(&tools);
                }
                (
                    payload,
                    input,
                    (!system.is_empty()).then(|| json!(system)),
                    tools,
                )
            } else {
                let built_messages = Self::build_messages(system, messages);
                let tools = Self::build_tools(tools);
                let mut payload = json!({
                    "model": &model_for_fingerprint,
                    "messages": &built_messages,
                    "stream": true,
                });
                Self::add_max_token_parameter(&mut payload, &model_for_fingerprint, 32_768u32);
                self.add_reasoning_effort_parameter(&mut payload, &model_for_fingerprint);
                if !tools.is_empty() {
                    payload["tools"] = json!(&tools);
                }
                let system_value = built_messages
                    .first()
                    .filter(|message| {
                        message.get("role").and_then(|role| role.as_str()) == Some("system")
                    })
                    .cloned();
                (payload, built_messages, system_value, tools)
            };
        let tools_value = if built_tools.is_empty() {
            None
        } else {
            Some(Value::Array(built_tools.clone()))
        };
        jcode_provider_core::fingerprint::log_provider_canonical_input(
            "copilot",
            &model_for_fingerprint,
            if uses_responses_api {
                "responses"
            } else {
                "chat_completions"
            },
            &canonical_payload,
            &fingerprint_input,
            system_value.as_ref(),
            tools_value.as_ref(),
            Some(built_tools.len()),
            &[("user_initiated", is_user_initiated.to_string())],
        );

        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>(100);

        let provider = CopilotApiProvider {
            backend: self.backend.clone(),
            model: self.model.clone(),
            fetched_models: self.fetched_models.clone(),
            catalog_source: self.catalog_source.clone(),
            init_ready: self.init_ready.clone(),
            init_done: self.init_done.clone(),
            premium_mode: self.premium_mode.clone(),
            user_turn_count: self.user_turn_count.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            official_runtime: self.official_runtime.clone(),
            created_at: self.created_at,
        };

        tokio::spawn(async move {
            provider
                .stream_request(canonical_payload, uses_responses_api, is_user_initiated, tx)
                .await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn complete_split_with_context(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        system_static: &str,
        system_dynamic: &str,
        resume_session_id: Option<&str>,
        request_context: &ProviderRequestContext,
    ) -> Result<EventStream> {
        let dynamic_messages = messages_with_dynamic_system_context(messages, system_dynamic);
        if matches!(&self.backend, CopilotBackend::OfficialCli { .. }) {
            let working_dir = request_context.working_dir.clone().ok_or_else(|| {
                anyhow!("Official Copilot CLI requests require a session working directory")
            })?;
            let effective_system = match (system_static.trim(), system_dynamic.trim()) {
                ("", "") => String::new(),
                (static_part, "") => static_part.to_string(),
                ("", dynamic_part) => dynamic_part.to_string(),
                (static_part, dynamic_part) => format!("{static_part}\n\n{dynamic_part}"),
            };
            let official_messages = if resume_session_id.is_some() {
                dynamic_messages.as_slice()
            } else {
                messages
            };
            return self
                .complete_official(
                    official_messages,
                    tools,
                    &effective_system,
                    resume_session_id,
                    working_dir,
                    request_context.current_user_prompt.as_deref(),
                )
                .await;
        }
        self.complete(&dynamic_messages, tools, system_static, resume_session_id)
            .await
    }

    fn name(&self) -> &str {
        "copilot"
    }

    fn model(&self) -> String {
        self.model
            .try_read()
            .map(|m| m.clone())
            .unwrap_or_else(|_| DEFAULT_MODEL.to_string())
    }

    fn set_model(&self, model: &str) -> Result<()> {
        // See `strip_own_model_prefix`: `--provider copilot` routes through this
        // runtime directly, so session restore hands it `copilot:<model>`.
        let trimmed = jcode_provider_core::strip_own_model_prefix(model, "copilot:");
        if trimmed.is_empty() {
            anyhow::bail!("Copilot model cannot be empty");
        }
        if trimmed.contains("[1m]") {
            anyhow::bail!(
                "1M context window models are not supported via Copilot. Use the Anthropic API directly."
            );
        }
        if let Ok(mut current) = self.model.try_write() {
            *current = trimmed.to_string();
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Cannot change model while a request is in progress"
            ))
        }
    }

    fn available_models(&self) -> Vec<&'static str> {
        FALLBACK_MODELS.to_vec()
    }

    fn available_models_display(&self) -> Vec<String> {
        if let Ok(models) = self.fetched_models.read()
            && !models.is_empty()
        {
            return models.clone();
        }
        FALLBACK_MODELS
            .iter()
            .map(|model| model.to_string())
            .collect()
    }

    fn available_models_for_switching(&self) -> Vec<String> {
        self.available_models_display()
    }

    async fn prefetch_models(&self) -> Result<()> {
        if matches!(self.backend.transport(), CopilotTransport::OfficialCli) {
            let discovered = self.discover_official_models().await?;
            self.update_official_models(discovered);
            return Ok(());
        }
        let grace_ms = Self::startup_prefetch_grace_ms();
        if self.created_at.elapsed().as_millis() < u128::from(grace_ms) {
            jcode_base::logging::info(&format!(
                "Skipping Copilot model prefetch during startup grace window ({}ms)",
                grace_ms
            ));
            return Ok(());
        }
        self.detect_tier_and_set_default().await;
        Ok(())
    }

    fn model_switch_session_key(&self) -> Option<&'static str> {
        matches!(&self.backend, CopilotBackend::OfficialCli { .. })
            .then_some("copilot:official-cli")
    }

    fn supports_conversation_rewind(&self) -> bool {
        !matches!(&self.backend, CopilotBackend::OfficialCli { .. })
    }

    fn supports_compaction(&self) -> bool {
        matches!(self.backend.transport(), CopilotTransport::Native)
    }

    fn model_catalog_detail(&self) -> String {
        self.model_catalog_detail_impl()
    }

    fn set_premium_mode(&self, mode: PremiumMode) {
        CopilotApiProvider::set_premium_mode(self, mode);
    }

    fn premium_mode(&self) -> PremiumMode {
        CopilotApiProvider::get_premium_mode(self)
    }

    fn context_window(&self) -> usize {
        jcode_provider_core::context_limit_for_model_with_provider(&self.model(), Some(self.name()))
            .unwrap_or(128_000)
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.fork_for_session())
    }

    fn reasoning_effort(&self) -> Option<String> {
        if matches!(self.backend.transport(), CopilotTransport::OfficialCli) {
            return None;
        }
        if !copilot_model_supports_reasoning_effort(&self.model()) {
            return None;
        }
        self.current_reasoning_effort()
    }

    fn set_reasoning_effort(&self, effort: &str) -> Result<()> {
        if matches!(self.backend.transport(), CopilotTransport::OfficialCli) {
            bail!("Reasoning effort selection is not supported by the official-cli ACP transport");
        }
        let model = self.model();
        if !copilot_model_supports_reasoning_effort(&model) {
            anyhow::bail!(
                "Reasoning effort is not supported for Copilot model '{}' (only claude-sonnet-5)",
                model
            );
        }
        let normalized = effort.trim().to_lowercase();
        if !SONNET5_EFFORTS.contains(&normalized.as_str()) {
            anyhow::bail!(
                "Unsupported reasoning effort '{}' for Copilot claude-sonnet-5. Supported: {}",
                effort,
                SONNET5_EFFORTS.join(", ")
            );
        }
        let mut guard = self
            .reasoning_effort
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = Some(normalized);
        Ok(())
    }

    fn available_efforts(&self) -> Vec<&'static str> {
        if matches!(self.backend.transport(), CopilotTransport::OfficialCli) {
            return vec![];
        }
        if copilot_model_supports_reasoning_effort(&self.model()) {
            SONNET5_EFFORTS.to_vec()
        } else {
            vec![]
        }
    }

    fn active_auth_method_label(&self) -> Option<&'static str> {
        match self.backend.transport() {
            CopilotTransport::Native => Some("GitHub OAuth token exchange"),
            CopilotTransport::OfficialCli => Some("Official Copilot CLI"),
        }
    }

    fn handles_tools_internally(&self) -> bool {
        matches!(self.backend.transport(), CopilotTransport::OfficialCli)
    }

    fn transport(&self) -> Option<String> {
        Some(self.backend.transport().as_str().to_string())
    }

    fn set_transport(&self, transport: &str) -> Result<()> {
        let requested = CopilotTransport::parse(Some(transport)).map_err(anyhow::Error::msg)?;
        let current = self.backend.transport();
        if requested == current {
            return Ok(());
        }
        bail!(
            "Copilot transport is fixed at construction ({}); set JCODE_COPILOT_TRANSPORT={} and restart jcode",
            current.as_str(),
            requested.as_str()
        )
    }

    fn available_transports(&self) -> Vec<&'static str> {
        vec!["native", "official-cli"]
    }
}

#[cfg(test)]
#[path = "copilot_tests.rs"]
mod tests;
