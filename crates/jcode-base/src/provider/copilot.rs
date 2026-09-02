//! Copilot pure model-catalog data (compatibility shim).
//!
//! The GitHub Copilot provider *runtime* (`CopilotApiProvider`) now lives in
//! the downstream `jcode-provider-copilot-runtime` crate so provider edits do
//! not rebuild the base -> app-core -> tui spine. The binary's composition
//! root registers it via [`crate::provider::external`]. Base keeps only the
//! pure model-catalog data (from `jcode-provider-copilot`) that its routing
//! logic needs, plus a credentials probe that delegates to auth.

pub use jcode_provider_copilot::{
    CopilotTransport, DEFAULT_MODEL, FALLBACK_MODELS, is_known_display_model,
    official_cli_path_from_env,
};
pub use jcode_provider_core::PremiumMode;

pub fn selected_transport() -> anyhow::Result<CopilotTransport> {
    CopilotTransport::from_env().map_err(anyhow::Error::msg)
}

pub fn official_cli_selected() -> bool {
    matches!(selected_transport(), Ok(CopilotTransport::OfficialCli))
}

pub fn official_cli_configured() -> bool {
    official_cli_selected() && official_cli_path_from_env().is_ok()
}

/// Whether GitHub Copilot credentials are present (GitHub OAuth token).
///
/// Kept here (not only in `auth::copilot`) because provider routing has
/// historically probed credentials through the provider module.
pub fn has_credentials() -> bool {
    crate::auth::copilot::has_copilot_credentials()
}

pub fn is_configured() -> bool {
    official_cli_configured() || has_credentials()
}

pub fn unavailable_message() -> String {
    match selected_transport() {
        Ok(CopilotTransport::OfficialCli) => {
            "GitHub Copilot official-cli transport is unavailable. Check JCODE_COPILOT_CLI_PATH and the official CLI ACP handshake.".to_string()
        }
        Ok(CopilotTransport::Native) => {
            "GitHub Copilot credentials not available. Run `jcode login --provider copilot` first."
                .to_string()
        }
        Err(error) => error.to_string(),
    }
}
