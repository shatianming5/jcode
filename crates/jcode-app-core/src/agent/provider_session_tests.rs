use super::*;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct OpaqueSessionProvider {
    model: Arc<Mutex<String>>,
}

#[async_trait]
impl Provider for OpaqueSessionProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<crate::provider::EventStream> {
        unreachable!("provider session tests do not complete requests")
    }

    fn name(&self) -> &str {
        "copilot"
    }

    fn model(&self) -> String {
        self.model.lock().unwrap().clone()
    }

    fn set_model(&self, model: &str) -> Result<()> {
        *self.model.lock().unwrap() = model.to_string();
        Ok(())
    }

    fn model_switch_session_key(&self) -> Option<&'static str> {
        Some("copilot:official-cli")
    }

    fn supports_conversation_rewind(&self) -> bool {
        false
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

async fn agent_with_provider_session(
    provider: Arc<dyn Provider>,
    local_session_id: &str,
    provider_session_id: &str,
) -> Agent {
    let registry = Registry::new(Arc::clone(&provider)).await;
    let mut session = Session::create_with_id(local_session_id.to_string(), None, None);
    session.model = Some(provider.model());
    let mut agent = Agent::new_with_session(provider, registry, session, None);
    agent.provider_session_id = Some(provider_session_id.to_string());
    agent.session.provider_session_id = Some(provider_session_id.to_string());
    agent
}

#[tokio::test]
async fn attached_agent_restores_persisted_provider_session_id() {
    let provider: Arc<dyn Provider> = Arc::new(OpaqueSessionProvider {
        model: Arc::new(Mutex::new("claude-sonnet-4.6".to_string())),
    });
    let registry = Registry::new(Arc::clone(&provider)).await;
    let mut session = Session::create_with_id("local-restored".to_string(), None, None);
    session.provider_session_id = Some("upstream-restored".to_string());

    let agent = Agent::new_with_session(provider, registry, session, None);

    assert_eq!(
        agent.provider_session_id.as_deref(),
        Some("upstream-restored")
    );
    assert_eq!(
        agent.session.provider_session_id.as_deref(),
        Some("upstream-restored")
    );
}

#[tokio::test]
async fn official_copilot_model_switch_preserves_each_local_sessions_upstream_id() {
    let provider: Arc<dyn Provider> = Arc::new(OpaqueSessionProvider {
        model: Arc::new(Mutex::new("claude-sonnet-4.6".to_string())),
    });
    let mut first =
        agent_with_provider_session(Arc::clone(&provider), "local-one", "upstream-one").await;
    let second =
        agent_with_provider_session(Arc::clone(&provider), "local-two", "upstream-two").await;

    let previous_key = first.model_switch_session_key();
    first.set_model("gpt-5-mini").unwrap();
    assert_eq!(
        first.reconcile_provider_session_after_model_switch(previous_key),
        "preserved"
    );

    assert_eq!(first.provider_session_id.as_deref(), Some("upstream-one"));
    assert_eq!(
        first.session.provider_session_id.as_deref(),
        Some("upstream-one")
    );
    assert_eq!(second.provider_session_id.as_deref(), Some("upstream-two"));
    assert_eq!(
        second.session.provider_session_id.as_deref(),
        Some("upstream-two")
    );
}

#[tokio::test]
async fn opaque_official_session_rewind_is_rejected_before_local_history_changes() {
    let provider: Arc<dyn Provider> = Arc::new(OpaqueSessionProvider {
        model: Arc::new(Mutex::new("claude-sonnet-4.6".to_string())),
    });
    let mut agent = agent_with_provider_session(provider, "local-rewind", "upstream-rewind").await;
    agent
        .session
        .add_message(Role::User, Message::user("first").content);
    agent
        .session
        .add_message(Role::Assistant, Message::assistant_text("answer").content);
    agent
        .session
        .add_message(Role::User, Message::user("second").content);
    let before_len = agent.messages().len();

    let error = agent.rewind_to_message(1).unwrap_err();

    assert!(error.contains("official Copilot CLI transport"), "{error}");
    assert_eq!(agent.messages().len(), before_len);
    assert_eq!(
        agent.provider_session_id.as_deref(),
        Some("upstream-rewind")
    );
    assert_eq!(
        agent.session.provider_session_id.as_deref(),
        Some("upstream-rewind")
    );
    assert!(agent.rewind_undo_snapshot.is_none());
}
