//! Cross-crate integration: raptor-context indexes a fixture repository, and the gateway's
//! `/raptor/context/build` route serves it through this shell's `ContextService` adapter.
//! Runs fully offline: no GPU, no llama-server, no network.

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use raptor_context::{ContextEngine, EngineDeps, NoEmbedder, NoGit};
use raptor_core::{DataDirs, EstimatingCounter, RaptorError, Result};
use raptor_gateway::{
    ApiKeyRecord, Completion, ContextService, Endpoint, GatewayConfig, GatewayDeps, ModelCatalog,
    Upstream,
};
use raptor_lib::adapters::{ContextAdapter, ContextHub, DepsFactory};
use raptor_lib::settings::{Settings, SettingsStore};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

const MARKER: &str = "PROTECTED-RULE-7: never bypass the refund ledger.";

/// Upstream that must never be reached by a context-only request.
struct NoUpstream;

#[async_trait]
impl Upstream for NoUpstream {
    async fn models(&self) -> Result<ModelCatalog> {
        Ok(ModelCatalog::default())
    }
    async fn call(&self, _endpoint: Endpoint, _body: Value) -> Result<Completion> {
        Err(RaptorError::Conflict("no model in this test".into()))
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn fixture_repo(root: &Path) {
    write(
        &root.join("AGENTS.md"),
        &format!("{MARKER}\nRun the billing tests before merging.\n"),
    );
    write(
        &root.join("src/billing.rs"),
        "use crate::ledger::record;\n\npub fn issue_refund(amount: u64) -> u64 {\n    record(amount);\n    amount\n}\n",
    );
    write(
        &root.join("src/ledger.rs"),
        "pub fn record(amount: u64) {\n    let _ = amount;\n}\n",
    );
    write(
        &root.join("tests/billing_test.rs"),
        "use repo::billing::issue_refund;\n\n#[test]\nfn refund_is_returned() {\n    assert_eq!(issue_refund(5), 5);\n}\n",
    );
}

fn deps() -> EngineDeps {
    EngineDeps {
        counter: Arc::new(EstimatingCounter),
        git: Box::new(NoGit),
        embedder: Arc::new(NoEmbedder),
        summarizer: None,
    }
}

#[tokio::test]
async fn gateway_context_route_keeps_protected_instructions_under_a_tiny_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let repo: PathBuf = tmp.path().join("repo");
    fixture_repo(&repo);
    let dirs = DataDirs::new(tmp.path().join("RaptorLLM"));
    dirs.ensure().unwrap();

    // Index the fixture through raptor-context directly, as the shell does on first use.
    let mut engine = ContextEngine::open(&dirs, &repo, deps()).unwrap();
    let report = engine.update().unwrap();
    assert!(
        report.scanned >= 3,
        "fixture files must be scanned: {report:?}"
    );
    assert!(report.reparsed.iter().any(|p| p.ends_with("billing.rs")));
    drop(engine);

    // Register the workspace, as the user would, and serve it through the gateway.
    let absolute = std::path::absolute(&repo).unwrap();
    let settings = Settings {
        workspaces: vec![absolute.clone()],
        ..Settings::default()
    };
    let store = Arc::new(SettingsStore::with_settings(
        tmp.path().join("config/settings.json"),
        settings,
    ));
    let factory: DepsFactory = Arc::new(deps);
    let hub = Arc::new(ContextHub::new(dirs.clone(), store, factory));
    let context: Arc<dyn ContextService> = Arc::new(ContextAdapter::new(hub));

    let (record, secret) = ApiKeyRecord::generate("integration").unwrap();
    let config = GatewayConfig {
        api_keys: vec![record],
        ..GatewayConfig::default()
    };
    let built = raptor_gateway::build(
        &config,
        GatewayDeps {
            upstream: Arc::new(NoUpstream),
            context: Some(context),
        },
    )
    .unwrap();

    let workspace = raptor_lib::adapters::workspace_id(&absolute);
    let body =
        json!({ "workspace": workspace, "query": "refund ledger", "token_budget": 4 }).to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/raptor/context/build")
        .header("host", "127.0.0.1:8790")
        .header("authorization", format!("Bearer {secret}"))
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let response = built.router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let pack: Value = serde_json::from_slice(&bytes).unwrap();

    let text = pack["text"].as_str().expect("text");
    assert!(
        text.contains(MARKER),
        "protected instruction was dropped:\n{text}"
    );
    let over = pack["details"]["over_budget_by"].as_u64().unwrap_or(0);
    assert!(
        over > 0,
        "a budget of 4 tokens cannot hold the protected set; the manifest must say so: {pack}"
    );
    assert!(pack["details"]["included"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(false));
}

#[tokio::test]
async fn unregistered_workspace_is_refused_by_the_gateway_route() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join("repo");
    fixture_repo(&repo);
    let dirs = DataDirs::new(tmp.path().join("RaptorLLM"));
    dirs.ensure().unwrap();
    let store = Arc::new(SettingsStore::with_settings(
        tmp.path().join("settings.json"),
        Settings::default(),
    ));
    let factory: DepsFactory = Arc::new(deps);
    let context: Arc<dyn ContextService> = Arc::new(ContextAdapter::new(Arc::new(
        ContextHub::new(dirs, store, factory),
    )));

    let (record, secret) = ApiKeyRecord::generate("integration").unwrap();
    let config = GatewayConfig {
        api_keys: vec![record],
        ..GatewayConfig::default()
    };
    let built = raptor_gateway::build(
        &config,
        GatewayDeps {
            upstream: Arc::new(NoUpstream),
            context: Some(context),
        },
    )
    .unwrap();

    let body = json!({ "workspace": "ws-0123456789abcdef", "query": "refund" }).to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/raptor/context/build")
        .header("host", "127.0.0.1:8790")
        .header("authorization", format!("Bearer {secret}"))
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let response = built.router.oneshot(request).await.unwrap();
    assert!(
        !response.status().is_success(),
        "unregistered workspace must not be served"
    );
}
