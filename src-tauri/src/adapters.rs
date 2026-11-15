//! Adapters that connect the subsystem crates to the gateway and to each other.
//!
//! - [`RuntimeUpstream`] implements the gateway's `Upstream` by forwarding to whichever
//!   llama-server the runtime currently reports as ready.
//! - [`ContextAdapter`] implements the gateway's `ContextService` over raptor-context. It
//!   only serves workspaces the user registered in settings, so a gateway client can
//!   never point the indexer at an arbitrary path.

use crate::semantic::{SemanticSetup, ServerTextGenerator};
use crate::settings::SettingsStore;
use crate::tokens::ServerAwareCounter;
use async_trait::async_trait;
use futures_util::StreamExt;
use raptor_context::{
    ContextEngine, ContextRequest, EngineDeps, ModelSummarizer, RecoverRequest, SearchRequest,
    SemanticStatus, SymbolHit, SymbolQuery,
};
use raptor_core::{ChatMessage, DataDirs, RaptorError, Result, Role};
use raptor_gateway::context_api as gw;
use raptor_gateway::{
    Completion, ContextPack, ContextService, Endpoint, HttpUpstream, ModelCatalog, Upstream,
};
use raptor_models::ModelManager;
use raptor_runtime::process::ServerState;
use raptor_runtime::tokenizer::TokenizeTransport;
use raptor_runtime::RuntimeEngine;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Reports the base URL of the llama-server that can currently take requests.
pub trait ServerProbe: Send + Sync {
    /// `Some` only while the server is ready or degraded-but-serving.
    fn endpoint(&self) -> Option<String>;
}

/// Lists the model ids installed in the library.
pub trait InstalledModels: Send + Sync {
    fn installed_ids(&self) -> Result<Vec<String>>;
}

impl ServerProbe for RuntimeEngine {
    fn endpoint(&self) -> Option<String> {
        let status = self.status();
        match status.server.state {
            ServerState::Ready | ServerState::Degraded { .. } => status.server.endpoint,
            _ => None,
        }
    }
}

impl InstalledModels for ModelManager {
    fn installed_ids(&self) -> Result<Vec<String>> {
        Ok(self.library().list()?.into_iter().map(|e| e.id).collect())
    }
}

/// Builds the upstream client for one server endpoint.
pub type Connect = Arc<dyn Fn(&str) -> Result<Arc<dyn Upstream>> + Send + Sync>;

/// `Upstream` over the runtime's current llama-server. The HTTP client is rebuilt only
/// when the server endpoint changes.
pub struct RuntimeUpstream {
    server: Arc<dyn ServerProbe>,
    library: Arc<dyn InstalledModels>,
    connect: Connect,
    cache: Mutex<Option<(String, Arc<dyn Upstream>)>>,
}

impl RuntimeUpstream {
    pub fn new(
        server: Arc<dyn ServerProbe>,
        library: Arc<dyn InstalledModels>,
        connect: Connect,
    ) -> Self {
        Self {
            server,
            library,
            connect,
            cache: Mutex::new(None),
        }
    }

    /// Forgets the client for the current server, and with it what it remembered about the model
    /// (`/props`). Called after a load, because a new model can be served at the same endpoint.
    pub fn forget_server_facts(&self) {
        *lock(&self.cache) = None;
    }

    /// Production wiring: loopback `HttpUpstream` per endpoint.
    pub fn loopback(server: Arc<dyn ServerProbe>, library: Arc<dyn InstalledModels>) -> Self {
        Self::new(server, library, loopback_connect())
    }
}

/// Connector that talks HTTP to a loopback llama-server. `HttpUpstream` refuses other hosts.
pub fn loopback_connect() -> Connect {
    Arc::new(|endpoint: &str| {
        HttpUpstream::new(endpoint, Vec::new()).map(|u| Arc::new(u) as Arc<dyn Upstream>)
    })
}

impl RuntimeUpstream {
    /// Returns the client for the server that is ready now, if any.
    fn active(&self) -> Result<Option<Arc<dyn Upstream>>> {
        let Some(endpoint) = self.server.endpoint() else {
            return Ok(None);
        };
        let mut cache = lock(&self.cache);
        if let Some((cached, up)) = cache.as_ref() {
            if *cached == endpoint {
                return Ok(Some(up.clone()));
            }
        }
        let up = (self.connect)(&endpoint)?;
        *cache = Some((endpoint, up.clone()));
        Ok(Some(up))
    }
}

/// llama-server names the model it serves by the file path it was started with. That path must
/// not reach API clients (it would expose the user's folders over LAN), so it is mapped to the
/// library id with the same file name. `None` when the name is a path with no library match.
/// A name that is not path-like passes through unchanged.
fn public_model_name(installed: &[String], reported: &str) -> Option<String> {
    if !reported.contains(['/', '\\']) || installed.iter().any(|id| id == reported) {
        return Some(reported.to_string());
    }
    let file = reported.rsplit(['/', '\\']).next().unwrap_or(reported);
    installed
        .iter()
        .find(|id| id.rsplit('/').next() == Some(file))
        .cloned()
}

/// Replaces a path-like top-level `model` field of a reply with its public name.
fn scrub_model_field(installed: &[String], body: &mut Value) {
    let Some(reported) = body.get("model").and_then(Value::as_str) else {
        return;
    };
    if let Some(public) = public_model_name(installed, reported) {
        body["model"] = Value::String(public);
    } else {
        body["model"] = Value::String("local".into());
    }
}

#[async_trait]
impl Upstream for RuntimeUpstream {
    async fn models(&self) -> Result<ModelCatalog> {
        let installed = self.library.installed_ids()?;
        let loaded = match self.active()? {
            Some(up) => up
                .models()
                .await?
                .loaded
                .and_then(|name| public_model_name(&installed, &name)),
            None => None,
        };
        Ok(ModelCatalog { installed, loaded })
    }

    async fn thinking_caps(&self) -> raptor_gateway::ThinkingCaps {
        match self.active() {
            Ok(Some(up)) => up.thinking_caps().await,
            _ => raptor_gateway::ThinkingCaps::default(),
        }
    }

    async fn call(&self, endpoint: Endpoint, body: Value) -> Result<Completion> {
        let Some(up) = self.active()? else {
            return Err(RaptorError::Conflict(
                "no model is loaded; load one from the Models page first".into(),
            ));
        };
        let installed = self.library.installed_ids()?;
        let started = Instant::now();
        Ok(match up.call(endpoint, body).await? {
            Completion::Json(mut v) => {
                scrub_model_field(&installed, &mut v);
                attach_speed(&mut v, None);
                Completion::Json(v)
            }
            Completion::Stream(s) => {
                let first_token_ms: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
                Completion::Stream(Box::pin(s.map(move |item| {
                    item.map(|payload| match serde_json::from_str::<Value>(&payload) {
                        Ok(mut v) => {
                            if v.get("model").is_some() {
                                scrub_model_field(&installed, &mut v);
                            }
                            let mut first = lock(&first_token_ms);
                            if first.is_none() && has_generated_text(&v) {
                                *first = Some(started.elapsed().as_secs_f64() * 1000.0);
                            }
                            attach_speed(&mut v, *first);
                            v.to_string()
                        }
                        Err(_) => payload,
                    })
                })))
            }
        })
    }
}

/// True when a stream chunk carries answer or reasoning text (the first one is the first token).
fn has_generated_text(chunk: &Value) -> bool {
    let delta = &chunk["choices"][0]["delta"];
    ["content", "reasoning_content"]
        .iter()
        .any(|k| delta[*k].as_str().is_some_and(|t| !t.is_empty()))
}

/// Adds a `raptor` object with this request's speed to a reply that carries llama-server's
/// `timings` (a complete reply, or the last stream chunk): `tokens_per_second` is generation
/// speed (reasoning tokens included), `time_to_first_token_ms` is measured by the gateway for
/// streams and derived from the timings (prompt processing plus one token) otherwise.
/// Untouched when there are no timings, so nothing is ever invented.
pub fn attach_speed(body: &mut Value, measured_first_token_ms: Option<f64>) {
    let Some(t) = body.get("timings") else {
        return;
    };
    let num = |k: &str| t.get(k).and_then(Value::as_f64).filter(|n| n.is_finite());
    let derived_ttft = match (num("prompt_ms"), num("predicted_per_token_ms")) {
        (Some(p), Some(per)) => Some(p + per),
        (Some(p), None) => Some(p),
        _ => None,
    };
    let mut raptor = serde_json::Map::new();
    if let Some(v) = num("predicted_per_second") {
        raptor.insert(
            "tokens_per_second".into(),
            json!((v * 100.0).round() / 100.0),
        );
    }
    if let Some(v) = num("prompt_per_second") {
        raptor.insert(
            "prompt_tokens_per_second".into(),
            json!((v * 100.0).round() / 100.0),
        );
    }
    if let Some(v) = measured_first_token_ms.or(derived_ttft) {
        raptor.insert("time_to_first_token_ms".into(), json!(v.round()));
    }
    if let Some(n) = t.get("predicted_n").and_then(Value::as_u64) {
        raptor.insert("generated_tokens".into(), json!(n));
    }
    if !raptor.is_empty() {
        body["raptor"] = Value::Object(raptor);
    }
}

/// Stable workspace id used by the gateway: `ws-` plus 16 hex digits of SHA-256 over the
/// absolute path.
pub fn workspace_id(root: &Path) -> String {
    let digest = Sha256::digest(root.display().to_string().as_bytes());
    format!("ws-{}", &hex::encode(digest)[..16])
}

/// Builds the context engine for a deps set. Production uses the estimating counter and
/// the git CLI; tests pass fakes.
pub type DepsFactory = Arc<dyn Fn() -> EngineDeps + Send + Sync>;

/// Files that carry protected instructions, in priority order.
pub const INSTRUCTION_FILES: [&str; 3] = ["RAPTOR.md", "AGENTS.md", "CLAUDE.md"];
const MAX_INSTRUCTION_BYTES: u64 = 64 * 1024;

/// Reads the first instruction file present at the workspace root. These are placed in
/// the protected category, so a tight budget never drops them.
pub fn read_instructions(root: &Path) -> Result<Option<String>> {
    for name in INSTRUCTION_FILES {
        let path = root.join(name);
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        if meta.len() > MAX_INSTRUCTION_BYTES {
            return Err(RaptorError::InvalidInput(format!(
                "{name} is larger than {} KiB and cannot be used as instructions",
                MAX_INSTRUCTION_BYTES / 1024
            )));
        }
        return Ok(Some(std::fs::read_to_string(&path)?));
    }
    Ok(None)
}

/// Semantic retrieval for one workspace, as the UI shows it (`context_semantic_status`).
#[derive(Debug, Clone, Serialize)]
pub struct SemanticView {
    /// The embedding model in settings, or null when none is configured.
    pub configured_model: Option<String>,
    /// Why semantic retrieval is off for this workspace; null when it is active or unconfigured.
    pub reason: Option<String>,
    /// The engine's own view: vectors stored, chunks without a vector, last backend error.
    pub status: SemanticStatus,
}

/// What the semantic backend of one engine was last set to, and why it is not active.
#[derive(Debug, Clone)]
struct SemanticApplied {
    /// Settings and server identity the backend was built for. A change re-applies it.
    key: String,
    /// Why semantic retrieval is off for this workspace, when it is.
    reason: Option<String>,
}

/// Open context engines, one per workspace root, plus the registry lookup.
pub struct ContextHub {
    dirs: DataDirs,
    settings: Arc<SettingsStore>,
    deps: DepsFactory,
    engines: Mutex<HashMap<PathBuf, Arc<Mutex<ContextEngine>>>>,
    /// Present when the hub is connected to a llama-server (embeddings and summaries).
    semantic: Option<Arc<SemanticSetup>>,
    applied: Mutex<HashMap<PathBuf, SemanticApplied>>,
}

impl ContextHub {
    pub fn new(dirs: DataDirs, settings: Arc<SettingsStore>, deps: DepsFactory) -> Self {
        Self {
            dirs,
            settings,
            deps,
            engines: Mutex::new(HashMap::new()),
            semantic: None,
            applied: Mutex::new(HashMap::new()),
        }
    }

    /// Production hub connected to the llama-server: the summarizer writes through the loaded
    /// model, and the embedding backend is set per workspace when the configured embedding
    /// model is the one loaded (see [`ContextHub::engine`]).
    pub fn connected(
        dirs: DataDirs,
        settings: Arc<SettingsStore>,
        server: Arc<dyn ServerProbe>,
        transport: Arc<dyn TokenizeTransport>,
        semantic: Arc<SemanticSetup>,
    ) -> Self {
        let summary_source = semantic.clone();
        let deps: DepsFactory = Arc::new(move || {
            let mut deps = EngineDeps::new(Arc::new(ServerAwareCounter::new(
                server.clone(),
                transport.clone(),
            )));
            let generator = ServerTextGenerator::new(
                summary_source.http.clone(),
                summary_source.active.clone(),
            );
            deps.summarizer = Some(Box::new(ModelSummarizer::new(generator)));
            deps
        });
        let mut hub = Self::new(dirs, settings, deps);
        hub.semantic = Some(semantic);
        hub
    }

    /// Makes the engine's embedding backend match the settings and the loaded model. Cheap
    /// when nothing changed. A change re-indexes the vectors; a failure is recorded, not raised.
    fn sync_semantic(&self, root: &Path, engine: &Mutex<ContextEngine>) -> Result<()> {
        let Some(semantic) = &self.semantic else {
            return Ok(());
        };
        let wanted = self.settings.snapshot().embedding_model;
        let active = semantic.active.active();
        let key = format!("{wanted:?}|{active:?}");
        if lock(&self.applied).get(root).is_some_and(|a| a.key == key) {
            return Ok(());
        }
        let (backend, reason) = match &wanted {
            None => (None, None),
            Some(model) => match semantic.embedder_for(model) {
                Ok(b) => (Some(b), None),
                Err(reason) => (None, Some(reason)),
            },
        };
        lock(engine).set_embedding_backend(backend)?;
        lock(&self.applied).insert(root.to_path_buf(), SemanticApplied { key, reason });
        Ok(())
    }

    /// Why semantic retrieval is not active for `root`, after the latest sync. `None` when it
    /// is active, or when no embedding model is configured.
    pub fn semantic_reason(&self, root: &Path) -> Option<String> {
        let key = std::path::absolute(root).ok()?;
        lock(&self.applied).get(&key).and_then(|a| a.reason.clone())
    }

    /// Production deps. Token counts come from the loaded model's tokenizer while the server
    /// is ready, and from the estimate otherwise; the manifest's `tokens_exact` says which.
    pub fn production(
        dirs: DataDirs,
        settings: Arc<SettingsStore>,
        server: Arc<dyn ServerProbe>,
        transport: Arc<dyn TokenizeTransport>,
    ) -> Self {
        let deps: DepsFactory = Arc::new(move || {
            EngineDeps::new(Arc::new(ServerAwareCounter::new(
                server.clone(),
                transport.clone(),
            )))
        });
        Self::new(dirs, settings, deps)
    }

    /// Engine for `root`, opened on first use and kept for the life of the app.
    pub fn engine(&self, root: &Path) -> Result<Arc<Mutex<ContextEngine>>> {
        let key = std::path::absolute(root)?;
        let cached = lock(&self.engines).get(&key).cloned();
        let engine = match cached {
            Some(e) => e,
            None => {
                let opened = Arc::new(Mutex::new(ContextEngine::open(
                    &self.dirs,
                    &key,
                    (self.deps)(),
                )?));
                lock(&self.engines)
                    .entry(key.clone())
                    .or_insert(opened)
                    .clone()
            }
        };
        self.sync_semantic(&key, &engine)?;
        Ok(engine)
    }

    /// Semantic retrieval state for `path`: the configured embedding model, the active
    /// status and the reason it is off. Opens the engine if needed.
    pub fn semantic_status(&self, root: &Path) -> Result<SemanticView> {
        let engine = self.engine(root)?;
        let status = lock(&engine).index_status().semantic;
        Ok(SemanticView {
            configured_model: self.settings.snapshot().embedding_model,
            reason: self.semantic_reason(root),
            status,
        })
    }

    /// Root path of a registered workspace id. Unknown ids are refused.
    pub fn registered_root(&self, id: &str) -> Result<PathBuf> {
        let b = id.as_bytes();
        let well_formed =
            b.len() == 19 && &b[..3] == b"ws-" && b[3..].iter().all(u8::is_ascii_hexdigit);
        if !well_formed {
            return Err(RaptorError::InvalidInput("invalid workspace id".into()));
        }
        self.settings
            .snapshot()
            .workspaces
            .into_iter()
            .find(|p| workspace_id(p) == id)
            .ok_or_else(|| RaptorError::NotFound(format!("workspace {id} is not registered")))
    }

    /// Builds the protected-first context pack for `query`. `budget` is the context budget
    /// in tokens; `None` uses the configured context target.
    pub fn build(&self, root: &Path, query: &str, budget: Option<usize>) -> Result<ContextPack> {
        let instructions = read_instructions(root)?;
        let model_ctx = match budget {
            Some(b) => b,
            None => self.settings.snapshot().context_target as usize,
        };
        let engine = self.engine(root)?;
        let mut eng = lock(&engine);
        let req = ContextRequest {
            task: query.to_string(),
            instructions,
            messages: vec![ChatMessage::new(Role::User, query)],
            model_ctx,
            reserved_output: 0,
            ..Default::default()
        };
        let resp = eng.build_context(req)?;
        let text = resp
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(ContextPack {
            text,
            details: serde_json::to_value(&resp.manifest)?,
        })
    }

    /// Re-indexes changed files, then searches.
    pub fn search(&self, root: &Path, req: &SearchRequest) -> Result<Value> {
        let engine = self.engine(root)?;
        let mut eng = lock(&engine);
        eng.update()?;
        Ok(serde_json::to_value(eng.search(req)?)?)
    }

    /// Redeems a `chunk:` handle, or searches for `query` within a small extra budget.
    pub fn recover(&self, root: &Path, query: &str) -> Result<Value> {
        let req = if query.starts_with("chunk:") {
            RecoverRequest {
                handle: Some(query.to_string()),
                ..Default::default()
            }
        } else {
            RecoverRequest {
                query: Some(query.to_string()),
                extra_budget: 2000,
                ..Default::default()
            }
        };
        let engine = self.engine(root)?;
        let mut eng = lock(&engine);
        Ok(serde_json::to_value(eng.recover(req)?)?)
    }

    /// Definitions named exactly `symbol`, and the impact of each defining file.
    pub fn impact(&self, root: &Path, symbol: &str) -> Result<Value> {
        let engine = self.engine(root)?;
        let eng = lock(&engine);
        let defs: Vec<SymbolHit> = eng
            .symbols(&SymbolQuery {
                name_contains: symbol.to_string(),
                kind: None,
                limit: 200,
            })
            .into_iter()
            .filter(|h| h.name == symbol)
            .collect();
        let mut files: Vec<String> = defs.iter().map(|d| d.path.clone()).collect();
        files.sort();
        files.dedup();
        let reports = files
            .iter()
            .map(|f| eng.impact(f))
            .collect::<Result<Vec<_>>>()?;
        Ok(json!({ "symbol": symbol, "definitions": defs, "impact": reports }))
    }

    /// Imports, dependents and related tests of one workspace-relative file. The path is
    /// normalized to `/` separators, the form the index uses.
    pub fn impact_file(&self, root: &Path, file: &str) -> Result<Value> {
        let normalized = file.replace('\\', "/");
        let engine = self.engine(root)?;
        let eng = lock(&engine);
        let report = eng.impact(&normalized)?;
        Ok(json!({ "file": normalized, "impact": report }))
    }

    /// Symbols whose name contains `name`.
    pub fn symbols(&self, root: &Path, name: &str) -> Result<Value> {
        let engine = self.engine(root)?;
        let eng = lock(&engine);
        let hits = eng.symbols(&SymbolQuery {
            name_contains: name.to_string(),
            kind: None,
            limit: 50,
        });
        Ok(serde_json::to_value(hits)?)
    }

    /// Status of every registered workspace. Only workspaces already opened report an index.
    pub fn status(&self) -> Result<Value> {
        let roots = self.settings.snapshot().workspaces;
        let open = lock(&self.engines);
        let items: Vec<Value> = roots
            .iter()
            .map(|root| {
                let index = open.get(root).map(|e| lock(e).index_status());
                json!({
                    "workspace": workspace_id(root),
                    "root": root.display().to_string(),
                    "opened": index.is_some(),
                    "index": index,
                })
            })
            .collect();
        Ok(json!({ "workspaces": items }))
    }
}

/// Runs blocking engine work off the async runtime.
pub async fn blocking<T, F>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r,
        Err(e) => Err(RaptorError::Internal(format!("context worker failed: {e}"))),
    }
}

/// The gateway's `ContextService`, backed by [`ContextHub`].
pub struct ContextAdapter {
    hub: Arc<ContextHub>,
}

impl ContextAdapter {
    pub fn new(hub: Arc<ContextHub>) -> Self {
        Self { hub }
    }
}

#[async_trait]
impl ContextService for ContextAdapter {
    async fn build(&self, req: gw::BuildRequest) -> Result<ContextPack> {
        req.validate()?;
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        blocking(move || hub.build(&root, &req.query, req.token_budget)).await
    }

    async fn search(&self, req: gw::SearchRequest) -> Result<Value> {
        req.validate()?;
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        let query = SearchRequest {
            query: req.query,
            limit: req.limit.unwrap_or(10),
            ..Default::default()
        };
        blocking(move || hub.search(&root, &query)).await
    }

    async fn recover(&self, req: gw::RecoverRequest) -> Result<Value> {
        req.validate()?;
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        blocking(move || hub.recover(&root, &req.query)).await
    }

    async fn impact(&self, req: gw::ImpactRequest) -> Result<Value> {
        req.validate()?;
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        blocking(move || hub.impact(&root, &req.symbol)).await
    }

    async fn impact_target(&self, req: gw::ImpactQuery) -> Result<Value> {
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        match req.target {
            gw::ImpactTarget::Symbol(symbol) => blocking(move || hub.impact(&root, &symbol)).await,
            gw::ImpactTarget::File(file) => blocking(move || hub.impact_file(&root, &file)).await,
        }
    }

    async fn symbols(&self, req: gw::SymbolsRequest) -> Result<Value> {
        req.validate()?;
        let root = self.hub.registered_root(&req.workspace)?;
        let hub = self.hub.clone();
        blocking(move || hub.symbols(&root, &req.name)).await
    }

    async fn status(&self) -> Result<Value> {
        let hub = self.hub.clone();
        blocking(move || hub.status()).await
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
pub(crate) mod fakes {
    use super::*;
    use futures_util::stream;

    /// Server probe whose endpoint can be changed by the test.
    #[derive(Default)]
    pub struct FakeServer(pub Mutex<Option<String>>);

    impl ServerProbe for FakeServer {
        fn endpoint(&self) -> Option<String> {
            lock(&self.0).clone()
        }
    }

    pub struct FakeLibrary(pub Vec<String>);

    impl InstalledModels for FakeLibrary {
        fn installed_ids(&self) -> Result<Vec<String>> {
            Ok(self.0.clone())
        }
    }

    /// Upstream that records every body it receives. Streamed replies are `streamed` when
    /// set, `pending` never ends (for cancellation tests), and otherwise a short canned reply.
    #[derive(Default)]
    pub struct FakeUpstream {
        pub bodies: Mutex<Vec<Value>>,
        pub loaded: Option<String>,
        pub streamed: Mutex<Option<Vec<Result<String>>>>,
        pub pending: bool,
        /// What the fake model's chat template says about reasoning effort.
        pub caps: raptor_gateway::ThinkingCaps,
    }

    #[async_trait]
    impl Upstream for FakeUpstream {
        async fn thinking_caps(&self) -> raptor_gateway::ThinkingCaps {
            self.caps.clone()
        }

        async fn models(&self) -> Result<ModelCatalog> {
            Ok(ModelCatalog {
                installed: vec![],
                loaded: self.loaded.clone(),
            })
        }

        async fn call(&self, _endpoint: Endpoint, body: Value) -> Result<Completion> {
            let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
            lock(&self.bodies).push(body);
            if !streaming {
                return Ok(Completion::Json(json!({"ok": true})));
            }
            if self.pending {
                return Ok(Completion::Stream(Box::pin(stream::pending())));
            }
            let items = lock(&self.streamed).take().unwrap_or_else(|| {
                vec![
                    Ok("{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}".to_string()),
                    Ok("[DONE]".to_string()),
                ]
            });
            Ok(Completion::Stream(Box::pin(stream::iter(items))))
        }
    }

    /// Counts how many clients were built, so reuse can be asserted.
    pub fn counting_connect(upstream: Arc<FakeUpstream>, count: Arc<Mutex<usize>>) -> Connect {
        Arc::new(move |_endpoint: &str| {
            *lock(&count) += 1;
            Ok(upstream.clone() as Arc<dyn Upstream>)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::*;
    use super::*;
    use crate::settings::Settings;
    use futures_util::StreamExt;
    use raptor_context::NoEmbedder;

    #[test]
    fn speed_is_attached_only_when_the_server_reported_timings() {
        let mut none = json!({"choices": []});
        attach_speed(&mut none, Some(5.0));
        assert!(none.get("raptor").is_none());

        let timings = json!({"prompt_ms": 236.0, "predicted_per_token_ms": 46.8,
            "predicted_per_second": 21.3499, "prompt_per_second": 72.05, "predicted_n": 72});
        let mut derived = json!({"timings": timings.clone()});
        attach_speed(&mut derived, None);
        assert_eq!(derived["raptor"]["tokens_per_second"], 21.35);
        assert_eq!(derived["raptor"]["prompt_tokens_per_second"], 72.05);
        assert_eq!(derived["raptor"]["time_to_first_token_ms"], 283.0);
        assert_eq!(derived["raptor"]["generated_tokens"], 72);

        let mut measured = json!({"timings": timings});
        attach_speed(&mut measured, Some(412.4));
        assert_eq!(measured["raptor"]["time_to_first_token_ms"], 412.0);
    }

    #[test]
    fn file_paths_are_mapped_to_library_ids_and_never_exposed() {
        let installed = vec!["local/a.gguf".to_string(), "acme/repo/b.gguf".to_string()];
        let win = r"D:\AI_Storage\models\b.gguf";
        assert_eq!(
            public_model_name(&installed, win).as_deref(),
            Some("acme/repo/b.gguf")
        );
        assert_eq!(
            public_model_name(&installed, "/home/u/x/a.gguf").as_deref(),
            Some("local/a.gguf")
        );
        // A path with no library match is dropped rather than leaked.
        assert_eq!(public_model_name(&installed, "/home/u/other.gguf"), None);
        // Ids and plain names pass through.
        assert_eq!(
            public_model_name(&installed, "local/a.gguf").as_deref(),
            Some("local/a.gguf")
        );
        assert_eq!(
            public_model_name(&installed, "gpt-like").as_deref(),
            Some("gpt-like")
        );

        let mut body = json!({ "model": win, "x": 1 });
        scrub_model_field(&installed, &mut body);
        assert_eq!(body["model"], "acme/repo/b.gguf");
        let mut unknown = json!({ "model": "/secret/dir/zzz.gguf" });
        scrub_model_field(&installed, &mut unknown);
        assert_eq!(unknown["model"], "local");
    }

    fn setup_upstream(
        loaded: Option<&str>,
    ) -> (
        RuntimeUpstream,
        Arc<FakeServer>,
        Arc<FakeUpstream>,
        Arc<Mutex<usize>>,
    ) {
        let server = Arc::new(FakeServer::default());
        let up = Arc::new(FakeUpstream {
            loaded: loaded.map(str::to_string),
            ..Default::default()
        });
        let count = Arc::new(Mutex::new(0usize));
        let rt = RuntimeUpstream::new(
            server.clone(),
            Arc::new(FakeLibrary(vec!["org/repo/a.gguf".into()])),
            counting_connect(up.clone(), count.clone()),
        );
        (rt, server, up, count)
    }

    #[tokio::test]
    async fn models_lists_installed_and_no_loaded_model_while_server_is_down() {
        let (rt, _server, _up, count) = setup_upstream(Some("ignored"));
        let cat = rt.models().await.unwrap();
        assert_eq!(cat.installed, vec!["org/repo/a.gguf".to_string()]);
        assert_eq!(cat.loaded, None);
        assert_eq!(*lock(&count), 0, "no client is built without a server");
    }

    #[tokio::test]
    async fn call_without_a_server_is_a_conflict_not_a_silent_success() {
        let (rt, _server, up, _) = setup_upstream(None);
        let err = rt
            .call(Endpoint::ChatCompletions, json!({"model": "m"}))
            .await;
        assert!(matches!(err, Err(RaptorError::Conflict(_))));
        assert!(lock(&up.bodies).is_empty());
    }

    #[tokio::test]
    async fn forwards_to_the_current_endpoint_and_reuses_the_client() {
        let (rt, server, up, count) = setup_upstream(Some("m.gguf"));
        *lock(&server.0) = Some("http://127.0.0.1:8081".into());
        let out = rt
            .call(
                Endpoint::ChatCompletions,
                json!({"model": "m.gguf", "messages": []}),
            )
            .await
            .unwrap();
        assert!(matches!(out, Completion::Json(_)));
        rt.call(
            Endpoint::ChatCompletions,
            json!({"model": "m.gguf", "stream": true}),
        )
        .await
        .unwrap();
        assert_eq!(lock(&up.bodies).len(), 2);
        assert_eq!(*lock(&count), 1, "same endpoint must reuse one client");
        let cat = rt.models().await.unwrap();
        assert_eq!(cat.loaded.as_deref(), Some("m.gguf"));
    }

    #[tokio::test]
    async fn rebuilds_the_client_when_the_endpoint_changes() {
        let (rt, server, _up, count) = setup_upstream(Some("m"));
        *lock(&server.0) = Some("http://127.0.0.1:8081".into());
        rt.models().await.unwrap();
        *lock(&server.0) = Some("http://127.0.0.1:8082".into());
        rt.models().await.unwrap();
        assert_eq!(*lock(&count), 2);
    }

    #[tokio::test]
    async fn streamed_payloads_pass_through_in_order() {
        let (rt, server, _up, _) = setup_upstream(Some("m"));
        *lock(&server.0) = Some("http://127.0.0.1:8081".into());
        let Completion::Stream(mut s) = rt
            .call(Endpoint::ChatCompletions, json!({"stream": true}))
            .await
            .unwrap()
        else {
            panic!("expected a stream");
        };
        let first = s.next().await.unwrap().unwrap();
        assert!(first.contains("hi"));
        assert_eq!(s.next().await.unwrap().unwrap(), "[DONE]");
    }

    #[test]
    fn loopback_connect_refuses_remote_hosts() {
        // The production connector must never point at a non-loopback llama-server.
        let remote = HttpUpstream::new("http://10.0.0.5:8081", vec![]);
        assert!(remote.is_err());
        let local = HttpUpstream::new("http://127.0.0.1:8081", vec![]);
        assert!(local.is_ok());
    }

    fn fixture_repo(root: &Path) {
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("AGENTS.md"),
            "Protected rule: never touch the refund ledger without a test.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/billing.rs"),
            "pub fn issue_refund(amount: u64) -> u64 {\n    amount\n}\n",
        )
        .unwrap();
    }

    fn hub_with(
        dirs: &DataDirs,
        settings_dir: &Path,
        repo: &Path,
        register: bool,
    ) -> Arc<ContextHub> {
        let mut s = Settings::default();
        if register {
            s.workspaces = vec![std::path::absolute(repo).unwrap()];
        }
        let store = Arc::new(SettingsStore::with_settings(
            settings_dir.join("settings.json"),
            s,
        ));
        let deps: DepsFactory = Arc::new(|| EngineDeps {
            counter: Arc::new(raptor_core::EstimatingCounter),
            git: Box::new(raptor_context::NoGit),
            embedder: Arc::new(NoEmbedder),
            summarizer: None,
        });
        Arc::new(ContextHub::new(dirs.clone(), store, deps))
    }

    #[tokio::test]
    async fn context_adapter_builds_for_registered_workspace_only() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let hub = hub_with(&dirs, t.path(), &repo, true);
        let svc = ContextAdapter::new(hub.clone());
        let id = workspace_id(&std::path::absolute(&repo).unwrap());

        let pack = svc
            .build(gw::BuildRequest {
                workspace: id.clone(),
                query: "refund".into(),
                token_budget: Some(4096),
            })
            .await
            .unwrap();
        assert!(
            pack.text.contains("never touch the refund ledger"),
            "{}",
            pack.text
        );
        assert!(pack.details.get("included").is_some());

        let unregistered = svc
            .build(gw::BuildRequest {
                workspace: "ws-0000000000000000".into(),
                query: "refund".into(),
                token_budget: None,
            })
            .await;
        assert!(matches!(unregistered, Err(RaptorError::NotFound(_))));

        let bad = svc
            .build(gw::BuildRequest {
                workspace: "../../etc".into(),
                query: "refund".into(),
                token_budget: None,
            })
            .await;
        assert!(matches!(bad, Err(RaptorError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn context_adapter_search_symbols_and_impact_use_the_index() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let hub = hub_with(&dirs, t.path(), &repo, true);
        let svc = ContextAdapter::new(hub);
        let id = workspace_id(&std::path::absolute(&repo).unwrap());

        let hits = svc
            .search(gw::SearchRequest {
                workspace: id.clone(),
                query: "issue_refund".into(),
                limit: Some(5),
            })
            .await
            .unwrap();
        assert!(hits.to_string().contains("billing.rs"), "{hits}");

        let syms = svc
            .symbols(gw::SymbolsRequest {
                workspace: id.clone(),
                name: "issue_refund".into(),
            })
            .await
            .unwrap();
        assert!(syms.to_string().contains("issue_refund"));

        let impact = svc
            .impact(gw::ImpactRequest {
                workspace: id.clone(),
                symbol: "issue_refund".into(),
            })
            .await
            .unwrap();
        assert!(impact.to_string().contains("billing.rs"));

        let status = svc.status().await.unwrap();
        assert_eq!(status["workspaces"][0]["workspace"], json!(id));
    }

    #[tokio::test]
    async fn impact_by_file_path_works_and_symbol_queries_still_work() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let hub = hub_with(&dirs, t.path(), &repo, true);
        let svc = ContextAdapter::new(hub);
        let id = workspace_id(&std::path::absolute(&repo).unwrap());
        // Index the workspace so the file is known to the engine.
        svc.search(gw::SearchRequest {
            workspace: id.clone(),
            query: "issue_refund".into(),
            limit: Some(5),
        })
        .await
        .unwrap();

        let by_file = svc
            .impact_target(gw::ImpactQuery {
                workspace: id.clone(),
                target: gw::ImpactTarget::File("src\\billing.rs".into()),
            })
            .await
            .unwrap();
        assert_eq!(by_file["file"], json!("src/billing.rs"), "{by_file}");
        assert!(by_file["impact"]["imports"].is_array(), "{by_file}");

        let by_symbol = svc
            .impact_target(gw::ImpactQuery {
                workspace: id.clone(),
                target: gw::ImpactTarget::Symbol("issue_refund".into()),
            })
            .await
            .unwrap();
        assert!(by_symbol.to_string().contains("billing.rs"), "{by_symbol}");

        let missing = svc
            .impact_target(gw::ImpactQuery {
                workspace: id,
                target: gw::ImpactTarget::File("src/nope.rs".into()),
            })
            .await;
        assert!(matches!(missing, Err(RaptorError::NotFound(_))));
    }

    /// Stand-in for the server tokenizer: one token per whitespace-separated word.
    struct WordTokenizer {
        fail: bool,
        calls: Mutex<usize>,
    }

    impl TokenizeTransport for WordTokenizer {
        fn count_tokens(&self, _base: &str, text: &str) -> Result<usize> {
            *lock(&self.calls) += 1;
            if self.fail {
                return Err(RaptorError::Network("tokenize down".into()));
            }
            Ok(text.split_whitespace().count())
        }
    }

    /// Builds a production-wired hub whose server probe answers `endpoint`.
    fn production_hub(
        dirs: &DataDirs,
        settings_dir: &Path,
        repo: &Path,
        endpoint: Option<&str>,
        transport: Arc<dyn TokenizeTransport>,
    ) -> Arc<ContextHub> {
        let s = Settings {
            workspaces: vec![std::path::absolute(repo).unwrap()],
            ..Settings::default()
        };
        let store = Arc::new(SettingsStore::with_settings(
            settings_dir.join("settings.json"),
            s,
        ));
        let server: Arc<dyn ServerProbe> =
            Arc::new(FakeServer(Mutex::new(endpoint.map(str::to_string))));
        Arc::new(ContextHub::production(
            dirs.clone(),
            store,
            server,
            transport,
        ))
    }

    #[test]
    fn context_build_counts_with_the_server_tokenizer_when_ready() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let tokenizer = Arc::new(WordTokenizer {
            fail: false,
            calls: Mutex::new(0),
        });
        let hub = production_hub(
            &dirs,
            t.path(),
            &repo,
            Some("http://127.0.0.1:8080"),
            tokenizer.clone(),
        );
        let pack = hub.build(&repo, "refund", Some(4096)).unwrap();
        assert_eq!(
            pack.details["tokens_exact"],
            json!(true),
            "{}",
            pack.details
        );
        assert!(*lock(&tokenizer.calls) > 0, "the server tokenizer was used");
    }

    #[test]
    fn context_build_falls_back_to_estimates_when_the_server_is_not_ready() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let tokenizer = Arc::new(WordTokenizer {
            fail: false,
            calls: Mutex::new(0),
        });
        let hub = production_hub(&dirs, t.path(), &repo, None, tokenizer.clone());
        let pack = hub.build(&repo, "refund", Some(4096)).unwrap();
        assert_eq!(pack.details["tokens_exact"], json!(false));
        assert_eq!(
            *lock(&tokenizer.calls),
            0,
            "no request without a ready server"
        );
        assert!(pack.text.contains("never touch the refund ledger"));
    }

    #[test]
    fn context_build_marks_estimates_when_the_tokenizer_request_fails() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fixture_repo(&repo);
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let tokenizer = Arc::new(WordTokenizer {
            fail: true,
            calls: Mutex::new(0),
        });
        let hub = production_hub(
            &dirs,
            t.path(),
            &repo,
            Some("http://127.0.0.1:8080"),
            tokenizer.clone(),
        );
        let pack = hub.build(&repo, "refund", Some(4096)).unwrap();
        assert_eq!(pack.details["tokens_exact"], json!(false));
        assert!(*lock(&tokenizer.calls) > 0);
    }

    #[test]
    fn workspace_id_is_stable_and_prefixed() {
        let a = workspace_id(Path::new("C:/repo"));
        assert!(a.starts_with("ws-") && a.len() == 19);
        assert_eq!(a, workspace_id(Path::new("C:/repo")));
        assert_ne!(a, workspace_id(Path::new("C:/other")));
    }
}

#[cfg(test)]
mod semantic_hub_tests {
    use super::fakes::*;
    use super::*;
    use crate::semantic::fakes::{active, FakeLlama};
    use crate::semantic::{ActiveModel, ActiveServer, LlamaHttp, SemanticSetup};
    use crate::settings::Settings;
    use std::sync::Mutex as StdMutex;

    struct NoTokens;

    impl TokenizeTransport for NoTokens {
        fn count_tokens(&self, _base: &str, text: &str) -> Result<usize> {
            Ok(text.split_whitespace().count())
        }
    }

    /// Active-model source whose state the test changes.
    struct SharedActive(Arc<StdMutex<Option<ActiveServer>>>);

    impl ActiveModel for SharedActive {
        fn active(&self) -> Option<ActiveServer> {
            self.0.lock().unwrap().clone()
        }
    }

    fn repo(root: &Path) {
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "Never skip the refund ledger.\n").unwrap();
        std::fs::write(
            root.join("src/billing.rs"),
            "pub fn issue_refund(amount: u64) -> u64 {\n    amount\n}\n",
        )
        .unwrap();
    }

    /// Hub connected to a fake server. `active_model` is the model the server reports loaded.
    fn connected(
        t: &Path,
        embedding: Option<&str>,
        active_model: Option<&str>,
        http: Arc<FakeLlama>,
    ) -> (Arc<ContextHub>, Arc<StdMutex<Option<ActiveServer>>>) {
        let dirs = DataDirs::new(t.join("RaptorLLM"));
        dirs.ensure().unwrap();
        let settings = Arc::new(SettingsStore::with_settings(
            t.join("settings.json"),
            Settings {
                embedding_model: embedding.map(str::to_string),
                ..Settings::default()
            },
        ));
        let slot = Arc::new(StdMutex::new(active_model.map(|m| active(m, 1))));
        let semantic = Arc::new(SemanticSetup {
            active: Arc::new(SharedActive(slot.clone())),
            http: http as Arc<dyn LlamaHttp>,
        });
        let server: Arc<dyn ServerProbe> = Arc::new(FakeServer(StdMutex::new(None)));
        let hub = Arc::new(ContextHub::connected(
            dirs,
            settings,
            server,
            Arc::new(NoTokens),
            semantic,
        ));
        (hub, slot)
    }

    #[test]
    fn semantic_retrieval_turns_on_when_the_embedding_model_is_loaded_and_vectors_are_built() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("repo");
        repo(&root);
        let http = Arc::new(FakeLlama::new());
        let (hub, _slot) = connected(t.path(), Some("emb.gguf"), Some("emb.gguf"), http.clone());
        let view = hub.semantic_status(&root).unwrap();
        assert_eq!(view.configured_model.as_deref(), Some("emb.gguf"));
        assert!(view.reason.is_none(), "{:?}", view.reason);
        assert!(view.status.enabled);
        assert_eq!(view.status.dimension, 4);
        assert!(view.status.vectors > 0, "{:?}", view.status);
        assert_eq!(view.status.chunks_without_vector, 0);
        assert!(http.count("/v1/embeddings") >= 2, "probe and indexing");
    }

    #[test]
    fn a_different_model_loaded_degrades_with_a_reason_and_lexical_search_still_works() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("repo");
        repo(&root);
        let http = Arc::new(FakeLlama::new());
        let (hub, _slot) = connected(t.path(), Some("emb.gguf"), Some("chat.gguf"), http.clone());
        let view = hub.semantic_status(&root).unwrap();
        assert!(!view.status.enabled);
        let reason = view.reason.expect("the reason is reported");
        assert!(reason.contains("not the embedding model"), "{reason}");
        assert_eq!(
            http.count("/v1/embeddings"),
            0,
            "no request to a model that is not loaded"
        );
        let engine = hub.engine(&root).unwrap();
        let hits = lock(&engine)
            .search(&raptor_context::SearchRequest {
                query: "issue_refund".into(),
                limit: 5,
                ..Default::default()
            })
            .unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn an_unsupported_embeddings_endpoint_is_reported_in_status() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("repo");
        repo(&root);
        let http = Arc::new(FakeLlama {
            embeddings_status: 501,
            ..FakeLlama::new()
        });
        let (hub, _slot) = connected(t.path(), Some("emb.gguf"), Some("emb.gguf"), http);
        let view = hub.semantic_status(&root).unwrap();
        assert!(!view.status.enabled);
        assert!(view.reason.unwrap().contains("did not answer"));
    }

    #[test]
    fn a_server_restart_rebuilds_the_backend_and_clearing_the_model_turns_it_off() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("repo");
        repo(&root);
        let http = Arc::new(FakeLlama::new());
        let (hub, slot) = connected(t.path(), Some("emb.gguf"), Some("emb.gguf"), http.clone());
        assert!(hub.semantic_status(&root).unwrap().status.enabled);
        let probes_before = http.count("/v1/embeddings");

        // Same model, new server generation: the backend is probed again.
        *slot.lock().unwrap() = Some(active("emb.gguf", 2));
        assert!(hub.semantic_status(&root).unwrap().status.enabled);
        assert!(http.count("/v1/embeddings") > probes_before);

        // Nothing changed: no further requests.
        let steady = http.count("/v1/embeddings");
        hub.semantic_status(&root).unwrap();
        assert_eq!(http.count("/v1/embeddings"), steady);

        // No embedding model configured: off, with no reason to report.
        hub.settings
            .update(|s| {
                s.embedding_model = None;
                Ok(())
            })
            .unwrap();
        let cleared = hub.semantic_status(&root).unwrap();
        assert!(!cleared.status.enabled);
        assert!(cleared.reason.is_none());
    }

    #[test]
    fn the_summarizer_writes_through_the_loaded_model() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("repo");
        repo(&root);
        let http = Arc::new(FakeLlama::new());
        let (hub, _slot) = connected(t.path(), None, Some("chat.gguf"), http.clone());
        let engine = hub.engine(&root).unwrap();
        let messages = vec![
            ChatMessage::new(Role::User, "We keep the ledger append-only."),
            ChatMessage::new(Role::Assistant, "Agreed."),
        ];
        let out = lock(&engine).summarize_session("s1", &messages);
        assert!(out.is_ok(), "{out:?}");
        assert_eq!(http.count("/v1/chat/completions"), 1);
    }
}
