//! HTTP server lifecycle, router assembly, and the middleware stack.
//!
//! Middleware, outermost first:
//! 1. CORS (answers preflights before authentication; allowlist only).
//! 2. Error shaping (turns axum's plain-text 404/405/413 into JSON errors).
//! 3. Authentication (skipped for `/health`), then the per-key rate limit.
//! 4. Body size limit.
//!
//! Listeners bind `127.0.0.1` in localhost mode and `0.0.0.0` in LAN mode.
//! LAN mode is refused unless an API key is configured and TLS is on, or the
//! caller opts into `allow_insecure_lan`. TLS, when configured, applies to any
//! bind mode.

use std::collections::{BTreeMap, HashSet};
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use axum_server::tls_rustls::{from_tcp_rustls, RustlsConfig};
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::auth::{AccessPolicy, ApiKeyRecord, Authenticator, KeyStore};
use crate::context_api::ContextService;
use crate::error::{ApiError, Shape};
use crate::openai::{ModelCatalog, Upstream};
use crate::queue::{Metrics, QueueConfig, RateLimiter, Scheduler};
use crate::tls::{self, TlsConfig, TlsIdentity};

/// Which interface the listener binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindMode {
    /// `127.0.0.1` only. Keyless access is allowed from loopback when configured.
    Localhost,
    /// `0.0.0.0`. Requires at least one API key and TLS, unless
    /// `allow_insecure_lan` is set.
    Lan,
}

/// Everything the gateway needs to start. Keys are passed already hashed.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayConfig {
    pub bind: BindMode,
    /// TCP port. `0` picks a free port (useful in tests).
    pub port: u16,
    /// Stored API keys (hashed). Required in LAN mode.
    pub api_keys: Vec<ApiKeyRecord>,
    /// Allow requests without a key from loopback peers. Default true.
    pub allow_keyless_loopback: bool,
    /// Exact origins allowed to make cross-origin requests. Empty means none.
    /// Wildcards are rejected.
    pub cors_allowed_origins: Vec<String>,
    /// Largest accepted request body.
    pub max_body_bytes: usize,
    /// Concurrency, queue depth, per-client limits and timeouts.
    pub queue: QueueConfig,
    /// Client model name to installed model id, e.g. `fast` -> `llama-3-8b-q4`.
    pub model_aliases: BTreeMap<String, String>,
    /// Root for the MCP `repo_read_file` tool. Unset disables that tool.
    pub workspace_root: Option<PathBuf>,
    /// How long a reconfigure or shutdown waits for in-flight requests.
    pub drain_timeout: Duration,
    /// Certificate for the listener. `None` serves plain HTTP.
    pub tls: Option<TlsConfig>,
    /// Permit LAN mode without TLS. Off by default. When set, the listener logs
    /// a warning and reports `insecure` in its status.
    pub allow_insecure_lan: bool,
    /// How long a fetched model catalog is reused. Zero disables the cache.
    pub catalog_ttl: Duration,
    /// Remove reasoning (`<think>` blocks, `reasoning_content`) from earlier assistant turns of a
    /// chat request before forwarding it. Default true. See [`crate::hygiene`].
    pub strip_history_reasoning: bool,
    /// Thinking effort added to a chat request that does not set one, when the loaded model
    /// supports it. Default Medium. A request with its own thinking fields is never changed, and
    /// a model without effort control is never sent the field. See [`crate::thinking`].
    pub default_effort: crate::thinking::Effort,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind: BindMode::Localhost,
            port: 8790,
            api_keys: Vec::new(),
            allow_keyless_loopback: true,
            cors_allowed_origins: Vec::new(),
            max_body_bytes: 32 * 1024 * 1024,
            queue: QueueConfig::default(),
            model_aliases: BTreeMap::new(),
            workspace_root: None,
            drain_timeout: Duration::from_secs(10),
            tls: None,
            allow_insecure_lan: false,
            catalog_ttl: Duration::from_secs(2),
            strip_history_reasoning: true,
            default_effort: crate::thinking::Effort::default(),
        }
    }
}

impl GatewayConfig {
    /// Checks the configuration without touching the network.
    ///
    /// Refuses LAN mode without a key, LAN mode without TLS (unless
    /// `allow_insecure_lan`), keyless-off without keys, duplicate key ids,
    /// wildcard or malformed CORS origins, bad TLS settings, and degenerate limits.
    pub fn validate(&self) -> Result<()> {
        if self.bind == BindMode::Lan && self.api_keys.is_empty() {
            return Err(RaptorError::Unauthorized(
                "LAN mode requires at least one API key; refusing to bind 0.0.0.0".into(),
            ));
        }
        if self.bind == BindMode::Lan && self.tls.is_none() && !self.allow_insecure_lan {
            return Err(RaptorError::InvalidInput(
                "LAN mode requires TLS; configure `tls`, or set `allow_insecure_lan` to serve API keys in clear text".into(),
            ));
        }
        if !self.allow_keyless_loopback && self.api_keys.is_empty() {
            return Err(RaptorError::Unauthorized(
                "no API keys configured and keyless localhost is disabled".into(),
            ));
        }
        let mut seen = HashSet::new();
        for key in &self.api_keys {
            if !seen.insert(key.id()) {
                return Err(RaptorError::InvalidInput(format!(
                    "duplicate api key id '{}'",
                    key.id()
                )));
            }
        }
        if self.max_body_bytes == 0 {
            return Err(RaptorError::InvalidInput(
                "max_body_bytes must be greater than zero".into(),
            ));
        }
        for origin in &self.cors_allowed_origins {
            validate_origin(origin)?;
        }
        for (alias, target) in &self.model_aliases {
            if alias.is_empty() || target.is_empty() {
                return Err(RaptorError::InvalidInput(
                    "model aliases must not be empty".into(),
                ));
            }
        }
        validate_tls(self.tls.as_ref())?;
        Scheduler::new(self.queue.clone())?;
        Ok(())
    }
}

fn validate_tls(tls: Option<&TlsConfig>) -> Result<()> {
    match tls {
        None => Ok(()),
        Some(TlsConfig::Files { cert_pem, key_pem }) => {
            if cert_pem.as_os_str().is_empty() || key_pem.as_os_str().is_empty() {
                return Err(RaptorError::InvalidInput(
                    "TLS certificate and key paths must not be empty".into(),
                ));
            }
            Ok(())
        }
        Some(TlsConfig::SelfSigned {
            dir,
            subject_alt_names,
        }) => {
            if dir.as_os_str().is_empty() {
                return Err(RaptorError::InvalidInput(
                    "self-signed TLS needs a directory to store the certificate".into(),
                ));
            }
            for name in subject_alt_names {
                let ok = !name.is_empty()
                    && name.len() <= 253
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | ':'));
                if !ok {
                    return Err(RaptorError::InvalidInput(format!(
                        "invalid subject alternative name: {name:?}"
                    )));
                }
            }
            Ok(())
        }
    }
}

fn validate_origin(origin: &str) -> Result<HeaderValue> {
    if origin == "*" {
        return Err(RaptorError::InvalidInput(
            "wildcard CORS origins are not allowed".into(),
        ));
    }
    let rest = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .ok_or_else(|| {
            RaptorError::InvalidInput(format!("CORS origin must be http(s)://host: {origin}"))
        })?;
    if rest.is_empty() || rest.contains('/') {
        return Err(RaptorError::InvalidInput(format!(
            "CORS origin must not contain a path: {origin}"
        )));
    }
    HeaderValue::from_str(origin)
        .map_err(|_| RaptorError::InvalidInput(format!("invalid CORS origin: {origin}")))
}

/// The collaborators the gateway calls out to.
#[derive(Clone)]
pub struct GatewayDeps {
    /// The inference backend.
    pub upstream: Arc<dyn Upstream>,
    /// The context engine. `None` disables the `/raptor/context` routes, context
    /// injection, and the context MCP tools.
    pub context: Option<Arc<dyn ContextService>>,
}

/// Caches the upstream model catalog for a short time, so each request does not
/// fetch it again. Errors are not cached.
#[derive(Clone)]
pub(crate) struct CatalogCache {
    inner: Arc<CatalogInner>,
}

struct CatalogInner {
    ttl: Duration,
    slot: Mutex<Option<(Instant, ModelCatalog)>>,
}

impl CatalogCache {
    pub(crate) fn new(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(CatalogInner {
                ttl,
                slot: Mutex::new(None),
            }),
        }
    }

    /// Returns the cached catalog while it is fresh, else asks `upstream`.
    pub(crate) async fn get(&self, upstream: &dyn Upstream) -> Result<ModelCatalog> {
        let ttl = self.inner.ttl;
        if ttl.is_zero() {
            return upstream.models().await;
        }
        {
            let slot = self.inner.slot.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((at, catalog)) = slot.as_ref() {
                if at.elapsed() < ttl {
                    return Ok(catalog.clone());
                }
            }
        }
        let catalog = upstream.models().await?;
        *self.inner.slot.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((Instant::now(), catalog.clone()));
        Ok(catalog)
    }

    /// Drops the cached catalog so the next request asks the upstream.
    pub(crate) fn invalidate(&self) {
        *self.inner.slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Shared state for every handler.
pub(crate) struct AppState {
    pub(crate) upstream: Arc<dyn Upstream>,
    pub(crate) context: Option<Arc<dyn ContextService>>,
    pub(crate) scheduler: Scheduler,
    pub(crate) auth: Authenticator,
    pub(crate) aliases: BTreeMap<String, String>,
    pub(crate) retry_after_secs: u64,
    pub(crate) workspace_root: Option<PathBuf>,
    pub(crate) catalog: CatalogCache,
    pub(crate) rate: RateLimiter,
    pub(crate) mcp: crate::mcp::Sessions,
    pub(crate) strip_history_reasoning: bool,
    pub(crate) default_effort: crate::thinking::Effort,
    /// Set once the volatile-system-prompt warning has been logged (once per gateway start).
    pub(crate) volatile_warned: std::sync::atomic::AtomicBool,
}

/// A validated router plus the scheduler whose metrics it reports.
pub struct Built {
    /// Routes and middleware, ready to serve or to call with `oneshot`.
    pub router: Router,
    /// The scheduler behind the router.
    pub scheduler: Scheduler,
    /// The live key set the router authenticates against. Revoking or replacing
    /// keys here takes effect on the next request.
    pub keys: KeyStore,
    catalog: CatalogCache,
}

/// Assembles the router after validating the configuration.
pub fn build(config: &GatewayConfig, deps: GatewayDeps) -> Result<Built> {
    config.validate()?;
    let keys = KeyStore::new(config.api_keys.clone())?;
    build_with(config, deps, keys)
}

/// Assembles the router over an existing key store. The caller has validated `config`.
fn build_with(config: &GatewayConfig, deps: GatewayDeps, keys: KeyStore) -> Result<Built> {
    let scheduler = Scheduler::new(config.queue.clone())?;
    let policy = AccessPolicy {
        allow_keyless_loopback: config.allow_keyless_loopback,
    };
    let catalog = CatalogCache::new(config.catalog_ttl);
    let state = Arc::new(AppState {
        upstream: deps.upstream,
        context: deps.context,
        scheduler: scheduler.clone(),
        auth: Authenticator::with_store(keys.clone(), policy),
        aliases: config.model_aliases.clone(),
        retry_after_secs: config.queue.retry_after_secs,
        workspace_root: config.workspace_root.clone(),
        catalog: catalog.clone(),
        rate: RateLimiter::new(),
        mcp: crate::mcp::Sessions::default(),
        strip_history_reasoning: config.strip_history_reasoning,
        default_effort: config.default_effort,
        volatile_warned: std::sync::atomic::AtomicBool::new(false),
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/raptor/metrics", get(metrics))
        .merge(crate::openai::routes())
        .merge(crate::adapters::routes())
        .merge(crate::context_api::routes())
        .merge(crate::mcp::routes())
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn(shape_errors))
        .layer(cors_layer(&config.cors_allowed_origins)?)
        .with_state(state);
    Ok(Built {
        router,
        scheduler,
        keys,
        catalog,
    })
}

fn cors_layer(origins: &[String]) -> Result<CorsLayer> {
    let values = origins
        .iter()
        .map(|o| validate_origin(o))
        .collect::<Result<Vec<_>>>()?;
    Ok(CorsLayer::new()
        .allow_origin(AllowOrigin::list(values))
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            HeaderName::from_static("x-api-key"),
            HeaderName::from_static("x-raptor-workspace"),
            HeaderName::from_static("anthropic-version"),
            HeaderName::from_static("anthropic-beta"),
            HeaderName::from_static("mcp-protocol-version"),
            HeaderName::from_static("mcp-session-id"),
        ])
        .expose_headers([HeaderName::from_static("mcp-session-id")])
        .max_age(Duration::from_secs(600)))
}

async fn auth_middleware(
    State(st): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if path == "/health" {
        return next.run(req).await;
    }
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip());
    let shape = if path.starts_with("/v1/messages") {
        Shape::Anthropic
    } else {
        Shape::OpenAi
    };
    match st.auth.authorize(peer, req.headers()) {
        Ok(principal) => {
            if let Err(retry_after) = st.rate.check(
                &principal.client_id,
                principal.limits.requests_per_minute,
                Instant::now(),
            ) {
                return ApiError::new(
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    "rate_limit_error",
                    "Too many requests for this API key.",
                )
                .with_code("rate_limit_exceeded")
                .with_retry_after(retry_after)
                .with_shape(shape)
                .into_response();
            }
            // Requiring JSON keeps simple cross-site form posts (text/plain, no
            // preflight) from reaching the handlers.
            if req.method() == Method::POST && !is_json(req.headers()) {
                return ApiError::new(
                    axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "invalid_request_error",
                    "Content-Type must be application/json.",
                )
                .with_shape(shape)
                .into_response();
            }
            req.extensions_mut().insert(principal);
            next.run(req).await
        }
        Err(e) => e.with_shape(shape).into_response(),
    }
}

/// True for `application/json` (any parameters) and `application/*+json`.
fn is_json(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            let media = v.split(';').next().unwrap_or("").trim();
            media.eq_ignore_ascii_case("application/json")
                || (media.starts_with("application/") && media.ends_with("+json"))
        })
}

/// Replaces axum's plain-text error bodies (404, 405, 413, 415, ...) with the
/// JSON error shape of the route. Responses that are already JSON are left alone.
async fn shape_errors(req: Request, next: Next) -> Response {
    let anthropic = req.uri().path().starts_with("/v1/messages");
    let response = next.run(req).await;
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) || is_json(response.headers()) {
        return response;
    }
    let message = match status.as_u16() {
        404 => "Not found.".to_string(),
        413 => "Request body is too large.".to_string(),
        _ => status
            .canonical_reason()
            .unwrap_or("Request failed.")
            .to_string(),
    };
    let shape = if anthropic {
        Shape::Anthropic
    } else {
        Shape::OpenAi
    };
    ApiError::new(status, "invalid_request_error", message)
        .with_shape(shape)
        .into_response()
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}))
}

async fn metrics(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let m: Metrics = st.scheduler.metrics();
    Json(
        json!({"queued": m.queued, "running": m.running, "completed": m.completed, "rejected": m.rejected}),
    )
}

/// Non-secret description of a running listener, for the UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ListenerStatus {
    /// The address actually bound.
    pub addr: SocketAddr,
    /// True when bound to all interfaces.
    pub lan: bool,
    /// True when the listener serves TLS.
    pub tls: bool,
    /// SHA-256 of the certificate (lowercase hex), for client pinning. `None` without TLS.
    pub tls_fingerprint_sha256: Option<String>,
    /// True for LAN mode without TLS (only possible with `allow_insecure_lan`).
    /// API keys and request bodies travel in clear text.
    pub insecure: bool,
    /// Number of API keys currently accepted.
    pub keys: usize,
}

/// The TLS material a running listener serves.
#[derive(Clone)]
struct TlsState {
    rustls: RustlsConfig,
    identity: TlsIdentity,
}

fn prepare_tls(config: &GatewayConfig) -> Result<Option<TlsState>> {
    let Some(cfg) = &config.tls else {
        return Ok(None);
    };
    let loaded = tls::load(cfg)?;
    Ok(Some(TlsState {
        rustls: tls::rustls_config(&loaded),
        identity: loaded.identity,
    }))
}

fn is_insecure(config: &GatewayConfig) -> bool {
    config.bind == BindMode::Lan && config.tls.is_none()
}

/// True when two configs differ only in their API keys.
fn same_except_keys(a: &GatewayConfig, b: &GatewayConfig) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    a.api_keys.clear();
    b.api_keys.clear();
    a == b
}

/// A running gateway listener.
pub struct Handle {
    addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<io::Result<()>>>,
    deps: GatewayDeps,
    scheduler: Scheduler,
    drain: Duration,
    config: GatewayConfig,
    keys: KeyStore,
    catalog: CatalogCache,
    tls: Option<TlsState>,
}

/// Entry point for starting the gateway.
pub struct Gateway;

impl Gateway {
    /// Validates `config`, prepares TLS, binds the listener, and starts serving.
    ///
    /// Errors before binding: an invalid config (for example LAN without a key
    /// or without TLS) returns an error and no socket is opened. Unreadable or
    /// mismatched TLS files fail the same way.
    pub async fn start(config: GatewayConfig, deps: GatewayDeps) -> Result<Handle> {
        config.validate()?;
        let keys = KeyStore::new(config.api_keys.clone())?;
        let built = build_with(&config, deps.clone(), keys.clone())?;
        let tls = prepare_tls(&config)?;
        let (addr, task, stop) = serve(&config, built.router, tls.as_ref()).await?;
        Ok(Handle {
            addr,
            stop: Some(stop),
            task: Some(task),
            deps,
            scheduler: built.scheduler,
            drain: config.drain_timeout,
            config,
            keys,
            catalog: built.catalog,
            tls,
        })
    }
}

async fn serve(
    config: &GatewayConfig,
    router: Router,
    tls: Option<&TlsState>,
) -> Result<(SocketAddr, JoinHandle<io::Result<()>>, oneshot::Sender<()>)> {
    let ip = match config.bind {
        BindMode::Localhost => IpAddr::V4(Ipv4Addr::LOCALHOST),
        BindMode::Lan => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
    };
    let std_listener = std::net::TcpListener::bind(SocketAddr::new(ip, config.port))
        .map_err(|e| RaptorError::Io(format!("bind {ip}:{}: {e}", config.port)))?;
    std_listener
        .set_nonblocking(true)
        .map_err(|e| RaptorError::Io(e.to_string()))?;
    let addr = std_listener
        .local_addr()
        .map_err(|e| RaptorError::Io(e.to_string()))?;
    let (tx, rx) = oneshot::channel::<()>();
    let drain = config.drain_timeout;
    let task = match tls {
        None => {
            let listener =
                TcpListener::from_std(std_listener).map_err(|e| RaptorError::Io(e.to_string()))?;
            tokio::spawn(async move {
                axum::serve(
                    listener,
                    router.into_make_service_with_connect_info::<SocketAddr>(),
                )
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
            })
        }
        Some(state) => {
            let handle = axum_server::Handle::new();
            let stopper = handle.clone();
            // Dropping the sender (handle dropped) and sending both mean stop.
            tokio::spawn(async move {
                let _ = rx.await;
                stopper.graceful_shutdown(Some(drain));
            });
            let server = from_tcp_rustls(std_listener, state.rustls.clone()).handle(handle);
            tokio::spawn(async move {
                server
                    .serve(router.into_make_service_with_connect_info::<SocketAddr>())
                    .await
            })
        }
    };
    let lan = config.bind == BindMode::Lan;
    tracing::info!(%addr, lan, tls = tls.is_some(), "raptor gateway listening");
    if is_insecure(config) {
        tracing::warn!(
            %addr,
            "LAN listener without TLS (allow_insecure_lan): API keys and request bodies travel in clear text"
        );
    }
    Ok((addr, task, tx))
}

impl Handle {
    /// The address actually bound (useful when `port` was 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Current queue counters for this listener.
    pub fn metrics(&self) -> Metrics {
        self.scheduler.metrics()
    }

    /// Description of the running listener: address, bind mode, TLS fingerprint,
    /// and whether it is insecure.
    pub fn status(&self) -> ListenerStatus {
        ListenerStatus {
            addr: self.addr,
            lan: self.config.bind == BindMode::Lan,
            tls: self.tls.is_some(),
            tls_fingerprint_sha256: self
                .tls
                .as_ref()
                .map(|t| t.identity.fingerprint_sha256.clone()),
            insecure: is_insecure(&self.config),
            keys: self.keys.len(),
        }
    }

    /// The certificate the listener presents, if TLS is on.
    pub fn tls_identity(&self) -> Option<&TlsIdentity> {
        self.tls.as_ref().map(|t| &t.identity)
    }

    /// The live key set. Add, revoke or replace keys here to hot-reload them:
    /// the listener keeps running and the change applies to the next request.
    pub fn keys(&self) -> KeyStore {
        self.keys.clone()
    }

    /// Drops the cached model catalog so the next request fetches it again.
    pub fn invalidate_model_cache(&self) {
        self.catalog.invalidate();
    }

    /// Regenerates a self-signed certificate (or re-reads user-provided files)
    /// and swaps it into the running listener, without a restart. Clients that
    /// pinned the old fingerprint must re-pin the returned one.
    pub async fn rotate_tls(&mut self) -> Result<TlsIdentity> {
        let cfg = self.config.tls.clone().ok_or_else(|| {
            RaptorError::InvalidInput("TLS is not enabled on this listener".into())
        })?;
        let loaded = tls::rotate(&cfg)?;
        let identity = loaded.identity.clone();
        let rustls = match &self.tls {
            Some(state) => state.rustls.clone(),
            None => {
                return Err(RaptorError::InvalidInput(
                    "TLS is not enabled on this listener".into(),
                ))
            }
        };
        rustls.reload_from_config(Arc::clone(&loaded.server_config));
        self.tls = Some(TlsState {
            rustls,
            identity: identity.clone(),
        });
        Ok(identity)
    }

    /// Replaces the configuration. The new config is validated first, so an
    /// invalid config leaves the current listener running.
    ///
    /// A change to API keys alone is applied to the live key set with no restart.
    /// Any other change drains the current listener (up to `drain_timeout`, then
    /// connections are dropped), binds a new one, and restarts queue counters and
    /// the model cache.
    pub async fn reconfigure(&mut self, config: GatewayConfig) -> Result<()> {
        config.validate()?;
        if same_except_keys(&self.config, &config) {
            self.keys.replace(config.api_keys.clone())?;
            self.catalog.invalidate();
            self.config = config;
            return Ok(());
        }
        let tls = prepare_tls(&config)?;
        self.keys.replace(config.api_keys.clone())?;
        let built = build_with(&config, self.deps.clone(), self.keys.clone())?;
        self.stop_inner().await?;
        let (addr, task, stop) = serve(&config, built.router, tls.as_ref()).await?;
        self.addr = addr;
        self.task = Some(task);
        self.stop = Some(stop);
        self.scheduler = built.scheduler;
        self.catalog = built.catalog;
        self.drain = config.drain_timeout;
        self.tls = tls;
        self.config = config;
        Ok(())
    }

    /// Stops accepting connections and waits for in-flight requests to finish,
    /// up to the drain timeout.
    pub async fn shutdown(mut self) -> Result<()> {
        self.stop_inner().await
    }

    async fn stop_inner(&mut self) -> Result<()> {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        let Some(mut task) = self.task.take() else {
            return Ok(());
        };
        match tokio::time::timeout(self.drain, &mut task).await {
            Ok(joined) => match joined {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(RaptorError::Io(format!("server error: {e}"))),
                Err(e) => Err(RaptorError::Internal(format!("server task failed: {e}"))),
            },
            Err(_) => {
                task.abort();
                Ok(())
            }
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // Dropping the sender triggers graceful shutdown, so the listener does not outlive its handle.
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tls_cfg() -> Option<TlsConfig> {
        Some(TlsConfig::SelfSigned {
            dir: PathBuf::from("unused"),
            subject_alt_names: vec![],
        })
    }

    #[test]
    fn lan_without_key_is_refused() {
        let cfg = GatewayConfig {
            bind: BindMode::Lan,
            tls: tls_cfg(),
            ..GatewayConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(RaptorError::Unauthorized(_))));
    }

    #[test]
    fn lan_with_key_and_tls_is_accepted() {
        let key = ApiKeyRecord::new("laptop", "nxk_0123456789abcdef").unwrap();
        let cfg = GatewayConfig {
            bind: BindMode::Lan,
            api_keys: vec![key],
            tls: tls_cfg(),
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn lan_without_tls_is_refused_by_default() {
        let key = ApiKeyRecord::new("laptop", "nxk_0123456789abcdef").unwrap();
        let cfg = GatewayConfig {
            bind: BindMode::Lan,
            api_keys: vec![key],
            ..GatewayConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(RaptorError::InvalidInput(_))));
    }

    #[test]
    fn lan_without_tls_needs_the_explicit_insecure_flag() {
        let key = ApiKeyRecord::new("laptop", "nxk_0123456789abcdef").unwrap();
        let cfg = GatewayConfig {
            bind: BindMode::Lan,
            api_keys: vec![key],
            allow_insecure_lan: true,
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_ok());
        assert!(is_insecure(&cfg));
        let secure = GatewayConfig {
            tls: tls_cfg(),
            ..cfg.clone()
        };
        assert!(!is_insecure(&secure));
    }

    #[test]
    fn duplicate_key_ids_are_refused() {
        let cfg = GatewayConfig {
            api_keys: vec![
                ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa1").unwrap(),
                ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa2").unwrap(),
            ],
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn tls_settings_are_validated() {
        let empty_dir = GatewayConfig {
            tls: Some(TlsConfig::SelfSigned {
                dir: PathBuf::new(),
                subject_alt_names: vec![],
            }),
            ..GatewayConfig::default()
        };
        assert!(empty_dir.validate().is_err());
        let bad_name = GatewayConfig {
            tls: Some(TlsConfig::SelfSigned {
                dir: PathBuf::from("d"),
                subject_alt_names: vec!["bad name/".into()],
            }),
            ..GatewayConfig::default()
        };
        assert!(bad_name.validate().is_err());
    }

    #[test]
    fn keyless_disabled_without_keys_locks_out_everyone() {
        let cfg = GatewayConfig {
            allow_keyless_loopback: false,
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cors_origins_must_be_exact() {
        assert!(validate_origin("https://app.example.com").is_ok());
        assert!(validate_origin("http://localhost:3000").is_ok());
        assert!(validate_origin("*").is_err());
        assert!(validate_origin("https://a.example.com/path").is_err());
        assert!(validate_origin("app.example.com").is_err());
        assert!(validate_origin("https://").is_err());
    }

    #[test]
    fn zero_limits_are_rejected() {
        let cfg = GatewayConfig {
            max_body_bytes: 0,
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_err());
        let q = QueueConfig {
            max_concurrent: 0,
            ..QueueConfig::default()
        };
        let cfg = GatewayConfig {
            queue: q,
            ..GatewayConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[tokio::test]
    async fn catalog_cache_reuses_until_ttl_or_invalidation() {
        use crate::openai::ModelCatalog;
        use async_trait::async_trait;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counting(AtomicUsize);
        #[async_trait]
        impl Upstream for Counting {
            async fn models(&self) -> Result<ModelCatalog> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(ModelCatalog::default())
            }
            async fn call(
                &self,
                _: crate::openai::Endpoint,
                _: serde_json::Value,
            ) -> Result<crate::openai::Completion> {
                Err(RaptorError::Internal("unused".into()))
            }
        }

        let up = Counting(AtomicUsize::new(0));
        let cache = CatalogCache::new(Duration::from_millis(40));
        cache.get(&up).await.unwrap();
        cache.get(&up).await.unwrap();
        assert_eq!(up.0.load(Ordering::SeqCst), 1);
        cache.invalidate();
        cache.get(&up).await.unwrap();
        assert_eq!(up.0.load(Ordering::SeqCst), 2);
        tokio::time::sleep(Duration::from_millis(60)).await;
        cache.get(&up).await.unwrap();
        assert_eq!(up.0.load(Ordering::SeqCst), 3);

        let no_cache = CatalogCache::new(Duration::ZERO);
        no_cache.get(&up).await.unwrap();
        no_cache.get(&up).await.unwrap();
        assert_eq!(up.0.load(Ordering::SeqCst), 5);
    }
}
