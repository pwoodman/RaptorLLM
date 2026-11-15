//! Context engine surface: the `ContextService` trait the desktop shell
//! implements over raptor-context, the `/raptor/context/*` routes, and automatic
//! context injection for generic clients.
//!
//! This crate does not depend on raptor-context. Responses are passed through as
//! JSON `Value`s, and the typed parts are the request shapes and [`ContextPack`].

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use raptor_core::{RaptorError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::server::AppState;

/// Request header that asks the gateway to inject workspace context into a chat request.
pub const WORKSPACE_HEADER: &str = "x-raptor-workspace";

const MAX_WORKSPACE_LEN: usize = 512;
const MAX_QUERY_BYTES: usize = 8 * 1024;
const MAX_SEARCH_LIMIT: usize = 100;
const MAX_TOKEN_BUDGET: usize = 1_000_000;

/// The context engine, implemented by the desktop shell.
#[async_trait]
pub trait ContextService: Send + Sync {
    /// Builds a token-budgeted context pack for a query.
    async fn build(&self, req: BuildRequest) -> Result<ContextPack>;
    /// Searches the indexed workspace. Returns the engine's own JSON.
    async fn search(&self, req: SearchRequest) -> Result<Value>;
    /// Recovers context lost to truncation or session boundaries.
    async fn recover(&self, req: RecoverRequest) -> Result<Value>;
    /// Lists what depends on a symbol. Symbol-only; kept as the required method
    /// so existing implementations compile unchanged.
    async fn impact(&self, req: ImpactRequest) -> Result<Value>;
    /// Lists what depends on a symbol or on a file path. This is what the
    /// `/raptor/context/impact` route and the `repo_impact` tool call.
    ///
    /// The default answers symbol queries through [`ContextService::impact`] and
    /// refuses file queries with `Unsupported`. Override it to support files.
    async fn impact_target(&self, req: ImpactQuery) -> Result<Value> {
        match req.target {
            ImpactTarget::Symbol(symbol) => {
                self.impact(ImpactRequest {
                    workspace: req.workspace,
                    symbol,
                })
                .await
            }
            ImpactTarget::File(_) => Err(RaptorError::Unsupported(
                "impact by file path is not supported by this context service".into(),
            )),
        }
    }
    /// Lists symbols matching a name.
    async fn symbols(&self, req: SymbolsRequest) -> Result<Value>;
    /// Index status for the `/raptor/context/status` route.
    async fn status(&self) -> Result<Value>;
}

/// Output of [`ContextService::build`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextPack {
    /// Text to place in front of the model.
    pub text: String,
    /// Engine details (sources, token counts, exactness). Passed through as-is.
    #[serde(default)]
    pub details: Value,
}

/// `POST /raptor/context/build` body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildRequest {
    pub workspace: String,
    pub query: String,
    #[serde(default)]
    pub token_budget: Option<usize>,
}

/// `POST /raptor/context/search` body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub workspace: String,
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `POST /raptor/context/recover` body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoverRequest {
    pub workspace: String,
    pub query: String,
}

/// `POST /raptor/context/impact` body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactRequest {
    pub workspace: String,
    pub symbol: String,
}

/// What an impact query is about. Exactly one of the two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImpactTarget {
    /// A symbol name as the index knows it.
    Symbol(String),
    /// A workspace-relative file path (`/` or `\` separated, no `..`, no drive letter).
    File(String),
}

/// A validated impact query, passed to [`ContextService::impact_target`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactQuery {
    pub workspace: String,
    pub target: ImpactTarget,
}

/// Wire form of an impact request, shared by the HTTP route and the `repo_impact`
/// tool: `{"workspace": ..., "symbol": ...}` or `{"workspace": ..., "path": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactInput {
    pub workspace: String,
    #[serde(default)]
    pub symbol: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

impl ImpactInput {
    /// Checks the input and selects the target. Giving both or neither of
    /// `symbol` and `path` is an error.
    pub fn into_query(self) -> Result<ImpactQuery> {
        validate_workspace(&self.workspace)?;
        let target = match (self.symbol, self.path) {
            (Some(symbol), None) => {
                validate_query(&symbol)?;
                ImpactTarget::Symbol(symbol)
            }
            (None, Some(path)) => {
                validate_relative_path(&path)?;
                ImpactTarget::File(path)
            }
            _ => {
                return Err(RaptorError::InvalidInput(
                    "impact needs exactly one of \"symbol\" or \"path\"".into(),
                ))
            }
        };
        Ok(ImpactQuery {
            workspace: self.workspace,
            target,
        })
    }
}

/// Rejects paths that could leave the workspace: absolute paths, drive letters,
/// `..` segments and control characters. Paths are not checked against the disk.
pub fn validate_relative_path(path: &str) -> Result<()> {
    validate_query(path)?;
    let escapes = path.chars().any(char::is_control)
        || path.starts_with(['/', '\\'])
        || path.contains(':')
        || path.split(['/', '\\']).any(|seg| seg == "..");
    if escapes {
        return Err(RaptorError::InvalidInput(
            "path must be relative to the workspace, without '..' or drive letters".into(),
        ));
    }
    Ok(())
}

/// Used by the `repo_symbols` MCP tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolsRequest {
    pub workspace: String,
    pub name: String,
}

/// Rejects empty, overlong, or control-character-bearing workspace ids.
pub fn validate_workspace(ws: &str) -> Result<()> {
    if ws.is_empty() || ws.len() > MAX_WORKSPACE_LEN || ws.chars().any(char::is_control) {
        return Err(RaptorError::InvalidInput("invalid workspace id".into()));
    }
    Ok(())
}

fn validate_query(q: &str) -> Result<()> {
    if q.trim().is_empty() {
        return Err(RaptorError::InvalidInput("query must not be empty".into()));
    }
    if q.len() > MAX_QUERY_BYTES {
        return Err(RaptorError::InvalidInput(format!(
            "query exceeds {MAX_QUERY_BYTES} bytes"
        )));
    }
    Ok(())
}

impl BuildRequest {
    /// Checks field bounds before the request reaches the engine.
    pub fn validate(&self) -> Result<()> {
        validate_workspace(&self.workspace)?;
        validate_query(&self.query)?;
        if self
            .token_budget
            .is_some_and(|b| b == 0 || b > MAX_TOKEN_BUDGET)
        {
            return Err(RaptorError::InvalidInput(
                "token_budget must be 1-1000000".into(),
            ));
        }
        Ok(())
    }
}

impl SearchRequest {
    /// Checks field bounds before the request reaches the engine.
    pub fn validate(&self) -> Result<()> {
        validate_workspace(&self.workspace)?;
        validate_query(&self.query)?;
        if self.limit.is_some_and(|l| l == 0 || l > MAX_SEARCH_LIMIT) {
            return Err(RaptorError::InvalidInput("limit must be 1-100".into()));
        }
        Ok(())
    }
}

impl RecoverRequest {
    /// Checks field bounds before the request reaches the engine.
    pub fn validate(&self) -> Result<()> {
        validate_workspace(&self.workspace)?;
        validate_query(&self.query)
    }
}

impl ImpactRequest {
    /// Checks field bounds before the request reaches the engine.
    pub fn validate(&self) -> Result<()> {
        validate_workspace(&self.workspace)?;
        validate_query(&self.symbol)
    }
}

impl SymbolsRequest {
    /// Checks field bounds before the request reaches the engine.
    pub fn validate(&self) -> Result<()> {
        validate_workspace(&self.workspace)?;
        validate_query(&self.name)
    }
}

/// Routes for `/raptor/context/*`.
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/raptor/context/build", post(build_route))
        .route("/raptor/context/search", post(search_route))
        .route("/raptor/context/recover", post(recover_route))
        .route("/raptor/context/impact", post(impact_route))
        .route("/raptor/context/status", get(status_route))
}

fn service(st: &AppState) -> std::result::Result<&dyn ContextService, ApiError> {
    st.context.as_deref().ok_or_else(|| {
        ApiError::new(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "No context service is configured.",
        )
        .with_code("context_unavailable")
    })
}

/// Parses a typed JSON body. Errors are OpenAI-shaped 400s.
pub(crate) fn parse_typed<T: DeserializeOwned>(body: &[u8]) -> std::result::Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("Invalid request body: {e}")))
}

fn finish(result: std::result::Result<Response, ApiError>) -> Response {
    result.unwrap_or_else(|e| e.into_response())
}

async fn build_route(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    finish(
        async {
            let req: BuildRequest = parse_typed(&body)?;
            req.validate().map_err(ApiError::from_raptor)?;
            let pack = service(&st)?
                .build(req)
                .await
                .map_err(ApiError::from_raptor)?;
            Ok(Json(pack).into_response())
        }
        .await,
    )
}

async fn search_route(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    finish(
        async {
            let req: SearchRequest = parse_typed(&body)?;
            req.validate().map_err(ApiError::from_raptor)?;
            let v = service(&st)?
                .search(req)
                .await
                .map_err(ApiError::from_raptor)?;
            Ok(Json(v).into_response())
        }
        .await,
    )
}

async fn recover_route(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    finish(
        async {
            let req: RecoverRequest = parse_typed(&body)?;
            req.validate().map_err(ApiError::from_raptor)?;
            let v = service(&st)?
                .recover(req)
                .await
                .map_err(ApiError::from_raptor)?;
            Ok(Json(v).into_response())
        }
        .await,
    )
}

async fn impact_route(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    finish(
        async {
            let input: ImpactInput = parse_typed(&body)?;
            let req = input.into_query().map_err(ApiError::from_raptor)?;
            let v = service(&st)?
                .impact_target(req)
                .await
                .map_err(ApiError::from_raptor)?;
            Ok(Json(v).into_response())
        }
        .await,
    )
}

async fn status_route(State(st): State<Arc<AppState>>) -> Response {
    finish(
        async {
            let v = service(&st)?
                .status()
                .await
                .map_err(ApiError::from_raptor)?;
            Ok(Json(v).into_response())
        }
        .await,
    )
}

/// Adds workspace context to an OpenAI-shaped chat body as a leading system
/// message. Uses the most recent user message as the query. Without user text,
/// or when the engine returns nothing, the body is left unchanged. An engine
/// error fails the request rather than silently dropping the context.
pub(crate) async fn inject_context(
    ctx: &dyn ContextService,
    workspace: &str,
    body: &mut Value,
) -> std::result::Result<(), ApiError> {
    validate_workspace(workspace).map_err(ApiError::from_raptor)?;
    let Some(query) = last_user_text(body) else {
        return Ok(());
    };
    let pack = ctx
        .build(BuildRequest {
            workspace: workspace.to_string(),
            query,
            token_budget: None,
        })
        .await
        .map_err(ApiError::from_raptor)?;
    if pack.text.trim().is_empty() {
        return Ok(());
    }
    let system = json!({
        "role": "system",
        "content": format!("Raptor LLM workspace context:\n{}", pack.text),
    });
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        messages.insert(0, system);
    }
    Ok(())
}

/// Text of the most recent user message, clamped to the query size limit.
fn last_user_text(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let user = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))?;
    let text = match user.get("content")? {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    if text.trim().is_empty() {
        return None;
    }
    Some(clamp_tail(text))
}

/// Keeps the last `MAX_QUERY_BYTES` bytes of `s`, on a char boundary.
fn clamp_tail(s: String) -> String {
    if s.len() <= MAX_QUERY_BYTES {
        return s;
    }
    let mut start = s.len() - MAX_QUERY_BYTES;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impact_input_selects_exactly_one_target() {
        let sym = ImpactInput {
            workspace: "w".into(),
            symbol: Some("Foo".into()),
            path: None,
        };
        assert_eq!(
            sym.into_query().unwrap().target,
            ImpactTarget::Symbol("Foo".into())
        );
        let file = ImpactInput {
            workspace: "w".into(),
            symbol: None,
            path: Some("src/lib.rs".into()),
        };
        assert_eq!(
            file.into_query().unwrap().target,
            ImpactTarget::File("src/lib.rs".into())
        );
        let none = ImpactInput {
            workspace: "w".into(),
            symbol: None,
            path: None,
        };
        assert!(matches!(
            none.into_query(),
            Err(RaptorError::InvalidInput(_))
        ));
        let both = ImpactInput {
            workspace: "w".into(),
            symbol: Some("Foo".into()),
            path: Some("a.rs".into()),
        };
        assert!(both.into_query().is_err());
    }

    #[test]
    fn relative_paths_stay_inside_the_workspace() {
        for ok in ["src/lib.rs", "a/./b.rs", "dir/sub/x.rs", "src\\win.rs"] {
            assert!(validate_relative_path(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "   ",
            "../x.rs",
            "a/../../x.rs",
            "a\\..\\x.rs",
            "/etc/passwd",
            "\\\\server\\share",
            "C:\\x.rs",
            "a\u{0}b.rs",
            "a\nb.rs",
        ] {
            assert!(validate_relative_path(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn workspace_validation() {
        assert!(validate_workspace("proj-1").is_ok());
        assert!(validate_workspace("").is_err());
        assert!(validate_workspace("bad\nname").is_err());
        assert!(validate_workspace(&"x".repeat(513)).is_err());
    }

    #[test]
    fn request_bounds_are_enforced() {
        let mut r = BuildRequest {
            workspace: "w".into(),
            query: "q".into(),
            token_budget: Some(0),
        };
        assert!(r.validate().is_err());
        r.token_budget = Some(4000);
        assert!(r.validate().is_ok());
        r.query = "   ".into();
        assert!(r.validate().is_err());
        let s = SearchRequest {
            workspace: "w".into(),
            query: "q".into(),
            limit: Some(101),
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn last_user_text_takes_latest_user_turn() {
        let body = json!({"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "latest"}, {"type": "image_url"}]}
        ]});
        assert_eq!(last_user_text(&body).as_deref(), Some("latest"));
        assert_eq!(last_user_text(&json!({"messages": []})), None);
    }

    #[test]
    fn clamp_keeps_char_boundary() {
        let s = "é".repeat(MAX_QUERY_BYTES);
        let out = clamp_tail(s);
        assert!(out.len() <= MAX_QUERY_BYTES);
        assert!(out.chars().all(|c| c == 'é'));
    }
}
