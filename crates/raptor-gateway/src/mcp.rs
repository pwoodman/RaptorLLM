//! MCP server over streamable HTTP (`POST /mcp`, `DELETE /mcp`), hand-rolled
//! JSON-RPC 2.0.
//!
//! Implements `initialize` (with protocol-version negotiation), `ping`,
//! `tools/list`, `tools/call`, `notifications/*` (answered with `202 Accepted`,
//! `notifications/initialized` included) and batches.
//!
//! Sessions: `initialize` returns an `Mcp-Session-Id` header. A later request
//! that carries that header must name a live session, or it gets `404`. Requests
//! without the header are accepted, so simple clients still work. `DELETE /mcp`
//! ends a session. The session table holds at most [`MAX_SESSIONS`] ids, and the
//! oldest is dropped when it is full. Server-initiated streams (GET) are not
//! implemented: `GET /mcp` returns 405.
//!
//! Tools:
//! * `context_build`, `context_search`, `context_recover`, `repo_symbols`,
//!   `repo_impact` are forwarded to the [`ContextService`]. Each declares an
//!   `outputSchema`, and its successful results carry `structuredContent`.
//! * `repo_read_file` reads a UTF-8 file under the configured workspace root.
//!   It is the one tool that touches the filesystem directly, so it carries the
//!   path confinement rules described on [`repo_read_file`].

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use raptor_core::{RaptorError, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;

use crate::context_api::{
    BuildRequest, ImpactInput, RecoverRequest, SearchRequest, SymbolsRequest,
};
use crate::server::AppState;

/// Latest MCP protocol revision this server speaks.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Every revision this server can speak, newest first. `initialize` answers
/// with the client's requested version when it is in this list, and with
/// [`PROTOCOL_VERSION`] otherwise.
pub const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// Header that carries the session id.
pub const SESSION_HEADER: &str = "mcp-session-id";

/// Upper bound on live sessions. The oldest is dropped beyond it.
pub const MAX_SESSIONS: usize = 64;

/// Largest file `repo_read_file` will return.
pub const MAX_READ_BYTES: u64 = 1024 * 1024;

const MAX_PATH_LEN: usize = 1024;

const TOOL_NAMES: [&str; 6] = [
    "context_build",
    "context_search",
    "context_recover",
    "repo_symbols",
    "repo_impact",
    "repo_read_file",
];

/// Live MCP session ids.
#[derive(Default)]
pub(crate) struct Sessions {
    ids: Mutex<VecDeque<String>>,
}

impl Sessions {
    /// Starts a session with a random 128-bit id.
    pub(crate) fn create(&self) -> Result<String> {
        let mut raw = [0u8; 16];
        getrandom::getrandom(&mut raw)
            .map_err(|e| RaptorError::Internal(format!("rng failure: {e}")))?;
        let id = hex::encode(raw);
        let mut ids = self.lock();
        if ids.len() >= MAX_SESSIONS {
            ids.pop_front();
        }
        ids.push_back(id.clone());
        Ok(id)
    }

    /// True when `id` is a live session. Compared in constant time.
    pub(crate) fn contains(&self, id: &str) -> bool {
        self.lock()
            .iter()
            .any(|s| bool::from(s.as_bytes().ct_eq(id.as_bytes())))
    }

    /// Ends a session. Returns `false` when it was not live.
    pub(crate) fn remove(&self, id: &str) -> bool {
        let mut ids = self.lock();
        let before = ids.len();
        ids.retain(|s| !bool::from(s.as_bytes().ct_eq(id.as_bytes())));
        ids.len() != before
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<String>> {
        self.ids.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Routes for `/mcp`.
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/mcp", post(mcp_post).delete(mcp_delete))
}

async fn mcp_post(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let message: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(rpc_error(Value::Null, -32700, "Parse error")),
            )
                .into_response();
        }
    };
    if let Some(v) = headers.get("mcp-protocol-version") {
        let known = v
            .to_str()
            .is_ok_and(|s| SUPPORTED_PROTOCOL_VERSIONS.contains(&s));
        if !known {
            return (
                StatusCode::BAD_REQUEST,
                Json(rpc_error(
                    Value::Null,
                    -32600,
                    "Unsupported MCP-Protocol-Version",
                )),
            )
                .into_response();
        }
    }
    let is_initialize = message.get("method").and_then(Value::as_str) == Some("initialize");
    let mut new_session = None;
    if is_initialize {
        match st.mcp.create() {
            Ok(id) => new_session = Some(id),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(rpc_error(Value::Null, -32603, &e.to_string())),
                )
                    .into_response();
            }
        }
    } else if let Some(id) = session_id(&headers) {
        if !st.mcp.contains(id) {
            return (
                StatusCode::NOT_FOUND,
                Json(rpc_error(
                    Value::Null,
                    -32001,
                    "Unknown or expired MCP session",
                )),
            )
                .into_response();
        }
    }

    let mut response = match message {
        Value::Array(items) => {
            if items.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(rpc_error(Value::Null, -32600, "Invalid Request")),
                )
                    .into_response();
            }
            let mut replies = Vec::new();
            for item in &items {
                if let Some(r) = handle_one(&st, item).await {
                    replies.push(r);
                }
            }
            if replies.is_empty() {
                StatusCode::ACCEPTED.into_response()
            } else {
                Json(Value::Array(replies)).into_response()
            }
        }
        single => match handle_one(&st, &single).await {
            Some(reply) => Json(reply).into_response(),
            None => StatusCode::ACCEPTED.into_response(),
        },
    };
    if let Some(id) = new_session {
        if let Ok(v) = HeaderValue::from_str(&id) {
            response.headers_mut().insert(SESSION_HEADER, v);
        }
    }
    response
}

/// `DELETE /mcp` with an `Mcp-Session-Id` ends that session.
async fn mcp_delete(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(id) = session_id(&headers) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(rpc_error(
                Value::Null,
                -32600,
                "Mcp-Session-Id header is required",
            )),
        )
            .into_response();
    };
    if st.mcp.remove(id) {
        StatusCode::OK.into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(rpc_error(
                Value::Null,
                -32001,
                "Unknown or expired MCP session",
            )),
        )
            .into_response()
    }
}

fn session_id(headers: &HeaderMap) -> Option<&str> {
    headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok())
}

struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Handles one JSON-RPC message. Returns `None` for notifications.
async fn handle_one(st: &AppState, msg: &Value) -> Option<Value> {
    let Some(obj) = msg.as_object() else {
        return Some(rpc_error(Value::Null, -32600, "Invalid Request"));
    };
    let id = obj.get("id").cloned();
    let reply_id = id.clone().unwrap_or(Value::Null);
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(rpc_error(reply_id, -32600, "Invalid Request"));
    }
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        return Some(rpc_error(reply_id, -32600, "Invalid Request"));
    };
    let params = obj.get("params").cloned().unwrap_or(Value::Null);

    let result: std::result::Result<Value, RpcError> = match method {
        "initialize" => {
            // Echo the client's version when this server speaks it, else offer the latest.
            let requested = params.get("protocolVersion").and_then(Value::as_str);
            let version = match requested {
                Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v,
                _ => PROTOCOL_VERSION,
            };
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "raptor-gateway", "version": env!("CARGO_PKG_VERSION")},
            }))
        }
        // Liveness check, valid before and after initialization.
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_definitions()})),
        "tools/call" => call_tool(st, &params).await,
        _ => Err(RpcError::new(-32601, "Method not found")),
    };
    // Notifications (no id) never get a reply, even on error.
    let id = id?;
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => rpc_error_with_code(id, e),
    })
}

fn rpc_error_with_code(id: Value, e: RpcError) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": e.code, "message": e.message}})
}

async fn call_tool(st: &AppState, params: &Value) -> std::result::Result<Value, RpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(-32602, "tools/call requires a tool name"))?;
    if !TOOL_NAMES.contains(&name) {
        return Err(RpcError::new(-32602, format!("Unknown tool: {name}")));
    }
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    // Tool failures are reported inside the result with isError, per MCP.
    Ok(match run_tool(st, name, &args).await {
        // Every declared output schema is an object, so a non-object result
        // would break the contract. It is reported as a tool error instead.
        Ok(v) if v.is_object() => structured_result(v),
        Ok(_) => tool_result(&format!("tool {name} returned a non-object result"), true),
        Err(e) => tool_result(&e.to_string(), true),
    })
}

fn tool_result(text: &str, is_error: bool) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": is_error})
}

/// A successful result: the JSON as `structuredContent`, and the same JSON as
/// text for clients that ignore structured output.
fn structured_result(v: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": v.to_string()}],
        "structuredContent": v,
        "isError": false,
    })
}

/// JSON Schema for each tool's `structuredContent`.
fn output_schema(name: &str) -> Value {
    match name {
        "context_build" => json!({"type": "object",
            "properties": {"text": {"type": "string"}, "details": {}},
            "required": ["text"]}),
        "repo_read_file" => json!({"type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}),
        _ => json!({"type": "object"}),
    }
}

async fn run_tool(st: &AppState, name: &str, args: &Value) -> Result<Value> {
    if name == "repo_read_file" {
        let path: String = parse_args(args, "path")?;
        let root = st
            .workspace_root
            .as_deref()
            .ok_or_else(|| RaptorError::Unsupported("no workspace root is configured".into()))?;
        let content = repo_read_file(root, &path)?;
        return Ok(json!({"path": path, "content": content}));
    }

    let svc = st
        .context
        .as_deref()
        .ok_or_else(|| RaptorError::Unsupported("no context service is configured".into()))?;
    match name {
        "context_build" => {
            let req: BuildRequest = parse_request(args)?;
            req.validate()?;
            serde_json::to_value(svc.build(req).await?)
                .map_err(|e| RaptorError::Internal(e.to_string()))
        }
        "context_search" => {
            let req: SearchRequest = parse_request(args)?;
            req.validate()?;
            svc.search(req).await
        }
        "context_recover" => {
            let req: RecoverRequest = parse_request(args)?;
            req.validate()?;
            svc.recover(req).await
        }
        "repo_symbols" => {
            let req: SymbolsRequest = parse_request(args)?;
            req.validate()?;
            svc.symbols(req).await
        }
        "repo_impact" => {
            let input: ImpactInput = parse_request(args)?;
            svc.impact_target(input.into_query()?).await
        }
        other => Err(RaptorError::Internal(format!(
            "tool dispatch missing for {other}"
        ))),
    }
}

fn parse_request<T: DeserializeOwned>(args: &Value) -> Result<T> {
    serde_json::from_value(args.clone())
        .map_err(|e| RaptorError::InvalidInput(format!("invalid arguments: {e}")))
}

fn parse_args<T: DeserializeOwned>(args: &Value, key: &str) -> Result<T> {
    let v = args
        .get(key)
        .cloned()
        .ok_or_else(|| RaptorError::InvalidInput(format!("missing argument '{key}'")))?;
    serde_json::from_value(v)
        .map_err(|e| RaptorError::InvalidInput(format!("invalid argument '{key}': {e}")))
}

/// Tool descriptors for `tools/list`.
pub fn tool_definitions() -> Value {
    let ws = json!({"type": "string", "description": "Workspace id"});
    let q = json!({"type": "string", "description": "Query text"});
    let mut tools = json!([
        {
            "name": "context_build",
            "description": "Build a token-budgeted context pack for a query.",
            "inputSchema": {"type": "object", "properties": {
                "workspace": ws, "query": q,
                "token_budget": {"type": "integer", "minimum": 1}
            }, "required": ["workspace", "query"]}
        },
        {
            "name": "context_search",
            "description": "Search the indexed workspace.",
            "inputSchema": {"type": "object", "properties": {
                "workspace": ws, "query": q,
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }, "required": ["workspace", "query"]}
        },
        {
            "name": "context_recover",
            "description": "Recover context lost to truncation or a session boundary.",
            "inputSchema": {"type": "object", "properties": {"workspace": ws, "query": q},
                "required": ["workspace", "query"]}
        },
        {
            "name": "repo_symbols",
            "description": "List symbols matching a name.",
            "inputSchema": {"type": "object", "properties": {
                "workspace": ws, "name": {"type": "string"}
            }, "required": ["workspace", "name"]}
        },
        {
            "name": "repo_impact",
            "description": "List what depends on a symbol or a workspace-relative file path. Give exactly one of symbol or path.",
            "inputSchema": {"type": "object", "properties": {
                "workspace": ws,
                "symbol": {"type": "string"},
                "path": {"type": "string"}
            }, "required": ["workspace"]}
        },
        {
            "name": "repo_read_file",
            "description": "Read a UTF-8 text file under the workspace root. Relative paths only.",
            "inputSchema": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"}
            }, "required": ["path"]}
        }
    ]);
    if let Some(list) = tools.as_array_mut() {
        for tool in list {
            let name = tool["name"].as_str().unwrap_or("").to_string();
            tool["outputSchema"] = output_schema(&name);
        }
    }
    tools
}

/// Reads `rel` from under `root`.
///
/// Rejected before touching the disk: empty or overlong paths, control
/// characters, absolute paths (`/`, `\`, UNC, `\\?\`), any `:` (drive letters
/// such as `C:x`, NTFS alternate streams such as `f.txt:s`), empty segments,
/// `.` and `..` segments (and segments made only of dots and spaces, which
/// Windows normalises to `..`), and reserved DOS device names (`CON`, `NUL`,
/// `COM1`, ...).
///
/// After that the path is resolved with symlinks followed, and the real path
/// must still lie under the canonical root. That blocks symlinks and junctions
/// pointing outside the workspace. Only regular files up to [`MAX_READ_BYTES`]
/// are returned, and the bytes must be valid UTF-8.
pub fn repo_read_file(root: &Path, rel: &str) -> Result<String> {
    let rel_path = confine_relative(rel)?;
    let root = root
        .canonicalize()
        .map_err(|e| RaptorError::Io(format!("workspace root: {e}")))?;
    let target = root.join(&rel_path);
    let real = target.canonicalize().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => RaptorError::NotFound(format!("no such file: {rel}")),
        _ => RaptorError::Io(format!("resolving {rel}: {e}")),
    })?;
    if !real.starts_with(&root) {
        return Err(RaptorError::InvalidInput(
            "path resolves outside the workspace".into(),
        ));
    }
    let meta = std::fs::metadata(&real)?;
    if !meta.is_file() {
        return Err(RaptorError::InvalidInput(format!(
            "not a regular file: {rel}"
        )));
    }
    if meta.len() > MAX_READ_BYTES {
        return Err(RaptorError::InvalidInput(format!(
            "file exceeds {MAX_READ_BYTES} bytes"
        )));
    }
    let bytes = std::fs::read(&real)?;
    String::from_utf8(bytes)
        .map_err(|_| RaptorError::InvalidInput(format!("not UTF-8 text: {rel}")))
}

/// Syntactic checks on a client-supplied relative path. Never touches the disk.
fn confine_relative(rel: &str) -> Result<PathBuf> {
    let bad = |msg: &str| RaptorError::InvalidInput(format!("{msg}: {rel:?}"));
    if rel.is_empty() || rel.len() > MAX_PATH_LEN {
        return Err(bad("path must be 1-1024 bytes"));
    }
    if rel.chars().any(|c| c == '\0' || c.is_control()) {
        return Err(bad("control characters are not allowed in paths"));
    }
    if rel.starts_with('/') || rel.starts_with('\\') {
        return Err(bad("absolute paths are not allowed"));
    }
    if rel.contains(':') {
        return Err(bad("drive letters and ':' are not allowed"));
    }
    let mut out = PathBuf::new();
    for segment in rel.split(['/', '\\']) {
        if segment.is_empty() {
            return Err(bad("empty path segment"));
        }
        if segment.chars().all(|c| c == '.' || c == ' ') {
            return Err(bad("'.' and '..' segments are not allowed"));
        }
        if is_reserved_device(segment) {
            return Err(bad("reserved device names are not allowed"));
        }
        out.push(segment);
    }
    Ok(out)
}

fn is_reserved_device(segment: &str) -> bool {
    let stem = segment
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || (stem.len() == 4
        && (stem.starts_with("COM") || stem.starts_with("LPT"))
        && stem.as_bytes()[3].is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_and_windows_forms_are_rejected_without_io() {
        let cases = [
            "../secret.txt",
            "src/../../secret.txt",
            "..\\secret.txt",
            "src\\..\\..\\secret.txt",
            "..",
            "a/../b",
            "/etc/passwd",
            "\\etc\\passwd",
            "\\\\server\\share\\file.txt",
            "\\\\?\\C:\\Windows\\win.ini",
            "C:\\Windows\\win.ini",
            "C:win.ini",
            "D:/secret",
            "notes.txt:hidden",
            "CON",
            "nul.txt",
            "dir/COM1.log",
            "LPT9",
            "...",
            ".. ",
            "src//lib.rs",
            "src/",
            "",
            "a\0b",
            "tab\tname",
        ];
        for c in cases {
            assert!(confine_relative(c).is_err(), "should reject {c:?}");
        }
    }

    #[test]
    fn plain_relative_paths_are_accepted() {
        assert_eq!(
            confine_relative("src/lib.rs").unwrap(),
            PathBuf::from("src").join("lib.rs")
        );
        assert!(confine_relative("src\\lib.rs").is_ok());
        assert!(confine_relative("my file.rs").is_ok());
        assert!(confine_relative("CONSOLE.md").is_ok());
    }

    #[test]
    fn reads_file_inside_workspace() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("src")).unwrap();
        std::fs::write(root.path().join("src").join("lib.rs"), "fn main() {}").unwrap();
        assert_eq!(
            repo_read_file(root.path(), "src/lib.rs").unwrap(),
            "fn main() {}"
        );
        assert_eq!(
            repo_read_file(root.path(), "src\\lib.rs").unwrap(),
            "fn main() {}"
        );
    }

    #[test]
    fn missing_directory_and_large_files_are_errors() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("dir")).unwrap();
        assert!(matches!(
            repo_read_file(root.path(), "missing.txt"),
            Err(RaptorError::NotFound(_))
        ));
        assert!(repo_read_file(root.path(), "dir").is_err());
        let big = vec![b'a'; (MAX_READ_BYTES + 1) as usize];
        std::fs::write(root.path().join("big.txt"), big).unwrap();
        assert!(repo_read_file(root.path(), "big.txt").is_err());
    }

    #[test]
    fn non_utf8_file_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("bin.dat"), [0xff, 0xfe, 0x00]).unwrap();
        assert!(repo_read_file(root.path(), "bin.dat").is_err());
    }

    #[test]
    fn absolute_path_to_real_file_outside_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "outside").unwrap();
        let abs = secret.to_string_lossy().to_string();
        assert!(repo_read_file(root.path(), &abs).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "outside").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            root.path().join("link.txt"),
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("linkdir")).unwrap();
        assert!(matches!(
            repo_read_file(root.path(), "link.txt"),
            Err(RaptorError::InvalidInput(_))
        ));
        assert!(matches!(
            repo_read_file(root.path(), "linkdir/secret.txt"),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[cfg(windows)]
    #[test]
    fn symlink_escape_is_rejected_when_links_are_permitted() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "outside").unwrap();
        match std::os::windows::fs::symlink_file(
            outside.path().join("secret.txt"),
            root.path().join("link.txt"),
        ) {
            Ok(()) => assert!(repo_read_file(root.path(), "link.txt").is_err()),
            Err(e) => eprintln!("skipping symlink case, cannot create symlinks here: {e}"),
        }
    }

    /// Directory junctions need no special privilege, so this case runs on
    /// ordinary Windows accounts. It is the usual way to escape a workspace.
    #[cfg(windows)]
    #[test]
    fn junction_escape_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "outside").unwrap();
        let link = root.path().join("jdir");
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(outside.path())
            .output();
        match status {
            Ok(out) if out.status.success() => {
                assert!(matches!(
                    repo_read_file(root.path(), "jdir/secret.txt"),
                    Err(RaptorError::InvalidInput(_))
                ));
            }
            _ => eprintln!("skipping junction case, mklink /J unavailable"),
        }
    }

    #[test]
    fn reserved_names_are_detected() {
        assert!(is_reserved_device("con"));
        assert!(is_reserved_device("NUL.txt"));
        assert!(is_reserved_device("com3.log"));
        assert!(!is_reserved_device("console.txt"));
        assert!(!is_reserved_device("COM10"));
    }
}
