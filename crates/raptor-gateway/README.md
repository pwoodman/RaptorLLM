# raptor-gateway

Local API gateway for Raptor LLM. One listener serves:

- an OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`),
- Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`), which Claude Code uses,
- an MCP endpoint (`/mcp`, JSON-RPC over HTTP),
- the context engine routes (`/raptor/context/*`).

Requests go to an inference `Upstream` (llama-server in production) and an optional `ContextService`. Both are traits this crate defines and the desktop shell implements. The crate depends only on `raptor-core`.

## Public API

Crate root re-exports everything below.

### Starting the gateway

```rust
let deps = GatewayDeps { upstream: Arc::new(my_upstream), context: Some(Arc::new(my_ctx)) };
let handle = Gateway::start(GatewayConfig::default(), deps).await?; // 127.0.0.1:8790
handle.addr();                          // bound address (useful with port 0)
handle.status();                        // ListenerStatus: lan, tls, tls_fingerprint_sha256, insecure, keys
handle.metrics();                       // queued / running / completed / rejected
handle.keys();                          // live KeyStore: add / revoke / replace / list
handle.rotate_tls().await?;             // new self-signed cert (or re-read files), no restart
handle.invalidate_model_cache();
handle.reconfigure(new_config).await?;  // validates first; key-only changes skip the restart
handle.shutdown().await?;
```

- `build(&GatewayConfig, GatewayDeps) -> Result<Built>`: validated `Router`, `Scheduler` and `KeyStore`, for embedding or `tower::ServiceExt::oneshot` tests.
- `GatewayConfig` fields: `bind: BindMode` (`Localhost` | `Lan`), `port`, `api_keys: Vec<ApiKeyRecord>`, `allow_keyless_loopback` (default `true`), `cors_allowed_origins` (exact origins, default none), `max_body_bytes` (32 MiB), `queue: QueueConfig`, `model_aliases`, `workspace_root`, `drain_timeout`, `tls: Option<TlsConfig>` (default `None`), `allow_insecure_lan` (default `false`), `catalog_ttl` (default 2 s; zero disables).
- `GatewayConfig::validate()` refuses `Lan` without a key (`Unauthorized`), `Lan` without `tls` unless `allow_insecure_lan` (`InvalidInput`), keyless-off without keys, duplicate key ids, wildcard or path-bearing CORS origins, bad TLS settings, and zero limits. `Gateway::start` validates before it reads TLS files or binds.

### TLS

```rust
pub enum TlsConfig {
    Files { cert_pem: PathBuf, key_pem: PathBuf },           // user-provided PEM chain and key
    SelfSigned { dir: PathBuf, subject_alt_names: Vec<String> }, // generated once, reused on restart
}
```

- Self-signed certificates are P-256, created with `rcgen`, and cover `localhost`, `127.0.0.1` and `::1` plus any extra names. They are stored as `gateway-cert.pem` and `gateway-key.pem` in `dir`. A stored pair is reused if it parses, and regenerated if it does not.
- `ListenerStatus::tls_fingerprint_sha256` is the lowercase hex SHA-256 of the leaf certificate, for client pinning. `tls::display_fingerprint` formats it as `AB:CD:...`.
- The TLS provider is `ring` through `rustls` and `axum-server`. No OpenSSL, CMake or NASM is needed.
- `tls::ensure_self_signed(dir, names)` and `tls::regenerate_self_signed(dir, names)` are also public, so the shell can show a fingerprint before the listener starts.

### Inference backend

```rust
#[async_trait]
pub trait Upstream: Send + Sync {
    async fn models(&self) -> Result<ModelCatalog>;            // installed + loaded
    async fn call(&self, endpoint: Endpoint, body: Value) -> Result<Completion>;
    async fn tokenize(&self, text: &str) -> Result<Option<u64>> { Ok(None) } // default: no tokenizer
}
pub enum Completion { Json(Value), Stream(SseStream) }        // SseStream yields raw `data:` payloads
```

The body is forwarded as received, except `model`, which is resolved to an installed or loaded id. Dropping a `Completion::Stream` must cancel the upstream work. `HttpUpstream::new(base_url, installed)` implements the trait over a loopback llama-server, including `/tokenize`.

The model catalog is cached for `catalog_ttl`. Errors are not cached. The cache is dropped on `reconfigure` and by `invalidate_model_cache`.

### Context engine

```rust
#[async_trait]
pub trait ContextService: Send + Sync {
    async fn build(&self, req: BuildRequest) -> Result<ContextPack>;
    async fn search(&self, req: SearchRequest) -> Result<Value>;
    async fn recover(&self, req: RecoverRequest) -> Result<Value>;
    async fn impact(&self, req: ImpactRequest) -> Result<Value>;          // symbol only (required)
    async fn impact_target(&self, req: ImpactQuery) -> Result<Value>;     // symbol or file; default delegates symbols, refuses files
    async fn symbols(&self, req: SymbolsRequest) -> Result<Value>;
    async fn status(&self) -> Result<Value>;
}
```

Requests are validated by the gateway before the call (non-empty, size-bounded). `ContextPack { text, details }`.

### Auth and keys

- `ApiKeyRecord::new(id, secret)` stores `SHA-256(salt || secret)` with a random 16-byte salt. The plaintext is not kept, and `Debug` prints `<redacted>`.
- `ApiKeyRecord::generate(id) -> (record, secret)` returns `nxk_` plus 64 hex characters. Show the secret to the user once.
- Each record also has a label (`with_label`), a creation time (`created_at()`), and optional per-key limits (`with_limits(KeyLimits)`):
  - `max_concurrent`: running plus waiting requests for this key. It replaces the queue's `per_client_limit` for this key only.
  - `requests_per_minute`: a rolling 60-second window. Refusals return 429 with `Retry-After`.
- Persistence: `ApiKeyRecord` serializes as `StoredApiKey { key_id, salt, digest, label?, created_unix?, max_concurrent?, requests_per_minute? }`. The optional fields are omitted when unset, and records from older versions still load. The secret never serializes.
- `KeyStore` is the live key set shared with the listener. `add`, `revoke(id)`, `replace(keys)` and `list() -> Vec<KeyInfo>` take effect on the next request, with no restart. `list` never includes secrets or digests. Requests already in flight finish.
- `Authenticator` / `AccessPolicy` / `Principal` are used by the middleware. `Principal` carries the matched key's `limits`.
- Presented keys (`Authorization: Bearer` or `x-api-key`) are compared in constant time against every record.

### Queue

`Scheduler::new(QueueConfig)`, `acquire(client_id)`, `acquire_with(client_id, per_client_override)`, `metrics() -> Metrics`. `Rejection` maps to 429 (per-client limit) or 503 (queue full, queue timeout, shutting down), each with `Retry-After`. `RateLimiter::check` is the per-key requests-per-minute window.

### HTTP surface

| Route | Auth | Notes |
|---|---|---|
| `GET /health` | no | `{"status":"ok"}` |
| `GET /raptor/metrics` | yes | scheduler counters |
| `GET /v1/models` | yes | installed models plus configured aliases (cached) |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | yes | SSE when `stream: true`, always ends with one `data: [DONE]` |
| `POST /v1/messages` | yes | Anthropic request and response, Anthropic error envelope |
| `POST /v1/messages/count_tokens` | yes | `{"input_tokens", "approximate": true, "method": "tokenize"\|"estimate"}`; no `max_tokens` needed |
| `POST /raptor/context/{build,search,recover,impact}` (impact takes `symbol` or `path`, exactly one), `GET /raptor/context/status` | yes | 503 `context_unavailable` without a service |
| `POST /mcp`, `DELETE /mcp` | yes | JSON-RPC. `DELETE` ends the session named by `Mcp-Session-Id` |

Headers: `x-raptor-workspace: <id>` on a chat request injects that workspace's context as a leading system message (the latest user message is the query; an engine error fails the request). Auth accepts `Authorization: Bearer` and `x-api-key`. Anthropic routes ignore the `?beta=...` query and the `anthropic-beta` and `anthropic-version` headers.

Errors are OpenAI-shaped `{"error":{"message","type","param","code"}}`, except on `/v1/messages`, which uses `{"type":"error","error":{...}}`. Any non-JSON error from axum (404, 405, 413, 415) is re-rendered in the route's shape. POST bodies must be `application/json`. Per-key rate limits return 429 `rate_limit_exceeded`.

MCP tools: `context_build`, `context_search`, `context_recover`, `repo_symbols`, `repo_impact` (`symbol` or `path`, forwarded to `ContextService::impact_target`), and `repo_read_file { path }` (reads under `workspace_root`). Each tool has an `outputSchema`, and successful results carry `structuredContent`.

## Security model

- Default bind is `127.0.0.1`. `Lan` binds `0.0.0.0` and requires at least one key and TLS. The only way around the TLS rule is `allow_insecure_lan`. That logs a warning on every start and reconfigure, and `ListenerStatus::insecure` reports it.
- Keyless access is allowed only from loopback peers with a loopback `Host` header. A foreign `Host` gets 403, which blocks DNS rebinding in both modes.
- Keys are compared in constant time against all records, and are never logged or echoed.
- CORS is an exact-origin allowlist. Empty by default, so no cross-origin reads.
- A self-signed private key file is created owner-only (`0600`) on Unix. If a stored one has group or other bits, they are removed on load, with a warning. User-provided key files are not changed, only warned about. On Windows the file inherits its directory's ACL.
- `repo_read_file` rejects `..` and `.` segments (including Windows dot-and-space forms), absolute and UNC paths, drive letters and `:` (alternate data streams), reserved device names, and control characters. It then canonicalises the real path, so symlinks and junctions must still resolve under the root. It returns only regular UTF-8 files up to 1 MiB.
- Chat, completion, embedding, messages, count-tokens and context requests are capped at `max_body_bytes`.

## Status

**Implemented and tested** (`cargo test -p raptor-gateway`: 176 tests, 93 unit and 83 integration; `clippy -D warnings` and `fmt` clean):

- **TLS**: optional for either bind mode. User-provided PEM files, or a self-signed certificate persisted under a caller-supplied directory with its SHA-256 fingerprint exposed. Tests cover a handshake pinned to the self-signed certificate through reqwest, a client without the pin being refused, user PEM files, a LAN listener with TLS, rotation on a live listener (the old pin is refused afterwards), and bad TLS files failing before the bind. Owner-only key permissions are tested on Unix only.
- **LAN policy**: LAN without a key is refused. LAN without TLS is refused unless `allow_insecure_lan`, which reports `insecure` in the status. Plain-HTTP localhost is unchanged.
- **Keys**: several keys, each with a label, creation time and optional per-key limits. Revocation takes effect on the live listener (tested over real sockets). A key-only `reconfigure` keeps the listener and its port. Per-key concurrency and rate limits are tested. Duplicate ids are rejected.
- **OpenAI pass-through**: tests show `response_format` (`json_schema` and `json_object`), `stream_options`, `n`, `stop`, `logprobs`, `top_logprobs` and `seed` reach the upstream unchanged except for `model`. The usage chunk streams byte-identical. Streamed tool-call deltas pass through byte-for-byte.
- **Anthropic**: streamed `tool_use` round trip (golden files), `tool_result` with `is_error`, `system` as an array of blocks, `max_tokens` required and positive, unknown beta query and headers tolerated. `POST /v1/messages/count_tokens` uses the upstream `/tokenize` when it exists, otherwise a length estimate, and is always reported as approximate.
- **MCP**: `Mcp-Session-Id` is issued on `initialize` and checked whenever a request sends it. `DELETE` ends a session. Protocol version negotiation (`2025-06-18`, `2025-03-26`, `2024-11-05`). `ping` and `notifications/initialized`. Output schemas and `structuredContent`. The `repo_read_file` confinement tests are unchanged.
- **Model catalog**: cached for 2 s by default, not cached on error, and dropped on `reconfigure` and on explicit invalidation.
- Earlier work is unchanged: queue fairness and limits, cancellation on client disconnect, the Host check, the CORS allowlist, and the `/v1/messages` and OpenAI error shapes.

**Stubbed or limited:**

- `ContextService` has no real implementation in this crate. The desktop shell supplies it. Tests use fakes.
- MCP has no GET SSE stream. Sessions are not required (a request without the header is accepted), are not expired by time, and at most 64 are kept. `notifications/initialized` is accepted but not enforced.
- Anthropic: image and document blocks are rejected (400). `thinking` blocks are dropped. Upstream `reasoning_content` is not translated.
- Token counts exclude chat-template overhead, so they are approximate even with `/tokenize`.
- Self-signed certificates use the rcgen default validity. It is not configurable here. Changing `subject_alt_names` does not reissue an existing certificate; call `rotate_tls` or `regenerate_self_signed`. Clients pin the fingerprint, so names are not checked.
- Windows: the key file is not ACL-restricted by this crate. It relies on the directory's inherited ACL, and no test covers Windows ACLs.
- Revoking a key does not end its in-flight requests. The per-key rate window is in memory and resets when the listener is rebuilt.
- `HttpUpstream` sets only a connect timeout. Non-streamed generation has no overall timeout, and a stalled stream lasts until the client disconnects.
- Plain-HTTP localhost remains the default. LAN with `allow_insecure_lan` sends keys in clear text, and the shell must decide when to use it.
- `repo_read_file` checks the path and then reads it. A file swapped for a link between those two steps is not caught (TOCTOU window).
- On Windows, the file symlink test is skipped without the symlink privilege. The junction test runs in ordinary accounts.
- `reconfigure` restarts queue counters and drops in-flight connections that outlast the drain timeout.
- The default port 8790 is a placeholder. The shell should set the real value.
