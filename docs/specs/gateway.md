# raptor-gateway - API Gateway spec
Owns: HTTP server (axum), OpenAI-compatible API, auth, LAN toggle, request queue, MCP surface, harness adapters.

Modules
- `server`: `Gateway::start(GatewayConfig) -> Handle` (axum + tokio). Default bind 127.0.0.1:<port> (configurable). LAN mode = bind 0.0.0.0 ONLY if an API key is configured (else `Err(Unauthorized)`); `reconfigure()` with graceful drain. Graceful shutdown.
- `openai`: `GET /v1/models` (installed models + aliases), `POST /v1/chat/completions` (SSE stream + non-stream; pass-through proxy to the upstream llama-server behind trait `Upstream`, fake in tests; correct SSE framing incl. `data: [DONE]`; OpenAI error JSON shape `{error:{message,type,code}}`; client disconnect => cancel upstream), `/v1/completions`, `/v1/embeddings` pass-through, `/health`. Tool calling and `response_format` JSON pass through untouched, including streamed tool_call deltas. Model alias mapping (e.g. `gpt-4`, `claude-*` -> the loaded model).
- `auth`: bearer API key (constant-time compare, keys stored salted+hashed), required when LAN is on; localhost may be keyless by config. 401 in OpenAI error shape. Never log keys. CORS locked down by default (allowlist).
- `queue`: scheduler with bounded concurrency per upstream slot, FIFO with per-client limits (api-key id / IP), queue-depth cap -> 429/503 with Retry-After, cancel on disconnect, metrics (queued, running, completed, rejected).
- `context_api`: trait `ContextService` (defined in THIS crate; the desktop shell implements it with raptor-context) + routes `POST /raptor/context/build`, `/search`, `/recover`, `/impact`, `GET /raptor/context/status`. Optional automatic injection for generic clients via header `x-raptor-workspace`.
- `mcp`: MCP server (streamable HTTP, JSON-RPC: `initialize`, `tools/list`, `tools/call`) exposing tools `context_build`, `context_search`, `context_recover`, `repo_symbols`, `repo_impact`, `repo_read_file` (workspace-confined; path traversal tested). Hand-rolled JSON-RPC is fine.
- `adapters`: OpenCode / generic OpenAI client normalisation, and Claude Code support via Anthropic `/v1/messages` -> OpenAI translation (non-stream + streaming + tool_use blocks). Golden-file tests of translated requests/responses.

Tests: >= 45 using `tower::ServiceExt::oneshot` and a fake upstream: SSE framing, auth matrix (localhost/LAN/key), queue limits, disconnect cancel, MCP path traversal, Anthropic translation golden tests.
Allowed deps: axum, tower, tower-http, hyper, reqwest, futures-util, bytes, async-stream, sha2, subtle.
