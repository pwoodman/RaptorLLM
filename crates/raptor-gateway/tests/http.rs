//! HTTP-level tests. They drive the router with `tower::ServiceExt::oneshot`
//! against a fake `Upstream` and fake `ContextService`, and use real sockets
//! only for the lifecycle tests at the end.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use raptor_core::{RaptorError, Result};
use raptor_gateway::{
    build, ApiKeyRecord, BindMode, BuildRequest, Completion, ContextPack, ContextService, Endpoint,
    Gateway, GatewayConfig, GatewayDeps, ModelCatalog, QueueConfig, Scheduler, TlsConfig, Upstream,
};
use serde_json::{json, Value};
use tower::ServiceExt;

const LOCAL: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const LOCAL_2: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));
const LAN: IpAddr = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20));
const KEY: &str = "nxk_test_secret_0123456789";

// ---------------------------------------------------------------- fakes

#[derive(Clone)]
enum Item {
    Data(String),
    Fail(String),
}

#[derive(Clone)]
enum Script {
    Json(Value),
    Stream(Vec<Item>),
    /// Streams forever. Sets the flag when the stream is dropped.
    Endless(Arc<AtomicBool>),
    FailCall(String),
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct FakeUpstream {
    script: Script,
    calls: Mutex<Vec<(Endpoint, Value)>>,
    models_calls: std::sync::atomic::AtomicUsize,
    tokenizer: bool,
    /// Sleep before answering a non-streamed call. Lets tests overlap requests.
    delay: Duration,
    /// What the fake model's chat template says about reasoning effort.
    caps: raptor_gateway::ThinkingCaps,
}

impl FakeUpstream {
    fn new(script: Script) -> Arc<Self> {
        Arc::new(Self {
            script,
            calls: Mutex::new(Vec::new()),
            models_calls: std::sync::atomic::AtomicUsize::new(0),
            tokenizer: false,
            delay: Duration::ZERO,
            caps: raptor_gateway::ThinkingCaps::default(),
        })
    }

    /// A fake whose `tokenize` answers 42 tokens for any text.
    fn with_tokenizer(script: Script) -> Arc<Self> {
        Arc::new(Self {
            script,
            calls: Mutex::new(Vec::new()),
            models_calls: std::sync::atomic::AtomicUsize::new(0),
            tokenizer: true,
            delay: Duration::ZERO,
            caps: raptor_gateway::ThinkingCaps::default(),
        })
    }

    /// A fake that waits `delay` before each non-streamed answer.
    fn slow(script: Script, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            script,
            calls: Mutex::new(Vec::new()),
            models_calls: std::sync::atomic::AtomicUsize::new(0),
            tokenizer: false,
            delay,
            caps: raptor_gateway::ThinkingCaps::default(),
        })
    }

    /// A fake whose model supports reasoning effort with the given template levels.
    fn thinking(script: Script, levels: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            script,
            calls: Mutex::new(Vec::new()),
            models_calls: std::sync::atomic::AtomicUsize::new(0),
            tokenizer: false,
            delay: Duration::ZERO,
            caps: raptor_gateway::ThinkingCaps {
                supported: true,
                levels: levels.iter().map(|l| l.to_string()).collect(),
            },
        })
    }

    fn models_call_count(&self) -> usize {
        self.models_calls.load(Ordering::SeqCst)
    }

    fn last(&self) -> (Endpoint, Value) {
        self.calls
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("upstream was called")
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl Upstream for FakeUpstream {
    async fn thinking_caps(&self) -> raptor_gateway::ThinkingCaps {
        self.caps.clone()
    }

    async fn tokenize(&self, _text: &str) -> Result<Option<u64>> {
        Ok(self.tokenizer.then_some(42))
    }

    async fn models(&self) -> Result<ModelCatalog> {
        self.models_calls.fetch_add(1, Ordering::SeqCst);
        Ok(ModelCatalog {
            installed: vec!["qwen-7b".into(), "llama-8b".into()],
            loaded: Some("qwen-7b".into()),
        })
    }

    async fn call(&self, endpoint: Endpoint, body: Value) -> Result<Completion> {
        self.calls.lock().unwrap().push((endpoint, body));
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        match &self.script {
            Script::Json(v) => Ok(Completion::Json(v.clone())),
            Script::Stream(items) => {
                let items: Vec<Result<String>> = items
                    .iter()
                    .cloned()
                    .map(|i| match i {
                        Item::Data(d) => Ok(d),
                        Item::Fail(m) => Err(RaptorError::Runtime(m)),
                    })
                    .collect();
                Ok(Completion::Stream(Box::pin(stream::iter(items))))
            }
            Script::Endless(flag) => {
                let guard = DropFlag(Arc::clone(flag));
                let s = async_stream::stream! {
                    let _guard = guard;
                    let mut n: u64 = 0;
                    loop {
                        tokio::task::yield_now().await;
                        yield Ok::<String, RaptorError>(format!("{{\"n\":{n}}}"));
                        n += 1;
                    }
                };
                Ok(Completion::Stream(Box::pin(s)))
            }
            Script::FailCall(m) => Err(RaptorError::Runtime(m.clone())),
        }
    }
}

struct FakeContext {
    builds: Mutex<Vec<BuildRequest>>,
    fail: bool,
}

impl FakeContext {
    fn new(fail: bool) -> Arc<Self> {
        Arc::new(Self {
            builds: Mutex::new(Vec::new()),
            fail,
        })
    }
}

#[async_trait]
impl ContextService for FakeContext {
    async fn build(&self, req: BuildRequest) -> Result<ContextPack> {
        self.builds.lock().unwrap().push(req);
        if self.fail {
            return Err(RaptorError::Runtime("index unavailable".into()));
        }
        Ok(ContextPack {
            text: "fn main() {}".into(),
            details: json!({"tokens": 4}),
        })
    }
    async fn search(&self, req: raptor_gateway::SearchRequest) -> Result<Value> {
        Ok(json!({"hits": [req.query]}))
    }
    async fn recover(&self, req: raptor_gateway::RecoverRequest) -> Result<Value> {
        Ok(json!({"recovered": req.query}))
    }
    async fn impact(&self, req: raptor_gateway::ImpactRequest) -> Result<Value> {
        Ok(json!({"symbol": req.symbol, "dependents": []}))
    }
    async fn symbols(&self, req: raptor_gateway::SymbolsRequest) -> Result<Value> {
        Ok(json!({"name": req.name, "symbols": []}))
    }
    async fn status(&self) -> Result<Value> {
        Ok(json!({"indexed": true}))
    }
}

// ---------------------------------------------------------------- helpers

fn localhost_cfg() -> GatewayConfig {
    GatewayConfig::default()
}

/// LAN mode needs a key and TLS. `build` only validates, so the TLS directory
/// is never touched here.
fn lan_cfg() -> GatewayConfig {
    GatewayConfig {
        bind: BindMode::Lan,
        api_keys: vec![ApiKeyRecord::new("laptop", KEY).unwrap()],
        tls: Some(TlsConfig::SelfSigned {
            dir: std::path::PathBuf::from("unused"),
            subject_alt_names: vec![],
        }),
        ..GatewayConfig::default()
    }
}

fn app(
    upstream: Arc<FakeUpstream>,
    cfg: GatewayConfig,
    ctx: Option<Arc<FakeContext>>,
) -> (Router, Scheduler) {
    let deps = GatewayDeps {
        upstream: upstream as Arc<dyn Upstream>,
        context: ctx.map(|c| c as Arc<dyn ContextService>),
    };
    let built = build(&cfg, deps).expect("valid config");
    (built.router, built.scheduler)
}

fn request(
    method: &str,
    path: &str,
    peer: IpAddr,
    headers: &[(&str, &str)],
    body: Option<Value>,
) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(path);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let body = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let mut req = b.body(body).unwrap();
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::new(peer, 50000)));
    req
}

fn raw_request(
    method: &str,
    path: &str,
    peer: IpAddr,
    headers: &[(&str, &str)],
    body: &str,
) -> Request<Body> {
    let mut b = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let mut req = b.body(Body::from(body.to_string())).unwrap();
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::new(peer, 50000)));
    req
}

fn raw_request_typed(
    method: &str,
    path: &str,
    peer: IpAddr,
    content_type: &str,
    body: &str,
) -> Request<Body> {
    let mut req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", content_type)
        .body(Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::new(peer, 50000)));
    req
}

async fn send(router: &Router, req: Request<Body>) -> Response {
    router.clone().oneshot(req).await.unwrap()
}

async fn read(resp: Response) -> (StatusCode, HeaderMap, String) {
    let (parts, body) = resp.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    (
        parts.status,
        parts.headers,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

fn chat(model: &str, stream: bool) -> Value {
    json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}]})
}

fn json_body(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}"))
}

fn openai_chat_json() -> Value {
    json!({
        "id": "chatcmpl-9",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen-7b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    })
}

// ---------------------------------------------------------------- SSE framing and passthrough

#[tokio::test]
async fn chat_stream_is_exact_sse_ending_in_done() {
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data(r#"{"id":"1","choices":[{"delta":{"content":"Hi"}}]}"#.into()),
        Item::Data("[DONE]".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (status, headers, body) = read(r).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "text/event-stream");
    assert_eq!(headers["cache-control"], "no-cache");
    assert_eq!(
        body,
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n"
    );
}

#[tokio::test]
async fn done_is_appended_once_when_upstream_omits_it() {
    let up = FakeUpstream::new(Script::Stream(vec![Item::Data("{\"a\":1}".into())]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (_, _, body) = read(r).await;
    assert_eq!(body, "data: {\"a\":1}\n\ndata: [DONE]\n\n");
}

#[tokio::test]
async fn nothing_is_forwarded_after_upstream_done() {
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data("{\"a\":1}".into()),
        Item::Data("[DONE]".into()),
        Item::Data("{\"late\":true}".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (_, _, body) = read(r).await;
    assert_eq!(body, "data: {\"a\":1}\n\ndata: [DONE]\n\n");
    assert_eq!(body.matches("[DONE]").count(), 1);
}

#[tokio::test]
async fn streamed_tool_call_deltas_pass_through_byte_for_byte() {
    let delta = r#"{"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":  1, \"b\":\"\\u00e9\"}"}}]}}]}"#;
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data(delta.into()),
        Item::Data("[DONE]".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (_, _, body) = read(r).await;
    assert!(
        body.starts_with(&format!("data: {delta}\n\n")),
        "tool delta was altered: {body}"
    );
}

#[tokio::test]
async fn multiline_payload_is_framed_as_multiple_data_lines() {
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data("line1\nline2".into()),
        Item::Data("[DONE]".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (_, _, body) = read(r).await;
    assert_eq!(body, "data: line1\ndata: line2\n\ndata: [DONE]\n\n");
}

#[tokio::test]
async fn mid_stream_error_is_sent_as_error_event_then_done() {
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data("{\"a\":1}".into()),
        Item::Fail("boom".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    let (_, _, body) = read(r).await;
    assert!(
        body.starts_with("data: {\"a\":1}\n\ndata: {\"error\":"),
        "{body}"
    );
    assert!(body.contains("\"type\":\"server_error\""));
    assert!(body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn non_stream_json_is_returned_unchanged() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", false)),
        ),
    )
    .await;
    let (status, _, body) = read(r).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json_body(&body), openai_chat_json());
}

#[tokio::test]
async fn model_is_resolved_and_rest_of_body_is_untouched() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "response_format": {"type": "json_object"},
        "temperature": 0.25
    });
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(body.clone()),
        ),
    )
    .await;
    assert_eq!(read(r).await.0, StatusCode::OK);
    let (endpoint, sent) = up.last();
    assert_eq!(endpoint, Endpoint::ChatCompletions);
    let mut expected = body;
    expected["model"] = json!("qwen-7b");
    assert_eq!(sent, expected);
}

#[tokio::test]
async fn alias_maps_to_installed_model() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let mut cfg = localhost_cfg();
    cfg.model_aliases = BTreeMap::from([("fast".to_string(), "llama-8b".to_string())]);
    let (router, _) = app(Arc::clone(&up), cfg, None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("fast", false)),
        ),
    )
    .await;
    assert_eq!(read(r).await.0, StatusCode::OK);
    assert_eq!(up.last().1["model"], "llama-8b");
}

#[tokio::test]
async fn unknown_model_is_404_in_openai_shape() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("mystery-model", false)),
        ),
    )
    .await;
    let (status, _, body) = read(r).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let v = json_body(&body);
    assert_eq!(v["error"]["code"], "model_not_found");
    assert_eq!(
        up.call_count(),
        0,
        "upstream must not be called for unknown models"
    );
}

#[tokio::test]
async fn models_lists_installed_and_aliases() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let mut cfg = localhost_cfg();
    cfg.model_aliases = BTreeMap::from([("fast".to_string(), "llama-8b".to_string())]);
    let (router, _) = app(up, cfg, None);
    let r = send(&router, request("GET", "/v1/models", LOCAL, &[], None)).await;
    let (status, _, body) = read(r).await;
    assert_eq!(status, StatusCode::OK);
    let ids: Vec<String> = json_body(&body)["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, vec!["qwen-7b", "llama-8b", "fast"]);
}

#[tokio::test]
async fn completions_and_embeddings_reach_their_endpoints() {
    let up = FakeUpstream::new(Script::Json(json!({"ok": true})));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let c = send(
        &router,
        request(
            "POST",
            "/v1/completions",
            LOCAL,
            &[],
            Some(json!({"model": "qwen-7b", "prompt": "x"})),
        ),
    )
    .await;
    assert_eq!(read(c).await.0, StatusCode::OK);
    assert_eq!(up.last().0, Endpoint::Completions);
    let e = send(
        &router,
        request(
            "POST",
            "/v1/embeddings",
            LOCAL,
            &[],
            Some(json!({"model": "qwen-7b", "input": "x"})),
        ),
    )
    .await;
    assert_eq!(read(e).await.0, StatusCode::OK);
    assert_eq!(up.last().0, Endpoint::Embeddings);
}

#[tokio::test]
async fn invalid_json_and_missing_fields_are_openai_400s() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let bad = send(
        &router,
        raw_request("POST", "/v1/chat/completions", LOCAL, &[], "{nope"),
    )
    .await;
    let (status, _, body) = read(bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json_body(&body)["error"]["type"], "invalid_request_error");
    let missing = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(json!({"model": "qwen-7b"})),
        ),
    )
    .await;
    assert_eq!(read(missing).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn oversized_body_is_rejected_with_json_error() {
    let mut cfg = localhost_cfg();
    cfg.max_body_bytes = 64;
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), cfg, None);
    let big = "x".repeat(500);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(json!({"model": "qwen-7b", "messages": [], "pad": big})),
        ),
    )
    .await;
    let (status, headers, body) = read(r).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(headers["content-type"], "application/json");
    assert!(json_body(&body)["error"].is_object());
    assert_eq!(up.call_count(), 0);
}

#[tokio::test]
async fn unknown_route_gets_json_404() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, headers, body) =
        read(send(&router, request("GET", "/nope", LOCAL, &[], None)).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(headers["content-type"], "application/json");
    assert!(json_body(&body)["error"]["message"].is_string());
}

#[tokio::test]
async fn upstream_failure_is_502_not_success() {
    let up = FakeUpstream::new(Script::FailCall("llama-server crashed".into()));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, _, body) = read(
        send(
            &router,
            request(
                "POST",
                "/v1/chat/completions",
                LOCAL,
                &[],
                Some(chat("qwen-7b", false)),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(json_body(&body)["error"]["code"], "upstream_error");
}

#[tokio::test]
async fn client_disconnect_cancels_upstream_and_releases_slot() {
    let flag = Arc::new(AtomicBool::new(false));
    let up = FakeUpstream::new(Script::Endless(Arc::clone(&flag)));
    let (router, sched) = app(up, localhost_cfg(), None);
    let resp = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    assert_eq!(sched.metrics().running, 1);
    let mut body = resp.into_body().into_data_stream();
    let first = body.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).starts_with("data: {\"n\":0}"));
    assert!(!flag.load(Ordering::SeqCst));
    drop(body); // the client goes away
    assert!(
        flag.load(Ordering::SeqCst),
        "upstream stream was not cancelled"
    );
    let m = sched.metrics();
    assert_eq!((m.running, m.queued, m.completed), (0, 0, 1));
}

// ---------------------------------------------------------------- auth matrix and CORS

#[tokio::test]
async fn localhost_keyless_is_allowed() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, _, _) =
        read(send(&router, request("GET", "/v1/models", LOCAL, &[], None)).await).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn remote_without_key_gets_openai_401_with_challenge() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let (status, headers, body) =
        read(send(&router, request("GET", "/v1/models", LAN, &[], None)).await).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(headers["www-authenticate"], "Bearer");
    let v = json_body(&body);
    assert_eq!(v["error"]["type"], "authentication_error");
    assert_eq!(v["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn remote_with_valid_bearer_key_is_allowed() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let auth = format!("Bearer {KEY}");
    let (status, _, _) = read(
        send(
            &router,
            request("GET", "/v1/models", LAN, &[("authorization", &auth)], None),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn wrong_key_is_refused_and_not_echoed() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let wrong = "nxk_guessed_guessed_guess_1";
    let auth = format!("Bearer {wrong}");
    let (status, _, body) = read(
        send(
            &router,
            request("GET", "/v1/models", LAN, &[("authorization", &auth)], None),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(
        !body.contains(wrong),
        "the presented key must not be echoed back"
    );
}

#[tokio::test]
async fn x_api_key_header_is_accepted() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let (status, _, _) = read(
        send(
            &router,
            request("GET", "/v1/models", LAN, &[("x-api-key", KEY)], None),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn keyless_localhost_can_be_disabled() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = GatewayConfig {
        allow_keyless_loopback: false,
        api_keys: vec![ApiKeyRecord::new("k", KEY).unwrap()],
        ..GatewayConfig::default()
    };
    let (router, _) = app(up, cfg, None);
    let (status, _, _) =
        read(send(&router, request("GET", "/v1/models", LOCAL, &[], None)).await).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let auth = format!("Bearer {KEY}");
    let (ok, _, _) = read(
        send(
            &router,
            request(
                "GET",
                "/v1/models",
                LOCAL,
                &[("authorization", &auth)],
                None,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(ok, StatusCode::OK);
}

#[tokio::test]
async fn post_without_json_content_type_is_415() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    // A cross-site form post arrives as text/plain and needs no preflight.
    let r = send(
        &router,
        raw_request_typed(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            "text/plain",
            &chat("qwen-7b", false).to_string(),
        ),
    )
    .await;
    let (status, _, body) = read(r).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(json_body(&body)["error"]["type"], "invalid_request_error");
    assert_eq!(up.call_count(), 0);
}

#[tokio::test]
async fn keyless_loopback_with_foreign_host_is_refused_in_lan_mode_too() {
    // DNS rebinding: a page on this machine resolves its own name to 127.0.0.1.
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = lan_cfg();
    let (router, _) = app(Arc::clone(&up), cfg, None);
    let (status, _, _) = read(
        send(
            &router,
            request(
                "GET",
                "/v1/models",
                LOCAL,
                &[("host", "rebind.example:8790")],
                None,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(up.call_count(), 0);
}

#[tokio::test]
async fn localhost_mode_rejects_foreign_host_header() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, _, _) = read(
        send(
            &router,
            request(
                "GET",
                "/v1/models",
                LOCAL,
                &[("host", "evil.example:8790")],
                None,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (ok, _, _) = read(
        send(
            &router,
            request(
                "GET",
                "/v1/models",
                LOCAL,
                &[("host", "localhost:8790")],
                None,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(ok, StatusCode::OK);
}

#[tokio::test]
async fn health_is_open_but_metrics_need_auth() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let (health, _, body) =
        read(send(&router, request("GET", "/health", LAN, &[], None)).await).await;
    assert_eq!(health, StatusCode::OK);
    assert_eq!(json_body(&body)["status"], "ok");
    let (metrics, _, _) =
        read(send(&router, request("GET", "/raptor/metrics", LAN, &[], None)).await).await;
    assert_eq!(metrics, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn cors_allows_only_listed_origins() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = GatewayConfig {
        cors_allowed_origins: vec!["https://app.example.com".into()],
        ..GatewayConfig::default()
    };
    let (router, _) = app(up, cfg, None);
    let preflight = |origin: &'static str| {
        request(
            "OPTIONS",
            "/v1/chat/completions",
            LOCAL,
            &[
                ("origin", origin),
                ("access-control-request-method", "POST"),
                (
                    "access-control-request-headers",
                    "authorization,content-type",
                ),
            ],
            None,
        )
    };
    let ok = send(&router, preflight("https://app.example.com")).await;
    assert_eq!(
        ok.headers()["access-control-allow-origin"],
        "https://app.example.com"
    );
    let evil = send(&router, preflight("https://evil.example.net")).await;
    assert!(evil.headers().get("access-control-allow-origin").is_none());
}

// ---------------------------------------------------------------- queue

fn queue_cfg(
    max_concurrent: usize,
    max_queued: usize,
    per_client: usize,
    retry: u64,
) -> GatewayConfig {
    GatewayConfig {
        queue: QueueConfig {
            max_concurrent,
            max_queued,
            per_client_limit: per_client,
            retry_after_secs: retry,
            queue_timeout: Some(Duration::from_secs(30)),
        },
        ..GatewayConfig::default()
    }
}

#[tokio::test]
async fn per_client_limit_returns_429_with_retry_after() {
    let up = FakeUpstream::new(Script::Endless(Arc::new(AtomicBool::new(false))));
    let (router, _) = app(up, queue_cfg(4, 4, 1, 7), None);
    let first = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let second = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    assert_eq!(second.headers()["retry-after"], "7");
    let (status, _, body) = read(second).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(json_body(&body)["error"]["code"], "rate_limit_exceeded");
    drop(first);
}

#[tokio::test]
async fn full_queue_returns_503_with_retry_after() {
    let up = FakeUpstream::new(Script::Endless(Arc::new(AtomicBool::new(false))));
    let (router, sched) = app(Arc::clone(&up), queue_cfg(1, 0, 8, 3), None);
    let first = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let second = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL_2,
            &[],
            Some(chat("qwen-7b", true)),
        ),
    )
    .await;
    assert_eq!(second.headers()["retry-after"], "3");
    let (status, _, body) = read(second).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json_body(&body)["error"]["code"], "queue_full");
    assert_eq!(
        up.call_count(),
        1,
        "rejected requests must not reach the upstream"
    );
    drop(first);
    assert_eq!(sched.metrics().running, 0);
}

#[tokio::test]
async fn metrics_endpoint_reports_counters() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    assert_eq!(
        read(
            send(
                &router,
                request(
                    "POST",
                    "/v1/chat/completions",
                    LOCAL,
                    &[],
                    Some(chat("qwen-7b", false))
                )
            )
            .await
        )
        .await
        .0,
        StatusCode::OK
    );
    let (status, _, body) =
        read(send(&router, request("GET", "/raptor/metrics", LOCAL, &[], None)).await).await;
    assert_eq!(status, StatusCode::OK);
    let v = json_body(&body);
    assert_eq!(v["completed"], 1);
    assert_eq!(v["running"], 0);
}

// ---------------------------------------------------------------- context

#[tokio::test]
async fn workspace_header_injects_context_as_leading_system_message() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let ctx = FakeContext::new(false);
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), Some(Arc::clone(&ctx)));
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[("x-raptor-workspace", "proj-1")],
            Some(chat("qwen-7b", false)),
        ),
    )
    .await;
    assert_eq!(read(r).await.0, StatusCode::OK);
    let builds = ctx.builds.lock().unwrap();
    assert_eq!(builds.len(), 1);
    assert_eq!(builds[0].workspace, "proj-1");
    assert_eq!(builds[0].query, "hi");
    let sent = up.last().1;
    assert_eq!(sent["messages"][0]["role"], "system");
    assert!(sent["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("fn main() {}"));
    assert_eq!(sent["messages"][1]["role"], "user");
}

#[tokio::test]
async fn no_header_means_no_context_lookup() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let ctx = FakeContext::new(false);
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), Some(Arc::clone(&ctx)));
    assert_eq!(
        read(
            send(
                &router,
                request(
                    "POST",
                    "/v1/chat/completions",
                    LOCAL,
                    &[],
                    Some(chat("qwen-7b", false))
                )
            )
            .await
        )
        .await
        .0,
        StatusCode::OK
    );
    assert!(ctx.builds.lock().unwrap().is_empty());
    assert_eq!(up.last().1["messages"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn context_engine_failure_fails_the_request() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let ctx = FakeContext::new(true);
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), Some(ctx));
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[("x-raptor-workspace", "proj-1")],
            Some(chat("qwen-7b", false)),
        ),
    )
    .await;
    assert_eq!(read(r).await.0, StatusCode::BAD_GATEWAY);
    assert_eq!(
        up.call_count(),
        0,
        "a failed context build must not run inference without context"
    );
}

#[tokio::test]
async fn workspace_header_without_context_service_is_503() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[("x-raptor-workspace", "proj-1")],
            Some(chat("qwen-7b", false)),
        ),
    )
    .await;
    let (status, _, body) = read(r).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json_body(&body)["error"]["code"], "context_unavailable");
}

#[tokio::test]
async fn context_routes_validate_and_forward() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), Some(FakeContext::new(false)));
    let ok = send(
        &router,
        request(
            "POST",
            "/raptor/context/search",
            LOCAL,
            &[],
            Some(json!({"workspace": "w", "query": "parser", "limit": 5})),
        ),
    )
    .await;
    let (status, _, body) = read(ok).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json_body(&body)["hits"][0], "parser");

    let empty = send(
        &router,
        request(
            "POST",
            "/raptor/context/build",
            LOCAL,
            &[],
            Some(json!({"workspace": "w", "query": "  "})),
        ),
    )
    .await;
    assert_eq!(read(empty).await.0, StatusCode::BAD_REQUEST);

    let status_resp = send(
        &router,
        request("GET", "/raptor/context/status", LOCAL, &[], None),
    )
    .await;
    assert_eq!(json_body(&read(status_resp).await.2)["indexed"], true);

    let impact = send(
        &router,
        request(
            "POST",
            "/raptor/context/impact",
            LOCAL,
            &[],
            Some(json!({"workspace": "w", "symbol": "Foo"})),
        ),
    )
    .await;
    assert_eq!(json_body(&read(impact).await.2)["symbol"], "Foo");
}

#[tokio::test]
async fn impact_takes_exactly_one_of_symbol_or_path() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), Some(FakeContext::new(false)));
    let impact = |body: Value| request("POST", "/raptor/context/impact", LOCAL, &[], Some(body));
    // Neither field, and both fields, are 400s.
    let neither = send(&router, impact(json!({"workspace": "w"}))).await;
    assert_eq!(read(neither).await.0, StatusCode::BAD_REQUEST);
    let both = send(
        &router,
        impact(json!({"workspace": "w", "symbol": "Foo", "path": "src/a.rs"})),
    )
    .await;
    assert_eq!(read(both).await.0, StatusCode::BAD_REQUEST);
    // The fake service only implements symbols, so a file path is refused by the
    // trait default with 400 Unsupported rather than answered as a symbol.
    let file = send(
        &router,
        impact(json!({"workspace": "w", "path": "src/a.rs"})),
    )
    .await;
    let (status, _, body) = read(file).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json_body(&body)["error"]["message"]
        .as_str()
        .unwrap()
        .contains("file path"));
    // Traversal and absolute paths never reach the service.
    for bad in ["../secret.rs", "/etc/passwd", "C:\\x.rs", "src\\..\\..\\x"] {
        let r = send(&router, impact(json!({"workspace": "w", "path": bad}))).await;
        assert_eq!(read(r).await.0, StatusCode::BAD_REQUEST, "{bad}");
    }
}

/// A service that answers file queries, to prove the route passes them through.
struct FileImpactContext;

#[async_trait]
impl ContextService for FileImpactContext {
    async fn build(&self, _: BuildRequest) -> Result<ContextPack> {
        Ok(ContextPack::default())
    }
    async fn search(&self, _: raptor_gateway::SearchRequest) -> Result<Value> {
        Ok(json!({}))
    }
    async fn recover(&self, _: raptor_gateway::RecoverRequest) -> Result<Value> {
        Ok(json!({}))
    }
    async fn impact(&self, req: raptor_gateway::ImpactRequest) -> Result<Value> {
        Ok(json!({"symbol": req.symbol}))
    }
    async fn impact_target(&self, req: raptor_gateway::ImpactQuery) -> Result<Value> {
        match req.target {
            raptor_gateway::ImpactTarget::File(p) => Ok(json!({"file": p})),
            raptor_gateway::ImpactTarget::Symbol(s) => Ok(json!({"symbol": s})),
        }
    }
    async fn symbols(&self, _: raptor_gateway::SymbolsRequest) -> Result<Value> {
        Ok(json!({}))
    }
    async fn status(&self) -> Result<Value> {
        Ok(json!({}))
    }
}

#[tokio::test]
async fn impact_file_query_reaches_an_overriding_service() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let deps = GatewayDeps {
        upstream: up as Arc<dyn Upstream>,
        context: Some(Arc::new(FileImpactContext)),
    };
    let router = build(&localhost_cfg(), deps).expect("valid config").router;
    let r = send(
        &router,
        request(
            "POST",
            "/raptor/context/impact",
            LOCAL,
            &[],
            Some(json!({"workspace": "w", "path": "src/lib.rs"})),
        ),
    )
    .await;
    assert_eq!(json_body(&read(r).await.2)["file"], "src/lib.rs");
}

#[tokio::test]
async fn context_routes_503_without_service() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, _, body) = read(
        send(
            &router,
            request("GET", "/raptor/context/status", LOCAL, &[], None),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json_body(&body)["error"]["code"], "context_unavailable");
}

// ---------------------------------------------------------------- MCP

async fn mcp(router: &Router, msg: Value) -> (StatusCode, Value) {
    let (status, _, body) =
        read(send(router, request("POST", "/mcp", LOCAL, &[], Some(msg))).await).await;
    let v = if body.is_empty() {
        Value::Null
    } else {
        json_body(&body)
    };
    (status, v)
}

fn mcp_app(root: Option<std::path::PathBuf>) -> Router {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = GatewayConfig {
        workspace_root: root,
        ..GatewayConfig::default()
    };
    app(up, cfg, Some(FakeContext::new(false))).0
}

#[tokio::test]
async fn mcp_initialize_and_tools_list() {
    let router = mcp_app(None);
    let (status, init) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(init["result"]["serverInfo"]["name"], "raptor-gateway");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    let (_, list) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    .await;
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec![
            "context_build",
            "context_search",
            "context_recover",
            "repo_symbols",
            "repo_impact",
            "repo_read_file"
        ]
    );
}

#[tokio::test]
async fn mcp_notifications_get_202_and_unknown_methods_get_error() {
    let router = mcp_app(None);
    let (status, body) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(body, Value::Null);
    let (_, err) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 9, "method": "resources/list"}),
    )
    .await;
    assert_eq!(err["error"]["code"], -32601);
    assert_eq!(err["id"], 9);
}

#[tokio::test]
async fn mcp_parse_error_and_batch() {
    let router = mcp_app(None);
    let bad = send(&router, raw_request("POST", "/mcp", LOCAL, &[], "{oops")).await;
    let (status, _, body) = read(bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json_body(&body)["error"]["code"], -32700);
    let (_, batch) = mcp(
        &router,
        json!([{"jsonrpc": "2.0", "id": "a", "method": "ping"}, {"jsonrpc": "2.0", "id": "b", "method": "ping"}]),
    )
    .await;
    assert_eq!(batch.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn mcp_context_build_tool_returns_pack_as_text() {
    let router = mcp_app(None);
    let (_, resp) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "context_build", "arguments": {"workspace": "w", "query": "main"}}}),
    )
    .await;
    assert_eq!(resp["result"]["isError"], false);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(json_body(text)["text"]
        .as_str()
        .unwrap()
        .contains("fn main"));
}

#[tokio::test]
async fn mcp_unknown_tool_is_jsonrpc_error() {
    let router = mcp_app(None);
    let (_, resp) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "shell_exec", "arguments": {}}}),
    )
    .await;
    assert_eq!(resp["error"]["code"], -32602);
}

#[tokio::test]
async fn mcp_repo_read_file_is_confined_to_workspace() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("lib.rs"), "pub fn ok() {}").unwrap();
    std::fs::write(outside.path().join("secret.txt"), "TOP SECRET").unwrap();
    let router = mcp_app(Some(root.path().to_path_buf()));

    let call = |path: String| json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "repo_read_file", "arguments": {"path": path}}});
    let (_, ok) = mcp(&router, call("lib.rs".into())).await;
    assert_eq!(ok["result"]["isError"], false);
    assert!(ok["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("pub fn ok"));

    let escapes = [
        "../secret.txt".to_string(),
        "..\\secret.txt".to_string(),
        outside
            .path()
            .join("secret.txt")
            .to_string_lossy()
            .into_owned(),
        "C:\\Windows\\win.ini".to_string(),
        "lib.rs:hidden".to_string(),
    ];
    for path in escapes {
        let (_, resp) = mcp(&router, call(path.clone())).await;
        assert_eq!(resp["result"]["isError"], true, "must refuse {path}");
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(!text.contains("TOP SECRET"), "content leaked for {path}");
    }
}

#[tokio::test]
async fn mcp_repo_read_file_without_root_reports_tool_error() {
    let router = mcp_app(None);
    let (_, resp) = mcp(
        &router,
        json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "repo_read_file", "arguments": {"path": "a.rs"}}}),
    )
    .await;
    assert_eq!(resp["result"]["isError"], true);
}

#[tokio::test]
async fn mcp_get_is_not_allowed() {
    let router = mcp_app(None);
    let (status, _, _) = read(send(&router, request("GET", "/mcp", LOCAL, &[], None)).await).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

// ---------------------------------------------------------------- Anthropic /v1/messages

fn anthropic_req(stream: bool) -> Value {
    json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role": "user", "content": "Say hello"}]
    })
}

#[tokio::test]
async fn messages_non_stream_translates_both_ways() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let (status, _, body) = read(
        send(
            &router,
            request(
                "POST",
                "/v1/messages",
                LOCAL,
                &[],
                Some(anthropic_req(false)),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v = json_body(&body);
    assert_eq!(v["type"], "message");
    assert_eq!(v["model"], "claude-sonnet-4-5");
    assert_eq!(v["content"][0], json!({"type": "text", "text": "Hello"}));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 3);
    // The upstream saw an OpenAI request with the loaded model.
    let (endpoint, sent) = up.last();
    assert_eq!(endpoint, Endpoint::ChatCompletions);
    assert_eq!(sent["model"], "qwen-7b");
    assert_eq!(sent["messages"][0]["content"], "Say hello");
}

#[tokio::test]
async fn messages_stream_emits_named_anthropic_events() {
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data(r#"{"id":"s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#.into()),
        Item::Data(r#"{"id":"s1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#.into()),
        Item::Data(r#"{"id":"s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#.into()),
        Item::Data("[DONE]".into()),
    ]));
    let (router, _) = app(up, localhost_cfg(), None);
    let (status, headers, body) = read(
        send(
            &router,
            request(
                "POST",
                "/v1/messages",
                LOCAL,
                &[],
                Some(anthropic_req(true)),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "text/event-stream");
    assert!(body.starts_with("event: message_start\ndata: "));
    assert!(body.contains(
        "event: content_block_delta\ndata: {\"delta\":{\"text\":\"Hel\",\"type\":\"text_delta\"}"
    ));
    assert!(body.contains("\"stop_reason\":\"end_turn\""));
    assert!(body.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
    assert!(
        !body.contains("[DONE]"),
        "Anthropic streams have no [DONE] marker"
    );
}

#[tokio::test]
async fn messages_errors_use_anthropic_envelope() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let no_max = json!({"model": "claude-sonnet-4-5", "messages": []});
    let (status, _, body) = read(
        send(
            &router,
            request("POST", "/v1/messages", LOCAL, &[], Some(no_max)),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v = json_body(&body);
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn messages_auth_failure_uses_anthropic_envelope_and_accepts_x_api_key() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, lan_cfg(), None);
    let (status, _, body) = read(
        send(
            &router,
            request("POST", "/v1/messages", LAN, &[], Some(anthropic_req(false))),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json_body(&body)["type"], "error");
    let (ok, _, _) = read(
        send(
            &router,
            request(
                "POST",
                "/v1/messages",
                LAN,
                &[("x-api-key", KEY)],
                Some(anthropic_req(false)),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(ok, StatusCode::OK);
}

#[tokio::test]
async fn messages_image_content_is_rejected_explicitly() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(Arc::clone(&up), localhost_cfg(), None);
    let body = json!({"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
    ]}]});
    let (status, _, _) = read(
        send(
            &router,
            request("POST", "/v1/messages", LOCAL, &[], Some(body)),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(up.call_count(), 0);
}

// ---------------------------------------------------------------- lifecycle (real sockets)

fn http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test]
async fn gateway_serves_on_loopback_and_refuses_lan_without_key() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let deps = GatewayDeps {
        upstream: up as Arc<dyn Upstream>,
        context: None,
    };

    let refused = Gateway::start(
        GatewayConfig {
            bind: BindMode::Lan,
            ..GatewayConfig::default()
        },
        deps.clone(),
    )
    .await;
    assert!(matches!(refused, Err(RaptorError::Unauthorized(_))));

    let mut handle = Gateway::start(
        GatewayConfig {
            port: 0,
            ..GatewayConfig::default()
        },
        deps,
    )
    .await
    .unwrap();
    let addr = handle.addr();
    assert!(addr.ip().is_loopback());
    let resp = http_client()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let models = http_client()
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(models.status(), 200, "keyless loopback should be allowed");

    // An invalid reconfigure is refused and the current listener keeps serving.
    let bad = handle
        .reconfigure(GatewayConfig {
            bind: BindMode::Lan,
            port: 0,
            ..GatewayConfig::default()
        })
        .await;
    assert!(bad.is_err());
    let still = http_client()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(still.status(), 200);

    // A valid reconfigure drains and rebinds on a new port.
    handle
        .reconfigure(GatewayConfig {
            port: 0,
            ..GatewayConfig::default()
        })
        .await
        .unwrap();
    let new_addr = handle.addr();
    let after = http_client()
        .get(format!("http://{new_addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), 200);

    handle.shutdown().await.unwrap();
    assert!(http_client()
        .get(format!("http://{new_addr}/health"))
        .send()
        .await
        .is_err());
}

// ---------------------------------------------------------------- TLS

fn self_signed_cfg(dir: &std::path::Path) -> Option<TlsConfig> {
    Some(TlsConfig::SelfSigned {
        dir: dir.to_path_buf(),
        subject_alt_names: vec![],
    })
}

/// A client that trusts exactly one certificate: the pin. No system roots.
fn pinned_client(pem: &str) -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .tls_built_in_root_certs(false)
        .add_root_certificate(reqwest::Certificate::from_pem(pem.as_bytes()).unwrap())
        .build()
        .unwrap()
}

/// The certificate the listener stores on disk, as PEM.
fn stored_pem(dir: &std::path::Path) -> String {
    std::fs::read_to_string(dir.join(raptor_gateway::tls::CERT_FILE)).unwrap()
}

fn plain_deps() -> GatewayDeps {
    GatewayDeps {
        upstream: FakeUpstream::new(Script::Json(openai_chat_json())) as Arc<dyn Upstream>,
        context: None,
    }
}

#[tokio::test]
async fn tls_handshake_succeeds_with_the_pinned_self_signed_certificate() {
    let dir = tempfile::tempdir().unwrap();
    let handle = Gateway::start(
        GatewayConfig {
            port: 0,
            tls: self_signed_cfg(dir.path()),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let status = handle.status();
    assert!(status.tls && !status.insecure && !status.lan);
    let fingerprint = status.tls_fingerprint_sha256.clone().expect("fingerprint");

    // The fingerprint shown to the user is the SHA-256 of the certificate on disk.
    let on_disk =
        <rustls::pki_types::CertificateDer as rustls::pki_types::pem::PemObject>::from_pem_slice(
            stored_pem(dir.path()).as_bytes(),
        )
        .unwrap();
    assert_eq!(
        fingerprint,
        raptor_gateway::fingerprint_of_der(on_disk.as_ref())
    );

    let url = format!("https://127.0.0.1:{}/health", handle.addr().port());
    let resp = pinned_client(&stored_pem(dir.path()))
        .get(&url)
        .send()
        .await
        .expect("pinned handshake");
    assert_eq!(resp.status(), 200);

    // Without the pin, the default trust store refuses the certificate.
    let unpinned = reqwest::Client::builder().no_proxy().build().unwrap();
    assert!(unpinned.get(&url).send().await.is_err());
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn tls_listener_serves_api_requests_with_keys_over_https() {
    let dir = tempfile::tempdir().unwrap();
    let (record, secret) = ApiKeyRecord::generate("tls-client").unwrap();
    let handle = Gateway::start(
        GatewayConfig {
            port: 0,
            api_keys: vec![record],
            allow_keyless_loopback: false,
            tls: self_signed_cfg(dir.path()),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let url = format!("https://127.0.0.1:{}/v1/models", handle.addr().port());
    let client = pinned_client(&stored_pem(dir.path()));
    let ok = client.get(&url).bearer_auth(&secret).send().await.unwrap();
    assert_eq!(ok.status(), 200, "key travels inside TLS and is accepted");
    let no_key = client.get(&url).send().await.unwrap();
    assert_eq!(no_key.status(), 401);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn lan_without_tls_is_refused_by_default() {
    let (record, _) = ApiKeyRecord::generate("laptop").unwrap();
    let refused = Gateway::start(
        GatewayConfig {
            bind: BindMode::Lan,
            port: 0,
            api_keys: vec![record.clone()],
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await;
    assert!(matches!(refused, Err(RaptorError::InvalidInput(_))));
    assert!(build(
        &GatewayConfig {
            bind: BindMode::Lan,
            api_keys: vec![record],
            ..GatewayConfig::default()
        },
        plain_deps()
    )
    .is_err());
}

#[tokio::test]
async fn lan_insecure_opt_in_reports_insecure_in_status() {
    let (record, _) = ApiKeyRecord::generate("laptop").unwrap();
    let handle = Gateway::start(
        GatewayConfig {
            bind: BindMode::Lan,
            port: 0,
            api_keys: vec![record],
            allow_insecure_lan: true,
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let status = handle.status();
    assert!(status.lan && status.insecure && !status.tls);
    assert_eq!(status.tls_fingerprint_sha256, None);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn lan_with_tls_is_secure_and_serves_on_all_interfaces() {
    let dir = tempfile::tempdir().unwrap();
    let (record, _) = ApiKeyRecord::generate("laptop").unwrap();
    let handle = Gateway::start(
        GatewayConfig {
            bind: BindMode::Lan,
            port: 0,
            api_keys: vec![record],
            tls: self_signed_cfg(dir.path()),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let status = handle.status();
    assert!(status.lan && status.tls && !status.insecure);
    let url = format!("https://127.0.0.1:{}/health", handle.addr().port());
    let resp = pinned_client(&stored_pem(dir.path()))
        .get(&url)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn rotation_replaces_the_certificate_without_restarting() {
    let dir = tempfile::tempdir().unwrap();
    let mut handle = Gateway::start(
        GatewayConfig {
            port: 0,
            tls: self_signed_cfg(dir.path()),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let addr = handle.addr();
    let old_pem = stored_pem(dir.path());
    let old_fp = handle.status().tls_fingerprint_sha256.unwrap();
    let url = format!("https://127.0.0.1:{}/health", addr.port());

    let new_identity = handle.rotate_tls().await.unwrap();
    assert_ne!(new_identity.fingerprint_sha256, old_fp);
    assert_eq!(handle.addr(), addr, "rotation must not rebind");
    let new_pem = stored_pem(dir.path());
    assert_ne!(new_pem, old_pem);

    assert_eq!(
        pinned_client(&new_pem)
            .get(&url)
            .send()
            .await
            .unwrap()
            .status(),
        200,
        "the new pin works on the live listener"
    );
    assert!(
        pinned_client(&old_pem).get(&url).send().await.is_err(),
        "the old pin is refused after rotation"
    );
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn rotate_without_tls_is_an_error() {
    let mut handle = Gateway::start(
        GatewayConfig {
            port: 0,
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    assert!(handle.rotate_tls().await.is_err());
    assert!(handle.tls_identity().is_none());
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn user_provided_pem_files_are_served() {
    let dir = tempfile::tempdir().unwrap();
    let key = rcgen::KeyPair::generate().unwrap();
    let cert =
        rcgen::CertificateParams::new(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .unwrap()
            .self_signed(&key)
            .unwrap();
    let cert_path = dir.path().join("user-cert.pem");
    let key_path = dir.path().join("user-key.pem");
    std::fs::write(&cert_path, cert.pem()).unwrap();
    std::fs::write(&key_path, key.serialize_pem()).unwrap();

    let handle = Gateway::start(
        GatewayConfig {
            port: 0,
            tls: Some(TlsConfig::Files {
                cert_pem: cert_path,
                key_pem: key_path,
            }),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    assert_eq!(handle.tls_identity().unwrap().cert_der, cert.der().to_vec());
    let url = format!("https://127.0.0.1:{}/health", handle.addr().port());
    let resp = pinned_client(&cert.pem()).get(&url).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn missing_tls_files_fail_before_the_port_is_bound() {
    let dir = tempfile::tempdir().unwrap();
    let result = Gateway::start(
        GatewayConfig {
            port: 0,
            tls: Some(TlsConfig::Files {
                cert_pem: dir.path().join("absent.pem"),
                key_pem: dir.path().join("absent-key.pem"),
            }),
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await;
    assert!(matches!(result, Err(RaptorError::InvalidInput(_))));
}

// ---------------------------------------------------------------- key management

#[tokio::test]
async fn revoked_key_stops_working_on_the_live_listener() {
    let a = ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa1").unwrap();
    let b = ApiKeyRecord::new("b", "bbbbbbbbbbbbbbbb2").unwrap();
    let handle = Gateway::start(
        GatewayConfig {
            port: 0,
            api_keys: vec![a, b],
            allow_keyless_loopback: false,
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let url = format!("http://{}/v1/models", handle.addr());
    let client = http_client();
    let get = |key: &'static str| client.get(&url).bearer_auth(key).send();
    assert_eq!(get("aaaaaaaaaaaaaaaa1").await.unwrap().status(), 200);
    assert_eq!(get("bbbbbbbbbbbbbbbb2").await.unwrap().status(), 200);

    assert!(handle.keys().revoke("a"));
    assert_eq!(get("aaaaaaaaaaaaaaaa1").await.unwrap().status(), 401);
    assert_eq!(get("bbbbbbbbbbbbbbbb2").await.unwrap().status(), 200);

    let listed = handle.keys().list();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "b");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn key_only_reconfigure_hot_reloads_without_restarting() {
    let a = ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa1").unwrap();
    let mut handle = Gateway::start(
        GatewayConfig {
            port: 0,
            api_keys: vec![a],
            allow_keyless_loopback: false,
            ..GatewayConfig::default()
        },
        plain_deps(),
    )
    .await
    .unwrap();
    let addr = handle.addr();
    let url = format!("http://{addr}/v1/models");
    let client = http_client();
    assert_eq!(
        client
            .get(&url)
            .bearer_auth("aaaaaaaaaaaaaaaa1")
            .send()
            .await
            .unwrap()
            .status(),
        200
    );

    let c = ApiKeyRecord::new("c", "cccccccccccccccc3").unwrap();
    handle
        .reconfigure(GatewayConfig {
            port: addr.port(),
            api_keys: vec![c],
            allow_keyless_loopback: false,
            ..GatewayConfig::default()
        })
        .await
        .unwrap();
    assert_eq!(handle.addr(), addr, "a key-only change keeps the listener");
    assert_eq!(
        client
            .get(&url)
            .bearer_auth("aaaaaaaaaaaaaaaa1")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    assert_eq!(
        client
            .get(&url)
            .bearer_auth("cccccccccccccccc3")
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn per_key_rate_limit_returns_429_with_retry_after() {
    let limited = ApiKeyRecord::new("slow", KEY)
        .unwrap()
        .with_limits(raptor_gateway::KeyLimits {
            max_concurrent: None,
            requests_per_minute: Some(2),
        })
        .unwrap();
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = GatewayConfig {
        api_keys: vec![limited],
        ..GatewayConfig::default()
    };
    let (router, _) = app(up, cfg, None);
    let hdr = [("authorization", "Bearer nxk_test_secret_0123456789")];
    for _ in 0..2 {
        let (status, _, _) =
            read(send(&router, request("GET", "/v1/models", LOCAL, &hdr, None)).await).await;
        assert_eq!(status, 200);
    }
    let (status, headers, body) =
        read(send(&router, request("GET", "/v1/models", LOCAL, &hdr, None)).await).await;
    assert_eq!(status, 429);
    assert!(headers.get("retry-after").is_some());
    assert_eq!(json_body(&body)["error"]["code"], "rate_limit_exceeded");

    // Keyless loopback is a different client and is not limited by this key's window.
    let (status, _, _) =
        read(send(&router, request("GET", "/v1/models", LOCAL_2, &[], None)).await).await;
    assert_eq!(status, 200);
}

async fn two_at_once(router: &Router, key: &str) -> (StatusCode, StatusCode) {
    let bearer = format!("Bearer {key}");
    let hdr = [("authorization", bearer.as_str())];
    let body = chat("qwen-7b", false).to_string();
    let first = send(
        router,
        raw_request("POST", "/v1/chat/completions", LOCAL, &hdr, &body),
    );
    let second = send(
        router,
        raw_request("POST", "/v1/chat/completions", LOCAL, &hdr, &body),
    );
    let (a, b) = tokio::join!(first, second);
    (a.status(), b.status())
}

#[tokio::test]
async fn per_key_concurrency_limit_replaces_the_default_client_cap() {
    // One request runs at a time and each client may hold one slot. A key with
    // max_concurrent = 2 may hold two (one runs, one waits). A default key gets
    // 429 on the second.
    let queue = QueueConfig {
        max_concurrent: 1,
        max_queued: 8,
        per_client_limit: 1,
        retry_after_secs: 1,
        queue_timeout: Some(Duration::from_secs(5)),
    };
    let wide = ApiKeyRecord::new("wide", KEY)
        .unwrap()
        .with_limits(raptor_gateway::KeyLimits {
            max_concurrent: Some(2),
            requests_per_minute: None,
        })
        .unwrap();
    let narrow = ApiKeyRecord::new("narrow", "nxk_narrow_0123456789").unwrap();

    let up = FakeUpstream::slow(Script::Json(openai_chat_json()), Duration::from_millis(150));
    let (router, _) = app(
        up,
        GatewayConfig {
            api_keys: vec![wide, narrow],
            allow_keyless_loopback: false,
            queue,
            ..GatewayConfig::default()
        },
        None,
    );
    let (a, b) = two_at_once(&router, KEY).await;
    assert_eq!((a, b), (StatusCode::OK, StatusCode::OK));
    let (a, b) = two_at_once(&router, "nxk_narrow_0123456789").await;
    assert_eq!(a, StatusCode::OK);
    assert_eq!(b, StatusCode::TOO_MANY_REQUESTS);
}

// ---------------------------------------------------------------- conformance: chat pass-through

#[tokio::test]
async fn structured_output_and_sampling_params_reach_upstream_untouched() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let body = json!({
        "model": "qwen-7b",
        "messages": [{"role": "user", "content": "give json"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": true,
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}
        },
        "n": 2,
        "stop": ["\n\n", "END"],
        "logprobs": true,
        "top_logprobs": 3,
        "seed": 7,
        "stream_options": {"include_usage": true},
        "stream": false
    });
    let (status, _, _) = read(
        send(
            &router,
            request(
                "POST",
                "/v1/chat/completions",
                LOCAL,
                &[],
                Some(body.clone()),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, 200);
    // Only `model` may differ from what the client sent.
    let mut expected = body;
    expected["model"] = json!("qwen-7b");
    assert_eq!(up.last().1, expected);

    // json_object is forwarded the same way.
    let up2 = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router2, _) = app(up2.clone(), localhost_cfg(), None);
    let body2 =
        json!({"model": "qwen-7b", "messages": [], "response_format": {"type": "json_object"}});
    send(
        &router2,
        request("POST", "/v1/chat/completions", LOCAL, &[], Some(body2)),
    )
    .await;
    assert_eq!(
        up2.last().1["response_format"],
        json!({"type": "json_object"})
    );
}

#[tokio::test]
async fn include_usage_stream_option_reaches_upstream_and_usage_chunk_passes_through() {
    let usage_chunk = r#"{"id":"c1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;
    let up = FakeUpstream::new(Script::Stream(vec![
        Item::Data(r#"{"id":"c1","choices":[{"delta":{"content":"hi"}}]}"#.into()),
        Item::Data(usage_chunk.into()),
        Item::Data("[DONE]".into()),
    ]));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let body = json!({"model": "qwen-7b", "stream": true,
        "stream_options": {"include_usage": true},
        "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, text) = read(
        send(
            &router,
            request("POST", "/v1/chat/completions", LOCAL, &[], Some(body)),
        )
        .await,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        up.last().1["stream_options"],
        json!({"include_usage": true})
    );
    assert!(
        text.contains(&format!("data: {usage_chunk}\n\n")),
        "usage chunk is byte-identical"
    );
    assert!(text.ends_with("data: [DONE]\n\n"));
}

// ---------------------------------------------------------------- conformance: anthropic adapter

fn anthropic_post(path: &str, body: Value, headers: &[(&str, &str)]) -> Request<Body> {
    request("POST", path, LOCAL, headers, Some(body))
}

#[tokio::test]
async fn anthropic_system_blocks_and_tool_result_error_flag_round_trip() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let body = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "Use tools."}],
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": true,
                "content": [{"type": "text", "text": "boom"}]}]}
        ]
    });
    let (status, _, _) = read(send(&router, anthropic_post("/v1/messages", body, &[])).await).await;
    assert_eq!(status, 200);
    let (_, fwd) = up.last();
    assert_eq!(
        fwd["messages"][0],
        json!({"role": "system", "content": "You are terse.\n\nUse tools."})
    );
    assert_eq!(
        fwd["messages"][2],
        json!({"role": "tool", "tool_call_id": "t1", "content": "Error: boom"})
    );
    assert_eq!(
        fwd["model"], "qwen-7b",
        "claude-* resolves to the loaded model"
    );
}

#[tokio::test]
async fn anthropic_max_tokens_is_required_and_positive() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let missing = json!({"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, body) =
        read(send(&router, anthropic_post("/v1/messages", missing, &[])).await).await;
    assert_eq!(status, 400);
    assert_eq!(json_body(&body)["type"], "error");
    let zero = json!({"model": "claude-x", "max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, _) = read(send(&router, anthropic_post("/v1/messages", zero, &[])).await).await;
    assert_eq!(status, 400);
    assert_eq!(
        up.call_count(),
        0,
        "invalid requests never reach the upstream"
    );
}

#[tokio::test]
async fn claude_code_beta_query_and_headers_are_tolerated() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let body = json!({"model": "claude-sonnet-4-5", "max_tokens": 16,
        "metadata": {"user_id": "u"}, "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]});
    let headers = [
        ("anthropic-version", "2023-06-01"),
        (
            "anthropic-beta",
            "claude-code-20250219,interleaved-thinking-2025-05-14",
        ),
    ];
    let (status, _, _) = read(
        send(
            &router,
            anthropic_post("/v1/messages?beta=true", body, &headers),
        )
        .await,
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn count_tokens_estimates_and_flags_approximate_without_a_tokenizer() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    // Claude Code's count request has no max_tokens.
    let body = json!({"model": "claude-sonnet-4-5",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Count the tokens in this sentence please."}],
        "tools": [{"name": "t", "input_schema": {"type": "object"}}]});
    let (status, _, text) = read(
        send(
            &router,
            anthropic_post("/v1/messages/count_tokens?beta=true", body, &[]),
        )
        .await,
    )
    .await;
    assert_eq!(status, 200);
    let v = json_body(&text);
    assert!(v["input_tokens"].as_u64().unwrap() > 10);
    assert_eq!(v["approximate"], true);
    assert_eq!(v["method"], "estimate");
    assert_eq!(up.call_count(), 0, "counting does not generate");
}

#[tokio::test]
async fn count_tokens_uses_the_upstream_tokenizer_when_available() {
    let up = FakeUpstream::with_tokenizer(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let body =
        json!({"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, text) = read(
        send(
            &router,
            anthropic_post("/v1/messages/count_tokens", body, &[]),
        )
        .await,
    )
    .await;
    assert_eq!(status, 200);
    let v = json_body(&text);
    assert_eq!(v["input_tokens"], 42);
    assert_eq!(v["method"], "tokenize");
    assert_eq!(v["approximate"], true);
}

#[tokio::test]
async fn count_tokens_rejects_an_invalid_body_in_anthropic_shape() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(up, localhost_cfg(), None);
    let body = json!({"messages": []});
    let (status, _, text) = read(
        send(
            &router,
            anthropic_post("/v1/messages/count_tokens", body, &[]),
        )
        .await,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(json_body(&text)["type"], "error");
}

// ---------------------------------------------------------------- MCP sessions and negotiation

async fn mcp_with(
    router: &Router,
    msg: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, HeaderMap, Value) {
    let (status, h, body) =
        read(send(router, request("POST", "/mcp", LOCAL, headers, Some(msg))).await).await;
    let v = if body.is_empty() {
        Value::Null
    } else {
        json_body(&body)
    };
    (status, h, v)
}

fn mcp_router() -> Router {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    app(up, GatewayConfig::default(), Some(FakeContext::new(false))).0
}

#[tokio::test]
async fn mcp_initialize_issues_a_session_that_later_requests_must_match() {
    let router = mcp_router();
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}});
    let (status, headers, _) = mcp_with(&router, init, &[]).await;
    assert_eq!(status, 200);
    let sid = headers
        .get("mcp-session-id")
        .expect("session id")
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(sid.len(), 32);

    let list = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
    let (ok, _, _) = mcp_with(&router, list.clone(), &[("mcp-session-id", &sid)]).await;
    assert_eq!(ok, 200);
    let (absent, _, _) = mcp_with(&router, list.clone(), &[]).await;
    assert_eq!(
        absent, 200,
        "requests without the header are still accepted"
    );

    let (unknown, _, err) = mcp_with(
        &router,
        list.clone(),
        &[("mcp-session-id", "00000000000000000000000000000000")],
    )
    .await;
    assert_eq!(unknown, 404);
    assert_eq!(err["error"]["code"], -32001);

    let del = request("DELETE", "/mcp", LOCAL, &[("mcp-session-id", &sid)], None);
    let (status, _, _) = read(send(&router, del).await).await;
    assert_eq!(status, 200);
    let (gone, _, _) = mcp_with(&router, list, &[("mcp-session-id", &sid)]).await;
    assert_eq!(gone, 404, "a deleted session is no longer valid");
}

#[tokio::test]
async fn mcp_session_delete_requires_a_live_session() {
    let router = mcp_router();
    let (status, _, _) =
        read(send(&router, request("DELETE", "/mcp", LOCAL, &[], None)).await).await;
    assert_eq!(status, 400);
    let (status, _, _) = read(
        send(
            &router,
            request("DELETE", "/mcp", LOCAL, &[("mcp-session-id", "abc")], None),
        )
        .await,
    )
    .await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn mcp_negotiates_protocol_version() {
    let router = mcp_router();
    let older = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}});
    let (_, _, v) = mcp_with(&router, older, &[]).await;
    assert_eq!(v["result"]["protocolVersion"], "2025-03-26");

    let unknown = json!({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}});
    let (_, _, v) = mcp_with(&router, unknown, &[]).await;
    assert_eq!(v["result"]["protocolVersion"], "2025-06-18");

    let (status, _, _) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        &[("mcp-protocol-version", "1999-01-01")],
    )
    .await;
    assert_eq!(
        status, 400,
        "an unsupported protocol version header is refused"
    );
    let (status, _, _) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
        &[("mcp-protocol-version", "2025-06-18")],
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn mcp_ping_and_initialized_notification() {
    let router = mcp_router();
    let (status, _, v) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": "p", "method": "ping"}),
        &[],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(v, json!({"jsonrpc": "2.0", "id": "p", "result": {}}));
    let (status, _, body) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        &[],
    )
    .await;
    assert_eq!(status, 202);
    assert_eq!(body, Value::Null);
}

#[tokio::test]
async fn mcp_tools_declare_output_schemas_and_return_structured_content() {
    let router = mcp_router();
    let (_, _, list) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        &[],
    )
    .await;
    for tool in list["result"]["tools"].as_array().unwrap() {
        assert_eq!(tool["outputSchema"]["type"], "object", "{tool}");
    }
    let (_, _, call) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "context_search", "arguments": {"workspace": "w", "query": "refund"}}}),
        &[],
    )
    .await;
    assert_eq!(call["result"]["isError"], false);
    assert_eq!(
        call["result"]["structuredContent"]["hits"],
        json!(["refund"])
    );
}

#[tokio::test]
async fn mcp_repo_read_file_structured_output_matches_its_schema() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "hello").unwrap();
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let cfg = GatewayConfig {
        workspace_root: Some(root.path().to_path_buf()),
        ..GatewayConfig::default()
    };
    let (router, _) = app(up, cfg, None);
    let (_, _, call) = mcp_with(
        &router,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "repo_read_file", "arguments": {"path": "a.txt"}}}),
        &[],
    )
    .await;
    assert_eq!(
        call["result"]["structuredContent"],
        json!({"path": "a.txt", "content": "hello"})
    );
}

// ---------------------------------------------------------------- model catalog cache

#[tokio::test]
async fn model_catalog_is_cached_within_the_ttl() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(
        up.clone(),
        GatewayConfig {
            catalog_ttl: Duration::from_secs(3600),
            ..GatewayConfig::default()
        },
        None,
    );
    for _ in 0..3 {
        let (status, _, _) =
            read(send(&router, request("GET", "/v1/models", LOCAL, &[], None)).await).await;
        assert_eq!(status, 200);
    }
    // A completion resolves its model through the same cache.
    send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(chat("qwen-7b", false)),
        ),
    )
    .await;
    assert_eq!(up.models_call_count(), 1);
}

#[tokio::test]
async fn zero_ttl_disables_the_catalog_cache() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let (router, _) = app(
        up.clone(),
        GatewayConfig {
            catalog_ttl: Duration::ZERO,
            ..GatewayConfig::default()
        },
        None,
    );
    for _ in 0..2 {
        send(&router, request("GET", "/v1/models", LOCAL, &[], None)).await;
    }
    assert_eq!(up.models_call_count(), 2);
}

#[tokio::test]
async fn reconfigure_and_explicit_invalidation_drop_the_catalog_cache() {
    let up = FakeUpstream::new(Script::Json(openai_chat_json()));
    let mut handle = Gateway::start(
        GatewayConfig {
            port: 0,
            catalog_ttl: Duration::from_secs(3600),
            ..GatewayConfig::default()
        },
        GatewayDeps {
            upstream: up.clone() as Arc<dyn Upstream>,
            context: None,
        },
    )
    .await
    .unwrap();
    let url = format!("http://{}/v1/models", handle.addr());
    let client = http_client();
    client.get(&url).send().await.unwrap();
    client.get(&url).send().await.unwrap();
    assert_eq!(up.models_call_count(), 1);

    handle.invalidate_model_cache();
    client.get(&url).send().await.unwrap();
    assert_eq!(up.models_call_count(), 2);

    handle
        .reconfigure(GatewayConfig {
            port: 0,
            catalog_ttl: Duration::from_secs(3600),
            ..GatewayConfig::default()
        })
        .await
        .unwrap();
    let url = format!("http://{}/v1/models", handle.addr());
    client.get(&url).send().await.unwrap();
    assert_eq!(up.models_call_count(), 3);
    handle.shutdown().await.unwrap();
}

// ---------------------------------------------------------------- agent hygiene

fn history_with_reasoning() -> Value {
    json!({
        "model": "qwen-7b",
        "messages": [
            {"role": "system", "content": "You are a coder."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>old plan</think>\n\nanswer one", "reasoning_content": "old plan"},
            {"role": "user", "content": "q2"}
        ]
    })
}

#[tokio::test]
async fn earlier_reasoning_is_stripped_before_the_upstream_sees_the_request() {
    let up = FakeUpstream::new(Script::Json(json!({"choices": []})));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(history_with_reasoning()),
        ),
    )
    .await;
    assert_eq!(r.status(), StatusCode::OK);
    let (_, sent) = up.last();
    assert_eq!(sent["messages"][2]["content"], "answer one");
    assert!(sent["messages"][2].get("reasoning_content").is_none());
    assert_eq!(sent["messages"][3]["content"], "q2");
}

#[tokio::test]
async fn stripping_can_be_turned_off_and_leaves_the_history_untouched() {
    let up = FakeUpstream::new(Script::Json(json!({"choices": []})));
    let cfg = GatewayConfig {
        strip_history_reasoning: false,
        ..GatewayConfig::default()
    };
    let (router, _) = app(up.clone(), cfg, None);
    send(
        &router,
        request(
            "POST",
            "/v1/chat/completions",
            LOCAL,
            &[],
            Some(history_with_reasoning()),
        ),
    )
    .await;
    let (_, sent) = up.last();
    assert_eq!(
        sent["messages"][2]["content"],
        "<think>old plan</think>\n\nanswer one"
    );
    assert_eq!(sent["messages"][2]["reasoning_content"], "old plan");
}

#[tokio::test]
async fn a_volatile_system_prompt_is_forwarded_unchanged() {
    let up = FakeUpstream::new(Script::Json(json!({"choices": []})));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let body = json!({
        "model": "qwen-7b",
        "messages": [
            {"role": "system", "content": "Current time: 12:34:56"},
            {"role": "user", "content": "hi"}
        ]
    });
    send(
        &router,
        request("POST", "/v1/chat/completions", LOCAL, &[], Some(body)),
    )
    .await;
    let (_, sent) = up.last();
    assert_eq!(sent["messages"][0]["content"], "Current time: 12:34:56");
}

// ---------------------------------------------------------------- thinking effort

async fn post_chat(router: &axum::Router, body: Value) -> StatusCode {
    send(
        router,
        request("POST", "/v1/chat/completions", LOCAL, &[], Some(body)),
    )
    .await
    .status()
}

fn plain_chat() -> Value {
    json!({"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]})
}

#[tokio::test]
async fn a_thinking_model_gets_the_default_medium_effort_when_the_client_sets_none() {
    // Qwen3.8's template levels: low, medium, xhigh.
    let up = FakeUpstream::thinking(
        Script::Json(json!({"choices": []})),
        &["low", "medium", "xhigh"],
    );
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    assert_eq!(post_chat(&router, plain_chat()).await, StatusCode::OK);
    let (_, sent) = up.last();
    assert_eq!(sent["reasoning_effort"], "medium");
    assert_eq!(sent["messages"], plain_chat()["messages"]);
}

#[tokio::test]
async fn the_configured_default_effort_is_mapped_to_the_models_own_level() {
    let up = FakeUpstream::thinking(
        Script::Json(json!({"choices": []})),
        &["low", "medium", "xhigh"],
    );
    let cfg = GatewayConfig {
        default_effort: raptor_gateway::Effort::High,
        ..localhost_cfg()
    };
    let (router, _) = app(up.clone(), cfg, None);
    post_chat(&router, plain_chat()).await;
    assert_eq!(up.last().1["reasoning_effort"], "xhigh");
}

#[tokio::test]
async fn an_explicit_client_choice_is_never_overridden() {
    let up = FakeUpstream::thinking(
        Script::Json(json!({"choices": []})),
        &["low", "medium", "xhigh"],
    );
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    for (field, value) in [
        ("reasoning_effort", json!("low")),
        ("thinking_budget_tokens", json!(64)),
        ("chat_template_kwargs", json!({"enable_thinking": false})),
    ] {
        let mut body = plain_chat();
        body[field] = value;
        let sent_by_client = body.clone();
        assert_eq!(post_chat(&router, body).await, StatusCode::OK);
        assert_eq!(up.last().1, sent_by_client, "{field}");
    }
}

#[tokio::test]
async fn a_model_without_effort_control_gets_a_byte_identical_request() {
    let up = FakeUpstream::new(Script::Json(json!({"choices": []})));
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    post_chat(&router, plain_chat()).await;
    let (_, sent) = up.last();
    assert!(sent.get("reasoning_effort").is_none());
    assert_eq!(sent, plain_chat());
}

#[tokio::test]
async fn embeddings_requests_never_get_an_effort() {
    let up = FakeUpstream::thinking(
        Script::Json(json!({"data": []})),
        &["low", "medium", "xhigh"],
    );
    let (router, _) = app(up.clone(), localhost_cfg(), None);
    let r = send(
        &router,
        request(
            "POST",
            "/v1/embeddings",
            LOCAL,
            &[],
            Some(json!({"model": "qwen-7b", "input": "x"})),
        ),
    )
    .await;
    assert_eq!(r.status(), StatusCode::OK);
    assert!(up.last().1.get("reasoning_effort").is_none());
}
