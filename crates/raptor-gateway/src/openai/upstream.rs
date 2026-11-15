//! The inference backend seen by the gateway.
//!
//! `Upstream` is the seam the desktop shell (or a test fake) implements. The
//! gateway hands it the client's request body almost untouched; only the
//! `model` field is rewritten. Streamed responses are exposed as the raw
//! `data:` payloads of the upstream SSE stream, so tool-call deltas and other
//! chunk JSON reach the client byte-for-byte as the upstream produced them.

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::{Stream, StreamExt};
use raptor_core::{RaptorError, Result};
use serde_json::Value;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::thinking::ThinkingCaps;

/// How long `/props` is trusted before it is read again.
const CAPS_TTL: Duration = Duration::from_secs(10);
/// A server that does not answer `/props` quickly is treated as having no effort control.
const CAPS_TIMEOUT: Duration = Duration::from_secs(3);

/// Upper bound on one unterminated SSE line, to keep a misbehaving upstream
/// from growing the parser buffer without limit.
const MAX_SSE_LINE: usize = 16 * 1024 * 1024;

/// OpenAI-compatible endpoints the gateway forwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    Completions,
    Embeddings,
}

impl Endpoint {
    /// Path on the upstream server.
    pub fn path(self) -> &'static str {
        match self {
            Endpoint::ChatCompletions => "/v1/chat/completions",
            Endpoint::Completions => "/v1/completions",
            Endpoint::Embeddings => "/v1/embeddings",
        }
    }
}

/// What the backend can serve.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCatalog {
    /// Model ids installed on disk.
    pub installed: Vec<String>,
    /// Model id currently loaded for inference, if any.
    pub loaded: Option<String>,
}

/// Stream of SSE `data:` payloads (without the `data: ` prefix), in order.
/// The upstream emits `[DONE]` as a payload when it finishes.
pub type SseStream = BoxStream<'static, Result<String>>;

/// A non-error upstream answer.
pub enum Completion {
    /// A complete JSON body.
    Json(Value),
    /// A stream of SSE payloads. Dropping it cancels the upstream request.
    Stream(SseStream),
}

/// The inference backend. Implemented by the desktop shell over llama-server,
/// and by fakes in tests.
#[async_trait]
pub trait Upstream: Send + Sync {
    /// Lists installed and loaded models.
    async fn models(&self) -> Result<ModelCatalog>;

    /// Forwards `body` to `endpoint`. The upstream returns a stream when
    /// `body["stream"]` is true. Dropping the returned future or stream must
    /// cancel the upstream work.
    async fn call(&self, endpoint: Endpoint, body: Value) -> Result<Completion>;

    /// Number of tokens `text` takes under the loaded model, or `None` when the
    /// backend cannot tell. The default is `None`, and the gateway then estimates.
    async fn tokenize(&self, text: &str) -> Result<Option<u64>> {
        let _ = text;
        Ok(None)
    }

    /// What the loaded model's chat template says about reasoning effort. The default is "not
    /// supported", so a backend that cannot tell is never sent a reasoning field.
    async fn thinking_caps(&self) -> ThinkingCaps {
        ThinkingCaps::default()
    }
}

/// `Upstream` over a llama-server HTTP endpoint on loopback.
///
/// Only loopback URLs are accepted: the upstream is a local inference process
/// and must never be a remote host. The system proxy is ignored. Connections
/// have a connect timeout only, because generation can legitimately take a
/// long time.
pub struct HttpUpstream {
    base: String,
    client: reqwest::Client,
    installed: Vec<String>,
    /// The last successful `/props` read. A failed read is not kept, so a server that was still
    /// loading is asked again on the next request.
    caps: Mutex<Option<(Instant, ThinkingCaps)>>,
}

impl HttpUpstream {
    /// `base_url` like `http://127.0.0.1:8081`. `installed` lists model ids
    /// that the shell knows are on disk.
    pub fn new(base_url: &str, installed: Vec<String>) -> Result<Self> {
        let url = reqwest::Url::parse(base_url)
            .map_err(|e| RaptorError::InvalidInput(format!("invalid upstream url: {e}")))?;
        if url.scheme() != "http" {
            return Err(RaptorError::InvalidInput("upstream must use http".into()));
        }
        let host = url.host_str().unwrap_or("");
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        if !loopback {
            return Err(RaptorError::InvalidInput(
                "upstream must be a loopback address".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| RaptorError::Runtime(format!("http client: {e}")))?;
        Ok(Self {
            base: base_url.trim_end_matches('/').to_string(),
            client,
            installed,
            caps: Mutex::new(None),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }
}

async fn upstream_error(resp: reqwest::Response) -> RaptorError {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    let snippet: String = text.chars().take(512).collect();
    RaptorError::Runtime(format!("upstream returned {status}: {snippet}"))
}

#[async_trait]
impl Upstream for HttpUpstream {
    async fn models(&self) -> Result<ModelCatalog> {
        let resp = self
            .client
            .get(self.url("/v1/models"))
            .send()
            .await
            .map_err(|e| RaptorError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(upstream_error(resp).await);
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| RaptorError::Runtime(e.to_string()))?;
        let loaded = v["data"][0]["id"].as_str().map(str::to_string);
        Ok(ModelCatalog {
            installed: self.installed.clone(),
            loaded,
        })
    }

    async fn call(&self, endpoint: Endpoint, body: Value) -> Result<Completion> {
        let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let resp = self
            .client
            .post(self.url(endpoint.path()))
            .json(&body)
            .send()
            .await
            .map_err(|e| RaptorError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(upstream_error(resp).await);
        }
        if streaming {
            Ok(Completion::Stream(sse_payloads(resp.bytes_stream())))
        } else {
            let v: Value = resp
                .json()
                .await
                .map_err(|e| RaptorError::Runtime(e.to_string()))?;
            Ok(Completion::Json(v))
        }
    }

    /// Posts to llama-server's `/tokenize`. Older builds without the route, and
    /// unreachable servers, yield `None` so the caller estimates instead.
    async fn tokenize(&self, text: &str) -> Result<Option<u64>> {
        let resp = match self
            .client
            .post(self.url("/tokenize"))
            .json(&serde_json::json!({ "content": text }))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("tokenize unavailable: {e}");
                return Ok(None);
            }
        };
        if !resp.status().is_success() {
            return Ok(None);
        }
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        Ok(v.get("tokens")
            .and_then(Value::as_array)
            .map(|t| t.len() as u64))
    }

    /// Reads llama-server's `/props` (kept for ten seconds). Any failure means "not supported".
    async fn thinking_caps(&self) -> ThinkingCaps {
        if let Some((at, caps)) = self.caps.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            if at.elapsed() < CAPS_TTL {
                return caps.clone();
            }
        }
        let props: Option<Value> = match self
            .client
            .get(self.url("/props"))
            .timeout(CAPS_TIMEOUT)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r.json().await.ok(),
            Ok(_) => None,
            Err(e) => {
                tracing::debug!("props unavailable: {e}");
                None
            }
        };
        let Some(props) = props else {
            return ThinkingCaps::default();
        };
        let caps = ThinkingCaps::from_props(&props);
        *self.caps.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), caps.clone()));
        caps
    }
}

/// Turns a raw byte stream into SSE `data:` payloads.
pub fn sse_payloads<S>(bytes: S) -> SseStream
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    Box::pin(async_stream::try_stream! {
        let mut parser = SseParser::default();
        let mut bytes = Box::pin(bytes);
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|e| RaptorError::Network(e.to_string()))?;
            for payload in parser.feed(&chunk)? {
                yield payload;
            }
        }
        // An event cut off by end-of-stream is discarded, as the SSE spec requires.
    })
}

/// Incremental parser for `text/event-stream`. Handles chunks that split lines
/// anywhere, CRLF line endings, comments, and multi-line `data:` fields.
#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
    data: Vec<String>,
}

impl SseParser {
    /// Consumes `chunk` and returns every payload completed by it.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            self.process_line(line, &mut out);
        }
        if self.buf.len() > MAX_SSE_LINE {
            return Err(RaptorError::Runtime(
                "upstream SSE line exceeds the size limit".into(),
            ));
        }
        Ok(out)
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<String>) {
        if line.is_empty() {
            if !self.data.is_empty() {
                out.push(self.data.join("\n"));
                self.data.clear();
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        if field == "data" {
            self.data.push(value.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_event_split_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: {\"a\":").unwrap().is_empty());
        assert_eq!(p.feed(b"1}\n\n").unwrap(), vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn handles_crlf_comments_and_multiline_data() {
        let mut p = SseParser::default();
        let out = p.feed(b": ping\r\n\r\ndata: a\r\ndata: b\r\n\r\n").unwrap();
        assert_eq!(out, vec!["a\nb".to_string()]);
    }

    #[test]
    fn done_marker_is_a_plain_payload() {
        let mut p = SseParser::default();
        assert_eq!(
            p.feed(b"data: [DONE]\n\n").unwrap(),
            vec!["[DONE]".to_string()]
        );
    }

    #[test]
    fn incomplete_trailing_event_is_not_emitted() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: partial\n").unwrap().is_empty());
    }

    #[test]
    fn rejects_unbounded_line() {
        let mut p = SseParser::default();
        let big = vec![b'x'; MAX_SSE_LINE + 1];
        assert!(p.feed(&big).is_err());
    }

    #[test]
    fn only_loopback_upstreams_are_accepted() {
        assert!(HttpUpstream::new("http://127.0.0.1:8081", vec![]).is_ok());
        assert!(HttpUpstream::new("http://localhost:8081/", vec![]).is_ok());
        assert!(HttpUpstream::new("http://192.168.1.5:8081", vec![]).is_err());
        assert!(HttpUpstream::new("https://127.0.0.1:8081", vec![]).is_err());
        assert!(HttpUpstream::new("not a url", vec![]).is_err());
    }
}
