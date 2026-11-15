//! Context-engine backends served by the running llama-server.
//!
//! - [`ServerEmbedder`] implements `EmbeddingBackend` over `POST /v1/embeddings`. It is only
//!   offered when the configured embedding model is the one loaded, and its vector width is
//!   measured with one probe request before use.
//! - [`ServerTextGenerator`] implements `TextGenerator` over `POST /v1/chat/completions`, so
//!   `ModelSummarizer` can write session summaries with the loaded model.
//!
//! Both reach the server through [`LlamaHttp`] (loopback only) and learn what is loaded from
//! [`ActiveModel`]. Tests use fakes for both. The context engine never fails a build because
//! of these backends: the reason they are unavailable is kept and reported.

use raptor_context::{EmbeddingBackend, TextGenerator};
use raptor_core::{RaptorError, Result};
use raptor_runtime::tokenizer::{parse_http_response, parse_loopback_base};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

/// Timeout for one embedding request (a batch of up to 32 chunks).
pub const EMBED_TIMEOUT: Duration = Duration::from_secs(60);
/// Timeout for one summary generation.
pub const CHAT_TIMEOUT: Duration = Duration::from_secs(300);
const ERROR_SNIPPET_CHARS: usize = 200;

/// The server that can take requests now, and the library model it serves.
/// `generation` changes on every successful load, so a cached backend is never reused across
/// server restarts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveServer {
    pub endpoint: String,
    pub model_id: String,
    pub generation: u64,
}

/// Reports the active server, or `None` while no model is ready.
pub trait ActiveModel: Send + Sync {
    fn active(&self) -> Option<ActiveServer>;
}

/// One JSON POST to a llama-server. `path` is always a constant chosen by this module.
pub trait LlamaHttp: Send + Sync {
    fn post_json(&self, base: &str, path: &str, body: &Value, timeout: Duration) -> Result<Value>;
}

/// Blocking loopback HTTP client. Refuses non-loopback endpoints, so prompts and documents
/// never leave the machine.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopbackHttp;

impl LlamaHttp for LoopbackHttp {
    fn post_json(&self, base: &str, path: &str, body: &Value, timeout: Duration) -> Result<Value> {
        let (ip, port) = parse_loopback_base(base)?;
        let payload = serde_json::to_string(body)?;
        let addr = SocketAddr::new(ip, port);
        let mut stream = TcpStream::connect_timeout(&addr, timeout)
            .map_err(|e| RaptorError::Network(format!("connect {addr}: {e}")))?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| RaptorError::Network(format!("{path} read: {e}")))?;
        let (status, resp) = parse_http_response(&raw)?;
        if status != 200 {
            let text = String::from_utf8_lossy(&resp);
            let snippet: String = text.chars().take(ERROR_SNIPPET_CHARS).collect();
            return Err(RaptorError::Runtime(format!(
                "{path} returned HTTP {status}: {snippet}"
            )));
        }
        serde_json::from_slice(&resp)
            .map_err(|e| RaptorError::Runtime(format!("{path} returned malformed JSON: {e}")))
    }
}

/// Reads `data[].embedding` from an OpenAI-style embeddings reply, ordered by `index`.
/// Refuses a reply with the wrong count, a gap in the indexes, or a non-finite value.
pub fn parse_embeddings(resp: &Value, expected: usize) -> Result<Vec<Vec<f32>>> {
    let data = resp["data"]
        .as_array()
        .ok_or_else(|| RaptorError::Runtime("embeddings reply has no data array".into()))?;
    if data.len() != expected {
        return Err(RaptorError::Runtime(format!(
            "embeddings reply has {} vectors for {expected} inputs",
            data.len()
        )));
    }
    let mut rows: Vec<Option<Vec<f32>>> = vec![None; expected];
    for (pos, item) in data.iter().enumerate() {
        let index = item["index"].as_u64().map_or(pos, |i| i as usize);
        let values = item["embedding"]
            .as_array()
            .ok_or_else(|| RaptorError::Runtime("embedding entry has no vector".into()))?;
        let mut vector = Vec::with_capacity(values.len());
        for v in values {
            let f = v
                .as_f64()
                .filter(|f| f.is_finite())
                .ok_or_else(|| RaptorError::Runtime("embedding contains a non-number".into()))?;
            vector.push(f as f32);
        }
        match rows.get_mut(index) {
            Some(slot) if slot.is_none() => *slot = Some(vector),
            _ => {
                return Err(RaptorError::Runtime(
                    "embeddings reply has a missing or repeated index".into(),
                ))
            }
        }
    }
    rows.into_iter()
        .map(|r| r.ok_or_else(|| RaptorError::Runtime("embeddings reply is incomplete".into())))
        .collect()
}

fn embed_raw(
    http: &dyn LlamaHttp,
    endpoint: &str,
    model_id: &str,
    inputs: &[&str],
) -> Result<Vec<Vec<f32>>> {
    let body = json!({ "model": model_id, "input": inputs });
    let resp = http.post_json(endpoint, "/v1/embeddings", &body, EMBED_TIMEOUT)?;
    parse_embeddings(&resp, inputs.len())
}

/// Embedding backend for one library model on one server generation.
pub struct ServerEmbedder {
    http: Arc<dyn LlamaHttp>,
    endpoint: String,
    model_id: String,
    dimension: usize,
}

impl ServerEmbedder {
    /// Measures the vector width with one request. A model that returns no vector is refused.
    pub fn probe(http: Arc<dyn LlamaHttp>, active: &ActiveServer, model_id: &str) -> Result<Self> {
        let vectors = embed_raw(
            http.as_ref(),
            &active.endpoint,
            model_id,
            &["dimension probe"],
        )?;
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0 {
            return Err(RaptorError::Runtime(
                "the embedding model returned an empty vector".into(),
            ));
        }
        Ok(Self {
            http,
            endpoint: active.endpoint.clone(),
            model_id: model_id.to_string(),
            dimension,
        })
    }
}

impl EmbeddingBackend for ServerEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = embed_raw(self.http.as_ref(), &self.endpoint, &self.model_id, texts)?;
        if vectors.iter().any(|v| v.len() != self.dimension) {
            return Err(RaptorError::Runtime(
                "the embedding width changed while indexing".into(),
            ));
        }
        Ok(vectors)
    }
}

/// Summary generation with whatever model the server has loaded.
pub struct ServerTextGenerator {
    http: Arc<dyn LlamaHttp>,
    active: Arc<dyn ActiveModel>,
}

impl ServerTextGenerator {
    pub fn new(http: Arc<dyn LlamaHttp>, active: Arc<dyn ActiveModel>) -> Self {
        Self { http, active }
    }
}

impl TextGenerator for ServerTextGenerator {
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        let active = self.active.active().ok_or_else(|| {
            RaptorError::Conflict("no model is loaded, so the summary cannot be written".into())
        })?;
        let body = json!({
            "model": active.model_id,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": false,
        });
        let resp = self.http.post_json(
            &active.endpoint,
            "/v1/chat/completions",
            &body,
            CHAT_TIMEOUT,
        )?;
        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| RaptorError::Runtime("chat reply has no message content".into()))
    }
}

/// Everything the context hub needs to build the server-backed collaborators.
pub struct SemanticSetup {
    pub active: Arc<dyn ActiveModel>,
    pub http: Arc<dyn LlamaHttp>,
}

impl SemanticSetup {
    /// Embedding backend for `model_id`, or the plain reason it cannot be used now.
    pub fn embedder_for(
        &self,
        model_id: &str,
    ) -> std::result::Result<Arc<dyn EmbeddingBackend>, String> {
        let Some(active) = self.active.active() else {
            return Err(format!(
                "no model is loaded; load the embedding model ({model_id}) to enable semantic search"
            ));
        };
        if active.model_id != model_id {
            return Err(format!(
                "the loaded model is not the embedding model ({model_id}); load it to enable semantic search"
            ));
        }
        ServerEmbedder::probe(self.http.clone(), &active, model_id)
            .map(|e| Arc::new(e) as Arc<dyn EmbeddingBackend>)
            .map_err(|e| format!("the embedding model did not answer: {e}"))
    }
}

#[cfg(test)]
pub(crate) mod fakes {
    use super::*;
    use std::sync::Mutex;

    pub struct FixedActive(pub Mutex<Option<ActiveServer>>);

    impl ActiveModel for FixedActive {
        fn active(&self) -> Option<ActiveServer> {
            self.0.lock().unwrap().clone()
        }
    }

    /// Answers `/v1/embeddings` with a 4-wide vector per input, and `/v1/chat/completions`
    /// with a canned reply. `embeddings_status` lets a test simulate a server without support.
    pub struct FakeLlama {
        pub calls: Mutex<Vec<String>>,
        pub embeddings_status: u16,
        pub reply: String,
    }

    impl FakeLlama {
        pub fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                embeddings_status: 200,
                reply: "The team agreed the ledger stays append-only.".into(),
            }
        }

        pub fn count(&self, path: &str) -> usize {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|p| *p == path)
                .count()
        }
    }

    /// Deterministic 4-dimensional embedding: byte sums of the text, spread over four slots.
    pub fn fake_vector(text: &str) -> Vec<f32> {
        let mut v = [0f32; 4];
        for (i, b) in text.bytes().enumerate() {
            v[i % 4] += f32::from(b);
        }
        v.iter().map(|x| x + 1.0).collect()
    }

    impl LlamaHttp for FakeLlama {
        fn post_json(&self, _base: &str, path: &str, body: &Value, _t: Duration) -> Result<Value> {
            self.calls.lock().unwrap().push(path.to_string());
            match path {
                "/v1/embeddings" => {
                    if self.embeddings_status != 200 {
                        return Err(RaptorError::Runtime(format!(
                            "{path} returned HTTP {}: embeddings are not supported",
                            self.embeddings_status
                        )));
                    }
                    let inputs: Vec<&str> = match &body["input"] {
                        Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
                        Value::String(s) => vec![s.as_str()],
                        _ => vec![],
                    };
                    let data: Vec<Value> = inputs
                        .iter()
                        .enumerate()
                        .map(|(i, t)| json!({ "index": i, "embedding": fake_vector(t) }))
                        .collect();
                    Ok(json!({ "data": data }))
                }
                "/v1/chat/completions" => Ok(json!({
                    "choices": [{ "message": { "role": "assistant", "content": self.reply } }]
                })),
                other => Err(RaptorError::NotFound(format!("unexpected path {other}"))),
            }
        }
    }

    pub fn active(model: &str, generation: u64) -> ActiveServer {
        ActiveServer {
            endpoint: "http://127.0.0.1:8081".into(),
            model_id: model.into(),
            generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::*;
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn embeddings_are_ordered_by_index_and_checked() {
        let reply = json!({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]}
        ]});
        let v = parse_embeddings(&reply, 2).unwrap();
        assert_eq!(v, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        // Wrong count, repeated index, and non-numbers are all refused.
        assert!(parse_embeddings(&reply, 3).is_err());
        let repeated = json!({"data": [
            {"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}
        ]});
        assert!(parse_embeddings(&repeated, 2).is_err());
        let text = json!({"data": [{"index": 0, "embedding": ["x"]}]});
        assert!(parse_embeddings(&text, 1).is_err());
        assert!(parse_embeddings(&json!({}), 1).is_err());
    }

    #[test]
    fn embedder_measures_width_and_serves_batches() {
        let http = Arc::new(FakeLlama::new());
        let e = ServerEmbedder::probe(http.clone(), &active("emb.gguf", 1), "emb.gguf").unwrap();
        assert_eq!(e.model_id(), "emb.gguf");
        assert_eq!(e.dimension(), 4);
        let out = e.embed_batch(&["alpha", "beta"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], fake_vector("alpha"));
        assert!(e.embed_batch(&[]).unwrap().is_empty());
        assert_eq!(
            http.count("/v1/embeddings"),
            2,
            "probe plus one batch; the empty batch sends nothing"
        );
    }

    #[test]
    fn a_server_without_embedding_support_is_reported_not_hidden() {
        let http = Arc::new(FakeLlama {
            embeddings_status: 501,
            ..FakeLlama::new()
        });
        let err = ServerEmbedder::probe(http, &active("emb.gguf", 1), "emb.gguf")
            .err()
            .unwrap();
        assert!(err.to_string().contains("501"), "{err}");
    }

    #[test]
    fn embedder_for_refuses_when_another_model_is_loaded_or_none() {
        let http: Arc<dyn LlamaHttp> = Arc::new(FakeLlama::new());
        let none = SemanticSetup {
            active: Arc::new(FixedActive(Mutex::new(None))),
            http: http.clone(),
        };
        let reason = none.embedder_for("emb.gguf").err().unwrap();
        assert!(reason.contains("no model is loaded"), "{reason}");

        let other = SemanticSetup {
            active: Arc::new(FixedActive(Mutex::new(Some(active("chat.gguf", 2))))),
            http: http.clone(),
        };
        let reason = other.embedder_for("emb.gguf").err().unwrap();
        assert!(reason.contains("not the embedding model"), "{reason}");

        let ok = SemanticSetup {
            active: Arc::new(FixedActive(Mutex::new(Some(active("emb.gguf", 3))))),
            http,
        };
        assert_eq!(ok.embedder_for("emb.gguf").unwrap().dimension(), 4);
    }

    #[test]
    fn text_generator_uses_the_loaded_model_and_needs_one() {
        let http = Arc::new(FakeLlama::new());
        let gen = ServerTextGenerator::new(
            http.clone(),
            Arc::new(FixedActive(Mutex::new(Some(active("chat.gguf", 1))))),
        );
        let text = gen.generate("summarise", 128).unwrap();
        assert!(text.contains("append-only"));
        assert_eq!(http.count("/v1/chat/completions"), 1);

        let idle = ServerTextGenerator::new(http, Arc::new(FixedActive(Mutex::new(None))));
        assert!(matches!(
            idle.generate("x", 8),
            Err(RaptorError::Conflict(_))
        ));
    }

    #[test]
    fn loopback_http_refuses_remote_endpoints_before_connecting() {
        let err = LoopbackHttp.post_json(
            "http://10.0.0.5:8081",
            "/v1/embeddings",
            &json!({}),
            Duration::from_secs(1),
        );
        assert!(err.is_err());
    }
}
