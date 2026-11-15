//! Exact token counts from llama-server's `/tokenize` endpoint, with a flagged
//! estimate when the server is unavailable.
//!
//! `TokenCounter` is synchronous, so the transport is a small blocking HTTP/1.1
//! client restricted to loopback addresses. A call takes milliseconds locally.

use crate::hash::sha256_hex;
use raptor_core::{EstimatingCounter, RaptorError, Result, TokenCounter};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const CACHE_LIMIT: usize = 4096;

/// Asks a server how many tokens `text` is. Behind a trait so tests use a fake.
pub trait TokenizeTransport: Send + Sync {
    fn count_tokens(&self, base_url: &str, text: &str) -> Result<usize>;
}

/// Blocking loopback transport. Refuses any non-loopback address, so prompt
/// text never leaves the machine.
#[derive(Debug, Clone, Copy)]
pub struct LoopbackTransport {
    pub timeout: Duration,
}

impl Default for LoopbackTransport {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
        }
    }
}

impl TokenizeTransport for LoopbackTransport {
    fn count_tokens(&self, base_url: &str, text: &str) -> Result<usize> {
        let (ip, port) = parse_loopback_base(base_url)?;
        let body = serde_json::json!({ "content": text }).to_string();
        let addr = SocketAddr::new(ip, port);
        let mut stream = TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|e| RaptorError::Network(format!("tokenize connect {addr}: {e}")))?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        let request = format!(
            "POST /tokenize HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| RaptorError::Network(format!("tokenize read: {e}")))?;
        let (status, resp) = parse_http_response(&raw)?;
        if status != 200 {
            return Err(RaptorError::Network(format!(
                "/tokenize returned HTTP {status}"
            )));
        }
        parse_tokenize_response(&resp)
    }
}

/// Splits `http://127.0.0.1:8080` (or `http://[::1]:8080`) and requires a loopback IP.
pub fn parse_loopback_base(base_url: &str) -> Result<(IpAddr, u16)> {
    let rest = base_url
        .strip_prefix("http://")
        .ok_or_else(|| RaptorError::InvalidInput("tokenizer base URL must be http://".into()))?
        .trim_end_matches('/');
    let (host, port) = rest
        .rsplit_once(':')
        .ok_or_else(|| RaptorError::InvalidInput("tokenizer base URL has no port".into()))?;
    let ip: IpAddr = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .map_err(|_| {
            RaptorError::InvalidInput("tokenizer base URL host must be an IP address".into())
        })?;
    if !ip.is_loopback() {
        return Err(RaptorError::InvalidInput(
            "tokenizer transport only talks to loopback addresses".into(),
        ));
    }
    let port: u16 = port
        .parse()
        .map_err(|_| RaptorError::InvalidInput("tokenizer base URL port is invalid".into()))?;
    Ok((ip, port))
}

/// Parses a raw HTTP/1.1 response into (status, body). Handles Content-Length
/// framing (or read-to-close) and chunked transfer encoding.
pub fn parse_http_response(raw: &[u8]) -> Result<(u16, Vec<u8>)> {
    let sep = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| {
            RaptorError::Network("malformed HTTP response: no header terminator".into())
        })?;
    let head = std::str::from_utf8(&raw[..sep])
        .map_err(|_| RaptorError::Network("malformed HTTP response: non-UTF-8 headers".into()))?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| RaptorError::Network(format!("malformed status line: {status_line:?}")))?;
    let chunked = lines.any(|h| {
        let h = h.to_ascii_lowercase();
        h.starts_with("transfer-encoding:") && h.contains("chunked")
    });
    let payload = &raw[sep + 4..];
    let body = if chunked {
        decode_chunked(payload)?
    } else {
        payload.to_vec()
    };
    Ok((status, body))
}

fn decode_chunked(data: &[u8]) -> Result<Vec<u8>> {
    let bad = |m: &str| RaptorError::Network(format!("malformed chunked body: {m}"));
    let mut out = Vec::new();
    let mut pos = 0usize;
    loop {
        let rel = data
            .get(pos..)
            .and_then(|s| s.windows(2).position(|w| w == b"\r\n"))
            .ok_or_else(|| bad("missing chunk size line"))?;
        let size_line =
            std::str::from_utf8(&data[pos..pos + rel]).map_err(|_| bad("non-UTF-8 size"))?;
        let size_hex = size_line.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|_| bad("invalid chunk size"))?;
        pos += rel + 2;
        if size == 0 {
            return Ok(out);
        }
        let end = pos
            .checked_add(size)
            .ok_or_else(|| bad("chunk size overflow"))?;
        let chunk = data.get(pos..end).ok_or_else(|| bad("truncated chunk"))?;
        out.extend_from_slice(chunk);
        if data.get(end..end + 2) != Some(b"\r\n") {
            return Err(bad("missing chunk terminator"));
        }
        pos = end + 2;
    }
}

/// Counts the entries in `{"tokens": [...]}`.
pub fn parse_tokenize_response(body: &[u8]) -> Result<usize> {
    #[derive(Deserialize)]
    struct Tokens {
        tokens: Vec<serde_json::Value>,
    }
    let parsed: Tokens = serde_json::from_slice(body)
        .map_err(|e| RaptorError::Runtime(format!("unreadable /tokenize response: {e}")))?;
    Ok(parsed.tokens.len())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// `TokenCounter` backed by the model's own tokenizer on a running server.
///
/// If the server cannot be reached, the count falls back to `EstimatingCounter`
/// and `is_exact()` returns false until the next successful count. `is_exact`
/// describes the most recent call, so consumers should check it right after counting.
pub struct ServerTokenCounter {
    base_url: String,
    transport: Arc<dyn TokenizeTransport>,
    fallback: EstimatingCounter,
    last_exact: AtomicBool,
    cache: Mutex<HashMap<String, usize>>,
}

impl ServerTokenCounter {
    /// Creates a counter that queries `base_url` (e.g. `http://127.0.0.1:8080`) through `transport`.
    pub fn new(base_url: impl Into<String>, transport: Arc<dyn TokenizeTransport>) -> Self {
        Self {
            base_url: base_url.into(),
            transport,
            fallback: EstimatingCounter,
            last_exact: AtomicBool::new(false),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Counter using the real loopback transport.
    pub fn loopback(base_url: impl Into<String>) -> Self {
        Self::new(base_url, Arc::new(LoopbackTransport::default()))
    }
}

impl TokenCounter for ServerTokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            self.last_exact.store(true, Ordering::SeqCst);
            return 0;
        }
        let key = sha256_hex(text.as_bytes());
        if let Some(n) = lock(&self.cache).get(&key).copied() {
            self.last_exact.store(true, Ordering::SeqCst);
            return n;
        }
        match self.transport.count_tokens(&self.base_url, text) {
            Ok(n) => {
                let mut cache = lock(&self.cache);
                if cache.len() >= CACHE_LIMIT {
                    cache.clear();
                }
                cache.insert(key, n);
                self.last_exact.store(true, Ordering::SeqCst);
                n
            }
            Err(e) => {
                tracing::warn!(error = %e, "tokenizer unavailable; using estimate");
                self.last_exact.store(false, Ordering::SeqCst);
                self.fallback.count(text)
            }
        }
    }

    fn is_exact(&self) -> bool {
        self.last_exact.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct FakeTransport {
        answer: std::result::Result<usize, String>,
        calls: AtomicUsize,
    }

    impl FakeTransport {
        fn ok(n: usize) -> Arc<Self> {
            Arc::new(Self {
                answer: Ok(n),
                calls: AtomicUsize::new(0),
            })
        }
        fn failing() -> Arc<Self> {
            Arc::new(Self {
                answer: Err("down".into()),
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl TokenizeTransport for FakeTransport {
        fn count_tokens(&self, _base: &str, _text: &str) -> Result<usize> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.answer.clone().map_err(RaptorError::Network)
        }
    }

    #[test]
    fn content_length_response_is_parsed() {
        let raw =
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_http_response(raw).unwrap(), (200, b"{}".to_vec()));
    }

    #[test]
    fn chunked_response_is_decoded() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(parse_http_response(raw).unwrap().1, b"hello world".to_vec());
    }

    #[test]
    fn chunk_extensions_are_ignored() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;ext=1\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_http_response(raw).unwrap().1, b"hello".to_vec());
    }

    #[test]
    fn truncated_chunk_is_an_error() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nhi\r\n";
        assert!(parse_http_response(raw).is_err());
    }

    #[test]
    fn malformed_responses_are_errors() {
        assert!(parse_http_response(b"HTTP/1.1 200 OK\r\nno terminator").is_err());
        assert!(parse_http_response(b"garbage\r\n\r\n").is_err());
    }

    #[test]
    fn tokenize_response_counts_entries() {
        assert_eq!(
            parse_tokenize_response(br#"{"tokens":[1,2,3]}"#).unwrap(),
            3
        );
        assert_eq!(parse_tokenize_response(br#"{"tokens":[]}"#).unwrap(), 0);
        assert!(parse_tokenize_response(br#"{"nope":1}"#).is_err());
    }

    #[test]
    fn base_url_must_be_loopback_http() {
        assert_eq!(
            parse_loopback_base("http://127.0.0.1:8080").unwrap(),
            ("127.0.0.1".parse().unwrap(), 8080)
        );
        assert_eq!(parse_loopback_base("http://[::1]:9000/").unwrap().1, 9000);
        assert!(parse_loopback_base("http://10.0.0.5:8080").is_err());
        assert!(parse_loopback_base("https://127.0.0.1:8080").is_err());
        assert!(parse_loopback_base("http://localhost:8080").is_err());
        assert!(parse_loopback_base("http://127.0.0.1").is_err());
    }

    #[test]
    fn exact_counts_are_cached() {
        let t = FakeTransport::ok(7);
        let c = ServerTokenCounter::new("http://127.0.0.1:8080", t.clone());
        assert_eq!(c.count("hello there"), 7);
        assert!(c.is_exact());
        assert_eq!(c.count("hello there"), 7);
        assert_eq!(t.calls(), 1);
    }

    #[test]
    fn empty_text_is_zero_without_a_request() {
        let t = FakeTransport::ok(99);
        let c = ServerTokenCounter::new("http://127.0.0.1:8080", t.clone());
        assert_eq!(c.count(""), 0);
        assert_eq!(t.calls(), 0);
    }

    #[test]
    fn unreachable_server_falls_back_and_is_flagged_inexact() {
        let t = FakeTransport::failing();
        let c = ServerTokenCounter::new("http://127.0.0.1:8080", t);
        let text = "a fairly long prompt that needs estimating";
        assert_eq!(c.count(text), EstimatingCounter.count(text));
        assert!(!c.is_exact());
    }

    #[test]
    fn exactness_recovers_after_a_good_count() {
        struct Flaky(AtomicUsize);
        impl TokenizeTransport for Flaky {
            fn count_tokens(&self, _b: &str, _t: &str) -> Result<usize> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(RaptorError::Network("first call fails".into()))
                } else {
                    Ok(4)
                }
            }
        }
        let c = ServerTokenCounter::new(
            "http://127.0.0.1:8080",
            Arc::new(Flaky(AtomicUsize::new(0))),
        );
        c.count("one");
        assert!(!c.is_exact());
        assert_eq!(c.count("two"), 4);
        assert!(c.is_exact());
    }

    #[test]
    fn loopback_transport_talks_to_a_real_socket() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = sock.read(&mut buf).unwrap();
            let body = r#"{"tokens":[1,2,3,4]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });
        let base = format!("http://127.0.0.1:{port}");
        let n = LoopbackTransport::default()
            .count_tokens(&base, "some text")
            .unwrap();
        assert_eq!(n, 4);
        server.join().unwrap();
    }

    #[test]
    fn loopback_transport_refuses_remote_hosts_without_connecting() {
        let err = LoopbackTransport::default()
            .count_tokens("http://8.8.8.8:8080", "secret prompt")
            .unwrap_err();
        assert!(matches!(err, RaptorError::InvalidInput(_)));
    }
}
