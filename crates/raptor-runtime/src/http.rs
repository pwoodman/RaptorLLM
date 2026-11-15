//! Async HTTP GET used for llama-server health checks. Behind a trait so the
//! process manager is testable without a socket.

use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use std::time::Duration;

/// A completed HTTP exchange. Non-2xx statuses are returned, not errors, so
/// callers can distinguish "loading" (503) from "down" (connection refused).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

#[async_trait]
pub trait HttpClient: Send + Sync {
    /// GET `url`, failing with `RaptorError::Network` on transport errors or timeout.
    async fn get(&self, url: &str, timeout: Duration) -> Result<HttpResponse>;
}

/// reqwest-backed client (rustls, no C toolchain required).
#[derive(Debug, Clone)]
pub struct ReqwestClient {
    client: reqwest::Client,
}

impl ReqwestClient {
    /// Builds the client. Redirects are disabled: llama-server never redirects,
    /// and following them could send probes to an unexpected host.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|e| RaptorError::Network(format!("http client init failed: {e}")))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl HttpClient for ReqwestClient {
    async fn get(&self, url: &str, timeout: Duration) -> Result<HttpResponse> {
        let resp = self
            .client
            .get(url)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| RaptorError::Network(format!("GET {url}: {e}")))?;
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .await
            .map_err(|e| RaptorError::Network(format!("GET {url}: body read failed: {e}")))?;
        Ok(HttpResponse { status, body })
    }
}
