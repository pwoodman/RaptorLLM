//! Token counting for context builds. While llama-server is Ready, counts come from the
//! loaded model's own tokenizer (`/tokenize`). Otherwise they come from the estimating counter.
//!
//! The context engine fixes its counter when a workspace engine opens, so this counter reads
//! the server state on every call. `is_exact` reports the most recent call, which is what the
//! manifest's `tokens_exact` records.

use crate::adapters::ServerProbe;
use raptor_core::{EstimatingCounter, TokenCounter};
use raptor_runtime::tokenizer::{ServerTokenCounter, TokenizeTransport};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

pub struct ServerAwareCounter {
    server: Arc<dyn ServerProbe>,
    transport: Arc<dyn TokenizeTransport>,
    /// The counter for the endpoint seen last. Its cache is kept while the endpoint is stable.
    current: Mutex<Option<(String, Arc<ServerTokenCounter>)>>,
    last_exact: AtomicBool,
}

impl ServerAwareCounter {
    pub fn new(server: Arc<dyn ServerProbe>, transport: Arc<dyn TokenizeTransport>) -> Self {
        Self {
            server,
            transport,
            current: Mutex::new(None),
            last_exact: AtomicBool::new(false),
        }
    }

    fn counter_for(&self, endpoint: &str) -> Arc<ServerTokenCounter> {
        let mut current = lock(&self.current);
        if let Some((known, counter)) = current.as_ref() {
            if known == endpoint {
                return counter.clone();
            }
        }
        let counter = Arc::new(ServerTokenCounter::new(
            endpoint.to_string(),
            self.transport.clone(),
        ));
        *current = Some((endpoint.to_string(), counter.clone()));
        counter
    }
}

impl TokenCounter for ServerAwareCounter {
    fn count(&self, text: &str) -> usize {
        match self.server.endpoint() {
            Some(endpoint) => {
                let counter = self.counter_for(&endpoint);
                let n = counter.count(text);
                self.last_exact.store(counter.is_exact(), Ordering::SeqCst);
                n
            }
            None => {
                self.last_exact.store(false, Ordering::SeqCst);
                EstimatingCounter.count(text)
            }
        }
    }

    fn is_exact(&self) -> bool {
        self.last_exact.load(Ordering::SeqCst)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::fakes::FakeServer;
    use raptor_core::RaptorError;
    use std::sync::atomic::AtomicUsize;

    struct CountingTransport {
        calls: AtomicUsize,
        fail: bool,
    }

    impl TokenizeTransport for CountingTransport {
        fn count_tokens(&self, _base: &str, text: &str) -> raptor_core::Result<usize> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(RaptorError::Network("down".into()))
            } else {
                Ok(text.len() / 2)
            }
        }
    }

    fn server(endpoint: Option<&str>) -> Arc<FakeServer> {
        Arc::new(FakeServer(Mutex::new(endpoint.map(str::to_string))))
    }

    #[test]
    fn ready_server_gives_exact_server_counts() {
        let t = Arc::new(CountingTransport {
            calls: AtomicUsize::new(0),
            fail: false,
        });
        let c = ServerAwareCounter::new(server(Some("http://127.0.0.1:8080")), t.clone());
        assert_eq!(c.count("abcdefgh"), 4);
        assert!(c.is_exact());
        assert_eq!(c.count("abcdefgh"), 4);
        assert_eq!(t.calls.load(Ordering::SeqCst), 1, "second count is cached");
    }

    #[test]
    fn stopped_server_gives_flagged_estimates_without_requests() {
        let t = Arc::new(CountingTransport {
            calls: AtomicUsize::new(0),
            fail: false,
        });
        let c = ServerAwareCounter::new(server(None), t.clone());
        let text = "a fairly long sentence to estimate";
        assert_eq!(c.count(text), EstimatingCounter.count(text));
        assert!(!c.is_exact());
        assert_eq!(t.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn failing_tokenizer_falls_back_and_is_flagged_inexact() {
        let t = Arc::new(CountingTransport {
            calls: AtomicUsize::new(0),
            fail: true,
        });
        let c = ServerAwareCounter::new(server(Some("http://127.0.0.1:8080")), t);
        let text = "another sentence here";
        assert_eq!(c.count(text), EstimatingCounter.count(text));
        assert!(!c.is_exact());
    }

    #[test]
    fn exactness_follows_the_server_state_per_call() {
        let t = Arc::new(CountingTransport {
            calls: AtomicUsize::new(0),
            fail: false,
        });
        let probe = server(Some("http://127.0.0.1:8080"));
        let c = ServerAwareCounter::new(probe.clone(), t);
        c.count("exact text");
        assert!(c.is_exact());
        *lock(&probe.0) = None;
        c.count("estimated text");
        assert!(!c.is_exact());
    }
}
