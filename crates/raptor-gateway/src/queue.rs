//! Request scheduler: bounded concurrency with a FIFO queue.
//!
//! * `max_concurrent` requests run at once (a fair tokio semaphore, so waiters
//!   are served in arrival order).
//! * At most `max_queued` requests wait. Beyond that, admission fails with 503.
//! * Each client may hold at most `per_client_limit` slots (running + waiting).
//!   Beyond that, admission fails with 429.
//!
//! Admission and cancellation are RAII: a waiting request that is dropped
//! (client disconnected) releases its queue slot, and a running request frees
//! its permit when its response stream ends or is dropped.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use raptor_core::{RaptorError, Result};
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::ApiError;

/// Scheduler limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueConfig {
    /// Requests allowed to run concurrently. Must be at least 1.
    pub max_concurrent: usize,
    /// Requests allowed to wait behind the running ones.
    pub max_queued: usize,
    /// Running plus waiting requests a single client may hold. Must be at least 1.
    pub per_client_limit: usize,
    /// Value for the `Retry-After` header on rejections.
    pub retry_after_secs: u64,
    /// Longest a request may wait for a slot before it is rejected with 503.
    pub queue_timeout: Option<Duration>,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1,
            max_queued: 16,
            per_client_limit: 8,
            retry_after_secs: 5,
            queue_timeout: Some(Duration::from_secs(300)),
        }
    }
}

/// Why a request was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The client already holds its maximum number of slots (429).
    PerClientLimit,
    /// The global queue is full (503).
    QueueFull,
    /// The request waited longer than `queue_timeout` (503).
    QueueTimeout,
    /// The scheduler is closed (503).
    Shutdown,
}

impl Rejection {
    /// Renders the rejection as an OpenAI-shaped HTTP error with a `Retry-After` hint.
    pub fn into_api_error(self, retry_after_secs: u64) -> ApiError {
        let e = match self {
            Rejection::PerClientLimit => ApiError::new(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "Too many concurrent requests for this client.",
            )
            .with_code("rate_limit_exceeded"),
            Rejection::QueueFull => ApiError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "The request queue is full; retry later.",
            )
            .with_code("queue_full"),
            Rejection::QueueTimeout => ApiError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "Timed out waiting for a free inference slot.",
            )
            .with_code("queue_timeout"),
            Rejection::Shutdown => ApiError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "The gateway is shutting down.",
            )
            .with_code("shutting_down"),
        };
        e.with_retry_after(retry_after_secs)
    }
}

/// Snapshot of scheduler counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Metrics {
    /// Requests admitted but not yet running.
    pub queued: usize,
    /// Requests currently holding a permit.
    pub running: usize,
    /// Requests whose permit has been released.
    pub completed: u64,
    /// Requests rejected at admission or by timeout.
    pub rejected: u64,
}

#[derive(Default)]
struct State {
    per_client: HashMap<String, usize>,
    waiting: usize,
    running: usize,
    completed: u64,
    rejected: u64,
}

struct Inner {
    cfg: QueueConfig,
    sem: Arc<Semaphore>,
    state: Mutex<State>,
}

fn lock(inner: &Inner) -> MutexGuard<'_, State> {
    // A poisoned lock still holds consistent counters; keep serving.
    inner.state.lock().unwrap_or_else(|e| e.into_inner())
}

fn release_client(st: &mut State, client: &str) {
    let now_zero = match st.per_client.get_mut(client) {
        Some(n) => {
            *n = n.saturating_sub(1);
            *n == 0
        }
        None => false,
    };
    if now_zero {
        st.per_client.remove(client);
    }
}

/// Shared handle to the scheduler. Cloning is cheap.
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Inner>,
}

impl Scheduler {
    /// Creates a scheduler. Errors on limits that would admit nothing.
    pub fn new(cfg: QueueConfig) -> Result<Self> {
        if cfg.max_concurrent == 0 {
            return Err(RaptorError::InvalidInput(
                "max_concurrent must be at least 1".into(),
            ));
        }
        if cfg.per_client_limit == 0 {
            return Err(RaptorError::InvalidInput(
                "per_client_limit must be at least 1".into(),
            ));
        }
        let sem = Arc::new(Semaphore::new(cfg.max_concurrent));
        Ok(Self {
            inner: Arc::new(Inner {
                cfg,
                sem,
                state: Mutex::new(State::default()),
            }),
        })
    }

    /// Waits for a slot for `client`. Dropping the returned future cancels the
    /// wait and frees the queue position.
    pub async fn acquire(&self, client: &str) -> std::result::Result<Permit, Rejection> {
        self.acquire_with(client, None).await
    }

    /// As [`Scheduler::acquire`], but with a per-client cap that replaces the
    /// queue's `per_client_limit` for this client (used for per-key limits).
    pub async fn acquire_with(
        &self,
        client: &str,
        per_client_override: Option<usize>,
    ) -> std::result::Result<Permit, Rejection> {
        let limit = per_client_override.unwrap_or(self.inner.cfg.per_client_limit);
        let mut slot = Slot::admit(&self.inner, client, limit)?;
        let acquire = Arc::clone(&self.inner.sem).acquire_owned();
        let granted = match self.inner.cfg.queue_timeout {
            Some(limit) => match tokio::time::timeout(limit, acquire).await {
                Ok(r) => r,
                Err(_) => {
                    lock(&self.inner).rejected += 1;
                    return Err(Rejection::QueueTimeout);
                }
            },
            None => acquire.await,
        };
        let permit = match granted {
            Ok(p) => p,
            Err(_) => {
                lock(&self.inner).rejected += 1;
                return Err(Rejection::Shutdown);
            }
        };
        slot.promote();
        Ok(Permit {
            inner: Arc::clone(&self.inner),
            client: client.to_string(),
            _permit: permit,
        })
    }

    /// Current counters.
    pub fn metrics(&self) -> Metrics {
        let st = lock(&self.inner);
        Metrics {
            queued: st.waiting,
            running: st.running,
            completed: st.completed,
            rejected: st.rejected,
        }
    }
}

/// A queue position that has not yet been granted a permit. Dropping it
/// without promoting releases the position.
struct Slot {
    inner: Arc<Inner>,
    client: String,
    armed: bool,
}

impl Slot {
    fn admit(
        inner: &Arc<Inner>,
        client: &str,
        limit: usize,
    ) -> std::result::Result<Self, Rejection> {
        let mut st = lock(inner);
        let in_flight = st.per_client.get(client).copied().unwrap_or(0);
        if in_flight >= limit {
            st.rejected += 1;
            return Err(Rejection::PerClientLimit);
        }
        if st.waiting + st.running >= inner.cfg.max_concurrent + inner.cfg.max_queued {
            st.rejected += 1;
            return Err(Rejection::QueueFull);
        }
        st.waiting += 1;
        *st.per_client.entry(client.to_string()).or_insert(0) += 1;
        Ok(Self {
            inner: Arc::clone(inner),
            client: client.to_string(),
            armed: true,
        })
    }

    fn promote(&mut self) {
        let mut st = lock(&self.inner);
        st.waiting = st.waiting.saturating_sub(1);
        st.running += 1;
        self.armed = false;
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.armed {
            let mut st = lock(&self.inner);
            st.waiting = st.waiting.saturating_sub(1);
            release_client(&mut st, &self.client);
        }
    }
}

/// Length of the rate-limit window.
const RATE_WINDOW: Duration = Duration::from_secs(60);

/// Per-client rolling-window request limiter (requests per minute).
///
/// Each check either records one request and succeeds, or fails with the
/// number of seconds until the oldest request leaves the window.
#[derive(Default)]
pub struct RateLimiter {
    windows: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl RateLimiter {
    /// Creates an empty limiter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a request from `client` at `now` if it is within `per_minute`
    /// (`None` is unlimited). On refusal returns the `Retry-After` value in seconds.
    pub fn check(
        &self,
        client: &str,
        per_minute: Option<u32>,
        now: Instant,
    ) -> std::result::Result<(), u64> {
        let Some(limit) = per_minute else {
            return Ok(());
        };
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let window = windows.entry(client.to_string()).or_default();
        while let Some(&front) = window.front() {
            if now.duration_since(front) >= RATE_WINDOW {
                window.pop_front();
            } else {
                break;
            }
        }
        if window.len() >= limit as usize {
            let oldest = window.front().copied().unwrap_or(now);
            let wait = RATE_WINDOW.saturating_sub(now.duration_since(oldest));
            return Err(wait.as_secs().max(1));
        }
        window.push_back(now);
        Ok(())
    }
}

/// A granted slot. Releases the concurrency permit and client count on drop.
pub struct Permit {
    inner: Arc<Inner>,
    client: String,
    _permit: OwnedSemaphorePermit,
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut st = lock(&self.inner);
        st.running = st.running.saturating_sub(1);
        st.completed += 1;
        release_client(&mut st, &self.client);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn cfg(max_concurrent: usize, max_queued: usize, per_client: usize) -> QueueConfig {
        QueueConfig {
            max_concurrent,
            max_queued,
            per_client_limit: per_client,
            retry_after_secs: 3,
            queue_timeout: Some(Duration::from_secs(10)),
        }
    }

    async fn wait_until(mut cond: impl FnMut() -> bool) {
        for _ in 0..2000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("condition not reached");
    }

    #[test]
    fn rejects_zero_limits() {
        assert!(Scheduler::new(cfg(0, 1, 1)).is_err());
        assert!(Scheduler::new(cfg(1, 1, 0)).is_err());
    }

    #[tokio::test]
    async fn admits_up_to_concurrency_and_counts() {
        let s = Scheduler::new(cfg(2, 0, 4)).unwrap();
        let a = s.acquire("a").await.unwrap();
        let b = s.acquire("b").await.unwrap();
        assert_eq!(s.metrics().running, 2);
        drop(a);
        drop(b);
        let m = s.metrics();
        assert_eq!((m.running, m.queued, m.completed), (0, 0, 2));
    }

    #[tokio::test]
    async fn full_queue_rejects_with_queue_full() {
        let s = Scheduler::new(cfg(1, 1, 8)).unwrap();
        let p1 = s.acquire("a").await.unwrap();
        let s2 = s.clone();
        let waiter = tokio::spawn(async move { s2.acquire("b").await.map(|_| ()) });
        wait_until(|| s.metrics().queued == 1).await;
        assert_eq!(s.acquire("c").await.err(), Some(Rejection::QueueFull));
        assert_eq!(s.metrics().rejected, 1);
        drop(p1);
        assert!(waiter.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn per_client_limit_is_independent_per_client() {
        let s = Scheduler::new(cfg(4, 4, 1)).unwrap();
        let _a = s.acquire("a").await.unwrap();
        assert_eq!(s.acquire("a").await.err(), Some(Rejection::PerClientLimit));
        assert!(s.acquire("b").await.is_ok());
    }

    #[tokio::test]
    async fn waiters_are_served_in_fifo_order() {
        let s = Scheduler::new(cfg(1, 10, 10)).unwrap();
        let gate = s.acquire("holder").await.unwrap();
        let order = Arc::new(StdMutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for i in 1..=3usize {
            let s2 = s.clone();
            let order2 = Arc::clone(&order);
            tasks.push(tokio::spawn(async move {
                let p = s2.acquire(&format!("c{i}")).await.unwrap();
                order2.lock().unwrap().push(i);
                drop(p);
            }));
            let want = i;
            wait_until(|| s.metrics().queued == want).await;
        }
        drop(gate);
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn cancelled_waiter_frees_its_position() {
        let s = Scheduler::new(cfg(1, 1, 8)).unwrap();
        let gate = s.acquire("a").await.unwrap();
        let s2 = s.clone();
        let waiter = tokio::spawn(async move { s2.acquire("b").await.map(|_| ()) });
        wait_until(|| s.metrics().queued == 1).await;
        waiter.abort();
        wait_until(|| s.metrics().queued == 0).await;
        // The freed position can be used again.
        let s3 = s.clone();
        let again = tokio::spawn(async move { s3.acquire("c").await.map(|_| ()) });
        wait_until(|| s.metrics().queued == 1).await;
        drop(gate);
        assert!(again.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn queue_timeout_rejects_waiter() {
        let mut c = cfg(1, 4, 4);
        c.queue_timeout = Some(Duration::from_millis(30));
        let s = Scheduler::new(c).unwrap();
        let _gate = s.acquire("a").await.unwrap();
        assert_eq!(s.acquire("b").await.err(), Some(Rejection::QueueTimeout));
        let m = s.metrics();
        assert_eq!((m.queued, m.rejected), (0, 1));
    }

    #[tokio::test]
    async fn per_key_override_replaces_the_client_cap() {
        let s = Scheduler::new(cfg(4, 4, 1)).unwrap();
        let _a = s.acquire_with("key:a", Some(2)).await.unwrap();
        // The default cap of 1 would have refused this second request.
        let _b = s.acquire_with("key:a", Some(2)).await.unwrap();
        assert_eq!(
            s.acquire_with("key:a", Some(2)).await.err(),
            Some(Rejection::PerClientLimit)
        );
        assert!(s.acquire("key:b").await.is_ok());
    }

    #[test]
    fn rate_limiter_is_a_rolling_window_per_client() {
        let rl = RateLimiter::new();
        let t0 = Instant::now();
        assert!(rl.check("key:a", Some(2), t0).is_ok());
        assert!(rl.check("key:a", Some(2), t0).is_ok());
        let retry = rl.check("key:a", Some(2), t0).unwrap_err();
        assert!((1..=60).contains(&retry));
        // Another client has its own window.
        assert!(rl.check("key:b", Some(2), t0).is_ok());
        // After the window has passed, the client may proceed again.
        let later = t0 + RATE_WINDOW + Duration::from_millis(1);
        assert!(rl.check("key:a", Some(2), later).is_ok());
        // No limit means no bookkeeping.
        assert!(rl.check("key:c", None, t0).is_ok());
    }

    #[tokio::test]
    async fn rejection_maps_to_status_and_retry_after() {
        let e = Rejection::QueueFull.into_api_error(9);
        assert_eq!(e.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(e.retry_after, Some(9));
        assert_eq!(e.code, Some("queue_full"));
        let r = Rejection::PerClientLimit.into_api_error(2);
        assert_eq!(r.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
    }
}
