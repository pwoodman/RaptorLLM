//! Reports llama-server state changes to the UI. The status is read once per interval and an
//! event is sent only when it differs from the last one sent, so a crash reaches the UI within
//! one interval. The loop ends when its cancellation token fires, which happens at app exit.

use raptor_runtime::EngineStatus;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Time between status reads. A crash is reported within this time.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// True when `status` differs from the last one reported. Records the new status when it does.
pub fn changed(last: &mut Option<String>, status: &EngineStatus) -> bool {
    let Ok(json) = serde_json::to_string(status) else {
        return false;
    };
    if last.as_deref() == Some(json.as_str()) {
        return false;
    }
    *last = Some(json);
    true
}

/// Calls `emit` with the status now and whenever it changes, until `cancel` fires.
pub async fn watch<S, E>(mut source: S, cancel: CancellationToken, interval: Duration, mut emit: E)
where
    S: FnMut() -> EngineStatus + Send,
    E: FnMut(&EngineStatus) + Send,
{
    let mut last = None;
    loop {
        let status = source();
        if changed(&mut last, &status) {
            emit(&status);
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(interval) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_runtime::process::{ServerState, ServerStatus};
    use std::sync::{Arc, Mutex};

    fn status_with(state: ServerState) -> EngineStatus {
        EngineStatus {
            hardware_detected: true,
            runtime: None,
            server: ServerStatus {
                state,
                endpoint: None,
                pid: None,
                restarts: 0,
            },
            gpu_verdict: None,
            device_preflight: None,
        }
    }

    fn label(s: &EngineStatus) -> &'static str {
        match s.server.state {
            ServerState::Stopped => "stopped",
            ServerState::Starting => "starting",
            ServerState::Ready => "ready",
            ServerState::Degraded { .. } => "degraded",
            ServerState::Crashed { .. } => "crashed",
        }
    }

    #[test]
    fn changed_reports_only_differences() {
        let mut last = None;
        let a = status_with(ServerState::Ready);
        assert!(changed(&mut last, &a));
        assert!(!changed(&mut last, &a));
        let b = status_with(ServerState::Crashed {
            reason: "process exited (code 1)".into(),
        });
        assert!(changed(&mut last, &b));
    }

    #[tokio::test]
    async fn crash_is_emitted_once_and_the_watcher_stops_on_cancel() {
        let current = Arc::new(Mutex::new(status_with(ServerState::Ready)));
        let seen = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let cancel = CancellationToken::new();
        let src = current.clone();
        let out = seen.clone();
        let interval = Duration::from_millis(10);
        let task = tokio::spawn(watch(
            move || src.lock().unwrap().clone(),
            cancel.clone(),
            interval,
            move |s| out.lock().unwrap().push(label(s)),
        ));

        tokio::time::sleep(Duration::from_millis(40)).await;
        *current.lock().unwrap() = status_with(ServerState::Crashed {
            reason: "exited".into(),
        });
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*seen.lock().unwrap(), vec!["ready", "crashed"]);

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("watcher stops after cancel")
            .unwrap();
    }

    #[test]
    fn poll_interval_is_one_second() {
        assert_eq!(POLL_INTERVAL, Duration::from_secs(1));
    }
}
