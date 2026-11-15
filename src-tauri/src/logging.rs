//! The app log: `logs/raptor-YYYY-MM-DD.log`, one file a day, the last seven days kept. Panics
//! are written to it too, so a crash report is just the logs folder.

use raptor_core::logfiles::{RollingLog, KEEP_DAYS};
use raptor_core::DataDirs;
use std::io::{self, Write};
use std::sync::Arc;
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone)]
struct LogFiles(Arc<RollingLog>);

struct LogWriter(Arc<RollingLog>);

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .write_bytes(buf)
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogFiles {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter(self.0.clone())
    }
}

/// Starts the app log under `dirs.logs()`. A failure (read-only disk, say) is returned and the
/// app carries on without file logging; it never blocks startup.
pub fn init(dirs: &DataDirs) -> raptor_core::Result<()> {
    let files = Arc::new(RollingLog::new(&dirs.logs(), "raptor", KEEP_DAYS)?);
    let filter = std::env::var("RAPTOR_LOG").unwrap_or_else(|_| "info".into());
    let level = filter
        .parse::<tracing::Level>()
        .unwrap_or(tracing::Level::INFO);
    tracing_subscriber::fmt()
        .with_writer(LogFiles(files))
        .with_ansi(false)
        .with_max_level(level)
        .try_init()
        .map_err(|e| raptor_core::RaptorError::Internal(format!("logging: {e}")))?;
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("panic: {info}");
        previous(info);
    }));
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Raptor LLM started");
    Ok(())
}

/// The log files in `dir`, newest first, as `(name, size_bytes, modified_unix_secs)`.
pub fn list_log_files(dir: &std::path::Path) -> Vec<(String, u64, u64)> {
    let mut out: Vec<(String, u64, u64)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".log") {
                return None;
            }
            let meta = e.metadata().ok()?;
            let modified = meta
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some((name, meta.len(), modified))
        })
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2).then(b.0.cmp(&a.0)));
    out
}
