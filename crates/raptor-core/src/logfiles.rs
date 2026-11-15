//! Daily rolling log files that keep the last N days, for the app log and the llama-server log.
//! Files are named `<prefix>-YYYY-MM-DD.log`; the date is UTC so rollover does not depend on a
//! time-zone database. Opening and every rollover delete the files that fall outside the window,
//! so the folder never grows without bound and a bug report only needs the folder.

use crate::{RaptorError, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Days of logs kept, today included.
pub const KEEP_DAYS: u32 = 7;

/// `YYYY-MM-DD` (UTC) for a Unix time in seconds.
pub fn date_stamp(unix_secs: u64) -> String {
    let (y, m, d) = civil_from_days((unix_secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since 1970-01-01 to (year, month, day). Howard Hinnant's civil-from-days algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_of(stamp: &str) -> Option<i64> {
    let mut parts = stamp.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // days-from-civil, the inverse of `civil_from_days`.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Deletes `<prefix>-<date>.log` files in `dir` that are older than `keep_days` days before
/// `today` (a `YYYY-MM-DD` stamp). Files with other names are never touched. Returns how many
/// were removed.
pub fn prune_old_logs(dir: &Path, prefix: &str, keep_days: u32, today: &str) -> usize {
    let Some(today_days) = days_of(today) else {
        return 0;
    };
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let head = format!("{prefix}-");
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stamp) = name
            .strip_prefix(&head)
            .and_then(|r| r.strip_suffix(".log"))
        else {
            continue;
        };
        let Some(file_days) = days_of(stamp) else {
            continue;
        };
        if today_days - file_days >= i64::from(keep_days) && fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

struct Open {
    stamp: String,
    file: File,
}

/// An append-only log that starts a new file each day and prunes old ones.
pub struct RollingLog {
    dir: PathBuf,
    prefix: String,
    keep_days: u32,
    open: Mutex<Option<Open>>,
}

impl RollingLog {
    /// Creates `dir` if needed and prunes files outside the window. The file for today is
    /// opened on the first write.
    pub fn new(dir: &Path, prefix: &str, keep_days: u32) -> Result<Self> {
        fs::create_dir_all(dir)?;
        prune_old_logs(dir, prefix, keep_days, &date_stamp(now_secs()));
        Ok(Self {
            dir: dir.to_path_buf(),
            prefix: prefix.to_string(),
            keep_days,
            open: Mutex::new(None),
        })
    }

    /// The file a write at `unix_secs` goes to.
    pub fn path_for(&self, unix_secs: u64) -> PathBuf {
        self.dir
            .join(format!("{}-{}.log", self.prefix, date_stamp(unix_secs)))
    }

    /// Appends `bytes` to today's file, rolling over (and pruning) when the date changed.
    pub fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.write_bytes_at(bytes, now_secs())
    }

    fn write_bytes_at(&self, bytes: &[u8], unix_secs: u64) -> Result<()> {
        let stamp = date_stamp(unix_secs);
        let mut guard = self
            .open
            .lock()
            .map_err(|_| RaptorError::Internal("log file lock poisoned".into()))?;
        if guard.as_ref().map(|o| o.stamp.as_str()) != Some(stamp.as_str()) {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.path_for(unix_secs))?;
            prune_old_logs(&self.dir, &self.prefix, self.keep_days, &stamp);
            *guard = Some(Open { stamp, file });
        }
        if let Some(open) = guard.as_mut() {
            open.file.write_all(bytes)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_utc_civil_dates() {
        assert_eq!(date_stamp(0), "1970-01-01");
        assert_eq!(date_stamp(951_782_400), "2000-02-29");
        assert_eq!(date_stamp(1_794_570_279), "2026-11-13");
        assert_eq!(days_of("2026-11-13"), Some(1_794_570_279 / 86_400));
        assert_eq!(days_of("not-a-date"), None);
        assert_eq!(days_of("2026-13-01"), None);
    }

    #[test]
    fn only_files_outside_the_window_are_removed() {
        let dir = tempfile::tempdir().unwrap();
        for stamp in [
            "2026-11-13", // today
            "2026-11-07", // 6 days ago: kept
            "2026-11-06", // 7 days ago: removed
            "2026-10-01",
        ] {
            fs::write(dir.path().join(format!("raptor-{stamp}.log")), "x").unwrap();
        }
        fs::write(dir.path().join("raptor-notes.txt"), "keep").unwrap();
        fs::write(
            dir.path().join("llama-server-2020-01-01.log"),
            "other prefix",
        )
        .unwrap();
        fs::write(dir.path().join("raptor-garbage.log"), "not a date").unwrap();

        assert_eq!(prune_old_logs(dir.path(), "raptor", 7, "2026-11-13"), 2);
        let mut left: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                "llama-server-2020-01-01.log",
                "raptor-2026-11-07.log",
                "raptor-2026-11-13.log",
                "raptor-garbage.log",
                "raptor-notes.txt"
            ]
        );
    }

    #[test]
    fn writes_roll_to_a_new_file_each_day_and_prune_on_rollover() {
        let dir = tempfile::tempdir().unwrap();
        let log = RollingLog::new(dir.path(), "app", 3).unwrap();
        let day = 86_400u64;
        let t0 = 1_794_570_279u64;
        log.write_bytes_at(b"one\n", t0).unwrap();
        log.write_bytes_at(b"two\n", t0 + 10).unwrap();
        log.write_bytes_at(b"three\n", t0 + day).unwrap();
        log.write_bytes_at(b"four\n", t0 + 2 * day).unwrap();
        assert_eq!(fs::read_to_string(log.path_for(t0)).unwrap(), "one\ntwo\n");
        assert_eq!(
            fs::read_to_string(log.path_for(t0 + day)).unwrap(),
            "three\n"
        );
        // Day 4 pushes day 1 out of a 3-day window.
        log.write_bytes_at(b"five\n", t0 + 3 * day).unwrap();
        assert!(!log.path_for(t0).exists());
        assert!(log.path_for(t0 + day).exists());
    }
}
