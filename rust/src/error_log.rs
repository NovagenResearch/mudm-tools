//! WS-D error-log facility: a failures-only `ErrorCollector` that streams JSONL
//! records write-through to `<run_dir>/errors.jsonl` (so it survives the
//! OOM/ENOSPC crash it exists to capture) and emits an honest `run_summary.json`.
//!
//! Successes are tracked via an `AtomicU32` (no per-success Vec push → no
//! O(corpus) hot-path lock traffic). Failures are O(n_errors), streamed, and
//! also mirrored in-memory. When `run_dir` is `None`, the collector is
//! in-memory only and writes no files — so existing callers (clean inputs,
//! no run_dir) see no behavior change and no new file.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Failure severity. `Fatal` sets the collector's fatal flag so the caller can
/// collect-then-raise from the GIL-held frame; `NonFatal` is a record-and-skip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Fatal,
    NonFatal,
}

impl Severity {
    fn as_str(self) -> &'static str {
        match self {
            Severity::Fatal => "fatal",
            Severity::NonFatal => "non_fatal",
        }
    }
}

/// One streamed failure record. Serialized as a single JSONL line.
#[derive(Clone, Debug, Serialize)]
pub struct ErrorRecord {
    pub ts: String,
    pub phase: String,
    pub severity: String,
    pub item: String,
    pub error_kind: String,
    pub message: String,
}

/// Honest run summary, written to `<run_dir>/run_summary.json` by `finish_summary`.
#[derive(Serialize)]
struct RunSummary<'a> {
    phase: &'a str,
    ok: u32,
    fail: u32,
    fatal: bool,
    elapsed_s: f64,
}

/// Milliseconds since the UNIX epoch, as a `String`. `std::time` is fine in Rust
/// (the no-std/wasm clock concern does not apply to this native extension).
fn now_ts() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

/// Failures-only error collector with optional write-through JSONL streaming.
pub struct ErrorCollector {
    /// `Some` when a `run_dir` is given → streams + flushes each record to
    /// `<run_dir>/errors.jsonl`. `None` → in-memory only.
    writer: Option<Mutex<BufWriter<File>>>,
    /// In-memory mirror of every failure record. Bounded O(n_errors).
    records: Mutex<Vec<ErrorRecord>>,
    ok: AtomicU32,
    fail: AtomicU32,
    fatal: AtomicBool,
    run_dir: Option<PathBuf>,
}

impl ErrorCollector {
    /// `Some(run_dir)` → create/open `<run_dir>/errors.jsonl` (append + create)
    /// and stream write-through. `None` → in-memory only, no files written.
    pub fn new(run_dir: Option<PathBuf>) -> io::Result<Self> {
        let writer = match &run_dir {
            Some(dir) => {
                let path = dir.join("errors.jsonl");
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)?;
                Some(Mutex::new(BufWriter::new(file)))
            }
            None => None,
        };
        Ok(ErrorCollector {
            writer,
            records: Mutex::new(Vec::new()),
            ok: AtomicU32::new(0),
            fail: AtomicU32::new(0),
            fatal: AtomicBool::new(false),
            run_dir,
        })
    }

    /// Record one failure: `fail += 1`; set the fatal flag if `sev == Fatal`;
    /// when a JSONL file is present, write one serde_json line AND flush
    /// (write-through survives a crash); always also push to the in-memory mirror.
    pub fn record_failure(&self, phase: &str, sev: Severity, item: &str, kind: &str, msg: &str) {
        self.fail.fetch_add(1, Ordering::Relaxed);
        if sev == Severity::Fatal {
            self.fatal.store(true, Ordering::Relaxed);
        }

        let record = ErrorRecord {
            ts: now_ts(),
            phase: phase.to_string(),
            severity: sev.as_str().to_string(),
            item: item.to_string(),
            error_kind: kind.to_string(),
            message: msg.to_string(),
        };

        if let Some(writer) = &self.writer {
            if let Ok(mut w) = writer.lock() {
                // serde_json on a flat #[derive(Serialize)] struct of owned
                // Strings cannot fail; ignore the (impossible) error so the
                // logging path itself never panics.
                if let Ok(line) = serde_json::to_string(&record) {
                    let _ = writeln!(w, "{}", line);
                    let _ = w.flush();
                }
            }
        }

        if let Ok(mut recs) = self.records.lock() {
            recs.push(record);
        }
    }

    /// `ok += 1` (Relaxed) — cheap success counter, no lock.
    pub fn inc_ok(&self) {
        self.ok.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ok_count(&self) -> u32 {
        self.ok.load(Ordering::Relaxed)
    }

    pub fn fail_count(&self) -> u32 {
        self.fail.load(Ordering::Relaxed)
    }

    pub fn had_fatal(&self) -> bool {
        self.fatal.load(Ordering::Relaxed)
    }

    /// When a `run_dir` is present, write `<run_dir>/run_summary.json` with
    /// `{phase, ok, fail, fatal, elapsed_s}`. When `None`, this is a no-op.
    pub fn finish_summary(&self, phase: &str, elapsed_s: f64) -> io::Result<()> {
        // Flush any buffered JSONL first so the summary and the log agree.
        if let Some(writer) = &self.writer {
            if let Ok(mut w) = writer.lock() {
                w.flush()?;
            }
        }
        if let Some(dir) = &self.run_dir {
            let summary = RunSummary {
                phase,
                ok: self.ok_count(),
                fail: self.fail_count(),
                fatal: self.had_fatal(),
                elapsed_s,
            };
            let json = serde_json::to_string(&summary)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let path = dir.join("run_summary.json");
            let mut f = File::create(&path)?;
            f.write_all(json.as_bytes())?;
            f.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_collector_streams_and_counts() {
        let dir = std::env::temp_dir().join(format!(
            "mudm_errlog_test_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Pinned signature: ErrorCollector::new(Some(dir)) streams to <dir>/errors.jsonl
        let c = ErrorCollector::new(Some(dir.clone())).unwrap();
        c.record_failure("ingest", Severity::NonFatal, "frag_00007.mjf", "parse", "bad header");
        c.inc_ok();
        c.inc_ok();
        c.finish_summary("ingest", 1.0).unwrap(); // writes run_summary.json

        let log = std::fs::read_to_string(dir.join("errors.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 1);
        assert!(log.contains("frag_00007.mjf"));
        assert!(log.contains("\"phase\":\"ingest\""));

        assert_eq!(c.ok_count(), 2);
        assert_eq!(c.fail_count(), 1);

        // run_summary.json must be written by finish_summary
        let summary = std::fs::read_to_string(dir.join("run_summary.json")).unwrap();
        assert!(summary.contains("\"phase\":\"ingest\""));
        assert!(summary.contains("\"ok\":2"));
        assert!(summary.contains("\"fail\":1"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
