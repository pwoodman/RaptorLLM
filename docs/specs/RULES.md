# Engineering rules (every agent, every crate)

Product: Raptor LLM - local AI made easy, max hardware performance, best context for coding agents.
Do NOT build: accounts, cloud sync, billing, an IDE, a custom inference engine/model format, an agent framework.

Architecture: Cargo workspace. Subsystems are independent crates that depend ONLY on `raptor-core`
(never on each other). Wiring happens in `src-tauri` (desktop) and `raptor-gateway`. The GUI knows nothing
about ROCm or context scoring - it calls typed facades.

Rules
1. Touch ONLY your own crate (`crates/raptor-<name>/`). Need a change in raptor-core or elsewhere? Do not edit it; report it.
2. Public API: small, documented (`///`), typed, `Result<T, raptor_core::RaptorError>`. No `unwrap()/expect()/panic!` outside tests.
3. Each crate is independently testable: no network, no real GPU, no real llama-server in unit tests. Put external effects behind traits (fakes in tests). Use `tempfile` for FS.
4. Honesty: never report success / GPU acceleration / verification without positive evidence. Surface real errors. No silent fallbacks (esp. GPU->CPU).
5. Security: validate every path/filename from network/UI (`raptor_core::paths::safe_file_name`). Never log secrets. Verify hashes before trusting downloads.
6. Failure handling is a feature: timeouts, cancellation, partial-state cleanup, idempotent retries.
7. Style: rustfmt clean; `cargo clippy -p <crate> --all-targets -- -D warnings` clean; `cargo test -p <crate>` green. Comments explain *why*, sparingly.
8. Write `crates/raptor-<name>/README.md` with `## Public API` and `## Status` (what is implemented vs stubbed, honestly).
9. Add dependencies to your crate's Cargo.toml only; prefer workspace deps; keep deps minimal and well-known.
10. Cargo here: in Git Bash run `export PATH="$HOME/.cargo/bin:$PATH"` first. Windows 11. Other agents build in the same workspace concurrently; "Blocking waiting for file lock" is normal - wait. Always scope commands with `-p <your-crate>`.
11. When writing files containing backslashes or quotes, use the Write/Edit tools, not shell heredocs.
12. Final report (<250 words): what you built, test count, what is stubbed, interface decisions, open issues. Be truthful; the manager re-runs build/test/clippy and will check claims.
13. Before trying an optimisation or experiment read `docs/EXPERIMENTS.md`; after it, record the result there (WORKS / REJECTED / BROKEN / UNTESTED, with the measurement and where it was measured). Real-hardware results and fake-based tests are never mixed up.
