import { useState } from "react";
import { ArrowDownToLine, RefreshCw } from "lucide-react";
import { check } from "@tauri-apps/plugin-updater";
import { Notice } from "../Notice";
import pkg from "../../../package.json";

type UpdateEvent =
  | { event: "Started"; data: { contentLength?: number } }
  | { event: "Progress"; data: { chunkLength: number } }
  | { event: "Finished" };

export function AppUpdates() {
  const [checking, setChecking] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [available, setAvailable] = useState<Awaited<ReturnType<typeof check>>>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<number | null>(null);

  async function checkForUpdates() {
    setChecking(true);
    setError("");
    setMessage("");
    setProgress(null);
    if (available) await available.close().catch(() => undefined);
    setAvailable(null);
    try {
      const update = await check();
      setAvailable(update);
      setMessage(update ? `Version ${update.version} is available.` : "You're up to date.");
    } catch (cause) {
      setError(`Update check failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    } finally {
      setChecking(false);
    }
  }

  async function installUpdate() {
    if (!available) return;
    setInstalling(true);
    setError("");
    setMessage("Downloading update…");
    setProgress(null);
    let received = 0;
    let totalBytes = 0;
    try {
      await available.downloadAndInstall((event: UpdateEvent) => {
        if (event.event === "Started" && event.data?.contentLength) {
          received = 0;
          totalBytes = event.data.contentLength;
          setProgress(0);
        } else if (event.event === "Progress" && event.data?.chunkLength) {
          received += event.data.chunkLength;
          if (totalBytes) setProgress(Math.min(100, Math.round((received / totalBytes) * 100)));
        } else if (event.event === "Finished") {
          setMessage("Download complete. Starting the installer…");
          setProgress(100);
        }
      });
    } catch (cause) {
      setError(`Update installation failed: ${cause instanceof Error ? cause.message : String(cause)}`);
      setInstalling(false);
    }
  }

  return (
    <section className="app-updates" aria-label="Application updates">
      <p>Version {pkg.version}. Updates are checked against signed GitHub Releases.</p>
      {message && <Notice tone="success">{message}</Notice>}
      {error && <Notice tone="danger">{error}</Notice>}
      {installing && <progress aria-label="Update download progress" max={100} value={progress ?? undefined} />}
      {available?.body && <p className="muted small">{available.body}</p>}
      <div className="button-row">
        <button className="btn" onClick={checkForUpdates} disabled={checking || installing}>
          <RefreshCw size={15} aria-hidden="true" /> {checking ? "Checking…" : "Check for updates"}
        </button>
        {available && (
          <button className="btn btn-primary" onClick={installUpdate} disabled={installing}>
            <ArrowDownToLine size={15} aria-hidden="true" /> {installing ? "Installing…" : `Install ${available.version}`}
          </button>
        )}
      </div>
    </section>
  );
}
