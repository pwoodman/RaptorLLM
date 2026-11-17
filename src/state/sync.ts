import { subscribe } from "../api";
import { applyFinished, applyProgress, refreshDownloads } from "./downloads";
import { refreshEngine, engineStore } from "./engine";
import { refreshLibrary } from "./library";
import { startLocalModelSync } from "./localModels";

/**
 * Connects the backend events to the stores and loads the initial snapshots.
 * Returns a cleanup function that removes every listener.
 */
export function startSync(): () => void {
  const subscriptions = [
    subscribe.serverState((status) =>
      engineStore.set((state) => ({
        status,
        statusError: null,
        // A stopped server no longer runs the model this session loaded.
        loaded: status.server.state.state === "stopped" ? null : state.loaded,
        loadReport: status.server.state.state === "stopped" ? null : state.loadReport,
      })),
    ),
    subscribe.gatewayState((gateway) => engineStore.set({ gateway })),
    subscribe.downloadProgress(applyProgress),
    subscribe.downloadFinished(applyFinished),
  ];

  void refreshEngine();
  void refreshLibrary();
  void refreshDownloads();
  const stopLocalModels = startLocalModelSync();

  return () => {
    stopLocalModels();
    void Promise.all(subscriptions).then((unlisteners) => unlisteners.forEach((unlisten) => unlisten()));
  };
}
