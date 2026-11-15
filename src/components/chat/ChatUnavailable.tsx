import { useStore } from "../../hooks/useStore";
import { engineStore } from "../../state/engine";
import { uiStore } from "../../state/ui";
import { Notice } from "../Notice";
import { QuickLoadModels } from "./QuickLoadModels";

/**
 * Why chat cannot run, with the installed models to load right here when the server is simply not
 * running. While it starts, or when its status cannot be read, only the reason is shown.
 */
export function ChatUnavailable({ reason }: { reason: string }) {
  const state = useStore(engineStore, (s) => s.status?.server.state.state ?? null);
  const busy = useStore(engineStore, (s) => s.busy);
  const canLoadHere = state === "stopped" || state === "crashed" || busy === "loading";
  return (
    <Notice
      tone="info"
      title={canLoadHere ? "Load a model to start chatting" : "Chat is not ready"}
      actions={
        <button type="button" className="btn btn-sm" onClick={() => uiStore.set({ page: "models" })}>
          Manage models
        </button>
      }
    >
      <p className="small muted">{reason}</p>
      {canLoadHere && <QuickLoadModels />}
    </Notice>
  );
}
