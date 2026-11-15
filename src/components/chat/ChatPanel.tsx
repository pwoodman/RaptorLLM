import { useRef, useState } from "react";
import { Download, SlidersHorizontal } from "lucide-react";
import { conversationsExportTo, dialogSaveFile } from "../../api";
import { useStore } from "../../hooks/useStore";
import { exportFileName } from "../../lib/conversationFile";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { chatStore, paramsFor } from "../../state/chat";
import { engineStore, inferenceBlockReason } from "../../state/engine";
import { uiStore } from "../../state/ui";
import { LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { ConversationFind } from "./ConversationFind";
import { Composer } from "./Composer";
import { ContextMeter } from "./ContextMeter";
import { MessageList } from "./MessageList";
import { ChatUnavailable } from "./ChatUnavailable";
import { ModelPicker } from "./ModelPicker";
import { ParamsPanel } from "./ParamsPanel";

/** The open conversation: header, search, server state, messages and composer. */
export function ChatPanel() {
  const activeId = useStore(chatStore, (s) => s.activeId);
  const active = useStore(chatStore, (s) => s.active);
  const activeError = useStore(chatStore, (s) => s.activeError);
  const systemPrompt = useStore(chatStore, (s) => (s.activeId ? paramsFor(s.activeId, s).systemPrompt : ""));
  const status = useStore(engineStore, (s) => s.status);
  const statusError = useStore(engineStore, (s) => s.statusError);
  const [showParams, setShowParams] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);

  const blocked = inferenceBlockReason(status, statusError);
  const degraded = status?.server.state.state === "degraded" ? status.server.state.reason : null;

  if (activeId === null) return null;
  if (active === null) {
    return activeError !== null ? (
      <Notice tone="danger" title="Could not open conversation">
        {activeError}
      </Notice>
    ) : (
      <LoadingState label="Opening conversation..." />
    );
  }

  const promptTexts = [systemPrompt, ...active.messages.map((m) => m.content)].filter((t) => t !== "");

  /** Asks where to save with the native dialog, then writes the file there. Nothing is written before that. */
  async function exportConversation() {
    if (active === null) return;
    setExporting(true);
    setExportError(null);
    setExportNotice(null);
    try {
      const path = await dialogSaveFile(exportFileName(active.title, active.id), "json");
      if (path === null) {
        setExportNotice("Export cancelled. Nothing was saved.");
        return;
      }
      const written = await conversationsExportTo(active.id, path);
      setExportNotice(`Saved ${formatBytes(written.bytes)} to ${written.path}`);
    } catch (e) {
      setExportError(messageOf(e));
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="chat-panel">
      <header className="chat-header">
        <h2 className="chat-title">{active.title}</h2>
        <div className="chat-tools">
          <ModelPicker />
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={exporting}
            onClick={() => void exportConversation()}
          >
            <Download size={14} aria-hidden="true" /> {exporting ? "Exporting..." : "Export"}
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-expanded={showParams}
            aria-controls="params-panel"
            onClick={() => setShowParams((open) => !open)}
          >
            <SlidersHorizontal size={14} aria-hidden="true" /> Settings
          </button>
        </div>
      </header>

      {exportError !== null && (
        <Notice tone="danger" title="Export failed">
          {exportError}
        </Notice>
      )}
      {exportNotice !== null && (
        <Notice tone="success" title="Export">
          {exportNotice}
        </Notice>
      )}

      <ContextMeter texts={promptTexts} />
      {showParams && <ParamsPanel conversationId={active.id} />}
      <ConversationFind rootRef={messagesRef} />

      {blocked !== null && <ChatUnavailable reason={blocked} />}
      {degraded !== null && (
        <Notice tone="warn" title="Inference server degraded">
          {degraded}
        </Notice>
      )}
      {activeError !== null && (
        <Notice tone="danger" title="Conversation error">
          {activeError}
        </Notice>
      )}

      <MessageList containerRef={messagesRef} />
      <Composer blockedReason={blocked} />
    </div>
  );
}
