import { useRef, useState } from "react";
import { Download, SlidersHorizontal } from "lucide-react";
import { conversationsExportTo, dialogSaveFile } from "../../api";
import { useStore } from "../../hooks/useStore";
import { exportFileName } from "../../lib/conversationFile";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { chatStore, paramsFor, selectConversation } from "../../state/chat";
import { engineStore, inferenceBlockReason } from "../../state/engine";
import { LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { LoadFailureCard } from "../models/LoadFailureCard";
import { SetupGuide } from "../setup/SetupGuide";
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
  const [modelError, setModelError] = useState<unknown>(null);
  const messagesRef = useRef<HTMLDivElement>(null);

  const blocked = inferenceBlockReason(status, statusError);
  const degraded = status?.server.state.state === "degraded" ? status.server.state.reason : null;

  if (activeId === null) return null;
  if (active === null) {
    return activeError !== null ? (
      <Notice tone="danger" title="Could not open conversation" actions={
        <button type="button" className="btn btn-sm" onClick={() => void selectConversation(activeId)}>Try again</button>
      }>
        {activeError}
      </Notice>
    ) : (
      <LoadingState label="Opening conversation..." />
    );
  }

  const promptTexts = [systemPrompt, ...active.messages.map((m) => m.content)];

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
        <h1 className="chat-title" title={active.title}>{active.title}</h1>
        <div className="chat-tools">
          {blocked === null && <ModelPicker onError={setModelError} />}
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
            <SlidersHorizontal size={14} aria-hidden="true" /> Chat settings
          </button>
        </div>
      </header>

      {modelError !== null && <LoadFailureCard error={modelError} />}
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

      {showParams ? (
        <div className="chat-settings">
          <ContextMeter texts={promptTexts} />
          <ParamsPanel conversationId={active.id} />
        </div>
      ) : (
        <ContextMeter texts={promptTexts} onlyNearLimit />
      )}
      <ConversationFind rootRef={messagesRef} />

      {blocked !== null && <SetupGuide chatFallback={<ChatUnavailable reason={blocked} />} />}
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

      <MessageList containerRef={messagesRef} blocked={blocked !== null} />
      <Composer conversationId={active.id} blockedReason={blocked} />
    </div>
  );
}
