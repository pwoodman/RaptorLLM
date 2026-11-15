import { useEffect, useRef } from "react";
import { chatStore, createConversation, isGenerating, stopGeneration } from "../state/chat";
import { messageOf } from "../lib/errors";
import { uiStore } from "../state/ui";

export type ShortcutAction = "newChat" | "focusSearch" | "showHelp" | "stop";

export interface ShortcutInfo {
  keys: string;
  action: string;
}

/** Every shortcut the app handles. The help dialog renders this list, so it stays the single source. */
export const SHORTCUTS: ShortcutInfo[] = [
  { keys: "Ctrl+N", action: "Start a new chat" },
  { keys: "Ctrl+K", action: "Search: finds inside the open conversation, or conversation titles when none is open" },
  { keys: "Enter", action: "Send the message (Ctrl+Enter also sends)" },
  { keys: "Shift+Enter", action: "New line in the message" },
  { keys: "Enter / Shift+Enter", action: "In the conversation search: next or previous match" },
  { keys: "Esc", action: "Stop the reply while the model is writing it" },
  { keys: "Ctrl+/", action: "Show this list of shortcuts" },
];

/** Maps a key press to an app action. Pure, so the mapping can be tested without a window. */
export function shortcutAction(event: Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey" | "shiftKey" | "altKey">): ShortcutAction | null {
  if (event.key === "Escape" && !event.ctrlKey && !event.metaKey && !event.altKey) return "stop";
  const mod = event.ctrlKey || event.metaKey;
  if (!mod || event.altKey) return null;
  const key = event.key.toLowerCase();
  if (key === "n" && !event.shiftKey) return "newChat";
  if (key === "k" && !event.shiftKey) return "focusSearch";
  if ((key === "/" || key === "?") && !event.shiftKey) return "showHelp";
  return null;
}

/** The chat search input shown when a conversation is open, or the title search otherwise. */
export const FIND_INPUT_ID = "chat-find-input";
export const TITLE_SEARCH_ID = "conv-search-input";

function focusById(id: string): boolean {
  const el = document.getElementById(id);
  if (!el) return false;
  el.focus();
  return true;
}

export interface GlobalShortcutHandlers {
  onShowHelp: () => void;
}

/** Registers the window-level shortcuts. Ignores key presses already handled by a control. */
export function useGlobalShortcuts({ onShowHelp }: GlobalShortcutHandlers): void {
  const helpRef = useRef(onShowHelp);
  helpRef.current = onShowHelp;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented) return;
      const action = shortcutAction(event);
      if (action === null) return;

      if (action === "stop") {
        if (!isGenerating(chatStore.get().pending)) return;
        event.preventDefault();
        void stopGeneration().catch((e: unknown) => uiStore.set({ flash: messageOf(e) }));
        return;
      }

      event.preventDefault();
      if (action === "showHelp") {
        helpRef.current();
      } else if (action === "newChat") {
        uiStore.set({ page: "chats" });
        createConversation().catch((e: unknown) => uiStore.set({ flash: `Could not start a chat: ${messageOf(e)}` }));
      } else if (action === "focusSearch") {
        const openConversation = chatStore.get().activeId !== null;
        uiStore.set({ page: "chats" });
        const id = openConversation ? FIND_INPUT_ID : TITLE_SEARCH_ID;
        // The chat page may not be mounted yet; focus once it renders.
        window.setTimeout(() => focusById(id), 0);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
