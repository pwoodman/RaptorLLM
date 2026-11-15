import { useEffect, useState } from "react";
import { MessageSquarePlus } from "lucide-react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { ConversationList } from "../components/chat/ConversationList";
import { EmptyState, ErrorState } from "../components/AsyncState";
import { ChatUnavailable } from "../components/chat/ChatUnavailable";
import { SetupGuide } from "../components/setup/SetupGuide";
import { useStore } from "../hooks/useStore";
import { chatStore, createConversation, refreshList } from "../state/chat";
import { engineStore, inferenceBlockReason } from "../state/engine";
import { refreshLibrary } from "../state/library";
import { uiStore } from "../state/ui";

function ChatWelcome() {
  const status = useStore(engineStore, (s) => s.status);
  const statusError = useStore(engineStore, (s) => s.statusError);
  const [error, setError] = useState<unknown>(null);
  const blocked = inferenceBlockReason(status, statusError);

  async function create() {
    setError(null);
    try {
      await createConversation();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <div className="chat-welcome">
      <SetupGuide />
      {blocked !== null && <ChatUnavailable reason={blocked} />}
      <EmptyState
        icon={MessageSquarePlus}
        title="Pick a conversation or start a new one"
        action={
          <button type="button" className="btn btn-primary" onClick={() => void create()}>
            New chat
          </button>
        }
      >
        Conversations are saved on this machine.
      </EmptyState>
      {error !== null && <ErrorState error={error} />}
    </div>
  );
}

export default function ChatsPage() {
  const activeId = useStore(chatStore, (s) => s.activeId);

  useEffect(() => {
    void refreshList();
    void refreshLibrary();
  }, []);

  return (
    <div className="chats">
      <ConversationList />
      <section className="chat-main" aria-label="Chat">
        {activeId === null ? <ChatWelcome /> : <ChatPanel />}
      </section>
    </div>
  );
}
