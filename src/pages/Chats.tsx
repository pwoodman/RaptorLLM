import { useEffect, useState } from "react";
import { MessageSquarePlus } from "lucide-react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { ConversationList } from "../components/chat/ConversationList";
import { ErrorState } from "../components/AsyncState";
import { ChatUnavailable } from "../components/chat/ChatUnavailable";
import { SetupGuide } from "../components/setup/SetupGuide";
import { useStore } from "../hooks/useStore";
import { chatStore, createConversation, refreshList } from "../state/chat";
import { engineStore, inferenceBlockReason } from "../state/engine";
import { refreshLibrary } from "../state/library";

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
      {blocked !== null ? (
        <SetupGuide chatFallback={<ChatUnavailable reason={blocked} />} />
      ) : (
        <div className="welcome-content">
          <MessageSquarePlus size={28} aria-hidden="true" />
          <h1>Start a local chat</h1>
          <p>Open a conversation or start a new one. Chats are saved on this computer.</p>
          <button type="button" className="btn btn-primary" onClick={() => void create()}>
            New chat
          </button>
        </div>
      )}
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
        {activeId === null ? <ChatWelcome /> : <ChatPanel key={activeId} />}
      </section>
    </div>
  );
}
