import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, conversationsSearch, RaptorApiError, type Conversation } from "../../api";
import { chatStore } from "../../state/chat";
import { ConversationList, SEARCH_LIMIT } from "./ConversationList";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    conversationsSearch: vi.fn(),
    api: { conversations: { get: vi.fn() } },
  };
});

const convo: Conversation = {
  version: 1,
  id: "c2",
  title: "Parser notes",
  created_at: 0,
  updated_at: 0,
  messages: [],
};

const EMPTY_SEARCH = { hits: [], conversations_scanned: 0, truncated: false, problems: [] };

describe("ConversationList content search", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.conversations.get).mockResolvedValue(convo);
    chatStore.set({
      list: [
        { id: "c1", title: "Refactor", created_at: 0, updated_at: 2, message_count: 3 },
        { id: "c2", title: "Parser notes", created_at: 0, updated_at: 1, message_count: 2 },
      ],
      listProblems: [],
      listError: null,
      activeId: null,
      active: null,
      activeError: null,
      pending: null,
    });
  });

  it("searches message content after a pause and marks the matching words in snippets", async () => {
    vi.mocked(conversationsSearch).mockResolvedValue({
      ...EMPTY_SEARCH,
      hits: [
        {
          conversation_id: "c2",
          title: "Parser notes",
          message_id: "m1",
          role: "assistant",
          snippet: "the tokenizer splits on whitespace",
          updated_at: 1,
        },
      ],
      conversations_scanned: 2,
    });
    const user = userEvent.setup();
    render(<ConversationList />);

    await user.type(screen.getByRole("searchbox", { name: /Search conversation titles and messages/ }), "tokenizer");

    const hits = await screen.findByRole("list", { name: "Message matches" });
    expect(within(hits).getByText("tokenizer").tagName).toBe("MARK");
    expect(screen.getByText("1 match in messages.")).toBeInTheDocument();
    expect(conversationsSearch).toHaveBeenCalledWith("tokenizer", SEARCH_LIMIT);
  });

  it("waits for typing to pause, so one search runs for a burst of keystrokes", async () => {
    vi.mocked(conversationsSearch).mockResolvedValue(EMPTY_SEARCH);
    const user = userEvent.setup();
    render(<ConversationList />);

    await user.type(screen.getByRole("searchbox"), "abc");

    await screen.findByText('No messages contain "abc".');
    expect(conversationsSearch).toHaveBeenCalledTimes(1);
    expect(conversationsSearch).toHaveBeenCalledWith("abc", SEARCH_LIMIT);
  });

  it("opens the conversation of a hit", async () => {
    vi.mocked(conversationsSearch).mockResolvedValue({
      ...EMPTY_SEARCH,
      hits: [
        { conversation_id: "c2", title: "Parser notes", message_id: "m1", role: "user", snippet: "parse this", updated_at: 1 },
      ],
    });
    const user = userEvent.setup();
    render(<ConversationList />);
    await user.type(screen.getByRole("searchbox"), "parse");

    await user.click(await screen.findByRole("button", { name: /^Parser notes/ }));

    await waitFor(() => expect(api.conversations.get).toHaveBeenCalledWith("c2"));
  });

  it("says plainly when no message matches", async () => {
    vi.mocked(conversationsSearch).mockResolvedValue({ ...EMPTY_SEARCH, truncated: false });
    const user = userEvent.setup();
    render(<ConversationList />);
    await user.type(screen.getByRole("searchbox"), "zzz");
    expect(await screen.findByText('No messages contain "zzz".')).toBeInTheDocument();
  });

  it("reports a failed search as an error rather than as no matches", async () => {
    vi.mocked(conversationsSearch).mockRejectedValue(new RaptorApiError({ kind: "Io", message: "disk unreadable" }));
    const user = userEvent.setup();
    render(<ConversationList />);
    await user.type(screen.getByRole("searchbox"), "x");

    expect(await screen.findByText("Message search failed")).toBeInTheDocument();
    expect(screen.getByText("disk unreadable")).toBeInTheDocument();
  });

  it("filters the titles at once, without waiting for the message search", async () => {
    vi.mocked(conversationsSearch).mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();
    render(<ConversationList />);
    await user.type(screen.getByRole("searchbox"), "parser");
    expect(screen.getByRole("button", { name: /^Parser notes/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Refactor/ })).toBeNull();
  });
});
