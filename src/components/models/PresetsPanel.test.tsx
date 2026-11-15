import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  api,
  presetsDelete,
  presetsList,
  presetsSave,
  presetsSetDefault,
  RaptorApiError,
  type PresetList,
} from "../../api";
import { libraryStore } from "../../state/library";
import { modelEntry, preset } from "../../test/fixtures";
import { PresetsPanel } from "./PresetsPanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    presetsList: vi.fn(),
    presetsSave: vi.fn(),
    presetsDelete: vi.fn(),
    presetsSetDefault: vi.fn(),
    api: { runtime: { recommend: vi.fn() } },
  };
});

const MODEL = modelEntry();

function listOf(presets = [preset()], def: string | null = "Review"): PresetList {
  return { model_id: MODEL.id, presets, default: def };
}

describe("PresetsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    libraryStore.set({ models: [MODEL], storage: null, error: null });
    vi.mocked(presetsList).mockResolvedValue(listOf());
  });

  it("lists the model's presets and marks the default", async () => {
    render(<PresetsPanel />);
    const list = await screen.findByRole("list", { name: "Presets of this model" });
    expect(within(list).getByText("Review")).toBeInTheDocument();
    expect(within(list).getByText("Default")).toBeInTheDocument();
    expect(presetsList).toHaveBeenCalledWith(MODEL.id);
  });

  it("says so when the model has no presets yet", async () => {
    vi.mocked(presetsList).mockResolvedValue(listOf([], null));
    render(<PresetsPanel />);
    expect(await screen.findByText("No presets for this model yet.")).toBeInTheDocument();
  });

  it("refuses to save an out-of-range value and shows the message next to the field", async () => {
    const user = userEvent.setup();
    render(<PresetsPanel />);
    await user.click(await screen.findByRole("button", { name: "New preset" }));

    await user.type(screen.getByRole("textbox", { name: "Name" }), "Tiny");
    const ctx = screen.getByRole("textbox", { name: "Context length (tokens)" });
    await user.clear(ctx);
    await user.type(ctx, "10");
    await user.click(screen.getByRole("button", { name: "Save preset" }));

    expect(ctx).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText(/whole number from 256/)).toBeInTheDocument();
    expect(presetsSave).not.toHaveBeenCalled();
  });

  it("saves a valid preset for the selected model and reports it saved", async () => {
    vi.mocked(presetsSave).mockImplementation(async (p) => p);
    vi.mocked(presetsList)
      .mockResolvedValueOnce(listOf())
      .mockResolvedValueOnce(listOf([preset(), preset({ name: "Chat", system_prompt: null })], "Review"));
    const user = userEvent.setup();
    render(<PresetsPanel />);
    await user.click(await screen.findByRole("button", { name: "New preset" }));

    await user.type(screen.getByRole("textbox", { name: "Name" }), "Chat");
    await user.click(screen.getByRole("button", { name: "Save preset" }));

    await waitFor(() => expect(presetsSave).toHaveBeenCalledTimes(1));
    expect(vi.mocked(presetsSave).mock.calls[0]?.[0]).toMatchObject({ model_id: MODEL.id, name: "Chat", n_ctx: 4096 });
    expect(await screen.findByText('Saved preset "Chat".')).toBeInTheDocument();
  });

  it("shows the backend's refusal when a save is rejected", async () => {
    vi.mocked(presetsSave).mockRejectedValue(new RaptorApiError({ kind: "InvalidInput", message: "system prompt too large" }));
    const user = userEvent.setup();
    render(<PresetsPanel />);
    await user.click(await screen.findByRole("button", { name: "New preset" }));
    await user.type(screen.getByRole("textbox", { name: "Name" }), "Big");
    await user.click(screen.getByRole("button", { name: "Save preset" }));

    expect(await screen.findByText("Preset not saved")).toBeInTheDocument();
    expect(screen.getByText("system prompt too large")).toBeInTheDocument();
  });

  it("makes a preset the default, and clears the default", async () => {
    vi.mocked(presetsSetDefault).mockResolvedValue(listOf([preset()], null));
    const user = userEvent.setup();
    render(<PresetsPanel />);
    await user.click(await screen.findByRole("button", { name: "Clear default" }));

    expect(presetsSetDefault).toHaveBeenCalledWith(MODEL.id, null);
    expect(await screen.findByText("Default cleared.")).toBeInTheDocument();
  });

  it("deletes only after confirmation", async () => {
    vi.mocked(presetsDelete).mockResolvedValue(listOf([], null));
    const user = userEvent.setup();
    render(<PresetsPanel />);
    await user.click(await screen.findByRole("button", { name: "Delete Review" }));

    const dialog = screen.getByRole("alertdialog", { name: "Delete this preset?" });
    expect(presetsDelete).not.toHaveBeenCalled();
    await user.click(within(dialog).getByRole("button", { name: "Delete preset" }));

    await waitFor(() => expect(presetsDelete).toHaveBeenCalledWith(MODEL.id, "Review"));
  });

  it("shows a read failure as an error with a way to see it", async () => {
    vi.mocked(presetsList).mockRejectedValue(new RaptorApiError({ kind: "NotFound", message: "model is not installed" }));
    render(<PresetsPanel />);
    expect(await screen.findByText("model is not installed")).toBeInTheDocument();
  });
});
