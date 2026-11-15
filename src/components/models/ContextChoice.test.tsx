import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ContextOption } from "../../api";

const mocks = vi.hoisted(() => ({ options: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: { ...actual.api, models: { ...actual.api.models, contextOptions: mocks.options } },
  };
});

import { ContextChoice, windowText } from "./ContextChoice";

function option(id: ContextOption["id"], ctx: number, kv: ContextOption["kv_cache"], over: Partial<ContextOption> = {}): ContextOption {
  return {
    id,
    label: id === "default" ? "Default" : id === "large" ? "Large (8-bit cache)" : "Maximum (4-bit cache)",
    ctx,
    kv_cache: kv,
    slots: id === "maximum" ? 1 : 2,
    speculation: id !== "maximum",
    projected_mib: id === "default" ? null : 14000,
    budget_mib: 14675,
    capped_by_model: false,
    note: `note for ${id}`,
    ...over,
  };
}

const ALL = [option("default", 8192, "f16"), option("large", 112640, "q8_0"), option("maximum", 190464, "q4_0")];

describe("windowText", () => {
  it("shows the size and says when it is the model's own maximum", () => {
    expect(windowText(ALL[1])).toBe("112,640 tokens");
    expect(windowText(option("maximum", 262144, "q4_0", { capped_by_model: true }))).toBe("262,144 tokens (model maximum)");
  });
});

describe("ContextChoice", () => {
  beforeEach(() => mocks.options.mockReset());

  it("reads the sizes only when asked, then offers them", async () => {
    mocks.options.mockResolvedValue(ALL);
    const onChange = vi.fn();
    render(<ContextChoice modelId="local/a" disabled={false} value={null} onChange={onChange} />);
    expect(mocks.options).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: /Context window/ }));
    await waitFor(() => expect(mocks.options).toHaveBeenCalledWith("local/a"));

    const select = await screen.findByRole("combobox");
    await userEvent.selectOptions(select, "maximum");
    expect(onChange).toHaveBeenLastCalledWith(ALL[2]);
    await userEvent.selectOptions(select, "default");
    expect(onChange).toHaveBeenLastCalledWith(null);
  });

  it("explains the tradeoff of the selected window", async () => {
    mocks.options.mockResolvedValue(ALL);
    render(<ContextChoice modelId="local/a" disabled={false} value={ALL[1]} onChange={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: /Context window/ }));
    expect(await screen.findByText("note for large")).toBeInTheDocument();
  });
});
