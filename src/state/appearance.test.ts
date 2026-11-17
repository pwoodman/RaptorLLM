import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { appearanceStore, readAppearance, saveAppearance, startAppearance } from "./appearance";

beforeEach(() => {
  localStorage.clear();
  appearanceStore.set({ theme: "system", glass: false });
});
afterEach(() => vi.restoreAllMocks());

describe("appearance persistence", () => {
  it("restores light/dark and glass choices after reopening", () => {
    saveAppearance({ theme: "light", glass: true });
    expect(readAppearance()).toEqual({ theme: "light", glass: true });
    saveAppearance({ theme: "dark" });
    expect(readAppearance()).toEqual({ theme: "dark", glass: true });
  });

  it("recovers invalid stored preferences", () => {
    localStorage.setItem("raptor.appearance", "broken");
    expect(readAppearance()).toEqual({ theme: "system", glass: false });
    localStorage.setItem("raptor.appearance", '{"theme":"unknown","glass":"yes"}');
    expect(readAppearance()).toEqual({ theme: "system", glass: false });
  });

  it("does not commit choices when storage fails", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("full"); });
    expect(() => saveAppearance({ theme: "light" })).toThrow();
    expect(appearanceStore.get().theme).toBe("system");
  });

  it("follows device changes only in system mode and cleans up listeners", () => {
    let listener = () => {};
    const media = { matches: true, addEventListener: vi.fn((_name, fn) => { listener = fn; }), removeEventListener: vi.fn() };
    vi.stubGlobal("matchMedia", () => media);
    const stop = startAppearance();
    expect(document.documentElement.dataset.theme).toBe("light");
    media.matches = false;
    listener();
    expect(document.documentElement.dataset.theme).toBe("dark");
    saveAppearance({ theme: "light", glass: true });
    listener();
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(document.documentElement.dataset.glass).toBe("true");
    stop();
    expect(media.removeEventListener).toHaveBeenCalledWith("change", listener);
    vi.unstubAllGlobals();
  });
});
