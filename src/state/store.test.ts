import { describe, expect, it, vi } from "vitest";
import { createStore } from "./store";

describe("createStore", () => {
  it("merges partial updates and function updates", () => {
    const store = createStore({ a: 1, b: "x" });
    store.set({ a: 2 });
    store.set((s) => ({ b: `${s.b}y` }));
    expect(store.get()).toEqual({ a: 2, b: "xy" });
  });

  it("notifies subscribers and stops after unsubscribe", () => {
    const store = createStore({ n: 0 });
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    store.set({ n: 1 });
    unsubscribe();
    store.set({ n: 2 });
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
