import { useSyncExternalStore } from "react";
import type { Store } from "../state/store";

/** Subscribes to a slice of a store. Selectors must return primitives or stable references. */
export function useStore<T extends object, U>(store: Store<T>, select: (state: T) => U): U {
  return useSyncExternalStore(store.subscribe, () => select(store.get()));
}
