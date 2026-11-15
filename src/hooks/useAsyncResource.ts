import { useCallback, useEffect, useState } from "react";

export interface AsyncResource<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
  /** Replaces the data after a mutation returned the new state. */
  setData: (data: T) => void;
}

/** Loads data on mount and whenever `deps` change. Stale responses are ignored. */
export function useAsyncResource<T>(load: () => Promise<T>, deps: readonly unknown[]): AsyncResource<T> {
  const [state, setState] = useState<{ data: T | null; error: unknown; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let current = true;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    load().then(
      (data) => {
        if (current) setState({ data, error: null, loading: false });
      },
      (error: unknown) => {
        if (current) setState({ data: null, error, loading: false });
      },
    );
    return () => {
      current = false;
    };
    // The caller owns the dependency list; `load` is typically an inline closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);
  const setData = useCallback((data: T) => setState({ data, error: null, loading: false }), []);

  return { ...state, reload, setData };
}
