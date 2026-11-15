import { RaptorApiError, type RaptorErrorKind } from "../api";

export interface UiError {
  title: string;
  message: string;
}

const TITLES: Record<RaptorErrorKind, string> = {
  InvalidInput: "Invalid input",
  NotFound: "Not found",
  Conflict: "Action not possible right now",
  Io: "File system error",
  Network: "Network error",
  Integrity: "Integrity check failed",
  Runtime: "Inference runtime error",
  Unsupported: "Not supported",
  Unauthorized: "Not authorized",
  Internal: "Unexpected error",
};

/** Turns any thrown value into a title and message the UI can show as-is. */
export function describeError(e: unknown): UiError {
  if (e instanceof RaptorApiError) {
    return { title: TITLES[e.kind], message: e.message };
  }
  return { title: TITLES.Internal, message: e instanceof Error ? e.message : String(e) };
}

export function messageOf(e: unknown): string {
  return describeError(e).message;
}

export function isErrorKind(e: unknown, kind: RaptorErrorKind): boolean {
  return e instanceof RaptorApiError && e.kind === kind;
}
