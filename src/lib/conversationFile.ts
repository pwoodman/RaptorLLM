/** Helpers for exporting and importing a conversation as a JSON file. */

export const MAX_IMPORT_BYTES = 20 * 1024 * 1024;

/** A filesystem-safe file name derived from the conversation title. */
export function exportFileName(title: string, id: string): string {
  const slug = title
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-")
    .slice(0, 60)
    .replace(/^-+|-+$/g, "");
  return `${slug || "conversation"}-${id.slice(0, 8)}.json`;
}

/** Saves text through a browser download. The file goes to the system download location. */
export function saveTextFile(fileName: string, text: string): void {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
