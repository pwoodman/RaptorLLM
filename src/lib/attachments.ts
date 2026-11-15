/**
 * Text and code files dropped into the composer. Files are read in the browser, capped in size,
 * and sent inside clearly delimited blocks. Images and other binary files are refused.
 */

export const MAX_ATTACHMENT_BYTES = 256 * 1024;
export const MAX_ATTACHMENTS = 8;
const SNIFF_BYTES = 8192;

const BINARY_EXTENSIONS = new Set([
  "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tif", "tiff", "heic", "svgz",
  "pdf", "zip", "gz", "tgz", "7z", "rar", "bz2", "xz", "exe", "dll", "so", "dylib", "bin", "o", "obj", "class",
  "wasm", "gguf", "safetensors", "mp3", "mp4", "wav", "mov", "avi", "mkv", "webm", "ogg", "flac",
  "ttf", "otf", "woff", "woff2", "sqlite", "db", "docx", "xlsx", "pptx", "odt", "ods",
]);

export interface Attachment {
  name: string;
  size: number;
  content: string;
}

export type AttachmentCheck = { ok: true } | { ok: false; reason: string };

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
}

/** Refuses files that are clearly not text before any bytes are read. */
export function checkAttachmentFile(file: { name: string; size: number; type: string }, alreadyAttached: number): AttachmentCheck {
  if (alreadyAttached >= MAX_ATTACHMENTS) {
    return { ok: false, reason: `Attach at most ${MAX_ATTACHMENTS} files per message.` };
  }
  if (/^(image|video|audio)\//.test(file.type) || file.type === "application/pdf") {
    return { ok: false, reason: `${file.name} is not a text file. Images and binary files cannot be attached.` };
  }
  if (BINARY_EXTENSIONS.has(extensionOf(file.name))) {
    return { ok: false, reason: `${file.name} is a binary file. Only text and code files can be attached.` };
  }
  if (file.size === 0) return { ok: false, reason: `${file.name} is empty.` };
  if (file.size > MAX_ATTACHMENT_BYTES) {
    return { ok: false, reason: `${file.name} is ${formatKb(file.size)}. The limit is ${formatKb(MAX_ATTACHMENT_BYTES)} per file.` };
  }
  return { ok: true };
}

/** Reads a checked file as UTF-8 text. A NUL byte in the first bytes means the file is binary. */
export async function readAttachment(file: File): Promise<{ ok: true; attachment: Attachment } | { ok: false; reason: string }> {
  const head = new Uint8Array(await file.slice(0, SNIFF_BYTES).arrayBuffer());
  if (head.includes(0)) {
    return { ok: false, reason: `${file.name} looks like a binary file. Only text and code files can be attached.` };
  }
  const content = await file.text();
  if (content.includes("\uFFFD")) {
    return { ok: false, reason: `${file.name} is not valid UTF-8 text. Only text and code files can be attached.` };
  }
  return { ok: true, attachment: { name: file.name, size: file.size, content } };
}

export const ATTACHMENT_BEGIN = "--- BEGIN ATTACHMENT";
export const ATTACHMENT_END = "--- END ATTACHMENT";

/** The message as stored and sent: the typed text, then one delimited block per attachment. */
export function composeMessage(text: string, attachments: readonly Attachment[]): string {
  const blocks = attachments.map((a) => {
    const name = a.name.replace(/[\r\n]+/g, " ");
    return `${ATTACHMENT_BEGIN}: ${name} (${a.size} bytes) ---\n${a.content}\n${ATTACHMENT_END}: ${name} ---`;
  });
  return [text, ...blocks].filter((part) => part !== "").join("\n\n");
}

export function formatKb(bytes: number): string {
  return `${Math.round(bytes / 1024)} KB`;
}
