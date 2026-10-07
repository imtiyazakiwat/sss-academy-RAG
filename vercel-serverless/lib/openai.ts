// Request normalisation (OpenAI "messages" body or the simple { text, system, image, history } body)
// into Gemini contents, plus OpenAI-shaped response builders.
import { HttpError } from "./http";

export type GeminiPart = { text: string } | { inlineData: { mimeType: string; data: string } };
export interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}

export interface NormalizedChat {
  system: string;
  contents: GeminiContent[];
  model?: string;
  stream: boolean;
  temperature?: number;
  maxTokens?: number;
}

const IMAGE_MIMES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
const BASE64_RE = /^[A-Za-z0-9+/=_-]+$/;

function bad(message: string, code = "invalid_request"): HttpError {
  return new HttpError(400, message, "invalid_request_error", code);
}

function sniffMime(b64: string): string {
  if (b64.startsWith("iVBOR")) return "image/png";
  if (b64.startsWith("/9j/")) return "image/jpeg";
  if (b64.startsWith("UklGR")) return "image/webp";
  if (b64.startsWith("R0lGO")) return "image/gif";
  return "image/png";
}

/** Returns null for empty placeholders (Pluely sends "data:image/png;base64," when there is no image). */
export function parseImage(raw: string): { mimeType: string; data: string } | null {
  const s = raw.trim();
  if (!s) return null;

  const m = /^data:([^;,]*)((?:;[^;,]*)*),([\s\S]*)$/i.exec(s);
  if (m) {
    const data = m[3].replace(/\s+/g, "");
    if (!data) return null;
    if (!/;base64/i.test(m[2])) throw bad("Image data URL must be base64 encoded.", "invalid_image");
    let mime = m[1].toLowerCase();
    if (mime === "image/jpg") mime = "image/jpeg";
    if (!IMAGE_MIMES.has(mime)) throw bad("Unsupported image type. Use JPEG, PNG, WebP or GIF.", "invalid_image");
    if (!BASE64_RE.test(data)) throw bad("Image is not valid base64.", "invalid_image");
    return { mimeType: mime, data };
  }

  if (/^https?:\/\//i.test(s)) {
    throw bad("Image URLs are not supported. Send the image as a base64 data URL.", "invalid_image");
  }

  const data = s.replace(/\s+/g, "");
  if (data.length > 100 && BASE64_RE.test(data)) return { mimeType: sniffMime(data), data };
  throw bad("Unsupported image value. Send a base64 data URL.", "invalid_image");
}

function roleOf(role: unknown): "system" | "user" | "model" | null {
  switch (role) {
    case "system":
    case "developer":
      return "system";
    case "user":
      return "user";
    case "assistant":
    case "model":
      return "model";
    default:
      return null;
  }
}

/** Convert OpenAI message content (string | parts[]) into text + Gemini parts. */
function contentToParts(content: unknown): { text: string; parts: GeminiPart[] } {
  if (typeof content === "string") {
    return { text: content, parts: content.trim() ? [{ text: content }] : [] };
  }
  const parts: GeminiPart[] = [];
  const texts: string[] = [];
  if (Array.isArray(content)) {
    for (const item of content) {
      if (!item || typeof item !== "object") continue;
      const p = item as Record<string, unknown>;
      if ((p.type === "text" || p.type === "input_text") && typeof p.text === "string") {
        texts.push(p.text);
        if (p.text.trim()) parts.push({ text: p.text });
      } else if (p.type === "image_url" || p.type === "input_image") {
        const ref = p.image_url ?? p.url;
        const url =
          typeof ref === "string"
            ? ref
            : ref && typeof ref === "object" && typeof (ref as { url?: unknown }).url === "string"
              ? (ref as { url: string }).url
              : "";
        const img = parseImage(url);
        if (img) parts.push({ inlineData: img });
      }
    }
  }
  return { text: texts.join("\n"), parts };
}

/** Body B (simple shape) -> OpenAI-style messages. */
function simpleToMessages(body: Record<string, unknown>): unknown[] {
  const text = [body.text, body.question, body.prompt].find((v) => typeof v === "string" && v.trim()) as
    | string
    | undefined;
  const image = typeof body.image === "string" ? body.image : "";
  const hasImage = !!image.trim() && parseImage(image) !== null;
  if (!text && !hasImage) throw bad("Question cannot be empty.");

  const messages: unknown[] = [];
  if (typeof body.system === "string" && body.system.trim()) {
    messages.push({ role: "system", content: body.system });
  }
  if (Array.isArray(body.history)) {
    for (const h of body.history) {
      if (!h || typeof h !== "object") continue;
      const item = h as Record<string, unknown>;
      if (typeof item.q === "string" || typeof item.a === "string") {
        if (typeof item.q === "string") messages.push({ role: "user", content: item.q });
        if (typeof item.a === "string") messages.push({ role: "assistant", content: item.a });
      } else {
        messages.push(item);
      }
    }
  }
  const content: unknown[] = [];
  if (text) content.push({ type: "text", text });
  if (hasImage) content.push({ type: "image_url", image_url: { url: image } });
  messages.push({ role: "user", content });
  return messages;
}

export function normalizeChat(body: unknown): NormalizedChat {
  if (!body || typeof body !== "object" || Array.isArray(body)) throw bad("Request body must be a JSON object.");
  const b = body as Record<string, unknown>;

  const messages = Array.isArray(b.messages) ? b.messages : simpleToMessages(b);

  const systemTexts: string[] = [];
  const contents: GeminiContent[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    const role = roleOf(m.role);
    if (!role) continue;
    const { text, parts } = contentToParts(m.content);
    if (role === "system") {
      if (text.trim()) systemTexts.push(text);
      continue;
    }
    if (!parts.length) continue; // empty turns are dropped
    const last = contents[contents.length - 1];
    if (last && last.role === role) last.parts.push(...parts); // Gemini needs strict alternation
    else contents.push({ role, parts: [...parts] });
  }

  if (!contents.length) throw bad("Question cannot be empty.");
  if (contents[contents.length - 1].role !== "user") throw bad("The last message must be a non-empty user message.");

  const maxRaw = b.max_tokens ?? b.max_completion_tokens;
  const maxTokens = typeof maxRaw === "number" && Number.isInteger(maxRaw) && maxRaw > 0 ? maxRaw : undefined;
  const temperature = typeof b.temperature === "number" && Number.isFinite(b.temperature) ? b.temperature : undefined;

  return {
    system: systemTexts.join("\n\n"),
    contents,
    model: typeof b.model === "string" ? b.model : undefined,
    stream: b.stream === true,
    temperature,
    maxTokens,
  };
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export function chatCompletion(opts: { id: string; model: string; text: string; finishReason: string; usage: Usage }) {
  return {
    id: opts.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: opts.model,
    choices: [
      { index: 0, finish_reason: opts.finishReason, message: { role: "assistant", content: opts.text } },
    ],
    usage: opts.usage,
    answer: opts.text,
  };
}

export function chatChunk(opts: {
  id: string;
  model: string;
  created: number;
  delta: { role?: "assistant"; content?: string };
  finishReason: string | null;
}) {
  return {
    id: opts.id,
    object: "chat.completion.chunk",
    created: opts.created,
    model: opts.model,
    choices: [{ index: 0, delta: opts.delta, finish_reason: opts.finishReason }],
  };
}
