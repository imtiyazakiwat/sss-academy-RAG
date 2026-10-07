// Gemini client: AI Studio (AIzaSy... keys) or Vertex AI express mode (other keys), chosen per key.
import { env } from "./env";
import { HttpError } from "./http";
import { KeyPool, redact } from "./keys";
import type { GeminiContent, Usage } from "./openai";

const MODEL_RE = /^gemini-[A-Za-z0-9._-]+$/;
const UPSTREAM_TIMEOUT_MS = 45_000;

export interface GenerateRequest {
  model: string;
  system: string;
  contents: GeminiContent[];
  temperature?: number;
  maxTokens?: number;
}

export interface GeminiEvent {
  text: string;
  finishReason?: string;
  usage?: Usage;
  blockReason?: string;
}

export function resolveModel(requested: string | undefined): string {
  return requested && MODEL_RE.test(requested) && env.geminiAllowedModels().includes(requested) ? requested : env.geminiModel();
}

export function mapFinishReason(reason: string | undefined): string {
  switch (reason) {
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
    case "BLOCKLIST":
    case "PROHIBITED_CONTENT":
      return "content_filter";
    default:
      return "stop";
  }
}

function endpoint(key: string, model: string, stream: boolean): { url: string; headers: Record<string, string> } {
  const method = stream ? "streamGenerateContent?alt=sse" : "generateContent";
  const headers: Record<string, string> = { "Content-Type": "application/json", "x-goog-api-key": key };
  if (key.startsWith("AIzaSy")) {
    return {
      url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:${method}`,
      headers,
    };
  }
  const project = env.vertexProject();
  const location = env.vertexLocation();
  const host = location === "global" ? "aiplatform.googleapis.com" : `${encodeURIComponent(location)}-aiplatform.googleapis.com`;
  headers["X-Goog-User-Project"] = project;
  return {
    url: `https://${host}/v1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}/publishers/google/models/${encodeURIComponent(model)}:${method}`,
    headers,
  };
}

function buildPayload(req: GenerateRequest, withThinking: boolean): Record<string, unknown> {
  const generationConfig: Record<string, unknown> = {
    maxOutputTokens: req.maxTokens ?? env.geminiMaxOutputTokens(),
  };
  // Gemini 3.8 rejects temperature/top_p/top_k.
  if (req.temperature !== undefined && !req.model.includes("3.8")) generationConfig.temperature = req.temperature;
  const budget = env.geminiThinkingBudget();
  if (withThinking && budget !== undefined) generationConfig.thinkingConfig = { thinkingBudget: budget };

  const payload: Record<string, unknown> = { contents: req.contents, generationConfig };
  if (req.system.trim()) payload.systemInstruction = { parts: [{ text: req.system }] };
  return payload;
}

function sleep(ms: number): Promise<void> {
  return ms > 0 ? new Promise((r) => setTimeout(r, ms)) : Promise.resolve();
}

async function errorFromResponse(res: Response, keys: string[]): Promise<HttpError> {
  let raw = "";
  try {
    raw = await res.text();
  } catch {
    /* ignore */
  }
  const message = extractMessage(raw); // non-JSON bodies are never echoed
  const lower = raw.toLowerCase();
  if (res.status === 429 || lower.includes("resource_exhausted")) {
    return new HttpError(429, "Gemini is rate-limited or the quota is spent. Try again shortly.", "rate_limit_error", "rate_limited");
  }
  if (res.status === 401 || res.status === 403) {
    return new HttpError(502, "Gemini API key error: authentication or permission failed.", "upstream_error", "upstream_auth");
  }
  if (res.status === 400) {
    const safe = redact(message, keys).slice(0, 300) || "Gemini rejected the request.";
    return new HttpError(400, safe, "invalid_request_error", "upstream_bad_request");
  }
  return new HttpError(502, "Gemini request failed. Try again shortly.", "upstream_error", "upstream_error");
}

function isRetryable(res: Response): boolean {
  return res.status === 429 || res.status === 503;
}

/**
 * POST to Gemini with key rotation, backoff and the thinking-config fallback.
 * Returns an OK Response (body unread) or throws HttpError.
 */
async function callUpstream(req: GenerateRequest, stream: boolean): Promise<Response> {
  const allKeys = env.geminiKeys();
  if (!allKeys.length) {
    throw new HttpError(500, "Server is not configured (GEMINI_API_KEY missing).", "server_error", "server_misconfigured");
  }
  const pool = new KeyPool(allKeys);
  const maxAttempts = env.geminiMaxRetries() + 1;
  let key = pool.next() as string;
  let withThinking = true;
  let thinkingFallbackUsed = false;
  let backoff = env.geminiRetryBaseMs();
  let last: HttpError | undefined;

  for (let attempt = 1; attempt <= maxAttempts; ) {
    const { url, headers } = endpoint(key, req.model, stream);
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(buildPayload(req, withThinking)),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch {
      throw new HttpError(502, "Could not reach Gemini. Try again shortly.", "upstream_error", "upstream_error");
    }
    if (res.ok) return res;

    if (res.status === 400 && withThinking && !thinkingFallbackUsed && env.geminiThinkingBudget() !== undefined) {
      const body = await res.text().catch(() => "");
      if (body.toLowerCase().includes("thinking")) {
        thinkingFallbackUsed = true;
        withThinking = false;
        continue; // retried once without thinkingConfig; not counted as a retry
      }
      last = new HttpError(400, redact(extractMessage(body), allKeys).slice(0, 300) || "Gemini rejected the request.", "invalid_request_error", "upstream_bad_request");
      throw last;
    }

    const retryable = isRetryable(res);
    last = await errorFromResponse(res, allKeys);
    if (!(retryable || last.code === "rate_limited") || attempt >= maxAttempts) throw last;

    // Prefer another key; otherwise wait with exponential backoff and jitter.
    const nextKey = pool.next();
    if (nextKey) {
      key = nextKey;
    } else {
      await sleep(backoff + Math.random() * backoff * 0.3);
      backoff *= 2;
    }
    attempt += 1;
  }
  throw last ?? new HttpError(502, "Gemini request failed. Try again shortly.", "upstream_error", "upstream_error");
}

function extractMessage(raw: string): string {
  try {
    const parsed = JSON.parse(raw) as { error?: { message?: string } } | Array<{ error?: { message?: string } }>;
    const first = Array.isArray(parsed) ? parsed[0] : parsed;
    return first?.error?.message ?? "";
  } catch {
    return "";
  }
}

interface GeminiResponseJson {
  candidates?: Array<{ content?: { parts?: Array<{ text?: string; thought?: boolean }> }; finishReason?: string }>;
  usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number; totalTokenCount?: number };
  promptFeedback?: { blockReason?: string };
}

function toEvent(json: GeminiResponseJson): GeminiEvent {
  let text = "";
  let finishReason: string | undefined;
  for (const c of json.candidates ?? []) {
    for (const p of c.content?.parts ?? []) {
      if (typeof p.text === "string" && !p.thought) text += p.text;
    }
    if (c.finishReason) finishReason = c.finishReason;
  }
  const u = json.usageMetadata;
  return {
    text,
    finishReason,
    blockReason: json.promptFeedback?.blockReason,
    usage: u
      ? {
          prompt_tokens: u.promptTokenCount ?? 0,
          completion_tokens: u.candidatesTokenCount ?? 0,
          total_tokens: u.totalTokenCount ?? 0,
        }
      : undefined,
  };
}

export interface GenerateResult {
  text: string;
  finishReason: string;
  usage: Usage;
}

export async function generate(req: GenerateRequest): Promise<GenerateResult> {
  const res = await callUpstream(req, false);
  let json: GeminiResponseJson;
  try {
    json = (await res.json()) as GeminiResponseJson;
  } catch {
    throw new HttpError(502, "Gemini returned an unreadable response.", "upstream_error", "upstream_error");
  }
  const ev = toEvent(json);
  if (!ev.text && ev.blockReason) {
    throw new HttpError(400, "The request was blocked by Gemini's safety filters.", "invalid_request_error", "content_blocked");
  }
  return {
    text: ev.text,
    finishReason: mapFinishReason(ev.finishReason),
    usage: ev.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

/** Parse Gemini's `alt=sse` stream; tolerates lines split across network chunks. */
async function* parseSse(body: ReadableStream<Uint8Array>): AsyncGenerator<GeminiEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const parseLine = (line: string): GeminiEvent | null => {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) return null;
    const payload = trimmed.slice(5).trim();
    if (!payload || payload === "[DONE]") return null;
    try {
      return toEvent(JSON.parse(payload) as GeminiResponseJson);
    } catch {
      return null;
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const ev = parseLine(line);
        if (ev) yield ev;
      }
    }
    buffer += decoder.decode();
    const ev = parseLine(buffer);
    if (ev) yield ev;
  } finally {
    reader.releaseLock();
  }
}

export interface StreamHandle {
  first: GeminiEvent | null;
  rest: AsyncGenerator<GeminiEvent>;
}

/**
 * Opens the upstream stream and reads the first event, so failures before any output
 * can still be returned as a normal JSON error with the right status code.
 */
export async function openStream(req: GenerateRequest): Promise<StreamHandle> {
  const res = await callUpstream(req, true);
  if (!res.body) throw new HttpError(502, "Gemini returned an empty stream.", "upstream_error", "upstream_error");
  const rest = parseSse(res.body);
  let first: GeminiEvent | null;
  try {
    const n = await rest.next();
    first = n.done ? null : n.value;
  } catch {
    throw new HttpError(502, "Gemini stream failed. Try again shortly.", "upstream_error", "upstream_error");
  }
  if (first && !first.text && first.blockReason) {
    throw new HttpError(400, "The request was blocked by Gemini's safety filters.", "invalid_request_error", "content_blocked");
  }
  return { first, rest };
}
