// Groq Whisper client (OpenAI-compatible transcription endpoint) with per-request key rotation.
import { DEFAULT_STT_MODEL, env, STT_ALLOWED_MODELS } from "./env";
import { HttpError } from "./http";
import { KeyPool } from "./keys";

const GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions";
const UPSTREAM_TIMEOUT_MS = 45_000;

export interface TranscribeRequest {
  audio: ArrayBuffer;
  mime: string;
  filename: string;
  model?: string;
  language?: string;
  prompt?: string;
}

export function resolveSttModel(requested: string | undefined): string {
  if (requested && STT_ALLOWED_MODELS.includes(requested)) return requested;
  const configured = env.sttModel();
  return STT_ALLOWED_MODELS.includes(configured) ? configured : DEFAULT_STT_MODEL;
}

/** "en-IN" -> "en"; empty -> undefined (Whisper auto-detects). */
export function normalizeLanguage(language: string | undefined): string | undefined {
  const base = (language ?? "").trim().split("-")[0].toLowerCase();
  return /^[a-z]{2,3}$/.test(base) ? base : undefined;
}

const RATE_LIMITED = () =>
  new HttpError(429, "Transcription is rate limited. Try again shortly.", "rate_limit_error", "rate_limited");
const FAILED = () => new HttpError(502, "Could not transcribe the audio.", "upstream_error", "upstream_error");

export async function transcribe(req: TranscribeRequest): Promise<string> {
  const keys = env.groqKeys();
  if (!keys.length) {
    throw new HttpError(500, "Server is not configured (GROQ_API_KEYS missing).", "server_error", "server_misconfigured");
  }
  const pool = new KeyPool(keys);
  const language = normalizeLanguage(req.language);
  const prompt = req.prompt?.trim() || env.sttPrompt();
  let sawRateLimit = false;

  for (let key = pool.next(); key; key = pool.next()) {
    const form = new FormData();
    form.append("file", new Blob([req.audio], { type: req.mime }), req.filename);
    form.append("model", resolveSttModel(req.model));
    form.append("response_format", "json");
    form.append("temperature", "0");
    if (language) form.append("language", language);
    if (prompt) form.append("prompt", prompt);

    let res: Response;
    try {
      res = await fetch(GROQ_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${key}` },
        body: form,
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch {
      throw FAILED();
    }

    if (res.ok) {
      try {
        const data = (await res.json()) as { text?: unknown };
        return typeof data.text === "string" ? data.text.trim() : "";
      } catch {
        throw FAILED();
      }
    }
    if (res.status === 429) {
      sawRateLimit = true;
      continue; // next key
    }
    if (res.status === 401 || res.status === 403) continue; // dead key: try the next one
    throw FAILED();
  }
  throw sawRateLimit ? RATE_LIMITED() : FAILED();
}
