// Typed readers for environment variables. Values are read on every call (not cached)
// so tests and `vercel dev` pick up changes. Never log the values of key variables.
import { parseKeys } from "./keys";

function str(name: string): string | undefined {
  const v = process.env[name];
  return v === undefined ? undefined : v.trim();
}

function nonEmpty(name: string, fallback: string): string {
  return str(name) || fallback;
}

function int(name: string, fallback: number, min: number): number {
  const raw = str(name);
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isInteger(n) && n >= min ? n : fallback;
}

export const DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite";
export const DEFAULT_STT_MODEL = "whisper-large-v3-turbo";
export const STT_ALLOWED_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"];

export const env = {
  appApiKey: (): string => str("APP_API_KEY") ?? "",
  geminiKeys: (): string[] => parseKeys(process.env.GEMINI_API_KEY),
  geminiModel: (): string => nonEmpty("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
  // undefined => omit thinkingConfig (explicit empty string). Default 0 (thinking off).
  geminiThinkingBudget: (): number | undefined => {
    const raw = str("GEMINI_THINKING_BUDGET");
    if (raw === undefined) return 0;
    if (raw === "") return undefined;
    const n = Number(raw);
    return Number.isInteger(n) ? n : 0;
  },
  geminiMaxOutputTokens: (): number => int("GEMINI_MAX_OUTPUT_TOKENS", 2048, 1),
  geminiMaxRetries: (): number => int("GEMINI_MAX_RETRIES", 2, 0),
  geminiRetryBaseMs: (): number => int("GEMINI_RETRY_BASE_MS", 1000, 0),
  vertexProject: (): string => nonEmpty("VERTEX_PROJECT_ID", "patil-hospital-pos"),
  vertexLocation: (): string => nonEmpty("VERTEX_LOCATION", "global"),
  groqKeys: (): string[] => {
    const many = parseKeys(process.env.GROQ_API_KEYS);
    return many.length ? many : parseKeys(process.env.GROQ_API_KEY);
  },
  sttModel: (): string => nonEmpty("STT_MODEL", DEFAULT_STT_MODEL),
  sttPrompt: (): string | undefined => str("STT_PROMPT") || undefined,
  corsOrigin: (): string => nonEmpty("CORS_ALLOW_ORIGIN", "*"),
};
