import { vi } from "vitest";

// Obviously fake credentials used only by tests.
export const APP_KEY = "test-app-key";
export const STUDIO_KEY = "AIza" + "SyFAKEFAKEFAKEFAKEFAKEFAKE000001";
export const STUDIO_KEY_2 = "AIza" + "SyFAKEFAKEFAKEFAKEFAKEFAKE000002";
export const VERTEX_KEY = "vertex-fake-key-0000000001";
export const GROQ_KEY_1 = "gsk" + "_FAKEFAKEFAKEFAKEFAKE0001";
export const GROQ_KEY_2 = "gsk" + "_FAKEFAKEFAKEFAKEFAKE0002";

export function setEnv(values: Record<string, string | undefined>): void {
  const managed = [
    "APP_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_ALLOWED_MODELS", "GEMINI_THINKING_BUDGET", "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_MAX_RETRIES", "GEMINI_RETRY_BASE_MS", "VERTEX_PROJECT_ID", "VERTEX_LOCATION", "GROQ_API_KEYS",
    "GROQ_API_KEY", "STT_MODEL", "STT_PROMPT", "CORS_ALLOW_ORIGIN",
  ];
  for (const k of managed) delete process.env[k];
  for (const [k, v] of Object.entries(values)) if (v !== undefined) process.env[k] = v;
}

export function authed(url: string, init: RequestInit = {}): Request {
  const headers = new Headers(init.headers);
  if (!headers.has("authorization")) headers.set("authorization", `Bearer ${APP_KEY}`);
  return new Request(url, { ...init, headers });
}

export function postJson(url: string, body: unknown, headers: Record<string, string> = {}): Request {
  return authed(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

export function geminiOk(text: string, extra: Record<string, unknown> = {}): Response {
  return new Response(
    JSON.stringify({
      candidates: [{ content: { role: "model", parts: [{ text }] }, finishReason: "STOP" }],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 7, totalTokenCount: 12 },
      ...extra,
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

export function upstreamError(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { code: status, message, status: "X" } }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export type FetchCall = { url: string; init: RequestInit };

/** Stub global fetch with a queue of responses (or functions). Returns the recorded calls. */
export function stubFetch(responses: Array<Response | ((call: FetchCall) => Response)>): FetchCall[] {
  const calls: FetchCall[] = [];
  const queue = [...responses];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, init: RequestInit = {}) => {
      const call = { url: String(url), init };
      calls.push(call);
      const next = queue.shift();
      if (!next) throw new Error("unexpected extra fetch");
      return typeof next === "function" ? next(call) : next;
    }),
  );
  return calls;
}

export function bodyOf(call: FetchCall): any {
  return JSON.parse(String(call.init.body));
}

export function headerOf(call: FetchCall, name: string): string | undefined {
  const h = call.init.headers as Record<string, string>;
  const key = Object.keys(h).find((k) => k.toLowerCase() === name.toLowerCase());
  return key ? h[key] : undefined;
}

export const PNG_B64 = "iVBORw0KGgo" + "A".repeat(120);
