import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleChat } from "../lib/chat";
import { handleHealth, handleModels } from "../lib/misc";
import { handleStt } from "../lib/stt";
import { APP_KEY, GROQ_KEY_1, STUDIO_KEY, VERTEX_KEY, authed, setEnv } from "./helpers";

const handlers = {
  health: [handleHealth, "GET"],
  models: [handleModels, "GET"],
  chat: [handleChat, "POST"],
  stt: [handleStt, "POST"],
} as const;

beforeEach(() => setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY, GROQ_API_KEYS: GROQ_KEY_1 }));
afterEach(() => vi.unstubAllGlobals());

describe.each(Object.entries(handlers))("%s endpoint auth + CORS", (name, [handler, method]) => {
  const url = `https://x.test/api/${name}`;

  it("401 without a token, with CORS headers and WWW-Authenticate", async () => {
    const res = await handler(new Request(url, { method }));
    expect(res.status).toBe(401);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(res.headers.get("www-authenticate")).toBe("Bearer");
    expect((await res.json()).error.code).toBe("unauthorized");
  });

  it("401 with a wrong Bearer token and wrong x-api-key", async () => {
    const a = await handler(new Request(url, { method, headers: { authorization: "Bearer nope" } }));
    const b = await handler(new Request(url, { method, headers: { "x-api-key": "nope" } }));
    expect(a.status).toBe(401);
    expect(b.status).toBe(401);
  });

  it("OPTIONS preflight is 204 without auth and echoes requested headers", async () => {
    const res = await handler(
      new Request(url, {
        method: "OPTIONS",
        headers: { "access-control-request-headers": "authorization,content-type", origin: "tauri://localhost" },
      }),
    );
    expect(res.status).toBe(204);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(res.headers.get("access-control-allow-methods")).toContain("POST");
    expect(res.headers.get("access-control-allow-headers")).toBe("authorization,content-type");
  });

  it("500 fail-closed when APP_API_KEY is unset", async () => {
    setEnv({});
    const res = await handler(new Request(url, { method, headers: { authorization: "Bearer anything" } }));
    expect(res.status).toBe(500);
    expect((await res.json()).error.code).toBe("server_misconfigured");
  });

  it("405 with CORS + Allow for a wrong method (authenticated)", async () => {
    const wrong = method === "GET" ? "POST" : "GET";
    const res = await handler(authed(url, { method: wrong, ...(wrong === "POST" ? { body: "{}" } : {}) }));
    expect(res.status).toBe(405);
    expect(res.headers.get("allow")).toContain(method);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
  });
});

describe("auth success paths", () => {
  it("accepts Bearer and x-api-key", async () => {
    const a = await handleHealth(authed("https://x.test/api/health"));
    const b = await handleHealth(new Request("https://x.test/api/health", { headers: { "x-api-key": APP_KEY } }));
    expect(a.status).toBe(200);
    expect(b.status).toBe(200);
  });

  it("honours CORS_ALLOW_ORIGIN", async () => {
    setEnv({ APP_API_KEY: APP_KEY, CORS_ALLOW_ORIGIN: "tauri://localhost" });
    const res = await handleHealth(authed("https://x.test/api/health"));
    expect(res.headers.get("access-control-allow-origin")).toBe("tauri://localhost");
  });

  it("oversize Content-Length is rejected before reading the body", async () => {
    const req = authed("https://x.test/api/chat", {
      method: "POST",
      headers: { "content-length": "4600000" },
      body: "{}",
    });
    const res = await handleChat(req);
    expect(res.status).toBe(413);
    expect((await res.json()).error.code).toBe("payload_too_large");
  });
});

describe("health", () => {
  it("reports configuration without any key material", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: `${STUDIO_KEY},${VERTEX_KEY}`, GROQ_API_KEYS: GROQ_KEY_1 });
    const res = await handleHealth(authed("https://x.test/api/health"));
    const text = await res.text();
    const body = JSON.parse(text);
    expect(body.status).toBe("ok");
    expect(body.gemini).toMatchObject({ configured: true, keys: 2, backend: "mixed", model: "gemini-3.5-flash-lite", allowed_models: ["gemini-3.5-flash-lite", "gemini-3.8-flash"] });
    expect(body.stt).toMatchObject({ configured: true, keys: 1, model: "whisper-large-v3-turbo" });
    for (const secret of [APP_KEY, STUDIO_KEY, VERTEX_KEY, GROQ_KEY_1]) expect(text).not.toContain(secret);
  });

  it("models lists the allowed Gemini models and the STT model", async () => {
    const res = await handleModels(authed("https://x.test/api/models"));
    const body = await res.json();
    expect(body.data.map((m: { id: string }) => m.id)).toEqual(["gemini-3.5-flash-lite", "gemini-3.8-flash", "whisper-large-v3-turbo"]);
  });
});
