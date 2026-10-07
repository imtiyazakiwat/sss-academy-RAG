// Real HTTP round trips through a tiny Vercel-like adapter: node:http -> Request -> api/*.ts exports.
// Routing uses the rewrites from vercel.json. Only upstream (Gemini/Groq) calls are mocked.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import * as chat from "../api/chat";
import * as health from "../api/health";
import * as models from "../api/models";
import * as stt from "../api/stt";
import { APP_KEY, GROQ_KEY_1, PNG_B64, STUDIO_KEY, setEnv } from "./helpers";

type Mod = Record<string, unknown>;
const modules: Record<string, Mod> = {
  "/api/chat": chat, "/api/stt": stt, "/api/health": health, "/api/models": models,
};
const rewrites: Array<{ source: string; destination: string }> = JSON.parse(
  readFileSync(join(__dirname, "..", "vercel.json"), "utf8"),
).rewrites;

let server: Server;
let base = "";
const realFetch = globalThis.fetch;
const upstream: string[] = [];
let streamMode = false;

beforeAll(async () => {
  setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY, GROQ_API_KEYS: GROQ_KEY_1 });
  vi.stubGlobal("fetch", async (url: string | URL, init?: RequestInit) => {
    const u = String(url);
    if (u.startsWith("http://127.0.0.1")) return realFetch(url, init);
    upstream.push(u);
    if (streamMode && u.includes(":streamGenerateContent")) {
      const ev = `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: "streamed" }] }, finishReason: "STOP" }] })}\n\n`;
      return new Response(ev, { status: 200 });
    }
    if (u.includes("groq.com")) return new Response(JSON.stringify({ text: " mocked transcript " }), { status: 200 });
    return new Response(
      JSON.stringify({ candidates: [{ content: { parts: [{ text: "mocked answer" }] }, finishReason: "STOP" }] }),
      { status: 200 },
    );
  });
  server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const route = rewrites.find((r) => r.source === url.pathname)?.destination ?? url.pathname;
    const fn = modules[route]?.[req.method ?? "GET"] as ((r: Request) => Promise<Response>) | undefined;
    if (!fn) {
      res.writeHead(404).end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const hasBody = !["GET", "HEAD", "OPTIONS"].includes(req.method ?? "GET");
    const r = new Request(`http://127.0.0.1${req.url}`, {
      method: req.method,
      headers: req.headers as Record<string, string>,
      body: hasBody ? Buffer.concat(chunks) : undefined,
    });
    const out = await fn(r);
    res.writeHead(out.status, Object.fromEntries(out.headers));
    res.end(Buffer.from(await out.arrayBuffer()));
  });
  await new Promise<void>((ok) => server.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(() => {
  server.close();
  vi.unstubAllGlobals();
});

const auth = { authorization: `Bearer ${APP_KEY}` };

describe("http end to end (upstreams mocked)", () => {
  it("GET /api/health and /health alias", async () => {
    for (const p of ["/api/health", "/health"]) {
      const res = await fetch(base + p, { headers: auth });
      expect(res.status).toBe(200);
      expect((await res.json()).status).toBe("ok");
    }
  });

  it("401 without token on every route", async () => {
    for (const [p, method] of [
      ["/api/health", "GET"], ["/v1/models", "GET"], ["/v1/chat/completions", "POST"], ["/v1/audio/transcriptions", "POST"],
    ]) {
      const res = await fetch(base + p, { method, body: method === "POST" ? "{}" : undefined });
      expect(res.status, p).toBe(401);
    }
  });

  it("OPTIONS preflight on the OpenAI-style routes", async () => {
    const res = await fetch(base + "/v1/chat/completions", {
      method: "OPTIONS",
      headers: { "access-control-request-headers": "authorization,content-type" },
    });
    expect(res.status).toBe(204);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
  });

  it("POST /v1/chat/completions (Pluely template body) -> choices[0].message.content", async () => {
    const res = await fetch(base + "/v1/chat/completions", {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify({
        model: "gemini-3.5-flash-lite",
        messages: [
          { role: "system", content: "Be brief." },
          { role: "user", content: [
            { type: "text", text: "hi" },
            { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
          ] },
        ],
      }),
    });
    expect(res.status).toBe(200);
    expect((await res.json()).choices[0].message.content).toBe("mocked answer");
  });

  it("POST /api/chat streaming over real HTTP ends with [DONE]", async () => {
    streamMode = true;
    const res = await fetch(base + "/api/chat", {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify({ text: "hi", stream: true }),
    });
    const text = await res.text();
    expect(text).toContain('"content":"streamed"');
    expect(text.trim().endsWith("data: [DONE]")).toBe(true);
  });

  it("POST /v1/audio/transcriptions multipart -> { text }", async () => {
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array(200).fill(1)], { type: "audio/wav" }), "audio.wav");
    form.append("model", "whisper-large-v3-turbo");
    const res = await fetch(base + "/v1/audio/transcriptions", { method: "POST", headers: auth, body: form });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ text: "mocked transcript" });
    expect(upstream.some((u) => u.includes("api.groq.com"))).toBe(true);
  });
});
