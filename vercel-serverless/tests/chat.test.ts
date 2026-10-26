import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleChat } from "../lib/chat";
import {
  APP_KEY, PNG_B64, STUDIO_KEY, STUDIO_KEY_2, VERTEX_KEY, bodyOf, geminiOk, headerOf, postJson, setEnv, stubFetch,
  upstreamError,
} from "./helpers";

const URL_ = "https://x.test/v1/chat/completions";

beforeEach(() => setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY, GEMINI_RETRY_BASE_MS: "0" }));
afterEach(() => vi.unstubAllGlobals());

const pluelyBody = {
  model: "gemini-3.5-flash-lite",
  messages: [
    { role: "system", content: "Be brief." },
    { role: "user", content: "What is ETL?" },
    { role: "assistant", content: "Extract, transform, load." },
    { role: "user", content: [
      { type: "text", text: "and SCD?" },
      { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
    ] },
  ],
};

describe("request translation", () => {
  it("builds the AI Studio request from a Pluely-shaped body", async () => {
    const calls = stubFetch([geminiOk("hello")]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(200);
    expect(calls[0].url).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
    );
    expect(headerOf(calls[0], "x-goog-api-key")).toBe(STUDIO_KEY);
    expect(headerOf(calls[0], "X-Goog-User-Project")).toBeUndefined();
    expect(calls[0].url).not.toContain(STUDIO_KEY);
    const b = bodyOf(calls[0]);
    expect(b.systemInstruction).toEqual({ parts: [{ text: "Be brief." }] });
    expect(b.contents.map((c: { role: string }) => c.role)).toEqual(["user", "model", "user"]);
    expect(b.contents[2].parts[0]).toEqual({ text: "and SCD?" });
    expect(b.contents[2].parts[1]).toEqual({ inlineData: { mimeType: "image/png", data: PNG_B64 } });
    expect(b.generationConfig).toEqual({ maxOutputTokens: 32768, thinkingConfig: { thinkingBudget: 0 } });
  });

  it("uses Vertex express URL and project header for non-AIzaSy keys", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: VERTEX_KEY, VERTEX_PROJECT_ID: "my-proj" });
    const calls = stubFetch([geminiOk("ok")]);
    await handleChat(postJson(URL_, pluelyBody));
    expect(calls[0].url).toBe(
      "https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/publishers/google/models/gemini-3.5-flash-lite:generateContent",
    );
    expect(headerOf(calls[0], "X-Goog-User-Project")).toBe("my-proj");
    expect(headerOf(calls[0], "x-goog-api-key")).toBe(VERTEX_KEY);
  });

  it("uses a regional Vertex host when location is not global", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: VERTEX_KEY, VERTEX_LOCATION: "us-central1" });
    const calls = stubFetch([geminiOk("ok")]);
    await handleChat(postJson(URL_, pluelyBody));
    expect(calls[0].url).toContain("https://us-central1-aiplatform.googleapis.com/v1/projects/");
  });

  it("merges consecutive same-role turns and concatenates system messages", async () => {
    const calls = stubFetch([geminiOk("ok")]);
    await handleChat(
      postJson(URL_, {
        messages: [
          { role: "system", content: "A" },
          { role: "developer", content: "B" },
          { role: "user", content: "one" },
          { role: "user", content: "two" },
        ],
      }),
    );
    const b = bodyOf(calls[0]);
    expect(b.systemInstruction.parts[0].text).toBe("A\n\nB");
    expect(b.contents).toEqual([{ role: "user", parts: [{ text: "one" }, { text: "two" }] }]);
  });

  it("drops the empty placeholder image Pluely sends when there is none", async () => {
    const calls = stubFetch([geminiOk("ok")]);
    await handleChat(
      postJson(URL_, {
        messages: [{ role: "user", content: [
          { type: "text", text: "hi" },
          { type: "image_url", image_url: { url: "data:image/png;base64," } },
        ] }],
      }),
    );
    expect(bodyOf(calls[0]).contents[0].parts).toEqual([{ text: "hi" }]);
  });

  it("sniffs the mime of raw base64 images", async () => {
    const calls = stubFetch([geminiOk("ok")]);
    const jpeg = "/9j/" + "A".repeat(120);
    await handleChat(
      postJson(URL_, { messages: [{ role: "user", content: [{ type: "text", text: "x" }, { type: "image_url", image_url: jpeg }] }] }),
    );
    expect(bodyOf(calls[0]).contents[0].parts[1].inlineData.mimeType).toBe("image/jpeg");
  });

  it("400 for http image URLs, unsupported mimes, non-user last turn and empty input", async () => {
    stubFetch([]);
    const cases: unknown[] = [
      { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://e.com/a.png" } }] }] },
      { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/bmp;base64,AAAA" } }] }] },
      { messages: [{ role: "user", content: "q" }, { role: "assistant", content: "a" }] },
      { messages: [{ role: "user", content: "   " }] },
      { text: "" },
    ];
    for (const body of cases) {
      const res = await handleChat(postJson(URL_, body));
      expect(res.status, JSON.stringify(body)).toBe(400);
      expect(res.headers.get("access-control-allow-origin")).toBe("*");
    }
  });

  it("400 for invalid JSON", async () => {
    stubFetch([]);
    const res = await handleChat(
      new Request(URL_, { method: "POST", headers: { authorization: `Bearer ${APP_KEY}` }, body: "{nope" }),
    );
    expect(res.status).toBe(400);
    expect((await res.json()).error.code).toBe("invalid_json");
  });

  it("accepts the simple body shape and returns the answer field", async () => {
    const calls = stubFetch([geminiOk("simple answer")]);
    const res = await handleChat(
      postJson(URL_, { system: "S", question: "Q?", history: [{ q: "p", a: "r" }], image: `data:image/png;base64,${PNG_B64}` }),
    );
    const body = await res.json();
    expect(body.answer).toBe("simple answer");
    const b = bodyOf(calls[0]);
    expect(b.systemInstruction.parts[0].text).toBe("S");
    expect(b.contents.map((c: { role: string }) => c.role)).toEqual(["user", "model", "user"]);
    expect(b.contents[2].parts).toHaveLength(2);
  });
});

describe("response shape", () => {
  it("non-stream: choices[0].message.content, answer, usage, finish mapping", async () => {
    stubFetch([
      new Response(
        JSON.stringify({
          candidates: [{ content: { parts: [{ text: "thinking", thought: true }, { text: "Hel" }, { text: "lo" }] }, finishReason: "MAX_TOKENS" }],
          usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 4, totalTokenCount: 7 },
        }),
        { status: 200 },
      ),
    ]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    const body = await res.json();
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(body.object).toBe("chat.completion");
    expect(body.choices[0].message).toEqual({ role: "assistant", content: "Hello" });
    expect(body.choices[0].finish_reason).toBe("length");
    expect(body.answer).toBe("Hello");
    expect(body.usage).toEqual({ prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 });
    expect(body.model).toBe("gemini-3.5-flash-lite");
  });

  it("400 content_blocked when there is no text and a blockReason", async () => {
    stubFetch([new Response(JSON.stringify({ promptFeedback: { blockReason: "SAFETY" } }), { status: 200 })]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(400);
    expect((await res.json()).error.code).toBe("content_blocked");
  });

  it("empty text without a block reason is an empty answer, not an error", async () => {
    stubFetch([new Response(JSON.stringify({ candidates: [{ finishReason: "STOP" }] }), { status: 200 })]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(200);
    expect((await res.json()).choices[0].message.content).toBe("");
  });
});

function sseResponse(chunks: string[]): Response {
  const enc = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(c) {
        for (const ch of chunks) c.enqueue(enc.encode(ch));
        c.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

const ev = (text: string, finish?: string) =>
  `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text }] }, ...(finish ? { finishReason: finish } : {}) }] })}\r\n\r\n`;

async function readSse(res: Response): Promise<string[]> {
  const text = await res.text();
  return text.split("\n\n").filter(Boolean).map((l) => l.replace(/^data: /, ""));
}

describe("streaming", () => {
  it("emits OpenAI chunks ending with [DONE], even when an SSE line is split across network chunks", async () => {
    const whole = ev("Hello ");
    const calls = stubFetch([sseResponse([whole.slice(0, 30), whole.slice(30), ev("world", "STOP")])]);
    const res = await handleChat(postJson(URL_, { ...pluelyBody, stream: true }));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(calls[0].url).toContain(":streamGenerateContent?alt=sse");

    const events = await readSse(res);
    expect(events[events.length - 1]).toBe("[DONE]");
    const chunks = events.slice(0, -1).map((e) => JSON.parse(e));
    expect(chunks.every((c) => c.object === "chat.completion.chunk")).toBe(true);
    expect(chunks[0].choices[0].delta).toEqual({ role: "assistant", content: "Hello " });
    expect(chunks[1].choices[0].delta).toEqual({ content: "world" });
    const last = chunks[chunks.length - 1];
    expect(last.choices[0].delta).toEqual({});
    expect(last.choices[0].finish_reason).toBe("stop");
    expect(chunks.map((c) => c.choices[0].delta.content ?? "").join("")).toBe("Hello world");
  });

  it("returns a JSON error (not a stream) when the upstream fails before any output", async () => {
    stubFetch([upstreamError(403, "API key not valid: " + STUDIO_KEY)]);
    const res = await handleChat(postJson(URL_, { ...pluelyBody, stream: true }));
    expect(res.status).toBe(502);
    expect(res.headers.get("content-type")).toContain("application/json");
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(await res.text()).not.toContain(STUDIO_KEY);
  });

  it("emits an in-band error chunk when the upstream dies mid-stream", async () => {
    const enc = new TextEncoder();
    let n = 0;
    stubFetch([
      new Response(
        new ReadableStream({
          pull(c) {
            if (n++ === 0) c.enqueue(enc.encode(ev("part")));
            else c.error(new Error("boom"));
          },
        }),
        { status: 200 },
      ),
    ]);
    const res = await handleChat(postJson(URL_, { ...pluelyBody, stream: true }));
    const events = await readSse(res);
    expect(events[events.length - 1]).toBe("[DONE]");
    expect(events.join("")).toContain("[error:");
  });
});

describe("retry, rotation and errors", () => {
  it("rotates to the second key after a 429", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: `${STUDIO_KEY},${STUDIO_KEY_2}`, GEMINI_RETRY_BASE_MS: "0" });
    const calls = stubFetch([upstreamError(429, "quota"), geminiOk("second")]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(headerOf(calls[0], "x-goog-api-key")).not.toBe(headerOf(calls[1], "x-goog-api-key"));
  });

  it("all keys 429 -> 429 rate_limited, bounded by GEMINI_MAX_RETRIES", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: `${STUDIO_KEY},${STUDIO_KEY_2}`, GEMINI_RETRY_BASE_MS: "0" });
    const calls = stubFetch([upstreamError(429, "a"), upstreamError(429, "b"), upstreamError(429, "c")]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(429);
    expect((await res.json()).error.code).toBe("rate_limited");
    expect(calls).toHaveLength(3);
  });

  it("retries a single key on 503 then succeeds", async () => {
    const calls = stubFetch([upstreamError(503, "busy"), geminiOk("ok")]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
  });

  it("retries once without thinkingConfig when a 400 mentions thinking", async () => {
    const calls = stubFetch([upstreamError(400, "Thinking budget 0 is invalid for this model"), geminiOk("ok")]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(bodyOf(calls[0]).generationConfig.thinkingConfig).toBeDefined();
    expect(bodyOf(calls[1]).generationConfig.thinkingConfig).toBeUndefined();
  });

  it("other 400s are returned as 400 (message truncated, keys redacted)", async () => {
    stubFetch([upstreamError(400, `bad ${STUDIO_KEY} ` + "x".repeat(600))]);
    const res = await handleChat(postJson(URL_, pluelyBody));
    const body = await res.json();
    expect(res.status).toBe(400);
    expect(body.error.message.length).toBeLessThanOrEqual(300);
    expect(JSON.stringify(body)).not.toContain(STUDIO_KEY);
  });

  it("401/403/500 upstream bodies and keys never reach the client", async () => {
    for (const status of [401, 403, 500]) {
      stubFetch([upstreamError(status, `secret-detail ${STUDIO_KEY}`)]);
      const res = await handleChat(postJson(URL_, pluelyBody));
      const text = await res.text();
      expect(res.status).toBe(502);
      expect(text).not.toContain("secret-detail");
      expect(text).not.toContain(STUDIO_KEY);
    }
  });

  it("network failure -> 502", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNRESET " + STUDIO_KEY); }));
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(502);
    expect(await res.text()).not.toContain(STUDIO_KEY);
  });

  it("500 server_misconfigured when GEMINI_API_KEY is missing", async () => {
    setEnv({ APP_API_KEY: APP_KEY });
    const res = await handleChat(postJson(URL_, pluelyBody));
    expect(res.status).toBe(500);
    expect((await res.json()).error.code).toBe("server_misconfigured");
  });
});

describe("model and generation config", () => {
  it("omits temperature for 3.8 models and sends it otherwise", async () => {
    const calls = stubFetch([geminiOk("a"), geminiOk("b")]);
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-3.8-flash", temperature: 0.3 }));
    await handleChat(postJson(URL_, { ...pluelyBody, temperature: 0.3 }));
    expect(calls[0].url).toContain("models/gemini-3.8-flash:");
    expect(bodyOf(calls[0]).generationConfig.temperature).toBeUndefined();
    expect(bodyOf(calls[1]).generationConfig.temperature).toBe(0.3);
  });

  it("honours allowlisted model overrides and falls back to GEMINI_MODEL otherwise", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY });
    const calls = stubFetch([geminiOk("a"), geminiOk("b"), geminiOk("c"), geminiOk("d")]);
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-3.8-flash" }));
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-custom-1.5" }));
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gpt-4o" }));
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini/../../evil" }));
    expect(calls[0].url).toContain("models/gemini-3.8-flash:");
    for (const i of [1, 2, 3]) expect(calls[i].url).toContain("models/gemini-3.5-flash-lite:");
  });
  it("GEMINI_MODEL is always allowed and GEMINI_ALLOWED_MODELS overrides the list", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY, GEMINI_MODEL: "gemini-env-model", GEMINI_ALLOWED_MODELS: "gemini-x-1, gemini-x-2" });
    const calls = stubFetch([geminiOk("a"), geminiOk("b"), geminiOk("c")]);
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-x-2" }));
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-env-model" }));
    await handleChat(postJson(URL_, { ...pluelyBody, model: "gemini-3.8-flash" }));
    expect(calls[0].url).toContain("models/gemini-x-2:");
    expect(calls[1].url).toContain("models/gemini-env-model:");
    expect(calls[2].url).toContain("models/gemini-env-model:");
  });
  it("max_tokens / max_completion_tokens and env overrides", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GEMINI_API_KEY: STUDIO_KEY, GEMINI_THINKING_BUDGET: "" });
    const calls = stubFetch([geminiOk("a"), geminiOk("b")]);
    await handleChat(postJson(URL_, { ...pluelyBody, max_tokens: 100 }));
    await handleChat(postJson(URL_, { ...pluelyBody, max_completion_tokens: 50 }));
    expect(bodyOf(calls[0]).generationConfig).toEqual({ maxOutputTokens: 100 });
    expect(bodyOf(calls[1]).generationConfig.maxOutputTokens).toBe(50);
  });
});
