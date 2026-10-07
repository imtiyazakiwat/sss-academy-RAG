import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleStt } from "../lib/stt";
import { APP_KEY, GROQ_KEY_1, GROQ_KEY_2, authed, headerOf, setEnv, stubFetch } from "./helpers";

const URL_ = "https://x.test/v1/audio/transcriptions";

beforeEach(() => setEnv({ APP_API_KEY: APP_KEY, GROQ_API_KEYS: GROQ_KEY_1 }));
afterEach(() => vi.unstubAllGlobals());

const groqOk = (text = "  hello world  ") =>
  new Response(JSON.stringify({ text }), { status: 200, headers: { "content-type": "application/json" } });
const groqErr = (status: number, detail = "nope") => new Response(JSON.stringify({ error: { message: detail } }), { status });

function multipart(fields: Record<string, string> = {}, bytes = 100, type = "audio/wav"): Request {
  const form = new FormData();
  form.append("file", new Blob([new Uint8Array(bytes).fill(1)], { type }), "audio.wav");
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  return authed(URL_, { method: "POST", body: form });
}

function sent(call: { init: RequestInit }): FormData {
  return call.init.body as FormData;
}

describe("multipart mode (what Pluely sends)", () => {
  it("forwards to Groq and returns { text } trimmed", async () => {
    const calls = stubFetch([groqOk()]);
    const res = await handleStt(multipart({ model: "whisper-large-v3-turbo", language: "en", response_format: "text" }));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ text: "hello world" });
    expect(res.headers.get("access-control-allow-origin")).toBe("*");

    expect(calls[0].url).toBe("https://api.groq.com/openai/v1/audio/transcriptions");
    expect(headerOf(calls[0], "authorization")).toBe(`Bearer ${GROQ_KEY_1}`);
    const form = sent(calls[0]);
    expect((form.get("file") as File).size).toBe(100);
    expect(form.get("model")).toBe("whisper-large-v3-turbo");
    expect(form.get("response_format")).toBe("json"); // client's response_format is ignored
    expect(form.get("temperature")).toBe("0");
    expect(form.get("language")).toBe("en");
  });

  it("omits language unless provided and reduces en-IN to en", async () => {
    const calls = stubFetch([groqOk(), groqOk()]);
    await handleStt(multipart());
    await handleStt(multipart({ language: "en-IN" }));
    expect(sent(calls[0]).has("language")).toBe(false);
    expect(sent(calls[1]).get("language")).toBe("en");
  });

  it("falls back to the default model for a disallowed model, allows whisper-large-v3", async () => {
    const calls = stubFetch([groqOk(), groqOk()]);
    await handleStt(multipart({ model: "gpt-4o-transcribe" }));
    await handleStt(multipart({ model: "whisper-large-v3" }));
    expect(sent(calls[0]).get("model")).toBe("whisper-large-v3-turbo");
    expect(sent(calls[1]).get("model")).toBe("whisper-large-v3");
  });

  it("forwards request prompt, else STT_PROMPT, else nothing", async () => {
    setEnv({ APP_API_KEY: APP_KEY, GROQ_API_KEYS: GROQ_KEY_1, STT_PROMPT: "ETL, SCD" });
    const calls = stubFetch([groqOk(), groqOk()]);
    await handleStt(multipart());
    await handleStt(multipart({ prompt: "custom" }));
    expect(sent(calls[0]).get("prompt")).toBe("ETL, SCD");
    expect(sent(calls[1]).get("prompt")).toBe("custom");
    setEnv({ APP_API_KEY: APP_KEY, GROQ_API_KEYS: GROQ_KEY_1 });
    const c2 = stubFetch([groqOk()]);
    await handleStt(multipart());
    expect(sent(c2[0]).has("prompt")).toBe(false);
  });

  it("accepts the field name 'audio' too", async () => {
    const calls = stubFetch([groqOk()]);
    const form = new FormData();
    form.append("audio", new Blob([new Uint8Array(10)], { type: "audio/webm" }), "a.webm");
    const res = await handleStt(authed(URL_, { method: "POST", body: form }));
    expect(res.status).toBe(200);
    expect((sent(calls[0]).get("file") as File).name).toBe("a.webm");
  });

  it("400 when the file is missing", async () => {
    stubFetch([]);
    const form = new FormData();
    form.append("model", "x");
    const res = await handleStt(authed(URL_, { method: "POST", body: form }));
    expect(res.status).toBe(400);
    expect((await res.json()).error.code).toBe("missing_file");
  });
});

describe("JSON base64 and raw modes", () => {
  it("JSON body with plain base64", async () => {
    const calls = stubFetch([groqOk("json text")]);
    const audio = Buffer.from(new Uint8Array(50).fill(7)).toString("base64");
    const res = await handleStt(
      authed(URL_, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ audio, language: "en" }) }),
    );
    expect(await res.json()).toEqual({ text: "json text" });
    expect((sent(calls[0]).get("file") as File).size).toBe(50);
    expect(sent(calls[0]).get("language")).toBe("en");
  });

  it("JSON body with a data: URL prefix strips the prefix and keeps the mime", async () => {
    const calls = stubFetch([groqOk()]);
    const audio = "data:audio/webm;codecs=opus;base64," + Buffer.from("abcdef").toString("base64");
    await handleStt(
      authed(URL_, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ audio }) }),
    );
    const file = sent(calls[0]).get("file") as File;
    expect(file.size).toBe(6);
    expect(file.type).toBe("audio/webm");
    expect(file.name).toBe("audio.webm");
  });

  it("raw audio body with query parameters", async () => {
    const calls = stubFetch([groqOk("raw")]);
    const res = await handleStt(
      authed(`${URL_}?language=hi-IN&model=whisper-large-v3`, {
        method: "POST",
        headers: { "content-type": "audio/wav" },
        body: new Uint8Array(64).fill(2),
      }),
    );
    expect(await res.json()).toEqual({ text: "raw" });
    expect((sent(calls[0]).get("file") as File).size).toBe(64);
    expect(sent(calls[0]).get("language")).toBe("hi");
    expect(sent(calls[0]).get("model")).toBe("whisper-large-v3");
  });

  it("415 for an unsupported content type, 400 for bad JSON / bad base64", async () => {
    stubFetch([]);
    expect((await handleStt(authed(URL_, { method: "POST", headers: { "content-type": "text/plain" }, body: "x" }))).status).toBe(415);
    const j = (body: string) => authed(URL_, { method: "POST", headers: { "content-type": "application/json" }, body });
    expect((await handleStt(j("{nope"))).status).toBe(400);
    expect((await handleStt(j(JSON.stringify({ audio: "***not base64***" })))).status).toBe(400);
    expect((await handleStt(j(JSON.stringify({})))).status).toBe(400);
  });
});

describe("limits", () => {
  it("empty upload -> 400", async () => {
    stubFetch([]);
    const res = await handleStt(multipart({}, 0));
    expect(res.status).toBe(400);
    expect((await res.json()).error.code).toBe("empty_audio");
  });

  it("oversize audio -> 413 in every mode", async () => {
    stubFetch([]);
    expect((await handleStt(multipart({}, 4_000_001))).status).toBe(413);
    const raw = await handleStt(
      authed(URL_, { method: "POST", headers: { "content-type": "audio/wav" }, body: new Uint8Array(4_000_001) }),
    );
    expect(raw.status).toBe(413);
    const b64 = Buffer.alloc(4_100_000).toString("base64");
    const js = await handleStt(
      authed(URL_, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ audio: b64 }) }),
    );
    expect(js.status).toBe(413);
  });

  it("audio exactly at the cap is accepted", async () => {
    stubFetch([groqOk()]);
    expect((await handleStt(multipart({}, 4_000_000))).status).toBe(200);
  });
});

describe("rotation and errors", () => {
  beforeEach(() => setEnv({ APP_API_KEY: APP_KEY, GROQ_API_KEYS: `${GROQ_KEY_1},${GROQ_KEY_2}` }));

  it("rotates to the next key on 429", async () => {
    const calls = stubFetch([groqErr(429), groqOk("ok")]);
    const res = await handleStt(multipart());
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(headerOf(calls[0], "authorization")).not.toBe(headerOf(calls[1], "authorization"));
  });

  it("rotates past a dead key (401)", async () => {
    const calls = stubFetch([groqErr(401), groqOk("ok")]);
    expect((await handleStt(multipart())).status).toBe(200);
    expect(calls).toHaveLength(2);
  });

  it("all keys 429 -> 429 with the friendly message, each key tried once", async () => {
    const calls = stubFetch([groqErr(429), groqErr(429), groqErr(429)]);
    const res = await handleStt(multipart());
    expect(res.status).toBe(429);
    expect((await res.json()).error.message).toBe("Transcription is rate limited. Try again shortly.");
    expect(calls).toHaveLength(2);
  });

  it("other upstream failures -> 502 generic message, no key or body leakage", async () => {
    stubFetch([groqErr(500, `internal ${GROQ_KEY_1} ${GROQ_KEY_2}`)]);
    const res = await handleStt(multipart());
    const text = await res.text();
    expect(res.status).toBe(502);
    expect(text).toContain("Could not transcribe the audio.");
    for (const s of [GROQ_KEY_1, GROQ_KEY_2, "internal", APP_KEY]) expect(text).not.toContain(s);
  });

  it("network failure -> 502", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("down"); }));
    expect((await handleStt(multipart())).status).toBe(502);
  });

  it("500 server_misconfigured without Groq keys", async () => {
    setEnv({ APP_API_KEY: APP_KEY });
    const res = await handleStt(multipart());
    expect(res.status).toBe(500);
  });

  it("401 without a token (no upstream call)", async () => {
    const calls = stubFetch([]);
    const res = await handleStt(new Request(URL_, { method: "POST", body: "x" }));
    expect(res.status).toBe(401);
    expect(calls).toHaveLength(0);
  });
});
