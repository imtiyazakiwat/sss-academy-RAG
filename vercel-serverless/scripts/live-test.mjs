// Live test of a deployed base URL. Usage: node scripts/live-test.mjs https://<host>
// Token: ./.app_api_key (fallback env APP_API_KEY). It is never printed or written.
// Needs /tmp/stt1.wav, /tmp/stt2.wav, /tmp/red.png, /tmp/big.wav (see README "Live testing").
import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { PROJ, TASK, stats, wer } from "./bench-common.mjs";

const BASE = (process.argv[2] || "").replace(/\/$/, "");
if (!BASE) {
  console.error("usage: node scripts/live-test.mjs https://<host>");
  process.exit(2);
}
const TOKEN = existsSync(`${PROJ}/.app_api_key`) ? readFileSync(`${PROJ}/.app_api_key`, "utf8").trim() : process.env.APP_API_KEY || "";
if (!TOKEN) {
  console.error("no token found");
  process.exit(2);
}
const MODEL = "gemini-3.5-flash-lite";
const REF1 = "What is slowly changing dimension type two and when do you use it in ETL testing";
const auth = { Authorization: `Bearer ${TOKEN}` };
const log = [];
let fails = 0;
const scrub = (s) => String(s).split(TOKEN).join("<APP>");
function report(ok, name, detail = "") {
  if (!ok) fails++;
  const line = `${ok ? "ok  " : "FAIL"}  ${name}  ${scrub(detail)}`;
  console.log(line);
  log.push(line);
}
async function call(path, init = {}) {
  const t0 = performance.now();
  const res = await fetch(BASE + path, { ...init, signal: AbortSignal.timeout(90000) });
  const text = await res.text();
  const ms = performance.now() - t0;
  let json;
  try { json = JSON.parse(text); } catch { /* not JSON */ }
  return { res, text, json, ms: Math.round(ms) };
}
const chatBody = (extra = {}) => JSON.stringify({ model: MODEL, messages: [{ role: "system", content: "Answer in 3 short bullets." }, { role: "user", content: [{ type: "text", text: "What is SCD Type 2?" }] }], ...extra });
const JSONH = { ...auth, "Content-Type": "application/json" };
const wav1 = readFileSync("/tmp/stt1.wav");
const wav2 = readFileSync("/tmp/stt2.wav");
const stt = (buf, name = "a.wav", fields = { language: "en" }) => {
  const fd = new FormData();
  fd.append("file", new Blob([buf], { type: "audio/wav" }), name);
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  return call("/v1/audio/transcriptions", { method: "POST", headers: auth, body: fd });
};
async function ttfb(path, body) {
  const t0 = performance.now();
  const res = await fetch(BASE + path, { method: "POST", headers: JSONH, body, signal: AbortSignal.timeout(90000) });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let first = null;
  let content = "";
  let done_ = false;
  let finish = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split(/\r?\n/);
    buf = lines.pop() ?? "";
    for (const l of lines) {
      if (!l.startsWith("data:")) continue;
      if (first === null) first = performance.now() - t0;
      const p = l.slice(5).trim();
      if (p === "[DONE]") { done_ = true; continue; }
      try { const j = JSON.parse(p); const c = j.choices?.[0]; content += c?.delta?.content ?? ""; if (c?.finish_reason) finish = true; } catch { /* ignore */ }
    }
  }
  return { res, first: Math.round(first ?? -1), total: Math.round(performance.now() - t0), content, done: done_, finish };
}

const cold = {};
const warm = { health: [], chat: [], stream: [], stt: [] };

// --- Cold phase: first authed request to each function.
let r = await call("/api/health", { headers: auth });
cold.health = r.ms;
report(r.res.status === 200 && r.json?.status === "ok" && r.json?.gemini?.configured === true && r.json?.gemini?.model === MODEL && !r.text.includes(TOKEN), "1 health 200 JSON (cold)", `${r.ms}ms backend=${r.json?.gemini?.backend} model=${r.json?.gemini?.model} runtime=${r.json?.runtime} region=${r.res.headers.get("x-vercel-id")?.split("::")[0]}`);
report(r.res.headers.get("x-vercel-id")?.startsWith("bom1::"), "15 region bom1", r.res.headers.get("x-vercel-id")?.split("::")[0] ?? "");
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: chatBody() });
cold.chat = r.ms;
const content = r.json?.choices?.[0]?.message?.content ?? "";
report(r.res.status === 200 && content.length > 0 && r.json.choices[0].finish_reason === "stop" && r.json.answer === content && r.json.usage?.total_tokens > 0 && r.json.model === MODEL && /histor|version|dimension|track/i.test(content), "5 chat non-stream (cold)", `${r.ms}ms "${content.replace(/\s+/g, " ").slice(0, 80)}"`);
let sr = await ttfb("/v1/chat/completions", chatBody({ stream: true }));
cold.stream = sr.first;
report(sr.res.status === 200 && (sr.res.headers.get("content-type") || "").includes("text/event-stream") && sr.first > 0 && sr.content.length > 0 && sr.finish && sr.done && sr.res.headers.get("access-control-allow-origin") === "*", "7 stream SSE (first call)", `ttfb=${sr.first}ms total=${sr.total}ms "${sr.content.replace(/\s+/g, " ").slice(0, 60)}"`);
r = await stt(wav1);
cold.stt = r.ms;
const keysOnly = r.json && Object.keys(r.json).join() === "text";
report(r.res.status === 200 && keysOnly && wer(REF1, r.json.text) < 0.15, "10 stt multipart (cold)", `${r.ms}ms wer=${r.json ? wer(REF1, r.json.text).toFixed(3) : "n/a"} "${r.json?.text}"`);

// --- Warm phase
for (let i = 0; i < 5; i++) {
  r = await call("/api/health", { headers: auth }); warm.health.push(r.ms);
  r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: chatBody() }); warm.chat.push(r.ms);
  sr = await ttfb("/v1/chat/completions", chatBody({ stream: true })); warm.stream.push(sr.first);
  r = await stt(wav1); warm.stt.push(r.ms);
}
const med = (a) => stats(a).median;
report(true, "warm medians", `health=${med(warm.health)}ms chat=${med(warm.chat)}ms stream_ttfb=${med(warm.stream)}ms stt=${med(warm.stt)}ms`);

// --- Health alias, auth
r = await call("/health", { headers: auth });
report(r.res.status === 200, "1b /health alias 200", String(r.res.status));
for (const [name, path, method] of [["health", "/api/health", "GET"], ["chat", "/v1/chat/completions", "POST"], ["stt", "/v1/audio/transcriptions", "POST"]]) {
  for (const [label, headers] of [["no token", {}], ["wrong bearer", { Authorization: "Bearer wrong" }], ["wrong x-api-key", { "x-api-key": "wrong" }]]) {
    r = await call(path, { method, headers: { ...headers, "Content-Type": "application/json" }, body: method === "POST" ? "{}" : undefined });
    report(r.res.status === 401 && r.json?.error?.code === "unauthorized" && r.res.headers.get("www-authenticate") === "Bearer" && r.res.headers.get("access-control-allow-origin") === "*", `2/3 ${name} ${label} -> 401 JSON`, `${r.res.status} ${r.json?.error?.code ?? "non-json"}`);
  }
}
r = await call("/api/health", { headers: { "x-api-key": TOKEN } });
report(r.res.status === 200, "4 x-api-key works", String(r.res.status));

// --- Model selection, images, bad input
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: chatBody({ model: "gemini-3.8-flash" }) });
report(r.res.status === 200 && r.json?.model === "gemini-3.8-flash" && !!r.json?.choices?.[0]?.message?.content, "6a model gemini-3.8-flash selectable", `${r.res.status} ${r.ms}ms model=${r.json?.model}`);
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: chatBody({ model: "gpt-4o" }) });
report(r.res.status === 200 && r.json?.model === MODEL, "6b model gpt-4o falls back to default", `${r.res.status} model=${r.json?.model}`);
const png = readFileSync("/tmp/red.png").toString("base64");
const img = (url) => JSON.stringify({ model: MODEL, messages: [{ role: "user", content: [{ type: "text", text: "What colour is this image? One word." }, { type: "image_url", image_url: { url } }] }] });
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: img(`data:image/png;base64,${png}`) });
report(r.res.status === 200 && /red/i.test(r.json?.choices?.[0]?.message?.content ?? ""), "8a image_url data-URI", `${r.res.status} ${r.ms}ms "${(r.json?.choices?.[0]?.message?.content ?? "").slice(0, 40)}"`);
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: img("data:image/png;base64,") });
report(r.res.status === 200, "8b empty image placeholder ignored", String(r.res.status));
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: '{"messages":[]}' });
report(r.res.status === 400 && r.json?.error, "9a empty messages -> 400", `${r.res.status} ${r.json?.error?.code}`);
r = await call("/v1/chat/completions", { method: "POST", headers: JSONH, body: "{not json" });
report(r.res.status === 400, "9b invalid JSON -> 400", String(r.res.status));
r = await call("/api/chat", { headers: auth });
report(r.res.status === 405 && !!r.res.headers.get("allow"), "9c GET /api/chat -> 405 + Allow", `${r.res.status} allow=${r.res.headers.get("allow")}`);

// --- STT variants
r = await stt(wav2, "b.wav", { language: "en", model: "whisper-large-v3-turbo" });
report(r.res.status === 200 && wer("In ETL testing we validate surrogate key generation for slowly changing dimension type two tables, and then perform data reconciliation between the source and the target using row counts and checksums", r.json?.text ?? "") < 0.15, "10b stt longer technical sample", `${r.ms}ms "${(r.json?.text ?? "").slice(0, 100)}"`);
const b64 = wav1.toString("base64");
for (const [label, audio] of [["raw base64", b64], ["data-URI", `data:audio/wav;base64,${b64}`]]) {
  r = await call("/v1/audio/transcriptions", { method: "POST", headers: JSONH, body: JSON.stringify({ audio, language: "en" }) });
  report(r.res.status === 200 && r.json?.text && wer(REF1, r.json.text) < 0.15, `11 stt base64 JSON (${label})`, `${r.ms}ms "${r.json?.text ?? ""}"`);
}
r = await call("/v1/audio/transcriptions?language=en", { method: "POST", headers: { ...auth, "Content-Type": "audio/wav" }, body: wav1 });
report(r.res.status === 200 && r.json?.text && wer(REF1, r.json.text) < 0.15, "12 stt raw body", `${r.ms}ms "${r.json?.text ?? ""}"`);
r = await stt(Buffer.alloc(0), "empty.wav");
report(r.res.status === 400 && r.json?.error?.code === "empty_audio", "13a empty audio -> 400 empty_audio", `${r.res.status} ${r.json?.error?.code}`);
r = await call("/v1/audio/transcriptions", { method: "POST", headers: { ...auth, "Content-Type": "text/plain" }, body: "hello" });
report(r.res.status === 415, "13b text/plain -> 415", String(r.res.status));
r = await call("/v1/audio/transcriptions", { method: "POST", headers: { ...auth, "Content-Type": "audio/wav" }, body: readFileSync("/tmp/big.wav") });
report(r.res.status === 413 && r.json?.error?.code === "payload_too_large", "13c >4MB -> 413 payload_too_large", `${r.res.status} ${r.json?.error?.code}`);

// --- CORS preflight
for (const path of ["/v1/chat/completions", "/v1/audio/transcriptions", "/api/health"]) {
  r = await call(path, { method: "OPTIONS", headers: { Origin: "tauri://localhost", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type" } });
  const h = r.res.headers;
  report(r.res.status === 204 && h.get("access-control-allow-origin") === "*" && /authorization/i.test(h.get("access-control-allow-headers") || "") && /content-type/i.test(h.get("access-control-allow-headers") || "") && /POST/.test(h.get("access-control-allow-methods") || "") && /OPTIONS/.test(h.get("access-control-allow-methods") || "") && h.get("access-control-max-age") === "86400", `14 OPTIONS ${path}`, String(r.res.status));
}

const table = `| path | cold (ms) | warm median of 5 (ms) |\n|---|---|---|\n| GET /api/health | ${cold.health} | ${med(warm.health)} |\n| POST chat (non-stream) | ${cold.chat} | ${med(warm.chat)} |\n| POST chat stream, TTFB | ${cold.stream} | ${med(warm.stream)} |\n| POST STT (short wav, multipart) | ${cold.stt} | ${med(warm.stt)} |`;
console.log("\n" + table);
console.log(`\n${fails === 0 ? "ALL PASS" : fails + " FAILED"}`);
writeFileSync(`${TASK}/live-results.md`, `# Live results for ${BASE}\n\nRun: ${new Date().toISOString()}\n\n\`\`\`\n${log.join("\n")}\n\`\`\`\n\n${table}\n\n${fails === 0 ? "ALL PASS" : fails + " FAILED"}\n`);
process.exit(fails ? 1 : 0);
