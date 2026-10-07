// STT benchmark: Groq Whisper (whisper-large-v3-turbo) vs Gemini (inline audio, transcribe-only prompt).
// Kept as the record of the comparison; it is not deployed (.vercelignore excludes scripts/).
// Usage: node --env-file=../.env scripts/bench-stt.mjs   (needs /tmp/stt1.wav, /tmp/stt2.wav, optional /tmp/stt3.wav)
import { readFileSync, existsSync } from "node:fs";
import { geminiKey, geminiEndpoint, groqKeys, stats, sleep, with429, wer, normalize, readResults, mergeResults } from "./bench-common.mjs";

const RUNS = 5;
const SAMPLES = [
  { id: "s1", file: "/tmp/stt1.wav", ref: "What is slowly changing dimension type two and when do you use it in ETL testing", terms: ["etl", "slowly changing dimension", "type two"] },
  {
    id: "s2",
    file: "/tmp/stt2.wav",
    ref: "In ETL testing we validate surrogate key generation for slowly changing dimension type two tables, and then perform data reconciliation between the source and the target using row counts and checksums",
    terms: ["etl", "surrogate key", "slowly changing dimension", "type two", "data reconciliation", "checksums"],
  },
  { id: "s3", file: "/tmp/stt3.wav", ref: "What is slowly changing dimension type two and when do you use it in ETL testing", terms: ["etl", "slowly changing dimension", "type two"] },
].filter((s) => existsSync(s.file));

const chat = readResults().chat;
const GEM_MODEL = chat?.winner_model || "gemini-3.5-flash-lite";
const GEM_THINKING = chat?.winner_thinking || { thinkingBudget: 0 };
const PROMPT = "Transcribe this audio verbatim. The language is English. Output only the transcript text, with no commentary, labels or quotation marks.";
const gkey = geminiKey();
const groqs = groqKeys();

async function groq(sample) {
  const bytes = readFileSync(sample.file);
  const t0 = performance.now();
  let status = 0;
  for (const k of groqs) {
    const fd = new FormData();
    fd.append("file", new Blob([bytes], { type: "audio/wav" }), "audio.wav");
    fd.append("model", "whisper-large-v3-turbo");
    fd.append("response_format", "json");
    fd.append("temperature", "0");
    fd.append("language", "en");
    try {
      const res = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", { method: "POST", headers: { Authorization: `Bearer ${k}` }, body: fd, signal: AbortSignal.timeout(60000) });
      status = res.status;
      if (res.ok) {
        const j = await res.json();
        return { status: 200, ms: performance.now() - t0, text: String(j.text || "").trim() };
      }
      if (![401, 403, 429].includes(res.status)) break;
    } catch {
      status = 0;
      break;
    }
  }
  return { status, ms: performance.now() - t0, text: "" };
}

async function gemini(sample, file = sample.file, mime = "audio/wav") {
  const { url, headers } = geminiEndpoint(gkey, GEM_MODEL, false);
  const generationConfig = { maxOutputTokens: 1024, thinkingConfig: GEM_THINKING };
  if (!GEM_MODEL.includes("3.8")) generationConfig.temperature = 0;
  const data = readFileSync(file).toString("base64");
  const body = JSON.stringify({ contents: [{ role: "user", parts: [{ inlineData: { mimeType: mime, data } }, { text: PROMPT }] }], generationConfig });
  const t0 = performance.now();
  try {
    const res = await fetch(url, { method: "POST", headers, body, signal: AbortSignal.timeout(60000) });
    if (!res.ok) {
      let msg = "";
      try {
        const j = JSON.parse(await res.text());
        msg = String((Array.isArray(j) ? j[0] : j)?.error?.message || "").slice(0, 160);
      } catch {
        /* ignore */
      }
      return { status: res.status, ms: performance.now() - t0, text: "", error: msg };
    }
    const j = await res.json();
    let text = "";
    for (const c of j.candidates || []) for (const p of c.content?.parts || []) if (typeof p.text === "string" && !p.thought) text += p.text;
    return { status: 200, ms: performance.now() - t0, text: text.trim() };
  } catch {
    return { status: 0, ms: performance.now() - t0, text: "" };
  }
}

const providers = { groq, gemini };
const acc = { groq: {}, gemini: {} };
for (const p of Object.keys(acc)) for (const s of SAMPLES) acc[p][s.id] = { ms: [], wers: [], hits: 0, termTotal: 0, failed: 0, first: null };

for (const s of SAMPLES) for (const p of Object.keys(providers)) { await with429(() => providers[p](s)); await sleep(2000); }
for (let i = 0; i < RUNS; i++) {
  for (const s of SAMPLES) {
    for (const [p, fn] of Object.entries(providers)) {
      const r = await with429(() => fn(s));
      const a = acc[p][s.id];
      if (r.status !== 200 || !r.text) a.failed++;
      else {
        a.ms.push(r.ms);
        a.wers.push(wer(s.ref, r.text));
        const n = normalize(r.text);
        for (const t of s.terms) {
          a.termTotal++;
          if (n.includes(normalize(t))) a.hits++;
        }
        if (!a.first) a.first = r.text;
      }
      await sleep(2000);
    }
  }
}

const summary = {};
for (const p of Object.keys(acc)) {
  const all = SAMPLES.flatMap((s) => acc[p][s.id].ms);
  const allWer = SAMPLES.flatMap((s) => acc[p][s.id].wers);
  const hits = SAMPLES.reduce((n, s) => n + acc[p][s.id].hits, 0);
  const terms = SAMPLES.reduce((n, s) => n + acc[p][s.id].termTotal, 0);
  summary[p] = {
    latency_ms: stats(all),
    mean_wer: allWer.length ? allWer.reduce((a, b) => a + b, 0) / allWer.length : null,
    term_hit_rate: terms ? hits / terms : 0,
    failed_runs: SAMPLES.reduce((n, s) => n + acc[p][s.id].failed, 0),
    per_sample: Object.fromEntries(SAMPLES.map((s) => [s.id, { latency_median_ms: stats(acc[p][s.id].ms).median, mean_wer: acc[p][s.id].wers.length ? acc[p][s.id].wers.reduce((a, b) => a + b, 0) / acc[p][s.id].wers.length : null, first_transcript: acc[p][s.id].first }])),
  };
}

// mime probes (informational)
const probes = {};
for (const [name, file, mime] of [["webm", "/tmp/stt1.webm", "audio/webm"], ["m4a", "/tmp/stt1.m4a", "audio/mp4"]]) {
  if (!existsSync(file)) continue;
  const r = await gemini(SAMPLES[0], file, mime);
  probes[name] = { accepted: r.status === 200 && !!r.text, status: r.status, transcript: r.text.slice(0, 120) };
  await sleep(2000);
}

const g = summary.gemini;
const q = summary.groq;
const checks = {
  zero_failed: g.failed_runs === 0,
  latency: g.latency_ms.median !== null && g.latency_ms.median <= 1.25 * q.latency_ms.median,
  wer: g.mean_wer !== null && g.mean_wer <= q.mean_wer + 0.005,
  terms: g.term_hit_rate >= q.term_hit_rate,
};
const decision = Object.values(checks).every(Boolean) ? "gemini" : "groq";
mergeResults("stt", { date: new Date().toISOString(), runs_per_sample: RUNS, gemini_model: GEM_MODEL, samples: SAMPLES.map((s) => ({ id: s.id, ref: s.ref })), summary, mime_probes: probes, checks, decision });

const f = (x) => (x === null ? "n/a" : Number(x).toFixed(3));
console.log("provider  med_ms  mean_ms  max_ms  mean_WER  term_hit  failed");
for (const [p, s] of Object.entries(summary)) console.log(`${p.padEnd(8)}  ${String(s.latency_ms.median).padEnd(6)}  ${String(s.latency_ms.mean).padEnd(7)}  ${String(s.latency_ms.max).padEnd(6)}  ${f(s.mean_wer).padEnd(8)}  ${f(s.term_hit_rate).padEnd(8)}  ${s.failed_runs}`);
console.log("checks:", JSON.stringify(checks), "mime probes:", JSON.stringify(Object.fromEntries(Object.entries(probes).map(([k, v]) => [k, v.accepted]))));
console.log("decision:", decision);
