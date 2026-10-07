// Chat model benchmark: Gemini Flash-Lite vs Gemini 3.8 Flash, streaming TTFT + total latency.
// Usage: node --env-file=../.env scripts/bench-chat.mjs
import { geminiKey, geminiEndpoint, stats, sleep, with429, mergeResults } from "./bench-common.mjs";

const PROMPT = "What is SCD Type 2 in ETL testing? Answer in 3 short bullet points.";
const RUNS = 5;
const CONFIGS = [
  { label: "A", name: "3.5-flash-lite budget0", model: "gemini-3.5-flash-lite", thinking: { thinkingBudget: 0 } },
  { label: "B", name: "3.8-flash budget0", model: "gemini-3.8-flash", thinking: { thinkingBudget: 0 } },
  { label: "C", name: "3.8-flash LOW", model: "gemini-3.8-flash", thinking: { thinkingLevel: "LOW" } },
  { label: "D", name: "3.8-flash MINIMAL", model: "gemini-3.8-flash", thinking: { thinkingLevel: "MINIMAL" } },
];

const key = geminiKey();

async function once(cfg) {
  const { url, headers } = geminiEndpoint(key, cfg.model, true);
  const generationConfig = { maxOutputTokens: 512, thinkingConfig: cfg.thinking };
  if (!cfg.model.includes("3.8")) generationConfig.temperature = 0;
  const body = JSON.stringify({ contents: [{ role: "user", parts: [{ text: PROMPT }] }], generationConfig });
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(url, { method: "POST", headers, body, signal: AbortSignal.timeout(60000) });
  } catch (e) {
    return { status: 0, error: "network" };
  }
  if (!res.ok) {
    let msg = "";
    try {
      const j = JSON.parse(await res.text());
      msg = String((Array.isArray(j) ? j[0] : j)?.error?.message || "").slice(0, 160);
    } catch {
      /* ignore */
    }
    return { status: res.status, error: msg };
  }
  let ttft = null;
  let text = "";
  let finish;
  let thoughts;
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const handle = (line) => {
    line = line.trim();
    if (!line.startsWith("data:")) return;
    try {
      const j = JSON.parse(line.slice(5));
      for (const c of j.candidates || []) {
        for (const p of c.content?.parts || []) {
          if (typeof p.text === "string" && !p.thought && p.text) {
            if (ttft === null) ttft = performance.now() - t0;
            text += p.text;
          }
        }
        if (c.finishReason) finish = c.finishReason;
      }
      if (j.usageMetadata?.thoughtsTokenCount !== undefined) thoughts = j.usageMetadata.thoughtsTokenCount;
    } catch {
      /* ignore */
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split(/\r?\n/);
    buf = lines.pop() ?? "";
    lines.forEach(handle);
  }
  handle(buf);
  const total = performance.now() - t0;
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return { status: 200, ttft, total, finish, words, thoughts: thoughts ?? 0, clean: finish === "STOP" && words >= 20 };
}

const data = Object.fromEntries(CONFIGS.map((c) => [c.label, { ...c, thinking: c.thinking, runs: [], rejected: null }]));

// warm-up
for (const cfg of CONFIGS) {
  const r = await with429(() => once(cfg));
  if (r.status === 400) data[cfg.label].rejected = r.error || "HTTP 400";
  await sleep(3000);
}
for (let i = 0; i < RUNS; i++) {
  for (const cfg of CONFIGS) {
    const d = data[cfg.label];
    if (d.rejected) continue;
    const r = await with429(() => once(cfg));
    if (r.status === 400) d.rejected = r.error || "HTTP 400";
    else d.runs.push(r);
    await sleep(3000);
  }
}

const summary = {};
for (const cfg of CONFIGS) {
  const d = data[cfg.label];
  const ok = d.runs.filter((r) => r.status === 200);
  const clean = ok.filter((r) => r.clean);
  summary[cfg.label] = {
    name: cfg.name,
    model: cfg.model,
    thinking: cfg.thinking,
    rejected: d.rejected,
    clean_runs: clean.length,
    failed_runs: d.runs.length - ok.length,
    ttft_ms: stats(clean.map((r) => r.ttft)),
    total_ms: stats(clean.map((r) => r.total)),
    median_words: stats(clean.map((r) => r.words)).median,
    median_thought_tokens: stats(clean.map((r) => r.thoughts)).median,
  };
}

const eligible = Object.entries(summary).filter(([, s]) => !s.rejected && s.clean_runs >= 4);
eligible.sort((a, b) => a[1].ttft_ms.median - b[1].ttft_ms.median);
let winner = eligible[0];
if (winner && eligible[1] && eligible[1][1].ttft_ms.median <= winner[1].ttft_ms.median * 1.1) {
  const close = eligible.filter(([, s]) => s.ttft_ms.median <= winner[1].ttft_ms.median * 1.1);
  close.sort((a, b) => a[1].total_ms.median - b[1].total_ms.median);
  winner = close[0];
}
const result = { prompt: PROMPT, runs_per_config: RUNS, date: new Date().toISOString(), configs: summary, winner: winner?.[0] ?? null, winner_model: winner?.[1].model ?? null, winner_thinking: winner?.[1].thinking ?? null };
mergeResults("chat", result);

console.log("cfg  name                    clean  ttft_med  ttft_mean  total_med  words  thoughts");
for (const [l, s] of Object.entries(summary)) {
  console.log(
    `${l}    ${s.name.padEnd(22)}  ${String(s.clean_runs).padEnd(5)}  ${String(s.ttft_ms.median).padEnd(8)}  ${String(s.ttft_ms.mean).padEnd(9)}  ${String(s.total_ms.median).padEnd(9)}  ${String(s.median_words).padEnd(5)}  ${s.median_thought_tokens}${s.rejected ? "  REJECTED: " + s.rejected : ""}`,
  );
}
console.log("winner:", result.winner, result.winner_model, JSON.stringify(result.winner_thinking));
