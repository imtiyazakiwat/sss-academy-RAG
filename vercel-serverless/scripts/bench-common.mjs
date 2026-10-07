// Shared helpers for the benchmark scripts. Never prints keys, URLs with keys, or headers.
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const PROJ = resolve(here, "..");
export const ROOT = resolve(PROJ, "..");
export const TASK = resolve(ROOT, ".agents/tasks/vercel-serverless-pluely-backend");
export const RESULTS = resolve(TASK, "bench-results.json");

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function geminiKey() {
  const k = (process.env.GEMINI_API_KEY || "").split(",").map((s) => s.trim()).filter(Boolean)[0];
  if (!k) throw new Error("GEMINI_API_KEY is not set");
  return k;
}

export function geminiEndpoint(key, model, stream) {
  const method = stream ? "streamGenerateContent?alt=sse" : "generateContent";
  const headers = { "Content-Type": "application/json", "x-goog-api-key": key };
  if (key.startsWith("AIzaSy")) {
    return { url: `https://generativelanguage.googleapis.com/v1beta/models/${model}:${method}`, headers };
  }
  const project = process.env.VERTEX_PROJECT_ID || "patil-hospital-pos";
  const location = process.env.VERTEX_LOCATION || "global";
  const host = location === "global" ? "aiplatform.googleapis.com" : `${location}-aiplatform.googleapis.com`;
  headers["X-Goog-User-Project"] = project;
  return { url: `https://${host}/v1/projects/${project}/locations/${location}/publishers/google/models/${model}:${method}`, headers };
}

export function groqKeys() {
  const out = [];
  const jsonPath = resolve(ROOT, "groq_keys.json");
  if (existsSync(jsonPath)) {
    try {
      const d = JSON.parse(readFileSync(jsonPath, "utf8"));
      for (const k of d.keys || []) out.push(String(k).trim());
    } catch {
      /* ignore */
    }
  }
  if (process.env.GROQ_API_KEY) out.push(process.env.GROQ_API_KEY.trim());
  return [...new Set(out.filter(Boolean))];
}

export function stats(arr) {
  if (!arr.length) return { n: 0, median: null, mean: null, min: null, max: null };
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  const median = s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  const r = (x) => Math.round(x * 10) / 10;
  return { n: s.length, median: r(median), mean: r(mean), min: r(s[0]), max: r(s[s.length - 1]) };
}

/** Retry a measured attempt on HTTP 429 (wait 15 s, max 3 tries); the failed try is not counted. */
export async function with429(fn) {
  for (let i = 0; i < 3; i++) {
    const r = await fn();
    if (r && r.status === 429 && i < 2) {
      await sleep(15000);
      continue;
    }
    return r;
  }
}

const DIGITS = { 0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine" };

export function normalize(text) {
  let t = String(text || "").toLowerCase();
  t = t.replace(/\d/g, (d) => ` ${DIGITS[d]} `);
  t = t.replace(/-/g, " ");
  t = t.replace(/[^a-z' ]+/g, " ");
  t = t.replace(/\s+/g, " ").trim();
  // collapse runs of single letters ("e t l" -> "etl")
  const words = t.split(" ");
  const out = [];
  let run = [];
  const flush = () => {
    if (run.length > 1) out.push(run.join(""));
    else if (run.length === 1) out.push(run[0]);
    run = [];
  };
  for (const w of words) {
    if (w.length === 1 && w !== "a" && w !== "i") run.push(w);
    else {
      flush();
      out.push(w);
    }
  }
  flush();
  return out.join(" ");
}

export function wer(ref, hyp) {
  const r = normalize(ref).split(" ").filter(Boolean);
  const h = normalize(hyp).split(" ").filter(Boolean);
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 1; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
    }
  }
  return r.length ? d[r.length][h.length] / r.length : 0;
}

export function mergeResults(section, value) {
  mkdirSync(TASK, { recursive: true });
  let cur = {};
  if (existsSync(RESULTS)) {
    try {
      cur = JSON.parse(readFileSync(RESULTS, "utf8"));
    } catch {
      cur = {};
    }
  }
  cur[section] = value;
  writeFileSync(RESULTS, JSON.stringify(cur, null, 2) + "\n");
}

export function readResults() {
  return existsSync(RESULTS) ? JSON.parse(readFileSync(RESULTS, "utf8")) : {};
}
