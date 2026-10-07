// Push production env vars to Vercel. Values travel only via stdin; stdout of the CLI is discarded.
// Usage (from vercel-serverless/): node --env-file=../.env scripts/push-env.mjs
// Prints variable NAMES only. Groq keys are read from ../groq_keys.json plus GROQ_API_KEY.
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { groqKeys, PROJ } from "./bench-common.mjs";

const appKey = readFileSync(`${PROJ}/.app_api_key`, "utf8").trim();
const vars = [
  ["APP_API_KEY", appKey],
  ["GEMINI_API_KEY", (process.env.GEMINI_API_KEY || "").trim()],
  ["GEMINI_MODEL", "gemini-3.5-flash-lite"],
  ["GROQ_API_KEYS", groqKeys().join(",")],
];

let failed = 0;
for (const [name, value] of vars) {
  if (!value) {
    console.log(`${name}: SKIPPED (empty)`);
    failed++;
    continue;
  }
  const r = spawnSync("vercel", ["env", "add", name, "production", "--force", "--sensitive", "--yes"], { input: value, cwd: PROJ, stdio: ["pipe", "ignore", "ignore"] });
  console.log(`${name}: ${r.status === 0 ? "ok" : "FAILED (exit " + r.status + ")"}`);
  if (r.status !== 0) failed++;
}
process.exit(failed ? 1 : 0);
