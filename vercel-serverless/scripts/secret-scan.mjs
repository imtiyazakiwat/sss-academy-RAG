// Value-level secret scan. Prints file paths and counts only, never values.
// Usage (from vercel-serverless/): node --env-file=../.env scripts/secret-scan.mjs
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { PROJ, ROOT, TASK, groqKeys } from "./bench-common.mjs";

const secrets = [...groqKeys(), ...(process.env.GEMINI_API_KEY || "").split(",").map((s) => s.trim())];
if (existsSync(`${PROJ}/.app_api_key`)) secrets.push(readFileSync(`${PROJ}/.app_api_key`, "utf8").trim());
const vals = [...new Set(secrets.filter((s) => s.length >= 8))];
const SKIP = new Set(["node_modules", ".vercel", ".app_api_key", ".git"]);
const hits = [];
function walk(dir) {
  for (const name of readdirSync(dir)) {
    if (SKIP.has(name)) continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p);
    else if (st.size < 5_000_000) {
      const text = readFileSync(p, "latin1");
      if (vals.some((v) => text.includes(v))) hits.push(p);
    }
  }
}
walk(PROJ);
walk(TASK);
let gitHits = 0;
for (const v of vals) {
  for (const args of [["grep", "-F", "-q", "-e", v, "HEAD", "--", "vercel-serverless"], ["grep", "--cached", "-F", "-q", "-e", v, "--", "vercel-serverless"]]) {
    try { execFileSync("git", ["-C", ROOT, ...args], { stdio: "ignore" }); gitHits++; } catch { /* exit 1 = no match */ }
  }
}
console.log(`checked ${vals.length} secret values`);
console.log(`file hits: ${hits.length}${hits.length ? "\n" + hits.join("\n") : ""}`);
console.log(`git (HEAD + index) hits: ${gitHits}`);
process.exit(hits.length || gitHits ? 1 : 0);
