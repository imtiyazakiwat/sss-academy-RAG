# Vercel serverless backend for Pluely (Gemini chat + Groq Whisper STT)

A direct forwarder written in TypeScript (Node 22, no runtime dependencies). It has no RAG, no prompt building and no
transcript post-processing. Pluely supplies the system prompt, history and image, and this API passes them to Gemini.
Audio goes to Groq `whisper-large-v3-turbo`, the same model as the local Python backend.

Every endpoint needs `Authorization: Bearer <APP_API_KEY>` (or `x-api-key`). Missing or wrong token returns 401. If
`APP_API_KEY` is not set on the server, every request returns 500 (fails closed). Only CORS preflight (`OPTIONS`) is open.

## Endpoints

| Route | Aliases | Purpose |
|---|---|---|
| `GET /api/health` | `/health` | Status, key counts, models (no key material) |
| `POST /api/chat` | `/v1/chat/completions`, `/chat/completions` | Gemini chat, OpenAI-compatible (optional streaming) |
| `POST /api/stt` | `/v1/audio/transcriptions`, `/audio/transcriptions` | Groq Whisper, returns `{ "text": "..." }` |
| `GET /api/models` | `/v1/models`, `/models` | Model list |

Limits: Vercel caps request bodies at about 4.5 MB. Audio is capped at 4,000,000 bytes (413 above that), and base64 JSON
inflates audio by about 33 percent. Send short audio chunks. `maxDuration` is 60 s for chat and STT.
Not verified here: Vercel's own 4.5 MB enforcement and the function `maxDuration` (needs a real deploy).

## Environment variables

Names only are in `.env.example`. Set values in Vercel, never in a committed file:

```bash
cd vercel-serverless
vercel link                          # pick the existing project to keep its URL, or create a new one
vercel env add APP_API_KEY production      # the shared secret Pluely will send
vercel env add GEMINI_API_KEY production   # one key or comma-separated list (rotates on 429)
vercel env add GROQ_API_KEYS production    # comma-separated Groq keys (fallback name: GROQ_API_KEY)
# optional: GEMINI_MODEL, GEMINI_THINKING_BUDGET, GEMINI_MAX_OUTPUT_TOKENS, GEMINI_MAX_RETRIES,
#           VERTEX_PROJECT_ID, VERTEX_LOCATION, STT_MODEL, STT_PROMPT, CORS_ALLOW_ORIGIN
vercel --prod
```

| Name | Required | Default |
|---|---|---|
| `APP_API_KEY` | yes | none (unset = 500) |
| `GEMINI_API_KEY` | yes (chat) | none. `AIzaSy...` keys use AI Studio, other keys use Vertex express |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` (requests may send any `gemini-*` id, e.g. `gemini-3.8-flash`) |
| `GEMINI_THINKING_BUDGET` | no | `0`; empty string omits `thinkingConfig` |
| `GEMINI_MAX_OUTPUT_TOKENS` | no | `2048` |
| `GEMINI_MAX_RETRIES` | no | `2` (switches key first on 429/503, else backs off) |
| `GEMINI_RETRY_BASE_MS` | no | `1000` |
| `VERTEX_PROJECT_ID` / `VERTEX_LOCATION` | no | `patil-hospital-pos` / `global` (same as local `config.py`) |
| `GROQ_API_KEYS` | yes (STT) | none |
| `STT_MODEL` / `STT_PROMPT` | no | `whisper-large-v3-turbo` / unset |
| `CORS_ALLOW_ORIGIN` | no | `*` |

## curl examples

```bash
export URL=https://YOUR-APP.vercel.app
export KEY=<your APP_API_KEY>

# health
curl -s $URL/api/health -H "Authorization: Bearer $KEY"

# chat, OpenAI shape (what Pluely sends). Answer is at choices[0].message.content
curl -s $URL/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"What is SCD Type 2?"}]}'

# chat, simple shape (answer is also at top-level "answer"); add "image":"data:image/png;base64,..." for a screenshot
curl -s $URL/api/chat \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"system":"Be brief.","text":"What is ETL?"}'

# chat, streaming (SSE, ends with data: [DONE])
curl -N $URL/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"stream":true,"messages":[{"role":"user","content":"Explain ETL testing"}]}'

# STT, multipart (what Pluely sends)
curl -s $URL/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav" -F language=en
# -> {"text":"..."}

# STT, raw body and base64 JSON
curl -s "$URL/api/stt?language=en" -H "Authorization: Bearer $KEY" -H "Content-Type: audio/wav" --data-binary @clip.wav
curl -s $URL/api/stt -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"audio\":\"$(base64 < clip.wav | tr -d '\n')\",\"language\":\"en\"}"
```

`scripts/smoke.sh https://YOUR-APP.vercel.app` runs the checks against a deployed URL (`APP_API_KEY` from the environment,
optional `SMOKE_AUDIO=clip.wav`).

## Pluely setup (Dev Space)

Replace `YOUR-APP.vercel.app` with your production domain. Full copies with notes are in `pluely/`.
Create the variables `API_KEY` (the `APP_API_KEY` value) and `MODEL` in Pluely for each provider.

Custom AI provider. Response Content Path: `choices[0].message.content`. Streaming: on (off also works). Variable
`MODEL` = `gemini-3.5-flash-lite`.

```bash
curl https://YOUR-APP.vercel.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {{API_KEY}}" \
  -d '{
    "model": "{{MODEL}}",
    "messages": [{"role": "system", "content": "{{SYSTEM_PROMPT}}"}, {"role": "user", "content": [{"type": "text", "text": "{{TEXT}}"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,{{IMAGE}}"}}]}]
  }'
```

Custom STT provider. Response Content Path: `text`. Streaming: off. Variable `MODEL` = `whisper-large-v3-turbo`.

```bash
curl -X POST "https://YOUR-APP.vercel.app/v1/audio/transcriptions" \
  -H "Authorization: Bearer {{API_KEY}}" \
  -F "file={{AUDIO}}" \
  -F "model={{MODEL}}" \
  -F language=en
```

Use the production domain. Vercel Deployment Protection can answer protected preview URLs with an HTML 401.

## Security notes

- No rate limiting beyond the upstream quotas. Anyone with `APP_API_KEY` can spend your Gemini and Groq quota.
- Keys are read from env vars only, sent in headers (never URLs), redacted from error text, and never returned.
- `.gitignore` excludes `.env*` (except `.env.example`), `.vercel` and `node_modules`. It also re-includes `lib/`
  because the repo-root `.gitignore` ignores every `lib/` directory.

## Verification record

Run in `vercel-serverless/` on Node v22.18.0, TypeScript 7.0.2, vitest 5.0.3:

- `npm run typecheck` (`tsc --noEmit`): no errors. Confirmed it detects errors by adding a deliberate type error, then removing it.
- `npm test` (`vitest run`): 5 files, 83 tests, all pass.
  - `auth-cors`: 401 for missing/wrong Bearer and `x-api-key`, 500 when `APP_API_KEY` is unset, OPTIONS 204 without auth,
    CORS headers on 401/405, 413 for oversize Content-Length, health has no key material (all four handlers).
  - `chat`: request translation (system, merged turns, images, placeholder image dropped, Vertex vs AI Studio URL and
    headers), response shape, streaming including split SSE lines, key rotation, retries, thinking fallback, error redaction.
  - `stt`: multipart, JSON base64, raw body, limits (400/413/415), key rotation, error mapping.
  - `http-e2e`: a real `node:http` server routes through the `vercel.json` rewrites into the `api/*.ts` exports and is called
    over real HTTP: health, 401 on every route, OPTIONS 204, chat (Pluely-shaped body), streaming to `[DONE]`, STT multipart.
- `bash -n scripts/smoke.sh`: syntax OK. Not run against any live URL.
- Secret scan: `grep -rE "AIzaSy[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}" vercel-serverless --exclude-dir=node_modules`
  finds nothing. `git check-ignore` confirms `.env`, `.env.local`, `node_modules` and `.vercel` are ignored.

Verified with mocks only: every Gemini and Groq call (global `fetch` is stubbed in all tests; fake keys only).
Not verified: real Gemini/Vertex and Groq responses, Vercel's runtime handling of `export function POST(request)`,
the rewrites, `maxDuration` and the 4.5 MB limit on a real deployment, and `vercel dev` (not run, it needs a linked project).
The Pluely field formats come from the Pluely docs and source as recorded in the plan, not from running Pluely.
