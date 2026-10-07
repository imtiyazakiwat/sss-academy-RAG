# Vercel serverless backend for Pluely (Gemini chat + Groq Whisper STT)

A direct forwarder written in TypeScript (Node 22, zero runtime dependencies). It authenticates, translates the request
shape, calls Gemini or Groq Whisper, and translates the response back. There is no RAG, no bundled knowledge base and no
prompt building. Pluely supplies the system prompt, history and image.

The Python app in the repo root is untouched. The older `../vercel-deploy/` (Python/FastAPI) is also untouched.

## Endpoints

Every endpoint except the CORS `OPTIONS` preflight needs `Authorization: Bearer <APP_API_KEY>` or `x-api-key: <APP_API_KEY>`.
Missing or wrong token gives `401`. If `APP_API_KEY` is not set on the server, every call fails closed with `500`.

| Path | Alias | Method | Purpose |
|---|---|---|---|
| `/api/chat` | `/v1/chat/completions`, `/chat/completions` | POST | OpenAI-compatible chat, forwarded to Gemini (text + optional image, streaming optional) |
| `/api/stt` | `/v1/audio/transcriptions`, `/audio/transcriptions` | POST | Speech to text via Groq Whisper, returns `{ "text": "..." }` |
| `/api/health` | `/health` | GET | Config status (key counts, model names, never key values) |
| `/api/models` | `/v1/models`, `/models` | GET | Model list |

Chat accepts either the OpenAI body (`messages`, what Pluely sends) or a simple body
`{ "system", "text" | "question" | "prompt", "image", "history": [{"q","a"}] }`. Non-stream responses contain
`choices[0].message.content` and a top-level `answer`.

Defaults: model `gemini-3.5-flash-lite` (override with `GEMINI_MODEL` or a request `model` like `gemini-3.8-flash`),
STT model `whisper-large-v3-turbo`.

Gemini backend is chosen per key, like the local app: keys starting `AIzaSy` use AI Studio, any other key uses Vertex AI
express mode (`VERTEX_PROJECT_ID`, `VERTEX_LOCATION`).

## Environment variables

Names only are in `.env.example`. Set real values on Vercel, never in a committed file:

```bash
cd vercel-serverless
vercel link                      # choose the existing project "vercel-deploy" to keep its URL, or create a new one
vercel env add APP_API_KEY production      # shared secret; this is what Pluely sends
vercel env add GEMINI_API_KEY production   # one key, or comma-separated list (rotates on 429)
vercel env add GROQ_API_KEYS production    # comma-separated Groq keys (fallback name: GROQ_API_KEY)
# optional
vercel env add GEMINI_MODEL production
vercel env add STT_MODEL production
vercel env add STT_PROMPT production
vercel env add VERTEX_PROJECT_ID production
vercel env add VERTEX_LOCATION production
```

Other optional names: `GEMINI_THINKING_BUDGET` (default `0`, empty string omits it), `GEMINI_MAX_OUTPUT_TOKENS`
(default `2048`), `GEMINI_MAX_RETRIES` (default `2`), `GEMINI_RETRY_BASE_MS` (default `1000`), `CORS_ALLOW_ORIGIN` (default `*`).

Generate a strong `APP_API_KEY`, for example `openssl rand -hex 32`.

## Deploy

```bash
vercel --prod
APP_API_KEY=... scripts/smoke.sh https://YOUR-APP.vercel.app          # optional: SMOKE_AUDIO=/path/short.wav
```

`vercel.json` sets `maxDuration` to 60 s for chat and STT (the Hobby ceiling without Fluid compute).

## Limits

- Vercel rejects request bodies above about 4.5 MB. Audio is capped at 4,000,000 bytes (`413` above that).
  Send short clips. JSON base64 inflates audio by about 33 %, so JSON mode fits roughly 3 MB of raw audio in practice.
- Images are sent inline as base64 inside the same 4.5 MB body limit. Allowed types: JPEG, PNG, WebP, GIF.
- There is no rate limiting beyond the upstream quotas. Anyone holding `APP_API_KEY` can spend your Gemini and Groq quota.
- Vercel Deployment Protection can answer with an HTML 401 on protected (preview) URLs. Use the production domain.

## curl examples

```bash
export URL=https://YOUR-APP.vercel.app
export KEY=<your APP_API_KEY>

# health
curl -s "$URL/api/health" -H "Authorization: Bearer $KEY"

# chat, OpenAI shape (what Pluely sends)
curl -s "$URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"Answer in 3 bullets."},{"role":"user","content":"What is SCD Type 2?"}]}'

# chat, simple shape
curl -s "$URL/api/chat" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"system":"Answer in 3 bullets.","text":"What is SCD Type 2?"}'

# chat with an image (base64 data URL) and streaming
curl -N "$URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}]}]}'

# speech to text, multipart upload
curl -s -X POST "$URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav" -F "model=whisper-large-v3-turbo" -F "language=en"

# speech to text, base64 JSON
curl -s -X POST "$URL/api/stt" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"audio\":\"$(base64 < clip.wav | tr -d '\n')\",\"language\":\"en\"}"

# speech to text, raw body
curl -s -X POST "$URL/api/stt?language=en" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: audio/wav" --data-binary @clip.wav
```

## Pluely setup

Both templates are also in `pluely/`. Replace `YOUR-APP.vercel.app` with your production domain.

### Custom AI provider (Dev Space, Custom Providers (AI))

Curl command:

```bash
curl https://YOUR-APP.vercel.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {{API_KEY}}" \
  -d '{
    "model": "{{MODEL}}",
    "messages": [{"role": "system", "content": "{{SYSTEM_PROMPT}}"}, {"role": "user", "content": [{"type": "text", "text": "{{TEXT}}"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,{{IMAGE}}"}}]}]
  }'
```

- Response Content Path: `choices[0].message.content`
- Streaming: ON (recommended, OFF also works)
- Variables: `API_KEY` = your `APP_API_KEY` value, `MODEL` = `gemini-3.5-flash-lite`

### Custom STT provider (Dev Space, Custom STT Providers)

Curl command:

```bash
curl -X POST "https://YOUR-APP.vercel.app/v1/audio/transcriptions" \
  -H "Authorization: Bearer {{API_KEY}}" \
  -F "file={{AUDIO}}" \
  -F "model={{MODEL}}" \
  -F language=en
```

- Response Content Path: `text`
- Streaming: OFF
- Variables: `API_KEY` = your `APP_API_KEY` value, `MODEL` = `whisper-large-v3-turbo`

Pluely sends its own system prompt, so answer style (for example 80-150 words in bullets) is controlled there.
The local app's RAG prompt rules are deliberately not ported.

## Development

```bash
npm install
npm run typecheck
npm test
```

## Verification record (what was run, before handoff)

Run from `vercel-serverless/` on Node v22.18.0, npm 10.9.3, TypeScript 7.0.2, Vitest 5.0.3.

| Check | Result |
|---|---|
| `npm run typecheck` (`tsc --noEmit`, strict) | passes, no errors. Confirmed it catches errors by adding a temporary bad file (removed) |
| `npm test` (`vitest run`) | 5 files, 83 tests, all pass |
| `bash -n scripts/smoke.sh` | syntax OK (script NOT run against any URL) |
| Secret scan `grep -rE "AIzaSy[A-Za-z0-9_-]{20,}\|gsk_[A-Za-z0-9]{20,}"` outside node_modules | no matches |
| Values from repo `.env` and `groq_keys.json` searched in this folder (values not printed) | 0 files contain any |
| `git check-ignore` | `.env`, `.env.local`, `node_modules`, `.vercel` are ignored; `lib/` is not (repo-root rule ignores `lib/`, re-included in this folder's `.gitignore`) |
| `vercel-deploy/` and the Python backend | not modified by this work |

What the tests cover, all with upstream calls MOCKED (stubbed `fetch`; no real Gemini or Groq call was made, no real keys were used):

- Auth on all 4 endpoints: 401 without/with wrong Bearer or `x-api-key`, 500 fail-closed when `APP_API_KEY` is unset, 405 with `Allow`, 413 on oversize `Content-Length`, OPTIONS 204 without auth, CORS headers on 2xx/4xx/5xx and SSE.
- Chat: request translation to Gemini (AI Studio URL and Vertex express URL/headers, `systemInstruction`, role merge, `inlineData`, empty placeholder image dropped, raw base64 mime sniffing, bad images and bad last turn give 400), OpenAI response shape with `answer`, SSE chunks ending in `[DONE]` including a line split across network chunks, mid-stream failure chunk, 429 key rotation and bounded retries, 503 retry, thinking-config fallback, 3.8 omits temperature, model override rules, error bodies and keys never leaked.
- STT: multipart, JSON base64 (with and without `data:` prefix), raw body, language `en-IN` to `en`, model allow-list, `STT_PROMPT`, empty (400) and oversize (413), 429 and 401 key rotation, friendly error messages, no key leakage.
- `tests/http-e2e.test.ts`: a real `node:http` server routes requests through the `vercel.json` rewrites into the `api/*.ts` exports and is called over real HTTP (health, 401 on all routes, preflight, chat, chat streaming, multipart STT). Upstream calls are mocked here too.

NOT verified (cannot be without deploying or real keys):

- Real Gemini (AI Studio and Vertex express) and real Groq responses, including the exact model ids `gemini-3.5-flash-lite` and `gemini-3.8-flash` (taken from the repo's own config).
- Vercel's runtime handling of the Web-standard `export function POST(request: Request)` handlers and the `vercel.json` rewrites. `vercel dev` and `vercel build` were not run because they need the project linked and logged in, and this task forbids deploys.
- Pluely's own behaviour with the templates (built from Pluely's documented and source-level behaviour).
