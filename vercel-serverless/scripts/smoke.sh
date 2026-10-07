#!/usr/bin/env bash
# Smoke test against a DEPLOYED base URL.
#   APP_API_KEY=... [SMOKE_AUDIO=/path/to/short.wav] scripts/smoke.sh https://sss-pluely-backend.vercel.app
# The key is read from the environment only; it is never written to a file or printed.
set -u

BASE="${1:-}"
if [ -z "$BASE" ] || [ -z "${APP_API_KEY:-}" ]; then
  echo "usage: APP_API_KEY=... $0 https://sss-pluely-backend.vercel.app" >&2
  exit 2
fi
BASE="${BASE%/}"
fail=0

check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "ok   $1 ($3)"; else echo "FAIL $1 (expected $2, got $3)"; fail=1; fi
}

code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/health")
check "no token -> 401" 401 "$code"

code=$(curl -s -o /dev/null -w '%{http_code}' -X OPTIONS "$BASE/v1/chat/completions" \
  -H "Origin: tauri://localhost" -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: authorization,content-type")
check "OPTIONS preflight -> 204" 204 "$code"

echo "--- health"
curl -s "$BASE/api/health" -H "Authorization: Bearer $APP_API_KEY"; echo

echo "--- chat (non-stream)"
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $APP_API_KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say OK in one word."}]}' |
  node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);console.log(j.choices?j.choices[0].message.content:s)}catch(e){console.log(s)}})'

if [ -n "${SMOKE_AUDIO:-}" ]; then
  echo "--- stt"
  curl -s -X POST "$BASE/v1/audio/transcriptions" \
    -H "Authorization: Bearer $APP_API_KEY" -F "file=@$SMOKE_AUDIO" -F language=en; echo
else
  echo "--- stt skipped (set SMOKE_AUDIO=/path/to/short.wav to test)"
fi

exit $fail
