// Shared-secret auth. Fails closed: if APP_API_KEY is not configured, nothing is served.
import { createHash, timingSafeEqual } from "node:crypto";
import { env } from "./env";
import { corsHeaders } from "./cors";

function digest(value: string): Buffer {
  return createHash("sha256").update(value).digest();
}

function safeEqual(a: string, b: string): boolean {
  // Hash first so lengths are equal and the comparison leaks nothing about the secret.
  return timingSafeEqual(digest(a), digest(b));
}

function extractToken(request: Request): string {
  const auth = request.headers.get("authorization") ?? "";
  const m = /^Bearer\s+(.+)$/i.exec(auth.trim());
  if (m) return m[1].trim();
  return (request.headers.get("x-api-key") ?? "").trim();
}

function fail(status: number, message: string, type: string, code: string, request: Request, extra = {}): Response {
  return new Response(JSON.stringify({ error: { message, type, code } }), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...corsHeaders(request),
      ...extra,
    },
  });
}

/** Returns a 401/500 Response when the request must be rejected, else null. */
export function requireAuth(request: Request): Response | null {
  const expected = env.appApiKey();
  if (!expected) {
    return fail(500, "Server is not configured (APP_API_KEY missing).", "server_error", "server_misconfigured", request);
  }
  const provided = extractToken(request);
  if (!provided || !safeEqual(provided, expected)) {
    return fail(401, "Invalid or missing API key", "auth_error", "unauthorized", request, {
      "WWW-Authenticate": "Bearer",
    });
  }
  return null;
}
