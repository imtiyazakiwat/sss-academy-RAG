// Shared HTTP plumbing: CORS, JSON/error responses, request guard (preflight, auth, method, size).
import { requireAuth } from "./auth";
import { corsHeaders } from "./cors";

export { corsHeaders };
export const MAX_BODY_BYTES = 4_500_000; // Vercel rejects larger request bodies (413) anyway.

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
    public type: string = "invalid_request_error",
    public code: string = "invalid_request",
  ) {
    super(message);
  }
}

export function json(
  data: unknown,
  status = 200,
  request?: Request,
  extra: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...corsHeaders(request),
      ...extra,
    },
  });
}

export function errorResponse(
  status: number,
  message: string,
  type: string,
  code: string,
  request?: Request,
  extra: Record<string, string> = {},
): Response {
  return json({ error: { message, type, code } }, status, request, extra);
}

export function httpErrorResponse(err: HttpError, request?: Request): Response {
  return errorResponse(err.status, err.message, err.type, err.code, request);
}

export function preflight(request: Request): Response {
  return new Response(null, { status: 204, headers: corsHeaders(request) });
}

/**
 * Runs the checks every endpoint shares, in this order:
 * OPTIONS preflight (no auth) -> auth (401/500) -> method (405) -> Content-Length (413).
 * Returns a Response to send, or null when the request may proceed.
 */
export function guard(request: Request, allowed: string[]): Response | null {
  if (request.method === "OPTIONS") return preflight(request);

  const denied = requireAuth(request);
  if (denied) return denied;

  if (!allowed.includes(request.method)) {
    return errorResponse(
      405,
      `Method ${request.method} not allowed`,
      "invalid_request_error",
      "method_not_allowed",
      request,
      { Allow: [...allowed, "OPTIONS"].join(", ") },
    );
  }

  const declared = Number(request.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) return tooLarge(request);
  return null;
}

export function tooLarge(request?: Request): Response {
  return errorResponse(
    413,
    "Request body too large (max about 4.5 MB). Send a shorter audio clip or smaller image.",
    "invalid_request_error",
    "payload_too_large",
    request,
  );
}
