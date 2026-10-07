import { env } from "./env";

export function corsHeaders(request?: Request): Record<string, string> {
  const requested = request?.headers.get("access-control-request-headers");
  return {
    "Access-Control-Allow-Origin": env.corsOrigin(),
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": requested || "Authorization, Content-Type, x-api-key",
    "Access-Control-Expose-Headers": "*",
    "Access-Control-Max-Age": "86400",
  };
}
