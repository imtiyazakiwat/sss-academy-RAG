import { handleHealth } from "../lib/misc";

export function GET(request: Request): Promise<Response> {
  return handleHealth(request);
}
export const POST = GET;
export const OPTIONS = GET;
