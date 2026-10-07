import { handleModels } from "../lib/misc";

export function GET(request: Request): Promise<Response> {
  return handleModels(request);
}
export const POST = GET;
export const OPTIONS = GET;
