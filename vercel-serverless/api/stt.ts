import { handleStt } from "../lib/stt";

export function POST(request: Request): Promise<Response> {
  return handleStt(request);
}
export const GET = POST;
export const OPTIONS = POST;
