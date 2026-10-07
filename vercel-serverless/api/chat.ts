import { handleChat } from "../lib/chat";

export function POST(request: Request): Promise<Response> {
  return handleChat(request);
}
export const GET = POST;
export const OPTIONS = POST;
