// POST /api/chat (also /v1/chat/completions): OpenAI-compatible in and out, forwarded to Gemini.
import { generate, mapFinishReason, openStream, resolveModel } from "./gemini";
import type { GeminiEvent, GenerateRequest } from "./gemini";
import { corsHeaders, errorResponse, guard, HttpError, httpErrorResponse, json, MAX_BODY_BYTES, tooLarge } from "./http";
import { chatChunk, chatCompletion, normalizeChat } from "./openai";

async function readJson(request: Request): Promise<unknown> {
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) throw new HttpError(413, "Request body too large.", "invalid_request_error", "payload_too_large");
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json");
  }
}

function sse(data: unknown): string {
  return `data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`;
}

export async function handleChat(request: Request): Promise<Response> {
  const blocked = guard(request, ["POST"]);
  if (blocked) return blocked;

  try {
    const chat = normalizeChat(await readJson(request));
    const gen: GenerateRequest = {
      model: resolveModel(chat.model),
      system: chat.system,
      contents: chat.contents,
      temperature: chat.temperature,
      maxTokens: chat.maxTokens,
    };
    const id = `chatcmpl-${Date.now()}`;

    if (!chat.stream) {
      const result = await generate(gen);
      return json(
        chatCompletion({ id, model: gen.model, text: result.text, finishReason: result.finishReason, usage: result.usage }),
        200,
        request,
      );
    }

    const { first, rest } = await openStream(gen);
    const created = Math.floor(Date.now() / 1000);
    const encoder = new TextEncoder();
    const chunk = (delta: { role?: "assistant"; content?: string }, finishReason: string | null) =>
      encoder.encode(sse(chatChunk({ id, model: gen.model, created, delta, finishReason })));

    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel() {
        cancelled = true; // client went away; stop enqueueing
      },
      async start(controller) {
        let finish: string | undefined;
        let roleSent = false;
        const emit = (ev: GeminiEvent) => {
          if (ev.finishReason) finish = ev.finishReason;
          if (!ev.text) return;
          if (cancelled) throw new Error("cancelled");
          controller.enqueue(chunk(roleSent ? { content: ev.text } : { role: "assistant", content: ev.text }, null));
          roleSent = true;
        };
        try {
          if (first) emit(first);
          for await (const ev of rest) emit(ev);
        } catch {
          if (cancelled) return;
          // Headers are already sent; surface a short message in-band.
          controller.enqueue(chunk({ content: "\n[error: Gemini stream interrupted. Try again.]" }, null));
          finish = undefined;
        }
        if (cancelled) return;
        controller.enqueue(chunk({}, mapFinishReason(finish)));
        controller.enqueue(encoder.encode(sse("[DONE]")));
        controller.close();
      },
    });

    return new Response(body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        ...corsHeaders(request),
      },
    });
  } catch (err) {
    if (err instanceof HttpError) {
      return err.status === 413 ? tooLarge(request) : httpErrorResponse(err, request);
    }
    return errorResponse(500, "Internal error.", "server_error", "internal_error", request);
  }
}
