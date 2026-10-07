// POST /api/stt (also /v1/audio/transcriptions): multipart, JSON base64, or raw audio body -> Groq Whisper.
import { errorResponse, guard, HttpError, httpErrorResponse, json } from "./http";
import { transcribe } from "./groq";

export const MAX_AUDIO_BYTES = 4_000_000;

interface AudioInput {
  audio: ArrayBuffer;
  mime: string;
  filename: string;
  model?: string;
  language?: string;
  prompt?: string;
}

const EXT_BY_MIME: Record<string, string> = {
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/wave": "wav",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/mp4": "m4a",
  "audio/m4a": "m4a",
  "audio/x-m4a": "m4a",
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
};

function extFor(mime: string): string {
  return EXT_BY_MIME[mime.split(";")[0].trim().toLowerCase()] ?? "wav";
}

function tooBig(): HttpError {
  return new HttpError(413, "Recording too long. Send a shorter clip (max about 4 MB of audio).", "invalid_request_error", "payload_too_large");
}

function emptyAudio(): HttpError {
  return new HttpError(400, "Empty audio upload.", "invalid_request_error", "empty_audio");
}

function optString(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function decodeBase64(b64: string): ArrayBuffer {
  const clean = b64.replace(/\s+/g, "");
  if (Math.floor((clean.length * 3) / 4) > MAX_AUDIO_BYTES + 3) throw tooBig();
  if (!/^[A-Za-z0-9+/_-]*={0,2}$/.test(clean)) {
    throw new HttpError(400, "Audio is not valid base64.", "invalid_request_error", "invalid_audio");
  }
  const buf = Buffer.from(clean, "base64");
  const out = new ArrayBuffer(buf.length);
  new Uint8Array(out).set(buf);
  return out;
}

async function readMultipart(request: Request): Promise<AudioInput> {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    throw new HttpError(400, "Could not parse multipart form data.", "invalid_request_error", "invalid_form");
  }
  const file = form.get("file") ?? form.get("audio");
  if (!file || typeof file === "string") {
    throw new HttpError(400, "Missing audio file (form field \"file\").", "invalid_request_error", "missing_file");
  }
  if (file.size > MAX_AUDIO_BYTES) throw tooBig();
  const mime = file.type || "audio/wav";
  return {
    audio: await file.arrayBuffer(),
    mime,
    filename: file.name || `audio.${extFor(mime)}`,
    model: optString(form.get("model")),
    language: optString(form.get("language")),
    prompt: optString(form.get("prompt")),
  };
}

async function readJsonBody(request: Request): Promise<AudioInput> {
  let body: Record<string, unknown>;
  try {
    const parsed: unknown = await request.json();
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("not object");
    body = parsed as Record<string, unknown>;
  } catch {
    throw new HttpError(400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json");
  }
  let data = optString(body.audio) ?? optString(body.file) ?? "";
  if (!data) throw new HttpError(400, "Missing \"audio\" (base64).", "invalid_request_error", "missing_file");
  let mime = optString(body.mime);
  const m = /^data:([^;,]*)(?:;[^,]*)?,/i.exec(data);
  if (m) {
    mime = mime ?? (m[1] || undefined);
    data = data.slice(m[0].length);
  }
  mime = mime ?? "audio/wav";
  return {
    audio: decodeBase64(data),
    mime,
    filename: optString(body.filename) ?? `audio.${extFor(mime)}`,
    model: optString(body.model),
    language: optString(body.language),
    prompt: optString(body.prompt),
  };
}

async function readRaw(request: Request, contentType: string): Promise<AudioInput> {
  const q = new URL(request.url).searchParams;
  const mime = contentType.split(";")[0].trim() || "audio/wav";
  const audio = await request.arrayBuffer();
  return {
    audio,
    mime: mime === "application/octet-stream" ? "audio/wav" : mime,
    filename: `audio.${extFor(mime === "application/octet-stream" ? "audio/wav" : mime)}`,
    model: optString(q.get("model")),
    language: optString(q.get("language")),
    prompt: optString(q.get("prompt")),
  };
}

export async function handleStt(request: Request): Promise<Response> {
  const blocked = guard(request, ["POST"]);
  if (blocked) return blocked;

  try {
    const contentType = request.headers.get("content-type") ?? "";
    const lower = contentType.toLowerCase();
    let input: AudioInput;
    if (lower.startsWith("multipart/form-data")) input = await readMultipart(request);
    else if (lower.startsWith("application/json")) input = await readJsonBody(request);
    else if (lower.startsWith("audio/") || lower.startsWith("application/octet-stream") || lower.startsWith("video/")) {
      input = await readRaw(request, contentType);
    } else {
      throw new HttpError(
        415,
        "Unsupported Content-Type. Use multipart/form-data, application/json (base64) or audio/*.",
        "invalid_request_error",
        "unsupported_media_type",
      );
    }

    if (input.audio.byteLength === 0) throw emptyAudio();
    if (input.audio.byteLength > MAX_AUDIO_BYTES) throw tooBig();

    const text = await transcribe(input);
    return json({ text }, 200, request);
  } catch (err) {
    if (err instanceof HttpError) return httpErrorResponse(err, request);
    return errorResponse(500, "Internal error.", "server_error", "internal_error", request);
  }
}
