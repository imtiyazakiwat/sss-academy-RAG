import { env } from "./env";
import { guard, json } from "./http";

const GET_ONLY = ["GET"];

function geminiBackend(keys: string[]): "vertex-express" | "ai-studio" | "mixed" | "none" {
  if (!keys.length) return "none";
  const studio = keys.filter((k) => k.startsWith("AIzaSy")).length;
  if (studio === keys.length) return "ai-studio";
  return studio === 0 ? "vertex-express" : "mixed";
}

export async function handleHealth(request: Request): Promise<Response> {
  const blocked = guard(request, GET_ONLY);
  if (blocked) return blocked;
  const gemini = env.geminiKeys();
  const groq = env.groqKeys();
  return json(
    {
      status: "ok",
      gemini: {
        configured: gemini.length > 0,
        keys: gemini.length,
        model: env.geminiModel(),
        backend: geminiBackend(gemini),
      },
      stt: { configured: groq.length > 0, keys: groq.length, model: env.sttModel() },
      runtime: "node",
    },
    200,
    request,
  );
}

export async function handleModels(request: Request): Promise<Response> {
  const blocked = guard(request, GET_ONLY);
  if (blocked) return blocked;
  return json(
    {
      object: "list",
      data: [
        { id: env.geminiModel(), object: "model", created: 1700000000, owned_by: "google" },
        { id: env.sttModel(), object: "model", created: 1700000000, owned_by: "groq" },
      ],
    },
    200,
    request,
  );
}
