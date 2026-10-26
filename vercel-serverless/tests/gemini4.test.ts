import { afterEach, describe, expect, it, vi } from "vitest";
import { env } from "../lib/env";
import { buildPayload, GEMINI4_MIN_OUTPUT_TOKENS, isGemini4, resolveModel, upstreamTimeoutMs } from "../lib/gemini";
import type { GenerateRequest } from "../lib/gemini";

const req = (o: Partial<GenerateRequest> = {}): GenerateRequest => ({
  model: "gemini-4-argon",
  system: "",
  contents: [{ role: "user", parts: [{ text: "hi" }] }],
  ...o,
});

const config = (payload: Record<string, unknown>) => payload.generationConfig as Record<string, unknown>;

afterEach(() => vi.unstubAllEnvs());

describe("Gemini 4", () => {
  it("recognises gemini-4 ids only", () => {
    expect(isGemini4("gemini-4-argon")).toBe(true);
    expect(isGemini4("gemini-4")).toBe(true);
    expect(isGemini4("gemini-3.8-flash")).toBe(false);
    expect(isGemini4("gemini-40-x")).toBe(false);
  });

  it("is selectable by default", () => {
    expect(env.geminiAllowedModels()).toContain("gemini-4-argon");
    expect(resolveModel("gemini-4-argon")).toBe("gemini-4-argon");
  });

  it("sends thinkingLevel, never thinkingBudget or temperature", () => {
    const c = config(buildPayload(req({ temperature: 0.2 }), true));
    expect(c.thinkingConfig).toEqual({ thinkingLevel: "medium" });
    expect(c).not.toHaveProperty("temperature");
  });

  it("reads the thinking level from env and ignores bad values", () => {
    vi.stubEnv("GEMINI4_THINKING_LEVEL", "HIGH");
    expect(config(buildPayload(req(), true)).thinkingConfig).toEqual({ thinkingLevel: "high" });
    vi.stubEnv("GEMINI4_THINKING_LEVEL", "extreme");
    expect(config(buildPayload(req(), true)).thinkingConfig).toEqual({ thinkingLevel: "medium" });
  });

  it("raises a small output budget but keeps a larger one", () => {
    expect(config(buildPayload(req({ maxTokens: 1000 }), true)).maxOutputTokens).toBe(GEMINI4_MIN_OUTPUT_TOKENS);
    expect(config(buildPayload(req({ maxTokens: 50_000 }), true)).maxOutputTokens).toBe(50_000);
  });

  it("drops thinkingConfig on the no-thinking fallback", () => {
    expect(config(buildPayload(req(), false))).not.toHaveProperty("thinkingConfig");
  });

  it("gives 3.x a 32768 default output budget, overridable by env and by the request", () => {
    expect(config(buildPayload(req({ model: "gemini-3.8-flash" }), true)).maxOutputTokens).toBe(32_768);
    vi.stubEnv("GEMINI_MAX_OUTPUT_TOKENS", "8000");
    expect(config(buildPayload(req({ model: "gemini-3.8-flash" }), true)).maxOutputTokens).toBe(8000);
    expect(config(buildPayload(req({ model: "gemini-3.8-flash", maxTokens: 500 }), true)).maxOutputTokens).toBe(500);
  });

  it("waits long for Gemini 4 and for large output budgets, 45 s for short 3.x answers", () => {
    expect(upstreamTimeoutMs(req({ model: "gemini-4-argon", maxTokens: 1000 }))).toBe(240_000);
    expect(upstreamTimeoutMs(req({ model: "gemini-3.8-flash" }))).toBe(240_000);
    expect(upstreamTimeoutMs(req({ model: "gemini-3.8-flash", maxTokens: 1000 }))).toBe(45_000);
  });

  it("leaves 3.x behaviour unchanged", () => {
    const c = config(buildPayload(req({ model: "gemini-3.5-flash-lite", temperature: 0.2, maxTokens: 1000 }), true));
    expect(c).toEqual({ maxOutputTokens: 1000, temperature: 0.2, thinkingConfig: { thinkingBudget: 0 } });
    expect(config(buildPayload(req({ model: "gemini-3.8-flash", temperature: 0.2 }), true))).not.toHaveProperty("temperature");
  });
});
