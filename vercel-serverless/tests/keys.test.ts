import { afterEach, describe, expect, it } from "vitest";
import { env } from "../lib/env";
import { KeyPool, parseKeys, redact } from "../lib/keys";
import { setEnv } from "./helpers";

afterEach(() => setEnv({}));

describe("parseKeys", () => {
  it("trims, drops empties and duplicates", () => {
    expect(parseKeys(" a , b,,c ,a")).toEqual(["a", "b", "c"]);
    expect(parseKeys(undefined)).toEqual([]);
    expect(parseKeys("  ,  ")).toEqual([]);
  });
});

describe("KeyPool", () => {
  it("hands out every key exactly once, starting at the random index", () => {
    const pool = new KeyPool(["a", "b", "c"], () => 0.5); // start index 1
    expect([pool.next(), pool.next(), pool.next(), pool.next()]).toEqual(["b", "c", "a", undefined]);
  });

  it("handles an empty pool", () => {
    expect(new KeyPool([]).next()).toBeUndefined();
  });
});

describe("env", () => {
  it("falls back to GROQ_API_KEY when GROQ_API_KEYS is unset", () => {
    setEnv({ GROQ_API_KEY: "single-key-value" });
    expect(env.groqKeys()).toEqual(["single-key-value"]);
    setEnv({ GROQ_API_KEYS: "k1,k2", GROQ_API_KEY: "ignored-key" });
    expect(env.groqKeys()).toEqual(["k1", "k2"]);
  });

  it("thinking budget defaults to 0 and empty string omits it", () => {
    setEnv({});
    expect(env.geminiThinkingBudget()).toBe(0);
    setEnv({ GEMINI_THINKING_BUDGET: "" });
    expect(env.geminiThinkingBudget()).toBeUndefined();
  });
});

describe("redact", () => {
  it("removes configured keys and key-shaped strings", () => {
    const out = redact("bad key secretkey123 and AIzaSyABCDEFGHIJKLMNOP", ["secretkey123"]);
    expect(out).not.toContain("secretkey123");
    expect(out).not.toContain("AIzaSyABCDEFGHIJKLMNOP");
  });
});
