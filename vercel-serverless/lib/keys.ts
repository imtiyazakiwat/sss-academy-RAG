// Comma-separated API key lists with per-request rotation.
// Serverless has no persistent disk, so each request starts at a random key and walks
// forward; every key is offered at most once.

export function parseKeys(value: string | undefined): string[] {
  if (!value) return [];
  const seen = new Set<string>();
  for (const part of value.split(",")) {
    const k = part.trim();
    if (k) seen.add(k);
  }
  return [...seen];
}

export class KeyPool {
  private readonly order: string[];
  private cursor = 0;

  constructor(keys: string[], random: () => number = Math.random) {
    const start = keys.length ? Math.floor(random() * keys.length) % keys.length : 0;
    this.order = [...keys.slice(start), ...keys.slice(0, start)];
  }

  get size(): number {
    return this.order.length;
  }

  /** Next untried key, or undefined once every key has been handed out. */
  next(): string | undefined {
    return this.cursor < this.order.length ? this.order[this.cursor++] : undefined;
  }

  get remaining(): number {
    return this.order.length - this.cursor;
  }
}

/** Replace any configured key values with a marker (defence in depth for error text). */
export function redact(text: string, keys: string[]): string {
  let out = text;
  for (const k of keys) {
    if (k.length >= 8) out = out.split(k).join("[redacted]");
  }
  return out
    .replace(/AIzaSy[A-Za-z0-9_-]{10,}/g, "[redacted]")
    .replace(/gsk_[A-Za-z0-9]{10,}/g, "[redacted]");
}
