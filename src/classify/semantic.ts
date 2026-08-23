/**
 * Semantic matching for `keywordTierRules`.
 *
 * Port of LiteLLM's `_semantic_tier_override`, which builds a `semantic_router` layer
 * with one route per tier, that tier's keywords as the route's utterances, and `max`
 * aggregation: a prompt matches a tier when it is close to *any* of the tier's keywords,
 * not to their average. The route utterances are embedded once and cached; only the
 * prompt is embedded per request.
 *
 * pi has no embeddings API, so the call goes straight to an OpenAI-compatible
 * `/embeddings` endpoint. Credentials come from pi's provider registry when it knows the
 * provider, else from the environment.
 */

import type { KeywordTierRule, RouterConfig } from "../config.ts";
import { type Tier, tierSeverity } from "../types.ts";
import { splitModelRef } from "./llm.ts";

/** Embed each input, returning one vector per input in order. */
export type Embedder = (inputs: readonly string[]) => Promise<number[][]>;

export interface SemanticMatch {
  tier: Tier;
  /** Cosine similarity of the winning utterance. */
  score: number;
  /** The keyword the prompt was closest to — reported for `/autoroute explain`, not as a
   *  literal match: upstream leaves `matched_keyword` empty on semantic hits. */
  nearestKeyword: string;
}

interface Route {
  tier: Tier;
  utterances: string[];
  vectors: number[][] | null;
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** One route per tier, in the order tiers first appear in the rules, as upstream builds them. */
export function routesFromRules(rules: readonly KeywordTierRule[]): { tier: Tier; utterances: string[] }[] {
  const byTier = new Map<Tier, string[]>();
  for (const rule of rules) {
    const utterances = byTier.get(rule.tier) ?? [];
    for (const keyword of rule.keywords) if (!utterances.includes(keyword)) utterances.push(keyword);
    byTier.set(rule.tier, utterances);
  }
  return [...byTier.entries()].map(([tier, utterances]) => ({ tier, utterances }));
}

export class SemanticMatcher {
  private readonly routes: Route[];
  private readonly threshold: number;
  private readonly embed: Embedder;
  /** The in-flight or finished index build, so concurrent cold starts embed the
   *  utterances exactly once rather than each firing the same calls. */
  private building: Promise<void> | null = null;

  constructor(rules: readonly KeywordTierRule[], threshold: number, embed: Embedder) {
    this.routes = routesFromRules(rules).map((route) => ({ ...route, vectors: null }));
    this.threshold = threshold;
    this.embed = embed;
  }

  get built(): boolean {
    return this.routes.every((route) => route.vectors !== null);
  }

  private async ensureBuilt(): Promise<void> {
    if (this.built) return;
    if (!this.building) {
      this.building = (async () => {
        const inputs = this.routes.flatMap((route) => route.utterances);
        const vectors = await this.embed(inputs);
        if (vectors.length !== inputs.length) {
          throw new Error(`embedding returned ${vectors.length} vectors for ${inputs.length} inputs`);
        }
        let offset = 0;
        for (const route of this.routes) {
          route.vectors = vectors.slice(offset, offset + route.utterances.length);
          offset += route.utterances.length;
        }
      })().catch((err: unknown) => {
        // A failed build is retried on the next prompt, not cached as a permanent failure.
        this.building = null;
        throw err;
      });
    }
    await this.building;
  }

  /**
   * The tier whose keywords the prompt is closest to, when that closeness clears the
   * threshold. Throws on embedding failure; the caller decides what a failure means.
   */
  async match(text: string): Promise<SemanticMatch | null> {
    await this.ensureBuilt();
    const [query] = await this.embed([text]);
    if (!query) throw new Error("embedding returned no vector for the prompt");

    let best: SemanticMatch | null = null;
    for (const route of this.routes) {
      if (!route.vectors) continue;
      for (const [i, vector] of route.vectors.entries()) {
        const score = cosineSimilarity(query, vector);
        // Ties go to the higher tier, matching the lexical rule that the highest tier wins.
        const beats =
          best === null ||
          score > best.score ||
          (score === best.score && tierSeverity(route.tier) > tierSeverity(best.tier));
        if (beats) best = { tier: route.tier, score, nearestKeyword: route.utterances[i] ?? "" };
      }
    }
    if (best === null || best.score < this.threshold) return null;
    return best;
  }
}

// ── The embeddings call ──────────────────────────────────────────────────────

/** Hosts for embedding providers pi does not know about. All speak the OpenAI shape. */
export const KNOWN_EMBEDDING_HOSTS: Readonly<Record<string, string>> = {
  voyage: "https://api.voyageai.com/v1",
  openai: "https://api.openai.com/v1",
  mistral: "https://api.mistral.ai/v1",
  openrouter: "https://openrouter.ai/api/v1",
  together: "https://api.together.xyz/v1",
  fireworks: "https://api.fireworks.ai/inference/v1",
  google: "https://generativelanguage.googleapis.com/v1beta/openai",
};

/** The slice of pi's registry used to resolve an embedding provider. */
export interface EmbeddingProviderRegistry {
  getProvider?(provider: string): { baseUrl?: string } | undefined;
  getApiKeyForProvider?(provider: string): Promise<string | undefined>;
}

export interface EmbedderOptions {
  /** `provider/model-id`. */
  model: string;
  /** Overrides the provider's base URL. */
  baseUrl?: string | undefined;
  /** Environment variable holding the API key; overrides the registry. */
  apiKeyEnv?: string | undefined;
  timeoutMs: number;
  registry?: EmbeddingProviderRegistry | undefined;
  env?: NodeJS.ProcessEnv | undefined;
  fetch?: typeof fetch | undefined;
}

export interface ResolvedEmbeddingEndpoint {
  baseUrl: string;
  apiKey: string | null;
  modelId: string;
}

/**
 * Where to send the embeddings call and what to authenticate it with.
 *
 * Base URL: explicit override, else pi's provider, else the built-in host table. Key:
 * explicit env var, else pi's key for the provider, else `<PROVIDER>_API_KEY`.
 */
export async function resolveEmbeddingEndpoint(options: EmbedderOptions): Promise<ResolvedEmbeddingEndpoint> {
  const ref = splitModelRef(options.model);
  if (!ref) throw new Error(`embeddingModel "${options.model}" is not provider/model-id`);
  const env = options.env ?? process.env;

  const baseUrl =
    options.baseUrl?.trim() || options.registry?.getProvider?.(ref.provider)?.baseUrl || KNOWN_EMBEDDING_HOSTS[ref.provider];
  if (!baseUrl) {
    throw new Error(
      `no embeddings endpoint known for provider "${ref.provider}"; set embeddingEndpoint.baseUrl`,
    );
  }

  let apiKey: string | null = null;
  if (options.apiKeyEnv) {
    apiKey = env[options.apiKeyEnv]?.trim() || null;
  } else {
    apiKey = (await options.registry?.getApiKeyForProvider?.(ref.provider))?.trim() || null;
    if (!apiKey) {
      const conventional = `${ref.provider.toUpperCase().replace(/[^A-Z0-9]/g, "_")}_API_KEY`;
      apiKey = env[conventional]?.trim() || null;
    }
  }

  return { baseUrl: baseUrl.replace(/\/+$/, ""), apiKey, modelId: ref.modelId };
}

/** An embedder for an OpenAI-compatible `/embeddings` endpoint. Credentials are resolved
 *  on first use and then kept, so a key that arrives later (an OAuth refresh, say) is
 *  still a restart away — the same contract as the rest of pi's provider plumbing. */
export function createEmbedder(options: EmbedderOptions): Embedder {
  let endpoint: Promise<ResolvedEmbeddingEndpoint> | null = null;
  const doFetch = options.fetch ?? globalThis.fetch;

  return async (inputs) => {
    if (inputs.length === 0) return [];
    if (!endpoint) {
      endpoint = resolveEmbeddingEndpoint(options).catch((err: unknown) => {
        endpoint = null;
        throw err;
      });
    }
    const { baseUrl, apiKey, modelId } = await endpoint;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs);
    try {
      const headers: Record<string, string> = { "content-type": "application/json" };
      if (apiKey) headers.authorization = `Bearer ${apiKey}`;
      const response = await doFetch(`${baseUrl}/embeddings`, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: modelId, input: inputs }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const body = (await response.text().catch(() => "")).slice(0, 200);
        throw new Error(`embeddings call failed: ${response.status}${body ? ` ${body}` : ""}`);
      }
      const payload = (await response.json()) as { data?: { index?: number; embedding?: unknown }[] };
      const rows = payload.data;
      if (!Array.isArray(rows) || rows.length !== inputs.length) {
        throw new Error(`embeddings call returned ${Array.isArray(rows) ? rows.length : "no"} vectors for ${inputs.length} inputs`);
      }
      const ordered: number[][] = new Array(inputs.length);
      rows.forEach((row, position) => {
        const index = typeof row.index === "number" ? row.index : position;
        if (!Array.isArray(row.embedding) || !row.embedding.every((n) => typeof n === "number")) {
          throw new Error("embeddings call returned a malformed vector");
        }
        ordered[index] = row.embedding as number[];
      });
      return ordered;
    } catch (err) {
      if (controller.signal.aborted) throw new Error(`embeddings call timed out after ${options.timeoutMs}ms`);
      throw err;
    } finally {
      clearTimeout(timer);
    }
  };
}

/** The matcher for `config`, or null when semantic matching is off. */
export function buildSemanticMatcher(
  config: RouterConfig,
  options: { registry?: EmbeddingProviderRegistry; env?: NodeJS.ProcessEnv; fetch?: typeof fetch } = {},
): SemanticMatcher | null {
  if (!config.semanticKeywordMatching || !config.embeddingModel || config.keywordTierRules.length === 0) return null;
  const embedder = createEmbedder({
    model: config.embeddingModel,
    baseUrl: config.embeddingEndpoint.baseUrl,
    apiKeyEnv: config.embeddingEndpoint.apiKeyEnv,
    timeoutMs: config.embeddingEndpoint.timeoutMs,
    registry: options.registry,
    env: options.env,
    fetch: options.fetch,
  });
  return new SemanticMatcher(config.keywordTierRules, config.matchThreshold, embedder);
}
