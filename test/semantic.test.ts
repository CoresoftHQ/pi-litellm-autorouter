import { describe, expect, it, vi } from "vitest";
import { resolveKeywordTierOverride } from "../src/classify/keywords.ts";
import {
  type Embedder,
  SemanticMatcher,
  cosineSimilarity,
  createEmbedder,
  resolveEmbeddingEndpoint,
  routesFromRules,
} from "../src/classify/semantic.ts";
import { buildConfig } from "../src/config.ts";
import type { ModelApplier } from "../src/resolve.ts";
import { route } from "../src/router.ts";
import type { ExtractedTurn } from "../src/types.ts";

/** Synthetic "embeddings": a table of vectors, orthogonal for anything unlisted. */
const VECTORS: Record<string, number[]> = {
  "kubernetes deployment": [1, 0, 0, 0],
  "container orchestration": [0.9, 0.1, 0, 0],
  hello: [0, 1, 0, 0],
  thanks: [0, 1, 0, 0],
  goodbye: [0, 0, 1, 0],
  "help me roll out my k8s cluster today": [1, 0, 0, 0],
  "hey there": [0, 0.6, 0.8, 0],
  "something else entirely": [0, 0, 0, 1],
};

function fakeEmbedder(): Embedder & { calls: string[][] } {
  const calls: string[][] = [];
  const embed = (async (inputs: readonly string[]) => {
    calls.push([...inputs]);
    return inputs.map((text) => VECTORS[text] ?? [0, 0, 0, 1]);
  }) as Embedder & { calls: string[][] };
  embed.calls = calls;
  return embed;
}

const RULES = [
  { keywords: ["kubernetes deployment", "container orchestration"], tier: "REASONING" as const },
  { keywords: ["hello", "thanks"], tier: "SIMPLE" as const },
];

const TIERS = { SIMPLE: "a/haiku", MEDIUM: "a/sonnet", COMPLEX: "a/sonnet", REASONING: "a/opus" };

function semanticConfig(overrides: Record<string, unknown> = {}) {
  const built = buildConfig([
    {
      defaultModel: "a/haiku",
      tiers: TIERS,
      keywordTierRules: RULES,
      semanticKeywordMatching: true,
      embeddingModel: "voyage/voyage-3-5",
      matchThreshold: 0.5,
      ...overrides,
    },
  ]);
  expect(built.errors).toEqual([]);
  return built.config;
}

function turn(currentAsk: string | null): ExtractedTurn {
  return { currentAsk, priorTurns: [], conversationContinuing: false, cumulativeTokens: 0 };
}

function applier(): ModelApplier {
  return {
    find: (provider, modelId) => ({ provider, id: modelId }),
    setModel: async () => true,
    setThinkingLevel: () => {},
  };
}

describe("cosineSimilarity", () => {
  it("is 1 for parallel vectors, 0 for orthogonal or zero ones", () => {
    expect(cosineSimilarity([1, 2], [2, 4])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });
});

describe("routesFromRules", () => {
  it("builds one route per tier in first-appearance order, merging rules and deduping", () => {
    expect(
      routesFromRules([
        { keywords: ["a", "b"], tier: "REASONING" },
        { keywords: ["c"], tier: "SIMPLE" },
        { keywords: ["b", "d"], tier: "REASONING" },
      ]),
    ).toEqual([
      { tier: "REASONING", utterances: ["a", "b", "d"] },
      { tier: "SIMPLE", utterances: ["c"] },
    ]);
  });
});

describe("SemanticMatcher", () => {
  it("routes a paraphrase with no literal keyword to the rule's tier", async () => {
    const matcher = new SemanticMatcher(RULES, 0.5, fakeEmbedder());
    const match = await matcher.match("help me roll out my k8s cluster today");
    expect(match).toEqual({ tier: "REASONING", score: 1, nearestKeyword: "kubernetes deployment" });
  });

  it("matches on the best utterance, not diluted by the tier's other keywords", async () => {
    // Mean over the three would be ~0.33 and fail the threshold; max alone clears it.
    const matcher = new SemanticMatcher(
      [{ keywords: ["kubernetes deployment", "thanks", "goodbye"], tier: "REASONING" }],
      0.5,
      fakeEmbedder(),
    );
    expect((await matcher.match("help me roll out my k8s cluster today"))?.tier).toBe("REASONING");
  });

  it("returns null below the threshold", async () => {
    const matcher = new SemanticMatcher(RULES, 0.5, fakeEmbedder());
    expect(await matcher.match("something else entirely")).toBeNull();
    // "hey there" is 0.6 from "hello": in at 0.5, out at 0.7.
    expect((await matcher.match("hey there"))?.tier).toBe("SIMPLE");
    expect(await new SemanticMatcher(RULES, 0.7, fakeEmbedder()).match("hey there")).toBeNull();
  });

  it("embeds the keywords once, even under concurrent first use, then only the prompt", async () => {
    const embed = fakeEmbedder();
    const matcher = new SemanticMatcher(RULES, 0.5, embed);
    await Promise.all([matcher.match("hey there"), matcher.match("hey there")]);
    const builds = embed.calls.filter((call) => call.length === 4);
    expect(builds).toHaveLength(1);
    expect(builds[0]).toEqual(["kubernetes deployment", "container orchestration", "hello", "thanks"]);
    await matcher.match("hey there");
    expect(embed.calls.filter((call) => call.length === 1)).toHaveLength(3);
  });

  it("retries the index build after a failure instead of caching it", async () => {
    let fail = true;
    const embed: Embedder = async (inputs) => {
      if (fail) throw new Error("boom");
      return inputs.map((text) => VECTORS[text] ?? [0, 0, 0, 1]);
    };
    const matcher = new SemanticMatcher(RULES, 0.5, embed);
    await expect(matcher.match("hey there")).rejects.toThrow("boom");
    expect(matcher.built).toBe(false);
    fail = false;
    expect((await matcher.match("hey there"))?.tier).toBe("SIMPLE");
    expect(matcher.built).toBe(true);
  });
});

describe("resolveEmbeddingEndpoint", () => {
  it("prefers explicit overrides", async () => {
    const resolved = await resolveEmbeddingEndpoint({
      model: "voyage/voyage-3-5",
      baseUrl: "https://proxy.example/v1/",
      apiKeyEnv: "MY_KEY",
      timeoutMs: 1,
      env: { MY_KEY: "k-explicit", VOYAGE_API_KEY: "k-conventional" },
      registry: { getApiKeyForProvider: async () => "k-registry" },
    });
    expect(resolved).toEqual({ baseUrl: "https://proxy.example/v1", apiKey: "k-explicit", modelId: "voyage-3-5" });
  });

  it("uses pi's provider and key when pi knows the provider", async () => {
    const resolved = await resolveEmbeddingEndpoint({
      model: "openai/text-embedding-3-small",
      timeoutMs: 1,
      env: {},
      registry: {
        getProvider: () => ({ baseUrl: "https://api.openai.com/v1" }),
        getApiKeyForProvider: async () => "k-registry",
      },
    });
    expect(resolved).toEqual({ baseUrl: "https://api.openai.com/v1", apiKey: "k-registry", modelId: "text-embedding-3-small" });
  });

  it("falls back to the built-in host table and <PROVIDER>_API_KEY", async () => {
    const resolved = await resolveEmbeddingEndpoint({
      model: "voyage/voyage-3-5",
      timeoutMs: 1,
      env: { VOYAGE_API_KEY: "k-conventional" },
      registry: { getProvider: () => undefined, getApiKeyForProvider: async () => undefined },
    });
    expect(resolved).toEqual({ baseUrl: "https://api.voyageai.com/v1", apiKey: "k-conventional", modelId: "voyage-3-5" });
  });

  it("names the fix when the provider is unknown", async () => {
    await expect(resolveEmbeddingEndpoint({ model: "nowhere/embed", timeoutMs: 1, env: {} })).rejects.toThrow(
      /embeddingEndpoint.baseUrl/,
    );
    await expect(resolveEmbeddingEndpoint({ model: "not-a-ref", timeoutMs: 1, env: {} })).rejects.toThrow(/provider\/model-id/);
  });
});

describe("createEmbedder", () => {
  it("posts the OpenAI shape and reorders vectors by index", async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe("https://api.voyageai.com/v1/embeddings");
      expect(init.headers).toMatchObject({ authorization: "Bearer k", "content-type": "application/json" });
      expect(JSON.parse(init.body as string)).toEqual({ model: "voyage-3-5", input: ["a", "b"] });
      return new Response(JSON.stringify({ data: [{ index: 1, embedding: [2] }, { index: 0, embedding: [1] }] }), {
        status: 200,
      });
    });
    const embed = createEmbedder({
      model: "voyage/voyage-3-5",
      timeoutMs: 1000,
      env: { VOYAGE_API_KEY: "k" },
      fetch: fetchMock as unknown as typeof fetch,
    });
    expect(await embed(["a", "b"])).toEqual([[1], [2]]);
    expect(await embed([])).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("turns an HTTP error into a thrown error with the status", async () => {
    const embed = createEmbedder({
      model: "voyage/voyage-3-5",
      timeoutMs: 1000,
      env: { VOYAGE_API_KEY: "k" },
      fetch: (async () => new Response("nope", { status: 401 })) as unknown as typeof fetch,
    });
    await expect(embed(["a"])).rejects.toThrow(/401 nope/);
  });

  it("abandons a slow call at the timeout", async () => {
    const embed = createEmbedder({
      model: "voyage/voyage-3-5",
      timeoutMs: 10,
      env: { VOYAGE_API_KEY: "k" },
      fetch: ((_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        })) as unknown as typeof fetch,
    });
    await expect(embed(["a"])).rejects.toThrow(/timed out after 10ms/);
  });
});

describe("resolveKeywordTierOverride", () => {
  it("stays lexical when semantic matching is off", async () => {
    const config = buildConfig([{ defaultModel: "a/haiku", tiers: TIERS, keywordTierRules: RULES }]).config;
    const result = await resolveKeywordTierOverride("please do a kubernetes deployment", config, null);
    expect(result.override).toEqual({
      tier: "REASONING",
      matchedKeyword: "kubernetes deployment",
      cause: "literal_keyword_match",
    });
  });

  it("does not fall back to lexical matching when semantic matching is on", async () => {
    const config = semanticConfig();
    const result = await resolveKeywordTierOverride("please do a kubernetes deployment", config, null);
    expect(result.override).toBeNull();
    expect(result.failure).toMatch(/unavailable/);
  });
});

describe("route — semantic keyword matching", () => {
  it("routes a paraphrase to the rule's tier with the semantic cause", async () => {
    const config = semanticConfig();
    const semantic = new SemanticMatcher(config.keywordTierRules, config.matchThreshold, fakeEmbedder());
    const { decision } = await route({
      turn: turn("help me roll out my k8s cluster today"),
      config,
      api: applier(),
      semantic,
      now: () => 0,
    });
    expect(decision.cause).toBe("semantic_keyword_match");
    expect(decision.tier).toBe("REASONING");
    expect(decision.chosenModel).toBe("a/opus");
    expect(decision.matchedKeyword).toBeNull();
    expect(decision.signals).toEqual(['semantic_match (1.00 ≈ "kubernetes deployment")']);
  });

  it("falls through to the scorer when the embedding call fails", async () => {
    const config = semanticConfig();
    const semantic = new SemanticMatcher(config.keywordTierRules, config.matchThreshold, async () => {
      throw new Error("embeddings call timed out after 3000ms");
    });
    const { decision } = await route({ turn: turn("hi"), config, api: applier(), semantic, now: () => 0 });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.signals).toContain("semantic_keyword_match_failed (embeddings call timed out after 3000ms)");
  });

  it("scores a prompt no rule is close to", async () => {
    const config = semanticConfig();
    const semantic = new SemanticMatcher(config.keywordTierRules, config.matchThreshold, fakeEmbedder());
    const { decision } = await route({
      turn: turn("something else entirely"),
      config,
      api: applier(),
      semantic,
      now: () => 0,
    });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals.some((s) => s.startsWith("semantic"))).toBe(false);
  });

  it("a literal match is never labelled semantic", async () => {
    const config = buildConfig([{ defaultModel: "a/haiku", tiers: TIERS, keywordTierRules: RULES }]).config;
    const { decision } = await route({ turn: turn("please do a kubernetes deployment"), config, api: applier(), now: () => 0 });
    expect(decision.cause).toBe("literal_keyword_match");
    expect(decision.matchedKeyword).toBe("kubernetes deployment");
  });
});
