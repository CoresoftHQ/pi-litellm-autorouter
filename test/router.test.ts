import { describe, expect, it, vi } from "vitest";
import { buildConfig } from "../src/config.ts";
import { route } from "../src/router.ts";
import type { ExtractedTurn } from "../src/types.ts";
import { type ModelApplier, candidatesForTier } from "../src/resolve.ts";

const TIERS = {
  SIMPLE: "anthropic/haiku",
  MEDIUM: "anthropic/sonnet",
  COMPLEX: "anthropic/sonnet",
  REASONING: "anthropic/opus",
};

function config(overrides: Record<string, unknown> = {}) {
  const built = buildConfig([{ defaultModel: "anthropic/haiku", tiers: TIERS, ...overrides }]);
  expect(built.errors).toEqual([]);
  return built.config;
}

function turn(currentAsk: string | null, extra: Partial<ExtractedTurn> = {}): ExtractedTurn {
  return {
    currentAsk,
    priorTurns: [],
    conversationContinuing: false,
    cumulativeTokens: 0,
    ...extra,
  };
}

/** An applier where every model exists and has credentials. */
function applier(overrides: Partial<ModelApplier> = {}): ModelApplier & { applied: string[] } {
  const applied: string[] = [];
  return {
    applied,
    find: (provider, modelId) => ({ provider, id: modelId }),
    setModel: async (model) => {
      const m = model as unknown as { provider: string; id: string };
      applied.push(`${m.provider}/${m.id}`);
      return true;
    },
    setThinkingLevel: () => {},
    ...overrides,
  } as ModelApplier & { applied: string[] };
}

const now = () => 1_000_000;

describe("route — classification", () => {
  it("routes a greeting to the SIMPLE tier", async () => {
    const api = applier();
    const { decision } = await route({ turn: turn("hi"), config: config(), api, now });
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });

  it("routes explicit reasoning to REASONING", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("think step by step and analyze this tradeoff: weigh the options for our schema"),
      config: config(),
      api,
      now,
    });
    expect(decision.tier).toBe("REASONING");
    expect(decision.chosenModel).toBe("anthropic/opus");
  });

  it("routes to defaultModel when there is nothing to classify", async () => {
    const api = applier();
    const { decision } = await route({ turn: turn(null), config: config(), api, now });
    expect(decision.cause).toBe("default_fallback");
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });
});

describe("route — precedence", () => {
  it("lets a keyword rule outrank the classifier", async () => {
    const api = applier();
    const cfg = config({ keywordTierRules: [{ keywords: ["migration"], tier: "REASONING" }] });
    const { decision } = await route({ turn: turn("do the migration"), config: cfg, api, now });
    expect(decision.cause).toBe("literal_keyword_match");
    expect(decision.tier).toBe("REASONING");
    expect(decision.matchedKeyword).toBe("migration");
  });

  it("escalates to the highest tier when several keyword rules match", async () => {
    // Rule order must never silently change the answer.
    const api = applier();
    const cfg = config({
      keywordTierRules: [
        { keywords: ["migration"], tier: "REASONING" },
        { keywords: ["typo"], tier: "SIMPLE" },
      ],
    });
    const { decision } = await route({ turn: turn("fix the typo in the migration"), config: cfg, api, now });
    expect(decision.tier).toBe("REASONING");
  });

  it("bumps exactly one tier on the escalation keyword", async () => {
    const api = applier();
    const cfg = config({ escalationKeywords: ["ESCALATE"] });
    const { decision } = await route({ turn: turn("hi ESCALATE"), config: cfg, api, now });
    expect(decision.escalated).toBe(true);
    expect(decision.tier).toBe("MEDIUM"); // SIMPLE + 1, not straight to the top
    expect(decision.escalationKeyword).toBe("ESCALATE");
  });

  it("does not fire escalation on a case mismatch", async () => {
    const api = applier();
    const cfg = config({ escalationKeywords: ["ESCALATE"] });
    const { decision } = await route({ turn: turn("please escalate this"), config: cfg, api, now });
    expect(decision.escalated).toBe(false);
  });

  it("raises a low classification to the plan-mode floor", async () => {
    const api = applier();
    const cfg = config({ planMode: { minTier: "COMPLEX" } });
    const { decision } = await route({ turn: turn("hi"), config: cfg, api, planModeActive: true, now });
    expect(decision.planFloored).toBe(true);
    expect(decision.tier).toBe("COMPLEX");
  });

  it("leaves a higher classification alone in plan mode", async () => {
    // The floor is a floor, not a pin: the classified tier still wins when it is higher.
    const api = applier();
    const cfg = config({ planMode: { minTier: "MEDIUM" } });
    const { decision } = await route({
      turn: turn("think step by step and analyze this: weigh the options carefully"),
      config: cfg,
      api,
      planModeActive: true,
      now,
    });
    expect(decision.tier).toBe("REASONING");
    expect(decision.planFloored).toBe(false);
  });

  it("does not floor when plan mode is inactive", async () => {
    const api = applier();
    const cfg = config({ planMode: { minTier: "COMPLEX" } });
    const { decision } = await route({ turn: turn("hi"), config: cfg, api, planModeActive: false, now });
    expect(decision.tier).toBe("SIMPLE");
  });

  it("skips the classifier when the floor is the top configured tier", async () => {
    const registry = { find: vi.fn(), complete: vi.fn() };
    const cfg = config({
      planMode: { minTier: "REASONING" },
      classifierType: "llm",
      classifierLLMConfig: { model: "anthropic/haiku" },
    });
    const { decision } = await route({
      turn: turn("hi"),
      config: cfg,
      api: applier(),
      registry: registry as never,
      planModeActive: true,
      now,
    });
    expect(decision.cause).toBe("plan_mode");
    expect(registry.complete).not.toHaveBeenCalled();
  });

  it("applies plan-mode floor on top of a keyword match", async () => {
    const api = applier();
    const cfg = config({
      keywordTierRules: [{ keywords: ["typo"], tier: "SIMPLE" }],
      planMode: { minTier: "COMPLEX" },
    });
    const { decision } = await route({ turn: turn("fix the typo"), config: cfg, api, planModeActive: true, now });
    expect(decision.cause).toBe("plan_mode");
    expect(decision.tier).toBe("COMPLEX");
  });
});

describe("route — one-shot override", () => {
  it("uses the named model and reports it as consumed", async () => {
    const api = applier();
    const { decision, consumedOneShot } = await route({
      turn: turn("hi"),
      config: config(),
      api,
      oneShot: "anthropic/opus",
      now,
    });
    expect(decision.cause).toBe("one_shot_override");
    expect(decision.chosenModel).toBe("anthropic/opus");
    expect(consumedOneShot).toBe(true);
  });

  it("outranks a keyword rule and the classifier", async () => {
    const api = applier();
    const cfg = config({ keywordTierRules: [{ keywords: ["migration"], tier: "SIMPLE" }] });
    const { decision } = await route({
      turn: turn("do the migration"),
      config: cfg,
      api,
      oneShot: "anthropic/opus",
      now,
    });
    expect(decision.chosenModel).toBe("anthropic/opus");
  });

  it("outranks the plan-mode floor", async () => {
    // The floor exists to stop planning being under-powered; naming a model is the more
    // explicit signal, and it is scoped to one prompt.
    const api = applier();
    const cfg = config({ planMode: { minTier: "REASONING" } });
    const { decision } = await route({
      turn: turn("hi"),
      config: cfg,
      api,
      planModeActive: true,
      oneShot: "anthropic/haiku",
      now,
    });
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });

  it("outranks a session pin without disturbing it", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true });
    const pin = { model: "anthropic/haiku", tier: "SIMPLE" as const, expiresAt: now() + 60_000 };
    const { decision, pinToWrite } = await route({
      turn: turn("hi"),
      config: cfg,
      api,
      pin,
      oneShot: "anthropic/opus",
      now,
    });
    expect(decision.chosenModel).toBe("anthropic/opus");
    // The pin is untouched, so the session returns to its own model next turn.
    expect(pinToWrite).toBeNull();
  });

  it("falls back to normal routing when the override cannot be applied", async () => {
    const api = applier({ find: (_p, id) => (id === "ghost" ? undefined : { provider: "anthropic", id }) });
    const { decision, consumedOneShot } = await route({
      turn: turn("hi"),
      config: config(),
      api,
      oneShot: "anthropic/ghost",
      now,
    });
    expect(decision.chosenModel).toBe("anthropic/haiku");
    expect(decision.fellBackBecause).toContain("could not be applied");
    // Spent even though it failed: an override that survived would hijack the next prompt.
    expect(consumedOneShot).toBe(true);
  });

  it("is not consumed when no override is set", async () => {
    const { consumedOneShot } = await route({ turn: turn("hi"), config: config(), api: applier(), now });
    expect(consumedOneShot).toBe(false);
  });
});

describe("route — session affinity", () => {
  const pin = { model: "anthropic/opus", tier: "REASONING" as const, expiresAt: now() + 60_000 };

  it("reuses the pin and skips classification", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true });
    const { decision } = await route({ turn: turn("hi"), config: cfg, api, pin, now });
    expect(decision.cause).toBe("session_affinity_pin");
    expect(decision.chosenModel).toBe("anthropic/opus");
  });

  it("ignores an expired pin", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true });
    const expired = { ...pin, expiresAt: now() - 1 };
    const { decision } = await route({ turn: turn("hi"), config: cfg, api, pin: expired, now });
    expect(decision.cause).toBe("heuristic_scorer");
  });

  it("re-pins higher on escalation", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true, escalationKeywords: ["ESCALATE"] });
    const lowPin = { model: "anthropic/haiku", tier: "SIMPLE" as const, expiresAt: now() + 60_000 };
    const { decision, pinToWrite } = await route({ turn: turn("hi ESCALATE"), config: cfg, api, pin: lowPin, now });
    expect(decision.cause).toBe("session_affinity_escalation");
    expect(decision.tier).toBe("MEDIUM");
    expect(pinToWrite?.tier).toBe("MEDIUM");
  });

  it("floors a pinned session in plan mode without rewriting the pin", async () => {
    // The turns in plan mode route at the floor; the pin keeps the session's own model so
    // the first turn after plan mode exits routes as if it never happened.
    const api = applier();
    const cfg = config({ sessionAffinity: true, planMode: { minTier: "REASONING" } });
    const lowPin = { model: "anthropic/haiku", tier: "SIMPLE" as const, expiresAt: now() + 60_000 };
    const { decision, pinToWrite } = await route({
      turn: turn("hi"),
      config: cfg,
      api,
      pin: lowPin,
      planModeActive: true,
      now,
    });
    expect(decision.cause).toBe("plan_mode");
    expect(decision.chosenModel).toBe("anthropic/opus");
    expect(pinToWrite).toBeNull();
  });

  it("does not write a pin on a plan-mode turn", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true, planMode: { minTier: "MEDIUM" } });
    const { pinToWrite } = await route({ turn: turn("hi"), config: cfg, api, planModeActive: true, now });
    expect(pinToWrite).toBeNull();
  });

  it("writes a pin on an ordinary turn when affinity is on", async () => {
    const api = applier();
    const cfg = config({ sessionAffinity: true });
    const { pinToWrite } = await route({ turn: turn("hi"), config: cfg, api, now });
    expect(pinToWrite?.model).toBe("anthropic/haiku");
    expect(pinToWrite?.expiresAt).toBe(now() + 3600 * 1000);
  });

  it("writes no pin when affinity is off", async () => {
    const { pinToWrite } = await route({ turn: turn("hi"), config: config(), api: applier(), now });
    expect(pinToWrite).toBeNull();
  });
});

describe("route — safety rails", () => {
  it("falls through to a lower tier when the first choice has no credentials", async () => {
    const api = applier({
      setModel: async (model) => {
        const m = model as unknown as { id: string };
        return m.id !== "opus"; // no key for opus
      },
    });
    const { decision } = await route({
      turn: turn("think step by step and analyze this: weigh the options carefully"),
      config: config(),
      api,
      now,
    });
    expect(decision.tier).toBe("REASONING");
    expect(decision.chosenModel).toBe("anthropic/sonnet");
    expect(decision.fellBackBecause).toContain("no available credentials");
  });

  it("skips a model that is not in the registry", async () => {
    const api = applier({ find: (_p, id) => (id === "opus" ? undefined : { provider: "anthropic", id }) });
    const { decision } = await route({
      turn: turn("think step by step and analyze this: weigh the options carefully"),
      config: config(),
      api,
      now,
    });
    expect(decision.chosenModel).toBe("anthropic/sonnet");
    expect(decision.fellBackBecause).toContain("not in pi's model registry");
  });

  it("leaves the model unchanged when nothing is usable", async () => {
    const api = applier({ setModel: async () => false });
    const { decision } = await route({ turn: turn("hi"), config: config(), api, now });
    expect(decision.chosenModel).toBeNull();
    expect(decision.fellBackBecause).toBeTruthy();
  });

  it("does not throw when setModel throws", async () => {
    const api = applier({
      setModel: async () => {
        throw new Error("provider exploded");
      },
    });
    const { decision } = await route({ turn: turn("hi"), config: config(), api, now });
    expect(decision.chosenModel).toBeNull();
    expect(decision.fellBackBecause).toContain("provider exploded");
  });

  it("applies the tier's thinking level after the model", async () => {
    const order: string[] = [];
    const api: ModelApplier = {
      find: (provider, modelId) => ({ provider, id: modelId }),
      setModel: async () => {
        order.push("setModel");
        return true;
      },
      setThinkingLevel: () => order.push("setThinkingLevel"),
    };
    const cfg = config({
      tiers: { ...TIERS, SIMPLE: { model: "anthropic/haiku", thinkingLevel: "high" } },
    });
    const { decision } = await route({ turn: turn("hi"), config: cfg, api, now });
    // pi clamps the level to the model's capabilities, so the order matters.
    expect(order).toEqual(["setModel", "setThinkingLevel"]);
    expect(decision.thinkingLevel).toBe("high");
  });
});

describe("route — llm classifier", () => {
  const llmConfig = (extra: Record<string, unknown> = {}) =>
    config({
      classifierType: "llm",
      classifierLLMConfig: { model: "anthropic/haiku", timeoutMs: 50 },
      ...extra,
    });

  it("uses the tier the classifier names", async () => {
    const registry = {
      find: () => ({ provider: "anthropic", id: "haiku" }),
      complete: async () => ({ content: "REASONING" }),
    };
    const { decision } = await route({
      turn: turn("anything"),
      config: llmConfig(),
      api: applier(),
      registry: registry as never,
      now,
    });
    expect(decision.cause).toBe("llm_classifier");
    expect(decision.tier).toBe("REASONING");
  });

  it("falls back to the heuristic scorer when the classifier fails", async () => {
    const registry = {
      find: () => ({ provider: "anthropic", id: "haiku" }),
      complete: async () => {
        throw new Error("upstream 500");
      },
    };
    const { decision } = await route({
      turn: turn("hi"),
      config: llmConfig(),
      api: applier(),
      registry: registry as never,
      now,
    });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals.some((s) => s.includes("llm_fallback"))).toBe(true);
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });

  it("falls back to defaultModel when configured to", async () => {
    const registry = {
      find: () => ({ provider: "anthropic", id: "haiku" }),
      complete: async () => ({ content: "not a tier at all" }),
    };
    const { decision } = await route({
      turn: turn("hi"),
      config: llmConfig({ classifierFallback: "default_model" }),
      api: applier(),
      registry: registry as never,
      now,
    });
    expect(decision.cause).toBe("default_model_fallback");
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });

  it("routes into the floor's pool, not defaultModel, when a plan-mode turn fails classification", async () => {
    // defaultModel carries no tier guarantee, so it cannot serve a floored request.
    const registry = {
      find: () => ({ provider: "anthropic", id: "haiku" }),
      complete: async () => ({ content: "gibberish" }),
    };
    const { decision } = await route({
      turn: turn("hi"),
      config: llmConfig({ classifierFallback: "default_model", planMode: { minTier: "COMPLEX" } }),
      api: applier(),
      registry: registry as never,
      planModeActive: true,
      now,
    });
    expect(decision.tier).toBe("COMPLEX");
    expect(decision.chosenModel).toBe("anthropic/sonnet");
  });

  it("abandons a classifier that exceeds its timeout", async () => {
    const registry = {
      find: () => ({ provider: "anthropic", id: "haiku" }),
      complete: (_m: unknown, _c: unknown, options?: { signal?: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          options?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    };
    const { decision } = await route({
      turn: turn("hi"),
      config: llmConfig(),
      api: applier(),
      registry: registry as never,
      now,
    });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals.some((s) => s.includes("timed out"))).toBe(true);
  });
});

describe("defaultModel thinking level", () => {
  const withLevel = () =>
    buildConfig([{ defaultModel: { model: "a/haiku", thinkingLevel: "low" }, tiers: { MEDIUM: "a/sonnet" } }]).config;

  it("applies it when classification falls back to defaultModel", async () => {
    const levels: string[] = [];
    const api = applier({ setThinkingLevel: (level) => levels.push(level) });
    const { decision } = await route({ turn: turn(null), config: withLevel(), api, now });
    expect(decision.cause).toBe("default_fallback");
    expect(decision.chosenModel).toBe("a/haiku");
    expect(decision.thinkingLevel).toBe("low");
    expect(levels).toEqual(["low"]);
  });

  it("applies it when an unconfigured tier routes to defaultModel", async () => {
    const levels: string[] = [];
    const api = applier({ setThinkingLevel: (level) => levels.push(level) });
    const { decision } = await route({ turn: turn("hi"), config: withLevel(), api, now });
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.chosenModel).toBe("a/haiku");
    expect(decision.thinkingLevel).toBe("low");
  });

  it("carries it on the credential rail too", () => {
    const candidates = candidatesForTier("MEDIUM", withLevel(), () => 0);
    expect(candidates).toEqual([{ model: "a/sonnet" }, { model: "a/haiku", thinkingLevel: "low" }]);
  });
});

describe("candidatesForTier — upstream's get_model_for_tier", () => {
  const POOLS = {
    SIMPLE: "a/haiku",
    MEDIUM: ["a/sonnet", "a/mini"],
    COMPLEX: ["a/sonnet", { model: "a/opus", thinkingLevel: "medium" }, "a/gpt"],
  };
  const built = () => buildConfig([{ defaultModel: "a/haiku", tiers: POOLS }]).config;

  it("picks uniformly from a list pool, like random.choice", () => {
    const models = (draw: number) => candidatesForTier("COMPLEX", built(), () => draw).map((t) => t.model);
    expect(models(0)[0]).toBe("a/sonnet");
    expect(models(0.5)[0]).toBe("a/opus");
    expect(models(0.99)[0]).toBe("a/gpt");
    // The rest of the pool, lower tiers, then defaultModel follow as pi's credential rail.
    expect(models(0.5)).toEqual(["a/opus", "a/sonnet", "a/gpt", "a/mini", "a/haiku"]);
  });

  it("returns a single-model tier as-is", () => {
    expect(candidatesForTier("SIMPLE", built(), () => 0.7)[0]).toEqual({ model: "a/haiku" });
  });

  it("sends an unconfigured tier to defaultModel, then to MEDIUM", () => {
    expect(candidatesForTier("REASONING", built(), () => 0.9)[0]).toEqual({ model: "a/haiku" });
    const noDefault = buildConfig([{ tiers: POOLS }]).config;
    expect(candidatesForTier("REASONING", noDefault, () => 0.9)[0]).toEqual({ model: "a/mini" });
  });

  it("routes through the injected rng end to end", async () => {
    const config = buildConfig([{ defaultModel: "a/haiku", tiers: { ...TIERS, SIMPLE: ["a/haiku", "a/mini"] } }]).config;
    const api = applier();
    const low = await route({ turn: turn("hi"), config, api, now, rng: () => 0 });
    const high = await route({ turn: turn("hi"), config, api, now, rng: () => 0.9 });
    expect(low.decision.chosenModel).toBe("a/haiku");
    expect(high.decision.chosenModel).toBe("a/mini");
  });
});
