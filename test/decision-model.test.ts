/**
 * The decision-model classifier (`classifierType: "decision"`): Jev, Clef, or any other
 * classifier model pi supports, called through pi's model registry.
 */

import { describe, expect, it, vi } from "vitest";
import { buildConfig } from "../src/config.ts";
import {
  CLASSIFIER_CIRCUIT_OPEN_SIGNAL,
  ClassifierCircuitBreaker,
  DEFAULT_DECISION_INSTRUCTIONS,
  DECISION_TIER_CRITERIA,
  type DecisionResult,
  classifyWithDecisionModel,
} from "../src/classify/decision-model.ts";
import { classifierFallbackNotice, explain } from "../src/decision.ts";
import { route } from "../src/router.ts";
import type { ModelLookup } from "../src/resolve.ts";
import type { ExtractedTurn } from "../src/types.ts";

const TIERS = {
  SIMPLE: "anthropic/haiku",
  MEDIUM: "anthropic/sonnet",
  COMPLEX: "anthropic/sonnet",
  REASONING: "anthropic/opus",
};

function config(settings: Record<string, unknown> = {}, extra: Record<string, unknown> = {}) {
  const built = buildConfig([
    { defaultModel: "anthropic/haiku", tiers: TIERS, classifierType: "decision", decisionClassifierConfig: settings, ...extra },
  ]);
  expect(built.errors).toEqual([]);
  return built.config;
}

function turn(currentAsk: string | null, extra: Partial<ExtractedTurn> = {}): ExtractedTurn {
  return { currentAsk, priorTurns: [], conversationContinuing: false, cumulativeTokens: 0, ...extra };
}

function lookup(): ModelLookup {
  return { find: (provider, modelId) => ({ provider, id: modelId }), hasConfiguredAuth: () => true };
}

const choice = (label: string, probabilities: Record<string, number> = { [label]: 0.9 }, confidence = 0.9) => ({
  type: "choice",
  choice: label,
  probabilities,
  confidence,
});

/** pi's registry, with the classifier catalog and `classify()` faked. */
function registry(
  answer: (context: unknown, options?: { signal?: AbortSignal }) => Promise<DecisionResult> | DecisionResult,
  catalog: string[] = ["typesafe/jev-latest", "openrouter/cloudflare/clef", "cloudflare-workers-ai/@cf/cloudflare/clef"],
) {
  const calls: { model: { provider: string; id: string }; context: unknown }[] = [];
  return {
    calls,
    findOfType: vi.fn((type: string, provider: string, id: string) =>
      type === "classifier" && catalog.includes(`${provider}/${id}`) ? { provider, id, type: "classifier" } : undefined,
    ),
    classify: vi.fn(async (model: never, context: never, options?: { signal?: AbortSignal }) => {
      calls.push({ model, context });
      return answer(context, options);
    }),
    find: vi.fn(),
    complete: vi.fn(),
  };
}

const ok = (label: string, extra: Partial<DecisionResult> = {}): DecisionResult => ({
  provider: "typesafe",
  model: "jev-latest",
  answers: { tier: choice(label, { SIMPLE: 0.05, MEDIUM: 0.1, COMPLEX: 0.8, REASONING: 0.05 }, 0.8) },
  stopReason: "stop",
  ...extra,
});

describe("decisionClassifierConfig", () => {
  const base = { defaultModel: "a/haiku", tiers: { SIMPLE: "a/haiku" } };

  it("defaults to TypeSafe's hosted Jev", () => {
    const { config, errors } = buildConfig([{ ...base, classifierType: "decision", decisionClassifierConfig: {} }]);
    expect(errors).toEqual([]);
    expect(config.decisionClassifierConfig).toEqual({
      model: "typesafe/jev-latest",
      timeoutMs: 3000,
      circuitBreakerEnabled: true,
      circuitBreakerCooldownSeconds: 30,
    });
  });

  it("names any classifier model pi supports with one model field", () => {
    const { config, errors } = buildConfig([
      {
        ...base,
        classifierType: "decision",
        decisionClassifierConfig: { model: " openrouter/cloudflare/clef ", timeoutMs: 15000 },
      },
    ]);
    expect(errors).toEqual([]);
    expect(config.decisionClassifierConfig?.model).toBe("openrouter/cloudflare/clef");
    expect(config.decisionClassifierConfig?.timeoutMs).toBe(15000);
  });

  it("requires the block for decision", () => {
    expect(buildConfig([{ ...base, classifierType: "decision" }]).errors).toEqual([
      'decisionClassifierConfig is required when classifierType is "decision"',
    ]);
  });

  it("only warns about a block another classifierType ignores", () => {
    const { errors, warnings } = buildConfig([{ ...base, classifierType: "heuristic", decisionClassifierConfig: {} }]);
    expect(errors).toEqual([]);
    expect(warnings).toContain('decisionClassifierConfig has no effect unless classifierType is "decision"');
  });

  it("validates every field, and forbids unknown ones", () => {
    const errors = buildConfig([
      {
        ...base,
        classifierType: "decision",
        decisionClassifierConfig: {
          model: "jev-latest",
          timeoutMs: 1.5,
          instructions: "  ",
          circuitBreakerEnabled: "yes",
          circuitBreakerCooldownSeconds: 0,
          timeout_ms: 3000,
        },
      },
    ]).errors;
    expect(errors).toEqual([
      "decisionClassifierConfig.timeout_ms is not a known setting",
      "decisionClassifierConfig.model must name a classifier model as provider/model-id, e.g. typesafe/jev-latest",
      "decisionClassifierConfig.timeoutMs must be a positive integer",
      "decisionClassifierConfig.instructions must be non-empty; omit it to use the default",
      "decisionClassifierConfig.circuitBreakerEnabled must be a boolean",
      "decisionClassifierConfig.circuitBreakerCooldownSeconds must be a number greater than 0",
    ]);
  });

  it("points connection settings at where pi keeps them", () => {
    const errors = buildConfig([
      {
        ...base,
        classifierType: "decision",
        decisionClassifierConfig: { apiKey: "k", apiBase: "https://x" },
      },
    ]).errors;
    expect(errors).toEqual([
      "decisionClassifierConfig.apiKey is not supported here: pi owns the connection; set the provider's key with /login or its environment variable",
      "decisionClassifierConfig.apiBase is not supported here: pi owns the connection; set the provider's baseUrl in models.json",
    ]);
  });

  it("rejects a classifierType it does not know", () => {
    expect(buildConfig([{ ...base, classifierType: "jev" }]).errors).toEqual([
      'classifierType must be "heuristic", "llm" or "decision"',
    ]);
  });
});

describe("classifyWithDecisionModel", () => {
  it("asks one System One choice question with the built-in instructions and criteria", async () => {
    const reg = registry(() => ok("COMPLEX"));
    const outcome = await classifyWithDecisionModel(
      turn("refactor the auth layer", {
        priorTurns: [{ role: "user", text: "plan the auth redesign" }],
        conversationContinuing: true,
        cumulativeTokens: 120,
      }),
      config(),
      reg,
      null,
    );

    expect(reg.findOfType).toHaveBeenCalledWith("classifier", "typesafe", "jev-latest");
    expect(reg.calls).toHaveLength(1);
    expect(reg.calls[0]?.context).toEqual({
      // The LLM classifier's user payload, sent as the state string.
      state:
        "\nRecent conversation (context only, do not classify these):\n[1] plan the auth redesign\n" +
        "\nConversation so far: ~120 tokens across the request\n\nClassify this message:\nrefactor the auth layer",
      questions: { tier: { type: "choice", instructions: DEFAULT_DECISION_INSTRUCTIONS, criteria: DECISION_TIER_CRITERIA } },
    });
    expect(outcome).toEqual({
      ok: true,
      classification: {
        tier: "COMPLEX",
        cause: "decision_classifier",
        classifierModel: "typesafe/jev-latest",
        signals: [
          "typesafe-classifier:COMPLEX",
          "typesafe-confidence=0.800000",
          "tier-probability:SIMPLE=0.050000",
          "tier-probability:MEDIUM=0.100000",
          "tier-probability:COMPLEX=0.800000",
          "tier-probability:REASONING=0.050000",
        ],
      },
    });
  });

  it("routes Clef, reports its cost, reports its cost, and honours custom instructions", async () => {
    const reg = registry(() =>
      ok("reasoning", { provider: "openrouter", model: "cloudflare/clef", usage: { cost: { total: 0.00012 } } }),
    );
    const outcome = await classifyWithDecisionModel(
      turn("weigh these tradeoffs"),
      config({ model: "openrouter/cloudflare/clef", instructions: "Rate the work." }),
      reg,
      null,
    );

    expect(reg.findOfType).toHaveBeenCalledWith("classifier", "openrouter", "cloudflare/clef");
    expect((reg.calls[0]?.context as { questions: { tier: { instructions: string } } }).questions.tier.instructions).toBe(
      "Rate the work.",
    );
    // Labels resolve case-insensitively.
    expect(outcome.ok && outcome.classification.tier).toBe("REASONING");
    expect(outcome.ok && outcome.classification.classifierModel).toBe("openrouter/cloudflare/clef");
    expect(outcome.ok && outcome.classification.classifierCost).toBe(0.00012);
  });

  it("prefixes its signals with the pi provider that answered", async () => {
    const reg = registry(() => ok("MEDIUM", { provider: "openrouter", model: "cloudflare/clef" }));
    const outcome = await classifyWithDecisionModel(
      turn("hi"),
      config({ model: "openrouter/cloudflare/clef" }),
      reg,
      null,
    );
    expect(reg.findOfType).toHaveBeenCalledWith("classifier", "openrouter", "cloudflare/clef");
    expect(outcome.ok && outcome.classification.signals[0]).toBe("openrouter-classifier:MEDIUM");
  });

  it("fails on a model pi has no classifier for, without calling anything", async () => {
    const reg = registry(() => ok("SIMPLE"));
    const outcome = await classifyWithDecisionModel(turn("hi"), config({ model: "laya/english" }), reg, null);
    expect(outcome).toEqual({ ok: false, reason: 'decision classifier model "laya/english" is not a classifier model pi knows', signals: [] });
    expect(reg.classify).not.toHaveBeenCalled();
  });

  it.each([
    ["an error result", { stopReason: "error", errorMessage: "401 Unauthorized" }, "decision classifier failed (401 Unauthorized)"],
    ["a missing tier answer", { stopReason: "stop", answers: {} }, "decision classifier failed (response is missing the 'tier' answer)"],
    ["an unknown label", { stopReason: "stop", answers: { tier: choice("EXPERT") } }, 'decision classifier failed (returned unknown tier "EXPERT")'],
  ] as const)("fails on %s", async (_name, result, reason) => {
    const outcome = await classifyWithDecisionModel(turn("hi"), config(), registry(() => result as DecisionResult), null);
    expect(outcome).toEqual({ ok: false, reason, signals: [] });
  });

  it("fails on a tier that has no models", async () => {
    const built = buildConfig([
      { tiers: { SIMPLE: "a/haiku" }, classifierType: "decision", decisionClassifierConfig: {} },
    ]).config;
    const outcome = await classifyWithDecisionModel(turn("hi"), built, registry(() => ok("COMPLEX")), null);
    expect(outcome).toEqual({
      ok: false,
      reason: "decision classifier failed (returned tier COMPLEX, which has no models configured)",
      signals: [],
    });
  });

  it("abandons a classifier that exceeds its timeout", async () => {
    const reg = registry(
      (_context, options) =>
        new Promise((resolve) =>
          options?.signal?.addEventListener("abort", () => resolve({ stopReason: "aborted", errorMessage: "aborted" })),
        ),
    );
    const outcome = await classifyWithDecisionModel(turn("hi"), config({ timeoutMs: 10 }), reg, null);
    expect(outcome).toEqual({ ok: false, reason: "decision classifier timed out after 10ms", signals: [] });
  });
});

describe("ClassifierCircuitBreaker", () => {
  it("opens on a timeout, skips the classifier for the cooldown, then lets one probe through", async () => {
    let now = 0;
    const breaker = new ClassifierCircuitBreaker(30, () => now);
    const slow = registry(
      (_context, options) =>
        new Promise((resolve) => options?.signal?.addEventListener("abort", () => resolve({ stopReason: "aborted" }))),
    );
    const cfg = config({ timeoutMs: 5 });

    expect((await classifyWithDecisionModel(turn("hi"), cfg, slow, breaker)).ok).toBe(false);
    expect(breaker.isOpen()).toBe(true);

    const fast = registry(() => ok("SIMPLE"));
    now = 29;
    expect(await classifyWithDecisionModel(turn("hi"), cfg, fast, breaker)).toEqual({
      ok: false,
      reason: "decision classifier circuit is open",
      signals: [CLASSIFIER_CIRCUIT_OPEN_SIGNAL],
    });
    expect(fast.classify).not.toHaveBeenCalled();

    now = 30;
    expect((await classifyWithDecisionModel(turn("hi"), cfg, fast, breaker)).ok).toBe(true);
    expect(breaker.isOpen()).toBe(false);
  });

  it("does not open on an ordinary error", async () => {
    const breaker = new ClassifierCircuitBreaker(30, () => 0);
    await classifyWithDecisionModel(turn("hi"), config(), registry(() => ({ stopReason: "error" })), breaker);
    expect(breaker.isOpen()).toBe(false);
  });

  it("reopens when the recovery probe fails for any reason", () => {
    let now = 0;
    const breaker = new ClassifierCircuitBreaker(30, () => now);
    breaker.recordFailure(breaker.acquirePermit()!, true);
    now = 31;
    const probe = breaker.acquirePermit()!;
    // Only one probe at a time.
    expect(breaker.acquirePermit()).toBeNull();
    breaker.recordFailure(probe, false);
    expect(breaker.acquirePermit()).toBeNull();
    now = 61;
    expect(breaker.acquirePermit()).not.toBeNull();
  });

  it("ignores a stale permit admitted before the circuit opened", () => {
    let now = 0;
    const breaker = new ClassifierCircuitBreaker(30, () => now);
    const early = breaker.acquirePermit()!;
    const other = breaker.acquirePermit()!;
    breaker.recordFailure(other, true);
    now = 31;
    breaker.acquirePermit();
    // The early call finishing late must not close the circuit the probe now owns.
    breaker.recordSuccess(early);
    expect(breaker.isOpen()).toBe(true);
  });
});

describe("route — decision", () => {
  it("routes on the decision model's tier and records which model decided", async () => {
    const { decision } = await route({
      turn: turn("refactor the auth layer"),
      config: config(),
      models: lookup(),
      registry: registry(() => ok("COMPLEX", { usage: { cost: { total: 0.0001 } } })) as never,
      now: () => 0,
    });
    expect(decision.cause).toBe("decision_classifier");
    expect(decision.tier).toBe("COMPLEX");
    expect(decision.chosenModel).toBe("anthropic/sonnet");
    expect(decision.classifierModel).toBe("typesafe/jev-latest");
    expect(explain(decision)).toContain("classifier: typesafe/jev-latest ($0.000100)");
  });

  it("falls back to the heuristic scorer, and says why", async () => {
    const { decision } = await route({
      turn: turn("hi"),
      config: config(),
      models: lookup(),
      registry: registry(() => ({ stopReason: "error", errorMessage: "no API key for typesafe" })) as never,
      now: () => 0,
    });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals).toContain("decision_fallback (decision classifier failed (no API key for typesafe))");
    expect(classifierFallbackNotice(decision)).toBe(
      "autoroute: decision classifier unavailable; used heuristic classification instead (decision classifier failed (no API key for typesafe)).",
    );
  });

  it("falls back to defaultModel when configured to", async () => {
    const { decision } = await route({
      turn: turn("hi"),
      config: config({}, { classifierFallback: "default_model" }),
      models: lookup(),
      registry: registry(() => ({ stopReason: "error", errorMessage: "boom" })) as never,
      now: () => 0,
    });
    expect(decision.cause).toBe("default_model_fallback");
    expect(decision.chosenModel).toBe("anthropic/haiku");
    expect(decision.signals).toContain("decision classifier failed (boom)");
  });

  it("keeps quiet while the circuit is open", async () => {
    const breaker = new ClassifierCircuitBreaker(30, () => 0);
    breaker.recordFailure(breaker.acquirePermit()!, true);
    const { decision } = await route({
      turn: turn("hi"),
      config: config(),
      models: lookup(),
      registry: registry(() => ok("SIMPLE")) as never,
      decisionBreaker: breaker,
      now: () => 0,
    });
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals).toContain(CLASSIFIER_CIRCUIT_OPEN_SIGNAL);
    expect(classifierFallbackNotice(decision)).toBeNull();
  });
});
