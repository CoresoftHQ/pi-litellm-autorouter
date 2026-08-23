import { describe, expect, it } from "vitest";
import {
  COLD_START_MASS,
  SAMPLE_CAP,
  applyDelta,
  betaVariate,
  cellMean,
  cellTotalSamples,
  initialCell,
  normalizedCost,
} from "../src/adaptive/bandit.ts";
import { buildAdaptiveRouter, turnFromRun } from "../src/adaptive/index.ts";
import { classifyRequestType } from "../src/adaptive/request-type.ts";
import { AdaptiveRouter } from "../src/adaptive/router.ts";
import { softFloorPick, targetForModel, tiersForModel } from "../src/adaptive/select.ts";
import {
  type Turn,
  banditDelta,
  detectResponseSignals,
  detectUserFeedback,
  toolCallSignature,
} from "../src/adaptive/signals.ts";
import { buildConfig } from "../src/config.ts";
import type { ModelApplier } from "../src/resolve.ts";
import { route } from "../src/router.ts";
import type { ExtractedTurn, RequestType } from "../src/types.ts";

/** A deterministic uniform generator: cycles through the given draws. */
function sequence(...draws: number[]): () => number {
  let i = 0;
  return () => {
    const draw = draws[i % draws.length] ?? 0.5;
    i++;
    return draw;
  };
}

/** A seeded xorshift, for tests that need many varied draws without flakiness. */
function seeded(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

function turn(currentAsk: string | null, extra: Partial<ExtractedTurn> = {}): ExtractedTurn {
  return { currentAsk, priorTurns: [], conversationContinuing: false, cumulativeTokens: 0, ...extra };
}

function applier(): ModelApplier & { applied: string[] } {
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
  } as ModelApplier & { applied: string[] };
}

/** Mirrors upstream's `hybrid_config` fixture: cheap lives in the low tiers, premium in the high. */
const HYBRID = {
  adaptive: true,
  adaptiveWeights: { quality: 0.7, cost: 0.3 },
  tierDistancePenalty: 0.15,
  tiers: {
    SIMPLE: [{ model: "a/cheap", qualityTier: 1 }],
    MEDIUM: ["a/cheap"],
    COMPLEX: [{ model: "a/premium", qualityTier: 3 }],
    REASONING: ["a/premium"],
  },
  defaultModel: "a/cheap",
};

const COSTS = new Map([
  ["a/cheap", 0.15],
  ["a/premium", 5],
]);

function hybridConfig(overrides: Record<string, unknown> = {}) {
  const built = buildConfig([{ ...HYBRID, ...overrides }]);
  expect(built.errors).toEqual([]);
  return built.config;
}

function costRegistry() {
  return {
    find: (provider: string, modelId: string) => ({
      provider,
      id: modelId,
      cost: { input: COSTS.get(`${provider}/${modelId}`) ?? 0 },
    }),
  };
}

/** Seed a cell with so much mass that a Thompson draw is effectively its mean. */
function pin(adaptive: AdaptiveRouter, requestType: RequestType, model: string, mean: number) {
  const mass = 1_000_000;
  adaptive.setCell(requestType, model, { alpha: mean * mass, beta: (1 - mean) * mass });
}

describe("bandit", () => {
  it("builds the cold-start prior from quality tier and strengths", () => {
    const plain = initialCell({ qualityTier: 2, strengths: [] }, "general");
    expect(cellMean(plain)).toBeCloseTo(0.5);
    expect(plain.alpha + plain.beta).toBeCloseTo(COLD_START_MASS);
    expect(cellTotalSamples(plain)).toBe(0);

    const strong = initialCell({ qualityTier: 3, strengths: ["code_generation"] }, "code_generation");
    expect(cellMean(strong)).toBeCloseTo(0.95); // 0.7 + 0.3 capped
    const elsewhere = initialCell({ qualityTier: 3, strengths: ["code_generation"] }, "writing");
    expect(cellMean(elsewhere)).toBeCloseTo(0.7);
  });

  it("applies deltas until the sample cap, then drops them", () => {
    let cell = initialCell({ qualityTier: 2, strengths: [] }, "general");
    cell = applyDelta(cell, 1, 0);
    expect(cellTotalSamples(cell)).toBe(1);
    const saturated = { alpha: SAMPLE_CAP - 1, beta: 0.5 };
    expect(applyDelta(saturated, 1, 0)).toBe(saturated);
  });

  it("samples Beta variates in [0, 1] whose mean tracks the posterior", () => {
    const rng = seeded(42);
    const draws = Array.from({ length: 4000 }, () => betaVariate(8, 2, rng));
    expect(draws.every((d) => d >= 0 && d <= 1)).toBe(true);
    const mean = draws.reduce((a, b) => a + b, 0) / draws.length;
    expect(mean).toBeCloseTo(0.8, 1);
    // Shape < 1 takes the boost path.
    const small = betaVariate(0.5, 0.5, rng);
    expect(small).toBeGreaterThanOrEqual(0);
    expect(small).toBeLessThanOrEqual(1);
  });

  it("normalises cost so the cheapest model scores 1 and no spread scores 0.5", () => {
    expect(normalizedCost(1, [1, 3])).toBe(1);
    expect(normalizedCost(3, [1, 3])).toBe(0);
    expect(normalizedCost(2, [2, 2])).toBe(0.5);
    expect(normalizedCost(2, [])).toBe(0.5);
  });
});

describe("classifyRequestType", () => {
  it.each<[string, RequestType]>([
    ["write a python script to parse logs", "code_generation"],
    ["implement a function that dedupes a list", "code_generation"],
    ["explain what this function does", "code_understanding"],
    ["fix the error in the stack trace below", "code_understanding"],
    ["review my pull request", "code_understanding"],
    ["design a system for rate limiting the api", "technical_design"],
    ["should I use postgres or mongodb for this", "technical_design"],
    ["solve the equation x^2 = 4", "analytical_reasoning"],
    ["draft an email to the team", "writing"],
    ["make this more concise", "writing"],
    ["what is the capital of France", "factual_lookup"],
    ["hi", "general"],
    ["", "general"],
  ])("%s → %s", (text, expected) => {
    expect(classifyRequestType(text)).toBe(expected);
  });
});

describe("signals", () => {
  const none = { toolResults: [], toolCalls: [] };

  it("reads a rephrase as misalignment but not a topic change or a repeat", () => {
    const rephrase = detectUserFeedback("fix the login bug in auth.ts", "why is login still broken after your change", [], false);
    expect(rephrase.misalignment).toBe(1);
    const topicChange = detectUserFeedback("fix the login bug", "write docs", [], false);
    expect(topicChange.misalignment).toBe(0);
    const repeat = detectUserFeedback("fix the login bug", "fix the login bug", [], false);
    expect(repeat.misalignment).toBe(0);
  });

  it("gates satisfaction on the caller's say-so", () => {
    expect(detectUserFeedback(null, "thanks, that worked", [], false).satisfaction).toBe(0);
    expect(detectUserFeedback(null, "thanks, that worked", [], true).satisfaction).toBe(1);
  });

  it("detects disengagement and tool failure", () => {
    expect(detectUserFeedback(null, "forget it, I'll do it myself", [], false).disengagement).toBe(1);
    expect(detectUserFeedback(null, "ok", [{ content: "", isError: true }], false).failure).toBe(1);
    expect(detectUserFeedback(null, "ok", [{ content: "", isError: false }], false).failure).toBe(0);
  });

  it("detects stagnation, loops and exhaustion on the response side", () => {
    const same = "I have updated the file as requested and the tests now pass";
    expect(detectResponseSignals(same, same, [], [], [], 200).stagnation).toBe(1);
    expect(detectResponseSignals("one thing", "entirely different reply", [], [], [], 200).stagnation).toBe(0);

    const call = { name: "bash", arguments: { command: "ls" } };
    const history = [toolCallSignature(call), toolCallSignature(call)];
    expect(detectResponseSignals(null, null, history, [call], [], 200).loop).toBe(1);
    expect(detectResponseSignals(null, null, history.slice(0, 1), [call], [], 200).loop).toBe(0);

    expect(detectResponseSignals(null, null, [], [], [], 429).exhaustion).toBe(1);
    expect(detectResponseSignals(null, null, [], [], [{ content: "Rate limit exceeded", isError: false }], 200).exhaustion).toBe(1);
    expect(detectResponseSignals(null, null, [], [], [], 500).exhaustion).toBe(0);
  });

  it("maps signals onto the posterior as upstream does", () => {
    expect(banditDelta({ misalignment: 1, stagnation: 1, disengagement: 0, satisfaction: 1, failure: 1, loop: 1, exhaustion: 1 })).toEqual([
      1, 3.5,
    ]);
    void none;
  });
});

describe("AdaptiveRouter.recordTurn", () => {
  const models = ["a/cheap", "a/premium"];
  function router(now = () => 1_000) {
    return new AdaptiveRouter({
      availableModels: models,
      modelToPrefs: new Map(),
      modelToCost: COSTS,
      now,
      rng: sequence(0.5),
    });
  }
  const turnOf = (user: string, assistant: string, extra: Partial<Turn> = {}): Turn => ({
    userContent: user,
    assistantContent: assistant,
    toolCalls: [],
    toolResults: [],
    responseStatus: 200,
    ...extra,
  });

  it("credits feedback in the next prompt to the model that produced the previous reply", () => {
    const r = router();
    const before = r.cell("general", "a/cheap");
    r.recordTurn("s1", "a/cheap", "general", turnOf("fix the login bug in auth.ts", "done"));
    // A rephrase of the same ask: misalignment against cheap, even though premium answers now.
    const delta = r.recordTurn("s1", "a/premium", "general", turnOf("why is login still broken after your change", "ok"));
    expect(delta.misalignment).toBe(1);
    expect(r.cell("general", "a/cheap").beta).toBeCloseTo(before.beta + 1);
    expect(r.cell("general", "a/premium")).toEqual(initialCell({ qualityTier: 2, strengths: [] }, "general"));
    expect(r.crossModelFeedbackTotal).toBe(1);
  });

  it("does not award a thanks before the clean-credit turn count", () => {
    const r = router();
    const prior = r.cell("general", "a/cheap").alpha;
    r.recordTurn("s1", "a/cheap", "general", turnOf("do x", "done"));
    r.recordTurn("s1", "a/cheap", "general", turnOf("thanks!", "you're welcome"));
    expect(r.cell("general", "a/cheap").alpha).toBe(prior);
    r.recordTurn("s1", "a/cheap", "general", turnOf("now do y", "done"));
    r.recordTurn("s1", "a/cheap", "general", turnOf("perfect, thanks", "np"));
    expect(r.cell("general", "a/cheap").alpha).toBeCloseTo(prior + 1);
  });

  it("charges a tool-call loop to the current model", () => {
    const r = router();
    const call = { name: "bash", arguments: { command: "ls" } };
    const before = r.cell("general", "a/cheap").beta;
    r.recordTurn("s1", "a/cheap", "general", turnOf("list files", "here are the files", { toolCalls: [call, call] }));
    r.recordTurn("s1", "a/cheap", "general", turnOf("and again", "same output as before", { toolCalls: [call] }));
    expect(r.cell("general", "a/cheap").beta).toBeCloseTo(before + 0.5);
  });

  it("lets a general follow-up inherit the previous turn's request type", () => {
    const r = router();
    r.recordTurn("s1", "a/cheap", "code_generation", turnOf("write a python script", "here"));
    const before = r.cell("code_generation", "a/cheap").beta;
    r.recordTurn("s1", "a/cheap", "general", turnOf("forget it", "ok"));
    expect(r.cell("code_generation", "a/cheap").beta).toBeCloseTo(before + 1);
  });

  it("round-trips its posteriors and ignores rows for models it no longer routes to", () => {
    const r = router();
    r.setCell("writing", "a/cheap", { alpha: 12, beta: 4 });
    const persisted = r.serialize();
    persisted.cells.push({ requestType: "writing", model: "a/gone", alpha: 1, beta: 1 });
    persisted.cells.push({ requestType: "bogus" as RequestType, model: "a/cheap", alpha: 1, beta: 1 });
    const fresh = router();
    expect(fresh.load(persisted)).toBe(models.length * 7);
    expect(fresh.cell("writing", "a/cheap")).toEqual({ alpha: 12, beta: 4 });
    expect(fresh.availableModels).toEqual(models);
    expect(fresh.load(null)).toBe(0);
  });
});

describe("softFloorPick", () => {
  it("uses the config defaults upstream ships", () => {
    const config = buildConfig([{ adaptive: true, tiers: { SIMPLE: "a/cheap" } }]).config;
    expect(config.adaptiveWeights).toEqual({ quality: 0.3, cost: 0.7 });
    expect(config.tierDistancePenalty).toBe(0.5);
    expect(config.adaptiveEligible).toBe("all");
  });

  it("cold-starts by sampling uniformly among unobserved models in the classified tier", () => {
    const config = buildConfig([
      { adaptive: true, tiers: { SIMPLE: ["a/cheap", "a/premium"], MEDIUM: "a/premium" } },
    ]).config;
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: sequence(0.9) })!;
    const pick = softFloorPick({ classifiedTier: "SIMPLE", userMessage: "hi", config, adaptive })!;
    expect(pick.model).toBe("a/premium");
    expect(pick.decision.phase).toBe("cold_start");
    expect(pick.decision.candidates.map((c) => c.model)).toEqual(["a/cheap", "a/premium"]);
  });

  it("prefers the home tier when posteriors are equal", () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(7) })!;
    pin(adaptive, "general", "a/cheap", 0.5);
    pin(adaptive, "general", "a/premium", 0.5);
    const pick = softFloorPick({ classifiedTier: "SIMPLE", userMessage: "hi", config, adaptive })!;
    expect(pick.model).toBe("a/cheap");
    expect(pick.decision.phase).toBe("adaptive");
  });

  it("crosses tiers when a posterior dominates", () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(7) })!;
    pin(adaptive, "general", "a/cheap", 0.05);
    pin(adaptive, "general", "a/premium", 0.95);
    const pick = softFloorPick({ classifiedTier: "SIMPLE", userMessage: "hi", config, adaptive })!;
    expect(pick.model).toBe("a/premium");
    const premium = pick.decision.candidates.find((c) => c.model === "a/premium")!;
    expect(premium.tierDistance).toBe(2);
  });

  it("gives a model reused across tiers zero distance in each of them", () => {
    const config = buildConfig([
      { adaptive: true, tiers: { SIMPLE: "a/cheap", MEDIUM: ["a/cheap", "a/premium"], COMPLEX: "a/premium" } },
    ]).config;
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(3) })!;
    pin(adaptive, "general", "a/cheap", 0.55);
    pin(adaptive, "general", "a/premium", 0.55);
    const pick = softFloorPick({ classifiedTier: "MEDIUM", userMessage: "hi", config, adaptive })!;
    expect(Object.fromEntries(pick.decision.candidates.map((c) => [c.model, c.tierDistance]))).toEqual({
      "a/cheap": 0,
      "a/premium": 0,
    });
  });

  it("excludes every candidate below a hard floor", () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(7) })!;
    pin(adaptive, "general", "a/cheap", 0.95);
    pin(adaptive, "general", "a/premium", 0.05);
    const unfloored = softFloorPick({ classifiedTier: "COMPLEX", userMessage: "hi", config, adaptive })!;
    expect(unfloored.model).toBe("a/cheap");
    const floored = softFloorPick({ classifiedTier: "COMPLEX", userMessage: "hi", config, adaptive, hardFloor: "COMPLEX" })!;
    expect(floored.model).toBe("a/premium");
    expect(floored.decision.candidates.map((c) => c.model)).toEqual(["a/premium"]);
  });

  it("samples only inside the classified tier in classified_tier mode", () => {
    const config = hybridConfig({ adaptiveEligible: "classified_tier" });
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(7) })!;
    pin(adaptive, "general", "a/cheap", 0.05);
    pin(adaptive, "general", "a/premium", 0.95);
    const pick = softFloorPick({ classifiedTier: "SIMPLE", userMessage: "hi", config, adaptive })!;
    expect(pick.model).toBe("a/cheap");
    expect(pick.decision.candidates.map((c) => c.model)).toEqual(["a/cheap"]);
  });

  it("carries the nearest tier's thinking level along with the chosen model", () => {
    const config = buildConfig([
      {
        adaptive: true,
        tiers: {
          SIMPLE: "a/cheap",
          COMPLEX: { model: "a/premium", thinkingLevel: "medium" },
          REASONING: { model: "a/premium", thinkingLevel: "high" },
        },
      },
    ]).config;
    expect(tiersForModel("a/premium", config)).toEqual(["COMPLEX", "REASONING"]);
    expect(targetForModel("a/premium", "SIMPLE", config)).toEqual({ model: "a/premium", thinkingLevel: "medium" });
    expect(targetForModel("a/premium", "REASONING", config)).toEqual({ model: "a/premium", thinkingLevel: "high" });
    expect(targetForModel("a/cheap", "REASONING", config)).toEqual({ model: "a/cheap" });
  });
});

describe("buildAdaptiveRouter", () => {
  it("reads prices from the registry and preferences from the tier entries", () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry() })!;
    expect(adaptive.availableModels).toEqual(["a/cheap", "a/premium"]);
    expect(adaptive.cost("a/premium")).toBe(5);
    expect(cellMean(adaptive.cell("general", "a/cheap"))).toBeCloseTo(0.3);
    expect(cellMean(adaptive.cell("general", "a/premium"))).toBeCloseTo(0.7);
  });

  it("returns null when adaptive is off", () => {
    expect(buildAdaptiveRouter(buildConfig([{ tiers: { SIMPLE: "a/cheap" } }]).config)).toBeNull();
  });
});

describe("turnFromRun", () => {
  it("gathers the final reply, tool calls and results from a pi run", () => {
    const t = turnFromRun("list files", [
      { role: "user", content: "list files" },
      { role: "assistant", content: [{ type: "toolCall", name: "bash", arguments: { command: "ls" } }], stopReason: "toolUse" },
      { role: "toolResult", content: [{ type: "text", text: "a.ts" }], isError: false },
      { role: "assistant", content: [{ type: "text", text: "Here are the files." }], stopReason: "stop" },
    ]);
    expect(t.userContent).toBe("list files");
    expect(t.assistantContent).toBe("Here are the files.");
    expect(t.toolCalls).toEqual([{ name: "bash", arguments: { command: "ls" } }]);
    expect(t.toolResults).toEqual([{ content: "a.ts", isError: false }]);
    expect(t.responseStatus).toBe(200);
  });

  it("takes the status of the final reply, not an earlier retried error", () => {
    expect(
      turnFromRun("x", [
        { role: "assistant", content: [], stopReason: "error", errorMessage: "boom" },
        { role: "assistant", content: [{ type: "text", text: "ok" }], stopReason: "stop" },
      ]).responseStatus,
    ).toBe(200);
  });

  it("reports an errored run as 500, or 429 when it reads as exhaustion", () => {
    expect(turnFromRun("x", [{ role: "assistant", content: [], stopReason: "error", errorMessage: "boom" }]).responseStatus).toBe(500);
    expect(
      turnFromRun("x", [{ role: "assistant", content: [], stopReason: "error", errorMessage: "429 rate limit exceeded" }])
        .responseStatus,
    ).toBe(429);
  });
});

describe("route — adaptive", () => {
  it("leads with the bandit's pick and records how it was chosen", async () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(11) })!;
    pin(adaptive, "general", "a/cheap", 0.05);
    pin(adaptive, "general", "a/premium", 0.95);
    const api = applier();
    const { decision } = await route({ turn: turn("hi"), config, api, adaptive, now: () => 0 });
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.chosenModel).toBe("a/premium");
    expect(decision.adaptive?.phase).toBe("adaptive");
    expect(decision.signals).toContain("adaptive:adaptive");
  });

  it("falls back down the ordinary chain when the pick cannot be applied", async () => {
    const config = hybridConfig();
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(11) })!;
    pin(adaptive, "general", "a/cheap", 0.05);
    pin(adaptive, "general", "a/premium", 0.95);
    const api = applier();
    api.setModel = async (model) => (model as unknown as { id: string }).id !== "premium";
    const { decision } = await route({ turn: turn("hi"), config, api, adaptive, now: () => 0 });
    expect(decision.adaptive?.chosenModel).toBe("a/premium");
    expect(decision.chosenModel).toBe("a/cheap");
    expect(decision.fellBackBecause).toMatch(/premium/);
  });

  it("keeps keyword overrides and session pins on the plain pool walk", async () => {
    const config = hybridConfig({
      keywordTierRules: [{ keywords: ["migration"], tier: "REASONING" }],
      sessionAffinity: true,
    });
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(11) })!;
    const api = applier();
    const keyword = await route({ turn: turn("run the migration"), config, api, adaptive, now: () => 0 });
    expect(keyword.decision.cause).toBe("literal_keyword_match");
    expect(keyword.decision.adaptive).toBeUndefined();

    const pinned = await route({
      turn: turn("hi"),
      config,
      api,
      adaptive,
      now: () => 0,
      pin: { model: "a/premium", tier: "COMPLEX", expiresAt: 10 },
    });
    expect(pinned.decision.cause).toBe("session_affinity_pin");
    expect(pinned.decision.adaptive).toBeUndefined();
  });

  it("hands the plan-mode floor to the bandit as a hard floor", async () => {
    const config = hybridConfig({ planMode: { minTier: "COMPLEX" } });
    const adaptive = buildAdaptiveRouter(config, { registry: costRegistry(), rng: seeded(11) })!;
    pin(adaptive, "general", "a/cheap", 0.95);
    pin(adaptive, "general", "a/premium", 0.05);
    const api = applier();
    const { decision } = await route({
      turn: turn("think step by step and analyze this tradeoff: weigh the options for our schema"),
      config,
      api,
      adaptive,
      now: () => 0,
      planModeActive: true,
    });
    expect(decision.tier).toBe("REASONING");
    expect(decision.planFloored).toBe(false);
    expect(decision.chosenModel).toBe("a/premium");
    expect(decision.adaptive?.candidates.map((c) => c.model)).toEqual(["a/premium"]);
  });

  it("routes without the bandit when none is supplied", async () => {
    const config = hybridConfig();
    const api = applier();
    const { decision } = await route({ turn: turn("hi"), config, api, now: () => 0 });
    expect(decision.chosenModel).toBe("a/cheap");
    expect(decision.adaptive).toBeUndefined();
  });
});
