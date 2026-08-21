import { describe, expect, it } from "vitest";
import { type CandidateModel, blendedPrice, buildInitialConfig, renderInitPreview } from "../src/init.ts";
import { buildConfig } from "../src/config.ts";

const model = (
  provider: string,
  id: string,
  input: number,
  output: number,
  extra: Partial<CandidateModel> = {},
): CandidateModel => ({ provider, id, cost: { input, output }, input: ["text"], ...extra });

// Roughly a real catalogue: a spread of prices, one reasoning-capable model at the top.
const CATALOGUE: CandidateModel[] = [
  model("anthropic", "haiku", 0.8, 4),
  model("anthropic", "sonnet", 3, 15),
  model("anthropic", "opus", 15, 75, { reasoning: true }),
  model("openai", "gpt-mini", 0.15, 0.6),
  model("openai", "gpt", 2.5, 10),
];

describe("blendedPrice", () => {
  it("weights input over output", () => {
    // Agent turns re-read a large context and write a little, so input dominates the bill.
    expect(blendedPrice(model("x", "y", 10, 0))).toBeGreaterThan(blendedPrice(model("x", "y", 0, 10)));
  });

  it("treats missing cost data as free", () => {
    expect(blendedPrice({ provider: "local", id: "llama" })).toBe(0);
  });
});

describe("buildInitialConfig", () => {
  it("orders tiers cheapest to priciest", () => {
    const { picks } = buildInitialConfig(CATALOGUE);
    const prices = (["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const).map((t) => blendedPrice(picks[t]));
    for (let i = 1; i < prices.length; i++) {
      expect(prices[i]!).toBeGreaterThanOrEqual(prices[i - 1]!);
    }
    expect(`${picks.SIMPLE.provider}/${picks.SIMPLE.id}`).toBe("openai/gpt-mini");
  });

  it("prefers a reasoning-capable model for the top tier", () => {
    const { picks, config } = buildInitialConfig(CATALOGUE);
    expect(`${picks.REASONING.provider}/${picks.REASONING.id}`).toBe("anthropic/opus");
    // A reasoning model gets a thinking level; the others are bare strings.
    expect((config.tiers as Record<string, unknown>).REASONING).toEqual({
      model: "anthropic/opus",
      thinkingLevel: "high",
    });
  });

  it("does not promote a cheap reasoning model into the top tier", () => {
    // "Supports thinking" must not outrank the price ordering, or a cheap reasoning model
    // would become the most expensive tier.
    const withCheapReasoner = [...CATALOGUE, model("openai", "o-mini", 0.2, 0.8, { reasoning: true })];
    const { picks } = buildInitialConfig(withCheapReasoner);
    expect(picks.REASONING.id).toBe("opus");
  });

  it("falls back to the priciest model when nothing supports reasoning", () => {
    const noReasoning = CATALOGUE.map((m) => ({ ...m, reasoning: false }));
    const { picks, config } = buildInitialConfig(noReasoning);
    expect(picks.REASONING.id).toBe("opus");
    expect((config.tiers as Record<string, unknown>).REASONING).toBe("anthropic/opus");
  });

  it("defaults to the cheapest model", () => {
    // defaultModel is the failed-classification fallback, so a wrong guess should be cheap.
    expect(buildInitialConfig(CATALOGUE).config.defaultModel).toBe("openai/gpt-mini");
  });

  it("filters by provider", () => {
    const { picks } = buildInitialConfig(CATALOGUE, { provider: "anthropic" });
    for (const tier of ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const) {
      expect(picks[tier].provider).toBe("anthropic");
    }
  });

  it("throws when a provider filter matches nothing", () => {
    expect(() => buildInitialConfig(CATALOGUE, { provider: "nope" })).toThrow(/no usable models/);
  });

  it("throws when there are no models at all", () => {
    expect(() => buildInitialConfig([])).toThrow(/no usable models/);
  });

  it("skips models that cannot accept text", () => {
    const withImageOnly = [...CATALOGUE, { provider: "x", id: "img", input: ["image"], cost: { input: 0, output: 0 } }];
    const { picks } = buildInitialConfig(withImageOnly);
    expect(Object.values(picks).every((m) => m.id !== "img")).toBe(true);
  });

  it("reuses one model across all tiers when only one is available", () => {
    const { picks, notes } = buildInitialConfig([model("solo", "only", 1, 2)]);
    expect(new Set(Object.values(picks).map((m) => m.id))).toEqual(new Set(["only"]));
    expect(notes.join()).toContain("only 1 distinct model");
  });

  it("spreads two models across four tiers", () => {
    const { picks } = buildInitialConfig([model("a", "cheap", 1, 1), model("a", "dear", 100, 100)]);
    expect(picks.SIMPLE.id).toBe("cheap");
    expect(picks.REASONING.id).toBe("dear");
  });

  it("warns when every model reports zero cost", () => {
    const free = [
      { provider: "local", id: "a", input: ["text"] },
      { provider: "local", id: "b", input: ["text"] },
    ];
    const { notes } = buildInitialConfig(free);
    expect(notes.join()).toContain("zero cost");
  });

  it("is deterministic across runs on the same catalogue", () => {
    const a = JSON.stringify(buildInitialConfig(CATALOGUE).config);
    const b = JSON.stringify(buildInitialConfig([...CATALOGUE].reverse()).config);
    expect(a).toBe(b);
  });

  it("deduplicates a catalogue that lists a model twice", () => {
    const { picks } = buildInitialConfig([...CATALOGUE, model("anthropic", "opus", 15, 75, { reasoning: true })]);
    expect(picks.REASONING.id).toBe("opus");
  });

  it("emits an llm classifier block using the cheapest model", () => {
    const { config, notes } = buildInitialConfig(CATALOGUE, { classifier: "llm" });
    expect(config.classifierType).toBe("llm");
    expect(config.classifierLLMConfig).toMatchObject({
      model: "openai/gpt-mini",
      classificationRubric: "agentic",
    });
    expect(config.classifierFallback).toBe("heuristic");
    expect(notes.join()).toContain("latency");
  });

  it("always says the ranking is price-only", () => {
    // The one thing a generated config must not imply is that it knows which model is good.
    expect(buildInitialConfig(CATALOGUE).notes.join()).toContain("price alone");
  });
});

describe("generated config round-trips through the loader", () => {
  it("validates with no errors", () => {
    const { config } = buildInitialConfig(CATALOGUE);
    const loaded = buildConfig([config]);
    expect(loaded.errors).toEqual([]);
    expect(loaded.config.enabled).toBe(true);
    expect(loaded.config.tiers.REASONING).toEqual([{ model: "anthropic/opus", thinkingLevel: "high" }]);
  });

  it("validates the llm variant too", () => {
    const { config } = buildInitialConfig(CATALOGUE, { classifier: "llm" });
    const loaded = buildConfig([config]);
    expect(loaded.errors).toEqual([]);
    expect(loaded.config.classifierLLMConfig?.classificationRubric).toBe("agentic");
  });

  it("validates even in the single-model degenerate case", () => {
    const { config } = buildInitialConfig([model("solo", "only", 1, 2)]);
    expect(buildConfig([config]).errors).toEqual([]);
  });
});

describe("renderInitPreview", () => {
  it("shows every tier with its model and price", () => {
    const preview = renderInitPreview(buildInitialConfig(CATALOGUE));
    for (const tier of ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]) {
      expect(preview).toContain(tier);
    }
    expect(preview).toContain("anthropic/opus");
    expect(preview).toMatch(/\$\d+\.\d\d\/Mtok/);
  });

  it("says so when a model has no cost data", () => {
    expect(renderInitPreview(buildInitialConfig([{ provider: "local", id: "a", input: ["text"] }]))).toContain(
      "no cost data",
    );
  });
});
