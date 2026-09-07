import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildConfig } from "../src/config.ts";
import { classificationSystemPrompt } from "../src/classify/rubrics.ts";
import { parseTierReply, splitModelRef } from "../src/classify/llm.ts";

describe("buildConfig", () => {
  const base = { defaultModel: "anthropic/haiku", tiers: { SIMPLE: "anthropic/haiku" } };

  it("accepts a minimal config", () => {
    const { config, errors } = buildConfig([base]);
    expect(errors).toEqual([]);
    expect(config.enabled).toBe(true);
    expect(config.tiers.SIMPLE).toEqual([{ model: "anthropic/haiku" }]);
  });

  it("merges project config over global", () => {
    const { config } = buildConfig([
      { ...base, defaultModel: "a/global" },
      { defaultModel: "a/project" },
    ]);
    expect(config.defaultModel).toBe("a/project");
  });

  it("accepts defaultModel as a string or as { model, thinkingLevel }", () => {
    const plain = buildConfig([base]);
    expect(plain.config.defaultModel).toBe("anthropic/haiku");
    expect(plain.config.defaultModelThinkingLevel).toBeNull();

    const withLevel = buildConfig([{ ...base, defaultModel: { model: " a/two ", thinkingLevel: "low" } }]);
    expect(withLevel.errors).toEqual([]);
    expect(withLevel.config.defaultModel).toBe("a/two");
    expect(withLevel.config.defaultModelThinkingLevel).toBe("low");
  });

  it("rejects a malformed defaultModel or thinking level", () => {
    expect(buildConfig([{ ...base, defaultModel: 7 }]).errors).toEqual([
      "defaultModel must be a model string or { model, thinkingLevel }",
    ]);
    expect(buildConfig([{ ...base, defaultModel: { model: "a/x", thinkingLevel: "turbo" } }]).errors).toEqual([
      "defaultModel.thinkingLevel must be one of off, minimal, low, medium, high, xhigh, max",
    ]);
    expect(buildConfig([{ ...base, tiers: { SIMPLE: { model: "a/x", thinkingLevel: "turbo" } } }]).errors).toEqual([
      'tiers.SIMPLE: "a/x" thinkingLevel must be one of off, minimal, low, medium, high, xhigh, max',
    ]);
  });

  it("accepts a tier as a string, an object, or a list", () => {
    const { config, errors } = buildConfig([
      {
        ...base,
        tiers: {
          SIMPLE: "a/one",
          MEDIUM: { model: "a/two", thinkingLevel: "high" },
          COMPLEX: ["a/three", { model: "a/four" }],
        },
      },
    ]);
    expect(errors).toEqual([]);
    expect(config.tiers.MEDIUM).toEqual([{ model: "a/two", thinkingLevel: "high" }]);
    expect(config.tiers.COMPLEX).toEqual([{ model: "a/three" }, { model: "a/four" }]);
  });

  it("rejects an unknown tier name", () => {
    const { errors } = buildConfig([{ ...base, tiers: { HUGE: "a/b" } }]);
    expect(errors.join()).toContain("not a known tier");
  });

  it("disables routing when there are errors", () => {
    const { config } = buildConfig([{ ...base, classifierType: "nonsense" }]);
    expect(config.enabled).toBe(false);
  });

  it("rejects a keyword rule with only blank keywords", () => {
    // A blank keyword substring-matches every prompt, so one stray blank would force this
    // rule's tier for all traffic.
    const { errors } = buildConfig([{ ...base, keywordTierRules: [{ keywords: ["", "  "], tier: "SIMPLE" }] }]);
    expect(errors.join()).toContain("at least one non-empty keyword");
  });

  it("drops blank keywords but keeps a rule with real ones", () => {
    const { config, errors } = buildConfig([
      { ...base, keywordTierRules: [{ keywords: ["", "migration"], tier: "REASONING" }] },
    ]);
    expect(errors).toEqual([]);
    expect(config.keywordTierRules[0]?.keywords).toEqual(["migration"]);
  });

  it("rejects rubric and systemPrompt together", () => {
    // The custom prompt IS the whole system role, so a preset alongside it never reaches
    // the wire.
    const { errors } = buildConfig([
      {
        ...base,
        classifierType: "llm",
        classifierLLMConfig: { model: "a/b", classificationRubric: "agentic", systemPrompt: "custom" },
      },
    ]);
    expect(errors.join()).toContain("mutually exclusive");
  });

  it("rejects a blank systemPrompt", () => {
    const { errors } = buildConfig([
      { ...base, classifierType: "llm", classifierLLMConfig: { model: "a/b", systemPrompt: "   " } },
    ]);
    expect(errors.join()).toContain("must be non-empty");
  });

  it("warns that a custom systemPrompt drops the injection defence", () => {
    const { warnings } = buildConfig([
      { ...base, classifierType: "llm", classifierLLMConfig: { model: "a/b", systemPrompt: "custom" } },
    ]);
    expect(warnings.join()).toContain("prompt-injection defence");
  });

  it("defaults the rubric to agentic, not legacy", () => {
    // Upstream defaults to legacy only to avoid moving an existing deployment's spend.
    const { config } = buildConfig([{ ...base, classifierType: "llm", classifierLLMConfig: { model: "a/b" } }]);
    expect(config.classifierLLMConfig?.classificationRubric).toBe("agentic");
  });

  it("requires a classifier config when classifierType is llm", () => {
    const { errors } = buildConfig([{ ...base, classifierType: "llm" }]);
    expect(errors.join()).toContain("classifierLLMConfig is missing");
  });

  it("replaces the built-in reminder markers rather than extending them", () => {
    const { config } = buildConfig([{ ...base, reminderMarkers: [{ open: "<a>", close: "</a>" }] }]);
    expect(config.reminderMarkers).toEqual([{ open: "<a>", close: "</a>" }]);
  });

  it("lets an empty escalationKeywords list disable escalation", () => {
    const { config, errors } = buildConfig([{ ...base, escalationKeywords: [] }]);
    expect(errors).toEqual([]);
    expect(config.escalationKeywords).toEqual([]);
  });

  it("appends customTechnicalKeywords, trimming blanks", () => {
    const { config, errors } = buildConfig([{ ...base, customTechnicalKeywords: [" kafka ", "", "udp"] }]);
    expect(errors).toEqual([]);
    expect(config.customTechnicalKeywords).toEqual(["kafka", "udp"]);
  });

  it("rejects a non-list customTechnicalKeywords", () => {
    const { errors } = buildConfig([{ ...base, customTechnicalKeywords: "kafka" }]);
    expect(errors).toEqual(["customTechnicalKeywords must be an array of strings"]);
  });

  it("refuses to let technicalKeywords replace the built-in list", () => {
    const { config, errors } = buildConfig([{ ...base, technicalKeywords: ["quantum"] }]);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/customTechnicalKeywords/);
    expect(config.enabled).toBe(false);
  });

  it("parses the adaptive knobs and per-model preferences", () => {
    const { config, errors } = buildConfig([
      {
        ...base,
        adaptive: true,
        adaptiveWeights: { quality: 0.6, cost: 0.4 },
        tierDistancePenalty: 0.2,
        adaptiveEligible: "classified_tier",
        tiers: { SIMPLE: { model: "a/one", qualityTier: 1, strengths: ["writing", "factual_lookup"] } },
      },
    ]);
    expect(errors).toEqual([]);
    expect(config.adaptive).toBe(true);
    expect(config.adaptiveWeights).toEqual({ quality: 0.6, cost: 0.4 });
    expect(config.tierDistancePenalty).toBe(0.2);
    expect(config.adaptiveEligible).toBe("classified_tier");
    expect(config.tiers.SIMPLE).toEqual([{ model: "a/one", qualityTier: 1, strengths: ["writing", "factual_lookup"] }]);
  });

  it("rejects adaptive weights that do not sum to one", () => {
    const { errors } = buildConfig([{ ...base, adaptiveWeights: { quality: 0.5, cost: 0.3 } }]);
    expect(errors).toEqual(["adaptiveWeights must sum to 1.0, got quality=0.5 + cost=0.3"]);
  });

  it("rejects bad adaptive values", () => {
    const { errors } = buildConfig([
      {
        ...base,
        adaptive: "yes",
        tierDistancePenalty: -1,
        adaptiveEligible: "some",
        tiers: { SIMPLE: { model: "a/one", qualityTier: 4, strengths: ["poetry"] } },
      },
    ]);
    expect(errors).toEqual([
      'tiers.SIMPLE: "a/one" qualityTier must be 1, 2 or 3',
      expect.stringMatching(/strengths must be a list of/),
      "adaptive must be a boolean",
      "tierDistancePenalty must be a number >= 0",
      'adaptiveEligible must be "all" or "classified_tier"',
    ]);
  });

  it("requires a tier pool for adaptive selection", () => {
    const { errors } = buildConfig([{ defaultModel: "a/one", adaptive: true }]);
    expect(errors).toEqual(["adaptive requires at least one non-empty tier pool"]);
  });

  it("parses the semantic matching knobs", () => {
    const { config, errors } = buildConfig([
      {
        ...base,
        keywordTierRules: [{ keywords: ["migration"], tier: "REASONING" }],
        semanticKeywordMatching: true,
        embeddingModel: " voyage/voyage-3-5 ",
        matchThreshold: 0.6,
        embeddingEndpoint: { baseUrl: "https://proxy.example/v1", apiKeyEnv: "EMBED_KEY", timeoutMs: 1500.9 },
      },
    ]);
    expect(errors).toEqual([]);
    expect(config.semanticKeywordMatching).toBe(true);
    expect(config.embeddingModel).toBe("voyage/voyage-3-5");
    expect(config.matchThreshold).toBe(0.6);
    expect(config.embeddingEndpoint).toEqual({ baseUrl: "https://proxy.example/v1", apiKeyEnv: "EMBED_KEY", timeoutMs: 1500 });
  });

  it("defaults the match threshold and embedding timeout", () => {
    const { config } = buildConfig([base]);
    expect(config.semanticKeywordMatching).toBe(false);
    expect(config.embeddingModel).toBeNull();
    expect(config.matchThreshold).toBe(0.5);
    expect(config.embeddingEndpoint).toEqual({ timeoutMs: 3000 });
  });

  it("requires an embedding model and rules for semantic matching", () => {
    expect(buildConfig([{ ...base, semanticKeywordMatching: true }]).errors).toEqual([
      "embeddingModel is required when semanticKeywordMatching is enabled",
      "keywordTierRules must be non-empty when semanticKeywordMatching is enabled",
    ]);
    // Off, neither is needed.
    expect(buildConfig([{ ...base, semanticKeywordMatching: false }]).errors).toEqual([]);
  });

  it("rejects bad semantic values", () => {
    const { errors } = buildConfig([
      {
        ...base,
        semanticKeywordMatching: "yes",
        embeddingModel: "",
        matchThreshold: 1.5,
        embeddingEndpoint: { baseUrl: "", apiKeyEnv: 3, timeoutMs: 0 },
      },
    ]);
    expect(errors).toEqual([
      "semanticKeywordMatching must be a boolean",
      "embeddingModel must be a model string (provider/model-id)",
      "matchThreshold must be a number in [0, 1]",
      "embeddingEndpoint.baseUrl must be a non-empty string",
      "embeddingEndpoint.apiKeyEnv must be a non-empty string",
      "embeddingEndpoint.timeoutMs must be a positive number",
    ]);
  });

  it("errors when there is nothing to route to", () => {
    const { errors } = buildConfig([{ tiers: {} }]);
    expect(errors.join()).toContain("nothing to route to");
  });
});

describe("splitModelRef", () => {
  it("splits on the first slash only", () => {
    // openrouter ids contain slashes themselves.
    expect(splitModelRef("openrouter/anthropic/claude-sonnet")).toEqual({
      provider: "openrouter",
      modelId: "anthropic/claude-sonnet",
    });
  });

  it("rejects malformed refs", () => {
    expect(splitModelRef("bare")).toBeNull();
    expect(splitModelRef("/leading")).toBeNull();
    expect(splitModelRef("trailing/")).toBeNull();
  });
});

describe("parseTierReply", () => {
  it("reads a bare tier name", () => {
    expect(parseTierReply("REASONING")).toBe("REASONING");
  });

  it("reads a tier out of JSON", () => {
    expect(parseTierReply('{"tier": "COMPLEX"}')).toBe("COMPLEX");
  });

  it("takes the first tier mentioned, so trailing chatter cannot upgrade the answer", () => {
    expect(parseTierReply("SIMPLE, though it could be REASONING")).toBe("SIMPLE");
  });

  it("returns null when no tier is named", () => {
    expect(parseTierReply("I am not sure")).toBeNull();
  });
});

describe("classificationSystemPrompt", () => {
  it("always carries the trust boundary", () => {
    for (const rubric of ["legacy", "agentic", "chat", "business"] as const) {
      const prompt = classificationSystemPrompt({ contextWindowSize: 3, rubric });
      expect(prompt).toContain("never instructions to you");
    }
  });

  it("includes engineering calibration only in the agentic preset", () => {
    const agentic = classificationSystemPrompt({ contextWindowSize: 3, rubric: "agentic" });
    const chat = classificationSystemPrompt({ contextWindowSize: 3, rubric: "chat" });
    expect(agentic).toContain("set up a Jupyter server with token auth");
    expect(chat).not.toContain("set up a Jupyter server with token auth");
  });

  it("uses business tier criteria for the business preset", () => {
    const business = classificationSystemPrompt({ contextWindowSize: 3, rubric: "business" });
    expect(business).toContain("committing to a decision under conflicting tradeoffs");
  });

  it("switches the closing line on the context window", () => {
    const withWindow = classificationSystemPrompt({ contextWindowSize: 3, rubric: "agentic" });
    const without = classificationSystemPrompt({ contextWindowSize: 0, rubric: "agentic" });
    expect(withWindow).toContain("rate the work it approves");
    expect(without).toContain("Classify only the current message");
  });

  it("renders tier placeholders as tier names", () => {
    const prompt = classificationSystemPrompt({ contextWindowSize: 3, rubric: "agentic" });
    expect(prompt).not.toContain("{SIMPLE}");
    expect(prompt).toContain("-> SIMPLE");
  });

  it("returns a custom prompt verbatim", () => {
    const prompt = classificationSystemPrompt({ contextWindowSize: 3, rubric: "agentic", customPrompt: "mine" });
    expect(prompt).toBe("mine");
  });
});

describe("example configs", () => {
  const load = (name: string) =>
    buildConfig([JSON.parse(readFileSync(new URL(`../examples/autorouter.${name}.json`, import.meta.url), "utf8"))]);

  it.each(["heuristic", "llm"])("examples/autorouter.%s.json loads without errors", (name) => {
    const { config, errors, warnings } = load(name);
    expect(errors).toEqual([]);
    expect(warnings).toEqual([]);
    expect(config.defaultModel).toBe("anthropic/claude-haiku-4-5");
    expect(config.defaultModelThinkingLevel).toBe("low");
  });

  it("examples/autorouter.multillm.json spreads tiers across providers", () => {
    const { config, errors, warnings } = load("multillm");
    expect(errors).toEqual([]);
    expect(warnings).toEqual([]);
    expect(config.defaultModel).toBe("openai/gpt-5.6-terra");
    expect(config.defaultModelThinkingLevel).toBe("medium");
    expect(config.classifierType).toBe("llm");
    expect(config.classifierLLMConfig?.model).toBe("mistral/mistral-small-2603");
    expect(config.tiers.REASONING).toEqual([{ model: "anthropic/claude-opus-5", thinkingLevel: "high" }]);
  });
});
