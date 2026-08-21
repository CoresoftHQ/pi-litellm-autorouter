import { describe, expect, it } from "vitest";
import { buildConfig } from "../src/config.ts";
import { classifyHeuristic, keywordMatches } from "../src/classify/heuristic.ts";

function config(overrides: Record<string, unknown> = {}) {
  return buildConfig([
    {
      defaultModel: "anthropic/haiku",
      tiers: {
        SIMPLE: "anthropic/haiku",
        MEDIUM: "anthropic/sonnet",
        COMPLEX: "anthropic/sonnet",
        REASONING: "anthropic/opus",
      },
      ...overrides,
    },
  ]).config;
}

describe("keywordMatches", () => {
  it("uses word boundaries for single words", () => {
    // The false positives upstream calls out by name.
    expect(keywordMatches("what is the capital of france", "api")).toBe(false);
    expect(keywordMatches("terrorism is bad", "error")).toBe(false);
    expect(keywordMatches("the api is down", "api")).toBe(true);
  });

  it("matches multi-word phrases as substrings", () => {
    expect(keywordMatches("please think step by step here", "step by step")).toBe(true);
  });

  it("matches CJK keywords as substrings", () => {
    // \b never fires between two CJK characters, so word-boundary matching would miss.
    expect(keywordMatches("我需要开发票", "发票")).toBe(true);
  });
});

describe("classifyHeuristic", () => {
  it("puts a greeting in SIMPLE", () => {
    const result = classifyHeuristic("hi", config());
    expect(result.tier).toBe("SIMPLE");
  });

  it("puts a short factual lookup in SIMPLE", () => {
    expect(classifyHeuristic("what is a semaphore", config()).tier).toBe("SIMPLE");
  });

  it("promotes on two or more reasoning markers", () => {
    const result = classifyHeuristic(
      "think step by step and analyze this: weigh the options for our database schema",
      config(),
    );
    expect(result.tier).toBe("REASONING");
    expect(result.cause).toBe("reasoning_override");
  });

  it("does not let stock reasoning phrases promote a trivial prompt", () => {
    // The override requires the score to clear reasoningOverrideMinScore, which tracks
    // simple_medium: two markers alone are not enough.
    const result = classifyHeuristic("hi thanks, evaluate and conclude", config());
    expect(result.cause).not.toBe("reasoning_override");
    expect(result.tier).toBe("SIMPLE");
  });

  it("promotes on markers alone when the floor is 0", () => {
    const result = classifyHeuristic("hi thanks, evaluate and conclude", config({ reasoningOverrideMinScore: 0 }));
    expect(result.cause).toBe("reasoning_override");
    expect(result.tier).toBe("REASONING");
  });

  it("matches a code keyword across an apostrophe", () => {
    // Pinning a real quirk of word-boundary matching, inherited from upstream: `\blet\b`
    // fires on the "let" in "let's", because the apostrophe is a non-word character. So
    // "let's think about it" scores codePresence. Surprising, but faithful — and enough to
    // push a short prompt over the reasoning-override floor on its own.
    const result = classifyHeuristic("let's think step by step here", config());
    expect(result.signals).toContain("code (let)");
  });

  it("scores technical content above a greeting", () => {
    const technical = classifyHeuristic(
      "our distributed microservice architecture has latency and throughput problems under concurrency",
      config(),
    );
    const greeting = classifyHeuristic("hello there", config());
    expect(technical.score!).toBeGreaterThan(greeting.score!);
  });

  it("reports per-dimension signals", () => {
    const result = classifyHeuristic("debug this python function please", config());
    expect(result.signals.some((s) => s.startsWith("code ("))).toBe(true);
    expect(result.dimensions?.map((d) => d.name)).toEqual([
      "tokenCount",
      "codePresence",
      "reasoningMarkers",
      "technicalTerms",
      "simpleIndicators",
      "multiStepPatterns",
      "questionComplexity",
    ]);
  });

  it("penalises very short prompts and rewards very long ones", () => {
    const short = classifyHeuristic("hey", config());
    expect(short.dimensions?.find((d) => d.name === "tokenCount")?.score).toBe(-1);
    const long = classifyHeuristic("word ".repeat(500), config());
    expect(long.dimensions?.find((d) => d.name === "tokenCount")?.score).toBe(1);
  });

  it("detects multi-step patterns", () => {
    const result = classifyHeuristic("first do the migration then update the callers", config());
    expect(result.signals).toContain("multi-step");
  });

  it("counts question marks only above three", () => {
    expect(classifyHeuristic("a? b? c?", config()).signals).not.toContain("3 questions");
    expect(classifyHeuristic("a? b? c? d?", config()).signals).toContain("4 questions");
  });

  it("respects custom boundaries", () => {
    // Everything lands in REASONING when every boundary is 0.
    const wide = config({ tierBoundaries: { simple_medium: -99, medium_complex: -99, complex_reasoning: -99 } });
    expect(classifyHeuristic("hi", wide).tier).toBe("REASONING");
  });
});
