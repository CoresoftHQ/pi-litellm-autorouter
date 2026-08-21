/**
 * The rule-based complexity scorer.
 *
 * Port of LiteLLM's `_score_and_classify`. Zero API calls, sub-millisecond, deterministic
 * — which is why it is the default classifier and the fallback for the LLM one.
 */

import type { RouterConfig } from "../config.ts";
import {
  DEFAULT_CODE_KEYWORDS,
  DEFAULT_REASONING_KEYWORDS,
  DEFAULT_SIMPLE_KEYWORDS,
  DEFAULT_TECHNICAL_KEYWORDS,
  MULTI_STEP_PATTERNS,
} from "../defaults.ts";
import { estimateTokens } from "../extract.ts";
import type { Classification, DimensionScore, Tier } from "../types.ts";

/** CJK is written without spaces and every CJK character is a regex word character, so
 *  `\b` never fires between two of them. Such keywords match as plain substrings. */
const CJK_CHARACTER = /[　-〿぀-ゟ゠-ヿ㐀-䶿一-鿿豈-﫿]/;

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Does `keyword` occur in `text` (both already lowercased)?
 *
 * Single-word keywords use word boundaries so "api" does not match "capital" and "error"
 * does not match "terrorism". Multi-word phrases and CJK keywords match as substrings.
 * The gate is on the *keyword*, so an ASCII keyword keeps word-boundary matching whatever
 * script the prompt is written in.
 */
export function keywordMatches(text: string, keyword: string): boolean {
  const kw = keyword.toLowerCase();
  if (kw.includes(" ") || CJK_CHARACTER.test(kw)) return text.includes(kw);
  return new RegExp(`\\b${escapeRegExp(kw)}\\b`).test(text);
}

function scoreTokenCount(estimated: number, thresholds: Record<string, number>): DimensionScore {
  const simple = thresholds.simple ?? 15;
  const complex = thresholds.complex ?? 400;
  if (estimated < simple) return { name: "tokenCount", score: -1.0, signal: `short (${estimated} tokens)` };
  if (estimated > complex) return { name: "tokenCount", score: 1.0, signal: `long (${estimated} tokens)` };
  return { name: "tokenCount", score: 0 };
}

function scoreKeywordMatch(
  text: string,
  keywords: readonly string[],
  name: string,
  signalLabel: string,
  thresholds: [low: number, high: number],
  scores: [none: number, low: number, high: number],
): { dimension: DimensionScore; matchCount: number } {
  const [lowThreshold, highThreshold] = thresholds;
  const [scoreNone, scoreLow, scoreHigh] = scores;

  const matches = keywords.filter((kw) => keywordMatches(text, kw));
  const matchCount = matches.length;
  if (matchCount < lowThreshold) {
    return { dimension: { name, score: scoreNone }, matchCount };
  }
  const detail = matches.slice(0, 3).join(", ");
  const score = matchCount >= highThreshold ? scoreHigh : scoreLow;
  return { dimension: { name, score, signal: `${signalLabel} (${detail})` }, matchCount };
}

function scoreMultiStep(text: string): DimensionScore {
  const hit = MULTI_STEP_PATTERNS.some((pattern) => pattern.test(text));
  return hit ? { name: "multiStepPatterns", score: 0.5, signal: "multi-step" } : { name: "multiStepPatterns", score: 0 };
}

function scoreQuestionComplexity(text: string): DimensionScore {
  const count = (text.match(/\?/g) ?? []).length;
  return count > 3
    ? { name: "questionComplexity", score: 0.5, signal: `${count} questions` }
    : { name: "questionComplexity", score: 0 };
}

/** The floor a reasoning-marker promotion must also clear. Tracks `simple_medium` unless
 *  explicitly set, so stock phrases on an otherwise trivial prompt cannot buy the top tier.
 *  Set it to 0 to promote on the markers alone. */
export function effectiveReasoningOverrideMinScore(config: RouterConfig): number {
  return config.reasoningOverrideMinScore ?? config.tierBoundaries.simple_medium ?? 0.15;
}

/**
 * Score `prompt` and map it to a tier.
 *
 * The system prompt is deliberately **not** scored. It is a per-session constant, so it
 * carries no information about how requests within a session differ, yet it saturates the
 * keyword thresholds — codePresence trips at 2 matches, which any agent identity prompt
 * clears on its first line — while spending 0.63 of the weight budget. That collapses the
 * scorer's dynamic range and escalates every request alike.
 */
export function classifyHeuristic(prompt: string, config: RouterConfig): Classification {
  const userText = prompt.toLowerCase();
  const estimated = estimateTokens(prompt);

  const code = scoreKeywordMatch(
    userText,
    config.codeKeywords ?? DEFAULT_CODE_KEYWORDS,
    "codePresence",
    "code",
    [1, 2],
    [0, 0.5, 1.0],
  );
  const reasoning = scoreKeywordMatch(
    userText,
    config.reasoningKeywords ?? DEFAULT_REASONING_KEYWORDS,
    "reasoningMarkers",
    "reasoning",
    [1, 2],
    [0, 0.7, 1.0],
  );
  const technical = scoreKeywordMatch(
    userText,
    config.technicalKeywords ?? DEFAULT_TECHNICAL_KEYWORDS,
    "technicalTerms",
    "technical",
    [2, 4],
    [0, 0.5, 1.0],
  );
  const simple = scoreKeywordMatch(
    userText,
    config.simpleKeywords ?? DEFAULT_SIMPLE_KEYWORDS,
    "simpleIndicators",
    "simple",
    [1, 2],
    [0, -1.0, -1.0],
  );

  const dimensions: DimensionScore[] = [
    scoreTokenCount(estimated, config.tokenThresholds),
    code.dimension,
    reasoning.dimension,
    technical.dimension,
    simple.dimension,
    scoreMultiStep(userText),
    scoreQuestionComplexity(prompt),
  ];

  const signals = dimensions.map((d) => d.signal).filter((s): s is string => s !== undefined);

  const weightedScore = dimensions.reduce(
    (sum, d) => sum + d.score * (config.dimensionWeights[d.name] ?? 0),
    0,
  );

  const boundaries = config.tierBoundaries;
  const clearsOverrideFloor = weightedScore >= effectiveReasoningOverrideMinScore(config);

  if (reasoning.matchCount >= 2 && clearsOverrideFloor) {
    return {
      tier: "REASONING",
      score: weightedScore,
      signals,
      cause: "reasoning_override",
      dimensions,
    };
  }

  let tier: Tier;
  if (weightedScore < (boundaries.simple_medium ?? 0.15)) tier = "SIMPLE";
  else if (weightedScore < (boundaries.medium_complex ?? 0.35)) tier = "MEDIUM";
  else if (weightedScore < (boundaries.complex_reasoning ?? 0.6)) tier = "COMPLEX";
  else tier = "REASONING";

  return { tier, score: weightedScore, signals, cause: "heuristic_scorer", dimensions };
}
