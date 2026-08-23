/**
 * Deterministic keyword → tier overrides, and the escalation keyword.
 *
 * Both run before/around the classifier, and both are floors on what a caller can reach:
 * escalation moves the result exactly one tier up and can never name a model.
 */

import type { KeywordTierRule, RouterConfig } from "../config.ts";
import { keywordMatches } from "./heuristic.ts";
import type { SemanticMatcher } from "./semantic.ts";
import { type DecisionCause, TIER_SEVERITY_ORDER, type Tier, tierSeverity } from "../types.ts";

export interface KeywordOverride {
  tier: Tier;
  /** Null on a semantic hit: that is a similarity match against the rule's keywords, not
   *  a literal one, so there is no single keyword to report. */
  matchedKeyword: string | null;
  cause: Extract<DecisionCause, "literal_keyword_match" | "semantic_keyword_match">;
  /** Extra detail for the decision log, e.g. the nearest keyword of a semantic hit. */
  signal?: string;
}

export type KeywordOverrideResult =
  | { override: KeywordOverride | null; failure?: undefined }
  | { override: null; failure: string };

/**
 * Resolve a `keywordTierRules` override, semantically or lexically per config.
 *
 * Mirrors upstream's `_resolve_keyword_tier_override`: with semantic matching on, the
 * lexical matcher is *not* consulted — and an embedding failure yields no override, so
 * the prompt falls through to the scorer rather than failing. The failure is reported so
 * the decision can record it.
 */
export async function resolveKeywordTierOverride(
  userMessage: string,
  config: RouterConfig,
  semantic: SemanticMatcher | null | undefined,
): Promise<KeywordOverrideResult> {
  if (config.keywordTierRules.length === 0) return { override: null };
  if (!config.semanticKeywordMatching) {
    return { override: lexicalTierOverride(userMessage, config.keywordTierRules) };
  }
  if (!semantic) return { override: null, failure: "semantic matcher unavailable" };
  try {
    const match = await semantic.match(userMessage);
    if (!match) return { override: null };
    return {
      override: {
        tier: match.tier,
        matchedKeyword: null,
        cause: "semantic_keyword_match",
        signal: `semantic_match (${match.score.toFixed(2)} ≈ "${match.nearestKeyword}")`,
      },
    };
  } catch (err) {
    return { override: null, failure: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * The tier the keyword rules force, if any.
 *
 * When several rules match, the **highest** tier wins. That is what keeps rule order from
 * silently changing behaviour: appending a rule can raise the answer but never lower one
 * an earlier rule already justified.
 */
export function lexicalTierOverride(
  userMessage: string,
  rules: readonly KeywordTierRule[],
): KeywordOverride | null {
  const text = userMessage.toLowerCase();
  let best: KeywordOverride | null = null;

  for (const rule of rules) {
    for (const keyword of rule.keywords) {
      if (!keywordMatches(text, keyword)) continue;
      if (!best || tierSeverity(rule.tier) > tierSeverity(best.tier)) {
        best = { tier: rule.tier, matchedKeyword: keyword, cause: "literal_keyword_match" };
      }
      break; // one match is enough to apply this rule
    }
  }
  return best;
}

/**
 * The escalation keyword present in `userMessage`, if any.
 *
 * Case-sensitive by design: a sentinel that fired on ordinary prose would let any message
 * mentioning the word buy a stronger model.
 */
export function matchedEscalationKeyword(userMessage: string, keywords: readonly string[]): string | null {
  for (const keyword of keywords) {
    if (keyword && userMessage.includes(keyword)) return keyword;
  }
  return null;
}

/** One tier up, saturating at the top. */
export function escalateTier(tier: Tier): Tier {
  const next = TIER_SEVERITY_ORDER[Math.min(tierSeverity(tier) + 1, TIER_SEVERITY_ORDER.length - 1)];
  return next ?? tier;
}

/** Raise `tier` to the floor when the floor is higher; the classified tier wins otherwise. */
export function applyFloor(tier: Tier, floor: Tier | null): Tier {
  if (!floor) return tier;
  return tierSeverity(floor) > tierSeverity(tier) ? floor : tier;
}

/** True when the plan-mode floor is the highest tier that has any models configured, in
 *  which case nothing downstream could outrank it and the classifier call can be skipped. */
export function floorIsTopConfiguredTier(config: RouterConfig): boolean {
  const floor = config.planModeMinTier;
  if (!floor) return false;
  const configured = TIER_SEVERITY_ORDER.filter((t) => config.tiers[t].length > 0);
  if (configured.length === 0) return false;
  const top = configured[configured.length - 1];
  return top !== undefined && tierSeverity(floor) >= tierSeverity(top);
}
