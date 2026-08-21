/**
 * Core types for the router.
 *
 * Tier names, config key names, and default values track LiteLLM's
 * `complexity_router_config` so a config can move between the two systems.
 */

/** Built-in tiers, in ascending severity. Order is load-bearing: escalation and
 *  plan-mode floors both walk this list. */
export const TIER_SEVERITY_ORDER = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const;

export type Tier = (typeof TIER_SEVERITY_ORDER)[number];

export function isTier(value: unknown): value is Tier {
  return typeof value === "string" && (TIER_SEVERITY_ORDER as readonly string[]).includes(value);
}

/** Ascending-severity index, or -1 for an unknown tier. */
export function tierSeverity(tier: Tier): number {
  return TIER_SEVERITY_ORDER.indexOf(tier);
}

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/** A model reference as written in config: `provider/modelId`. */
export type ModelRef = string;

/** One candidate a tier can route to. */
export interface TierTarget {
  model: ModelRef;
  thinkingLevel?: ThinkingLevel;
}

/** Why a particular model was chosen. Mirrors LiteLLM's `routing_decision.cause`
 *  so decisions stay comparable across the two implementations. */
export type DecisionCause =
  | "one_shot_override"
  | "heuristic_scorer"
  | "reasoning_override"
  | "llm_classifier"
  | "literal_keyword_match"
  | "semantic_keyword_match"
  | "plan_mode"
  | "session_affinity_pin"
  | "session_affinity_escalation"
  | "default_fallback"
  | "default_model_fallback"
  | "classifier_failed";

/** One dimension's contribution to the heuristic score. */
export interface DimensionScore {
  name: string;
  score: number;
  signal?: string;
}

/** What the classifier concluded, before overrides are applied. */
export interface Classification {
  tier: Tier;
  /** Weighted score for the heuristic scorer; undefined for the LLM classifier,
   *  which returns a tier without a numeric score. */
  score?: number;
  signals: string[];
  cause: DecisionCause;
  dimensions?: DimensionScore[];
}

/** The full record of one routing decision, persisted per turn. */
export interface RouteDecision {
  cause: DecisionCause;
  tier: Tier | null;
  score: number | null;
  signals: string[];
  matchedKeyword: string | null;
  escalationKeyword: string | null;
  escalated: boolean;
  planFloored: boolean;
  /** Model actually applied, as `provider/modelId`. Null when routing left the
   *  session's model untouched. */
  chosenModel: ModelRef | null;
  thinkingLevel: ThinkingLevel | null;
  latencyMs: number;
  /** Set when the intended model could not be applied and a fallback was used. */
  fellBackBecause?: string;
}

/** The extracted, classifiable view of a turn. */
export interface ExtractedTurn {
  /** The current ask, with reminder blocks stripped. Null when nothing human remains. */
  currentAsk: string | null;
  /** Prior turns, oldest-first, already truncated. */
  priorTurns: { role: "user" | "assistant"; text: string }[];
  /** True when this turn continues an existing conversation. */
  conversationContinuing: boolean;
  /** Cumulative size of the conversation, in estimated tokens. */
  cumulativeTokens: number;
}
