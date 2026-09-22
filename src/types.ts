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

export const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;

export type ThinkingLevel = (typeof THINKING_LEVELS)[number];

export function isThinkingLevel(value: unknown): value is ThinkingLevel {
  return typeof value === "string" && (THINKING_LEVELS as readonly string[]).includes(value);
}

/** A model reference as written in config: `provider/modelId`. */
export type ModelRef = string;

/** The adaptive router's fixed request taxonomy. Mirrors LiteLLM's `RequestType`; bandit
 *  posteriors are kept per (request type, model), so a model can be learned to be good at
 *  code and weak at prose. */
export const REQUEST_TYPES = [
  "code_generation",
  "code_understanding",
  "technical_design",
  "analytical_reasoning",
  "writing",
  "factual_lookup",
  "general",
] as const;

export type RequestType = (typeof REQUEST_TYPES)[number];

export function isRequestType(value: unknown): value is RequestType {
  return typeof value === "string" && (REQUEST_TYPES as readonly string[]).includes(value);
}

/** A model's self-declared standing for the adaptive router's cold-start prior. Mirrors
 *  LiteLLM's `model_info.adaptive_router_preferences`. */
export interface AdaptivePreferences {
  /** 1 = budget, 2 = balanced (the default), 3 = frontier. Sets the prior mean. */
  qualityTier: 1 | 2 | 3;
  /** Request types the model is believed to be strong at; each gets a prior bonus. */
  strengths: RequestType[];
}

/** One candidate a tier can route to. */
export interface TierTarget {
  model: ModelRef;
  thinkingLevel?: ThinkingLevel;
  /** Only read when `adaptive` is on. */
  qualityTier?: AdaptivePreferences["qualityTier"];
  strengths?: RequestType[];
}

/** Why a particular model was chosen. Mirrors LiteLLM's `routing_decision.cause`
 *  so decisions stay comparable across the two implementations. */
export type DecisionCause =
  | "one_shot_override"
  | "heuristic_scorer"
  | "reasoning_override"
  | "llm_classifier"
  | "jev_classifier"
  | "literal_keyword_match"
  | "semantic_keyword_match"
  | "plan_mode"
  | "session_affinity_pin"
  | "session_affinity_escalation"
  | "question_reply"
  | "question_reply_escalation"
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
  /** How the adaptive bandit chose, when `adaptive` is on and a tier was routed. Mirrors
   *  the `adaptive_router_decision` metadata LiteLLM stamps on the request. */
  adaptive?: AdaptiveDecision;
}

export interface AdaptiveCandidate {
  model: ModelRef;
  /** Cold-start phase: observations so far for this (request type, model) cell. */
  totalSamples?: number;
  /** Adaptive phase: the Thompson draw and the terms it was scored with. */
  qualitySample?: number;
  costScore?: number;
  tierDistance?: number;
  score?: number;
}

export interface AdaptiveDecision {
  /** `cold_start` picks uniformly among unobserved models in the classified tier;
   *  `adaptive` scores Thompson draws against cost and tier distance. */
  phase: "cold_start" | "adaptive";
  classifiedTier: Tier;
  requestType: RequestType;
  eligibleMode: "all" | "classified_tier";
  qualityWeight: number;
  costWeight: number;
  tierDistancePenalty: number;
  chosenModel: ModelRef;
  candidates: AdaptiveCandidate[];
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
