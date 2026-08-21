/**
 * The routing pipeline.
 *
 * Order matches LiteLLM's `async_pre_routing_hook` / `_classify_and_route`, which is
 * load-bearing rather than incidental:
 *
 *   1. session-affinity pin              (skips classification entirely)
 *   2. plan-mode floor, when it is the top configured tier (skips the classifier call)
 *   3. keyword tier rules
 *   4. classifier (llm, falling back per `classifierFallback`; else heuristic)
 *   then: escalation keyword (+1 tier), then plan-mode floor
 *
 * This module is deliberately free of pi imports: it takes an applier and a registry, so
 * the whole decision path is testable without a running agent.
 */

import type { RouterConfig } from "./config.ts";
import { classifyHeuristic } from "./classify/heuristic.ts";
import { type ClassifierRegistry, classifyWithLLM } from "./classify/llm.ts";
import {
  applyFloor,
  escalateTier,
  floorIsTopConfiguredTier,
  lexicalTierOverride,
  matchedEscalationKeyword,
} from "./classify/keywords.ts";
import { type ModelApplier, applyFirstUsable, candidatesForTier } from "./resolve.ts";
import type { Classification, ExtractedTurn, RouteDecision, Tier, TierTarget } from "./types.ts";

export interface SessionPin {
  model: string;
  tier: Tier | null;
  /** Epoch millis when this pin stops being honoured. */
  expiresAt: number;
}

export interface RouteInput {
  turn: ExtractedTurn;
  config: RouterConfig;
  api: ModelApplier;
  registry?: ClassifierRegistry;
  /** pi's own plan-mode state, read directly rather than sniffed out of prompt text. */
  planModeActive?: boolean;
  /** The active session pin, if session affinity is on and one has been set. */
  pin?: SessionPin | null;
  /** A model the user forced for this one prompt via `/autoroute next`. Outranks
   *  everything, including the plan-mode floor and a session pin. */
  oneShot?: string | null;
  /** Injected for testability; defaults to `Date.now`. */
  now?: () => number;
  callerSystemPrompt?: string;
}

export interface RouteOutput {
  decision: RouteDecision;
  /** The pin to store for this session, or null to leave the existing one alone. */
  pinToWrite: SessionPin | null;
  /** True when a one-shot override was used up on this turn and the caller should clear
   *  it. Set even if the model could not be applied — a failed override is still spent,
   *  or it would silently leak into the next prompt. */
  consumedOneShot: boolean;
}

function emptyDecision(): RouteDecision {
  return {
    cause: "default_fallback",
    tier: null,
    score: null,
    signals: [],
    matchedKeyword: null,
    escalationKeyword: null,
    escalated: false,
    planFloored: false,
    chosenModel: null,
    thinkingLevel: null,
    latencyMs: 0,
  };
}

/**
 * Does the prompt carry a plan-mode sentinel?
 *
 * pi's own plan-mode state is authoritative and is checked first. The text patterns are
 * an escape hatch for sentinels that ride in prompt text — the mechanism upstream has to
 * rely on entirely, and which is spoofable by anyone who pastes one. It is a floor either
 * way: a caller can spend *up* to the floor's tier, never down, and never outside the
 * configured pools.
 */
export function planModeSignal(input: RouteInput): string | null {
  if (input.planModeActive) return "pi plan mode";
  const text = input.turn.currentAsk;
  if (!text) return null;
  for (const pattern of input.config.planModePatterns) {
    if (pattern && text.includes(pattern)) return pattern;
  }
  return null;
}

async function classify(input: RouteInput): Promise<Classification | { failed: string }> {
  const { turn, config, registry } = input;
  if (!turn.currentAsk) return { failed: "no user message" };

  if (config.classifierType === "llm" && registry) {
    const outcome = await classifyWithLLM(turn, config, registry, input.callerSystemPrompt);
    if (outcome.ok) return outcome.classification;
    if (config.classifierFallback === "default_model") return { failed: outcome.reason };
    // "heuristic" fallback: score locally and carry on. The signal records that the LLM
    // classifier was asked and did not answer.
    const heuristic = classifyHeuristic(turn.currentAsk, config);
    return { ...heuristic, signals: [...heuristic.signals, `llm_fallback (${outcome.reason})`] };
  }

  return classifyHeuristic(turn.currentAsk, config);
}

/**
 * Decide and apply the model for one turn.
 *
 * Never throws: every path either applies a model or returns a decision explaining why
 * the session's current model was left alone.
 */
export async function route(input: RouteInput): Promise<RouteOutput> {
  const now = input.now ?? Date.now;
  const startedAt = now();
  const { config, turn, api } = input;
  const decision = emptyDecision();

  let consumedOneShot = false;

  const finish = async (
    tier: Tier | null,
    pinToWrite: SessionPin | null,
    explicitCandidates?: TierTarget[],
  ): Promise<RouteOutput> => {
    const candidates =
      explicitCandidates ?? (tier ? candidatesForTier(tier, config) : config.defaultModel ? [{ model: config.defaultModel }] : []);
    const { applied, problems } = await applyFirstUsable(candidates, api);
    decision.tier = tier;
    decision.latencyMs = now() - startedAt;
    if (applied) {
      decision.chosenModel = applied.model;
      decision.thinkingLevel = applied.thinkingLevel;
      if (applied.fellBackBecause) decision.fellBackBecause = applied.fellBackBecause;
      // A pin is only worth writing for a model that was actually applied.
      return {
        decision,
        pinToWrite: pinToWrite ? { ...pinToWrite, model: applied.model } : null,
        consumedOneShot,
      };
    }
    if (problems.length > 0) decision.fellBackBecause = problems.join("; ");
    return { decision, pinToWrite: null, consumedOneShot };
  };

  // ── 0. One-shot override ───────────────────────────────────────────────────
  // The most explicit signal there is: the user naming a model for this prompt. It
  // outranks the plan-mode floor and a session pin alike, and is spent whether or not it
  // could be applied — a failed override that survived would silently hijack the next
  // prompt too.
  if (input.oneShot) {
    consumedOneShot = true;
    decision.cause = "one_shot_override";
    decision.signals = ["one_shot_override"];
    const result = await finish(null, null, [{ model: input.oneShot }]);
    if (result.decision.chosenModel) return result;
    // Could not be applied. Rather than leave the turn on whatever model happened to be
    // active, fall through to ordinary routing — the decision record keeps the reason.
    decision.cause = "default_fallback";
    decision.fellBackBecause = `one-shot override "${input.oneShot}" could not be applied${
      result.decision.fellBackBecause ? `: ${result.decision.fellBackBecause}` : ""
    }`;
  }

  const escalationKeyword = turn.currentAsk
    ? matchedEscalationKeyword(turn.currentAsk, config.escalationKeywords)
    : null;
  decision.escalationKeyword = escalationKeyword;

  const planSentinel = planModeSignal(input);
  const planFloor = planSentinel !== null ? config.planModeMinTier : null;

  // ── 1. Session-affinity pin ────────────────────────────────────────────────
  const pin = input.pin;
  if (config.sessionAffinity && pin && pin.expiresAt > now()) {
    if (escalationKeyword && pin.tier) {
      // Escalation is an explicit ask to re-pin higher, so it rewrites the pin.
      const tier = escalateTier(pin.tier);
      decision.cause = "session_affinity_escalation";
      decision.escalated = true;
      return finish(tier, { model: "", tier, expiresAt: now() + config.sessionAffinityTtlSeconds * 1000 });
    }
    if (planFloor && pin.tier && applyFloor(pin.tier, planFloor) !== pin.tier) {
      // The floor outranks the pin because plan mode is a transient state of the session,
      // not a request to move it. The stored pin deliberately keeps the session's own model
      // so the first turn after plan mode exits routes as if it had never happened.
      const tier = applyFloor(pin.tier, planFloor);
      decision.cause = "plan_mode";
      decision.planFloored = true;
      decision.matchedKeyword = planSentinel;
      return finish(tier, null);
    }
    decision.cause = "session_affinity_pin";
    decision.tier = pin.tier;
    return finish(pin.tier, { ...pin, expiresAt: now() + config.sessionAffinityTtlSeconds * 1000 }, [
      { model: pin.model },
    ]);
  }

  // Nothing human to classify: route to the default model rather than guessing.
  if (!turn.currentAsk) {
    decision.cause = "default_fallback";
    return finish(null, null);
  }

  // ── 2. Plan-mode floor, when nothing could outrank it ──────────────────────
  if (planFloor && floorIsTopConfiguredTier(config)) {
    decision.cause = "plan_mode";
    decision.planFloored = true;
    decision.matchedKeyword = planSentinel;
    return finish(planFloor, null);
  }

  // ── 3. Keyword tier rules ──────────────────────────────────────────────────
  const override = lexicalTierOverride(turn.currentAsk, config.keywordTierRules);
  if (override) {
    let tier = override.tier;
    if (escalationKeyword) {
      tier = escalateTier(tier);
      decision.escalated = true;
    }
    const beforeFloor = tier;
    tier = applyFloor(tier, planFloor);
    decision.planFloored = tier !== beforeFloor;
    decision.cause = decision.planFloored ? "plan_mode" : "literal_keyword_match";
    decision.matchedKeyword = decision.planFloored ? planSentinel : override.matchedKeyword;
    return finish(tier, null);
  }

  // ── 4. Classifier ──────────────────────────────────────────────────────────
  const classification = await classify(input);

  if ("failed" in classification) {
    decision.cause = "default_model_fallback";
    decision.signals = [classification.failed];
    // A sentinel-carrying request skips this exit: defaultModel carries no tier guarantee,
    // so a plan-mode request must land in the floor's pool, which is the only destination
    // the floor can vouch for.
    if (config.defaultModel && !planSentinel) {
      return finish(null, null);
    }
    const tier = applyFloor("MEDIUM", planFloor);
    decision.planFloored = planFloor !== null;
    return finish(tier, null);
  }

  decision.score = classification.score ?? null;
  decision.signals = classification.signals;
  decision.cause = classification.cause;

  let tier = classification.tier;
  if (escalationKeyword) {
    tier = escalateTier(tier);
    decision.escalated = true;
    decision.signals = [...decision.signals, "escalation"];
  }
  const beforeFloor = tier;
  tier = applyFloor(tier, planFloor);
  if (tier !== beforeFloor) {
    decision.planFloored = true;
    decision.signals = [...decision.signals, "plan_mode_floor"];
  }

  // Sentinel presence, not the plan_mode cause, gates the pin write: a plan-mode turn
  // classified at or above the floor keeps its ordinary cause, yet pinning it would carry
  // a plan-mode-shaped choice past plan mode's exit.
  const pinToWrite =
    config.sessionAffinity && !planSentinel
      ? { model: "", tier, expiresAt: now() + config.sessionAffinityTtlSeconds * 1000 }
      : null;

  return finish(tier, pinToWrite);
}
