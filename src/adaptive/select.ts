/**
 * The adaptive pick: soft complexity floors over a Thompson-sampled pool.
 *
 * Port of LiteLLM's `ComplexityRouter._soft_floor_pick`. The classified tier is a *home*,
 * not a cage: with `adaptiveEligible: "all"` every pool model is scored and only penalised
 * by its tier distance, so a cheap model with a strong posterior can win a COMPLEX request
 * and a frontier model can be pulled down to a SIMPLE one it has proven itself on.
 */

import type { RouterConfig } from "../config.ts";
import { type AdaptiveRouter } from "./router.ts";
import { cellTotalSamples, normalizedCost, thompsonSample } from "./bandit.ts";
import { classifyRequestType } from "./request-type.ts";
import {
  type AdaptiveCandidate,
  type AdaptiveDecision,
  type ModelRef,
  TIER_SEVERITY_ORDER,
  type Tier,
  type TierTarget,
  tierSeverity,
} from "../types.ts";

/** Every model in any pool, in tier order then config order, each listed once. */
export function poolModels(config: RouterConfig): ModelRef[] {
  const models: ModelRef[] = [];
  for (const tier of TIER_SEVERITY_ORDER) {
    for (const target of config.tiers[tier]) {
      if (!models.includes(target.model)) models.push(target.model);
    }
  }
  return models;
}

/** The tiers whose pool contains `model`. */
export function tiersForModel(model: ModelRef, config: RouterConfig): Tier[] {
  return TIER_SEVERITY_ORDER.filter((tier) => config.tiers[tier].some((target) => target.model === model));
}

/**
 * The config entry to apply for `model` once the bandit has chosen it: the classified
 * tier's own entry when the model sits there, else the entry from the nearest tier that
 * lists it. The bandit picks a *model*; the thinking level rides along from config.
 */
export function targetForModel(model: ModelRef, classifiedTier: Tier, config: RouterConfig): TierTarget {
  const home = config.tiers[classifiedTier].find((target) => target.model === model);
  if (home) return home;
  const classifiedIdx = tierSeverity(classifiedTier);
  const nearest = tiersForModel(model, config).sort(
    (a, b) => Math.abs(tierSeverity(a) - classifiedIdx) - Math.abs(tierSeverity(b) - classifiedIdx),
  )[0];
  const target = nearest ? config.tiers[nearest].find((t) => t.model === model) : undefined;
  return target ?? { model };
}

export interface SoftFloorPickInput {
  classifiedTier: Tier;
  userMessage: string;
  config: RouterConfig;
  adaptive: AdaptiveRouter;
  /**
   * Excludes every candidate whose tiers all sit below it, turning the soft floor into a
   * hard minimum for requests that carry one (the plan-mode floor). The classified tier
   * arrives already clamped to the floor, so the cold-start pool and the `classified_tier`
   * mode satisfy it by construction; only `all` can reach below.
   */
  hardFloor?: Tier | null;
}

export interface SoftFloorPick {
  model: ModelRef;
  decision: AdaptiveDecision;
}

/** Returns null when the pools give the bandit nothing to choose from. */
export function softFloorPick(input: SoftFloorPickInput): SoftFloorPick | null {
  const { classifiedTier, config, adaptive } = input;
  const rng = adaptive.rng;
  const requestType = classifyRequestType(input.userMessage);
  const classifiedIdx = tierSeverity(classifiedTier);
  const classifiedCandidates = [...new Set(config.tiers[classifiedTier].map((t) => t.model))];

  const common = {
    classifiedTier,
    requestType,
    qualityWeight: config.adaptiveWeights.quality,
    costWeight: config.adaptiveWeights.cost,
    tierDistancePenalty: config.tierDistancePenalty,
  };

  // ── Cold start: explore the home tier uniformly until every model has data ──
  const coldStart = classifiedCandidates.filter((model) => cellTotalSamples(adaptive.cell(requestType, model)) === 0);
  if (coldStart.length > 0) {
    const chosen = coldStart[Math.min(coldStart.length - 1, Math.floor(rng() * coldStart.length))];
    if (chosen === undefined) return null;
    return {
      model: chosen,
      decision: {
        ...common,
        phase: "cold_start",
        eligibleMode: "classified_tier",
        chosenModel: chosen,
        candidates: coldStart.map((model) => ({
          model,
          totalSamples: cellTotalSamples(adaptive.cell(requestType, model)),
        })),
      },
    };
  }

  // ── Adaptive: Thompson-sample, score against cost and distance ────────────
  let candidates: ModelRef[];
  if (config.adaptiveEligible === "classified_tier") {
    candidates = classifiedCandidates;
    if (candidates.length === 0) return null;
  } else {
    candidates = adaptive.availableModels.slice();
  }

  const allCosts = candidates.map((model) => adaptive.cost(model));
  const floorSeverity = input.hardFloor ? tierSeverity(input.hardFloor) : null;

  let bestModel: ModelRef | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  const scored: AdaptiveCandidate[] = [];
  for (const model of candidates) {
    const modelTiers = tiersForModel(model, config);
    const tiers = modelTiers.length > 0 ? modelTiers : [classifiedTier];
    if (floorSeverity !== null && tiers.every((tier) => tierSeverity(tier) < floorSeverity)) continue;

    const qualitySample = thompsonSample(adaptive.cell(requestType, model), rng);
    const costScore = normalizedCost(adaptive.cost(model), allCosts);
    const distance =
      config.adaptiveEligible === "classified_tier"
        ? 0
        : Math.min(...tiers.map((tier) => Math.abs(tierSeverity(tier) - classifiedIdx)));
    const score =
      common.qualityWeight * qualitySample + common.costWeight * costScore - common.tierDistancePenalty * distance;
    scored.push({ model, qualitySample, costScore, tierDistance: distance, score });
    if (score > bestScore) {
      bestScore = score;
      bestModel = model;
    }
  }
  if (bestModel === null) return null;
  return {
    model: bestModel,
    decision: {
      ...common,
      phase: "adaptive",
      eligibleMode: config.adaptiveEligible,
      chosenModel: bestModel,
      candidates: scored,
    },
  };
}
