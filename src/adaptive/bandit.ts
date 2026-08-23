/**
 * Thompson sampling and prior initialisation for the adaptive router's bandit.
 *
 * Port of LiteLLM's `router_strategy/adaptive_router/bandit.py`. Each (request type,
 * model) cell is a Beta(alpha, beta) posterior:
 *   alpha = pseudo-successes, beta = pseudo-failures, mean = alpha / (alpha + beta),
 *   total samples = alpha + beta - COLD_START_MASS (the prior is informative, not data).
 *
 * Everything here is pure and synchronous; the only non-determinism is the injected RNG.
 */

import type { AdaptivePreferences, RequestType } from "../types.ts";

/** Uniform draw in [0, 1). Injected so tests can make the sampler deterministic. */
export type Rng = () => number;

// Calibration constants, kept byte-comparable with upstream's `adaptive_router/config.py`.
// Upstream marks them UNVALIDATED first-pass guesses; they are tuned there, not here.

/** Prior mean per declared quality tier. */
export const BASE_TIER_WEIGHT: Readonly<Record<1 | 2 | 3, number>> = { 1: 0.3, 2: 0.5, 3: 0.7 };
/** Added to the prior mean for a request type the model declares as a strength. */
export const STRENGTH_BONUS = 0.3;
/** Total prior mass, so ~10 real observations can move the posterior noticeably. */
export const COLD_START_MASS = 10.0;
/** Hard cap on alpha + beta. Updates that would exceed it are dropped, not rescaled. */
export const SAMPLE_CAP = 200;

export interface BanditCell {
  readonly alpha: number;
  readonly beta: number;
}

export function cellMean(cell: BanditCell): number {
  const total = cell.alpha + cell.beta;
  return total > 0 ? cell.alpha / total : 0.5;
}

/** Observations that have moved the posterior, excluding the prior's mass. */
export function cellTotalSamples(cell: BanditCell): number {
  return Math.max(0, Math.trunc(cell.alpha + cell.beta - COLD_START_MASS));
}

/**
 * Cold-start prior for a (model, request type) cell.
 *
 * mean = BASE_TIER_WEIGHT[qualityTier] + STRENGTH_BONUS if the type is a declared strength,
 * capped at 0.95 so no prior is over-confident.
 */
export function initialCell(prefs: AdaptivePreferences, requestType: RequestType): BanditCell {
  const base = BASE_TIER_WEIGHT[prefs.qualityTier];
  const bonus = prefs.strengths.includes(requestType) ? STRENGTH_BONUS : 0;
  const mean = Math.min(0.95, base + bonus);
  return { alpha: mean * COLD_START_MASS, beta: (1 - mean) * COLD_START_MASS };
}

/** Apply a learning update, enforcing the sample cap. */
export function applyDelta(cell: BanditCell, deltaAlpha: number, deltaBeta: number): BanditCell {
  const alpha = cell.alpha + deltaAlpha;
  const beta = cell.beta + deltaBeta;
  if (alpha + beta > SAMPLE_CAP) return cell;
  return { alpha, beta };
}

/** Standard normal via Box–Muller. */
function normalVariate(rng: Rng): number {
  let u = 0;
  // Guard the log against a zero draw.
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gamma(shape, 1) via Marsaglia & Tsang, with the shape < 1 boost. */
function gammaVariate(shape: number, rng: Rng): number {
  if (shape < 1) {
    // Gamma(a) = Gamma(a + 1) · U^(1/a)
    let u = 0;
    while (u === 0) u = rng();
    return gammaVariate(shape + 1, rng) * u ** (1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normalVariate(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    let u = 0;
    while (u === 0) u = rng();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Beta(alpha, beta) as a ratio of gammas, matching Python's `random.betavariate`. */
export function betaVariate(alpha: number, beta: number, rng: Rng): number {
  const x = gammaVariate(alpha, rng);
  if (x === 0) return 0;
  const y = gammaVariate(beta, rng);
  return x / (x + y);
}

/** One draw from the cell's posterior: a quality estimate in [0, 1]. */
export function thompsonSample(cell: BanditCell, rng: Rng = Math.random): number {
  return betaVariate(cell.alpha, cell.beta, rng);
}

/**
 * Map a raw cost onto [0, 1] where 1 = cheapest of `allCosts` and 0 = most expensive.
 * Returns 0.5 when there is no spread, so cost neither helps nor hurts.
 */
export function normalizedCost(modelCost: number, allCosts: readonly number[]): number {
  if (allCosts.length === 0) return 0.5;
  const lo = Math.min(...allCosts);
  const hi = Math.max(...allCosts);
  if (hi === lo) return 0.5;
  return 1 - (modelCost - lo) / (hi - lo);
}
