/**
 * Turning a tier into a model pi will actually run.
 *
 * The safety rail of the whole extension: a candidate that is not in pi's registry, that
 * pi has no credentials for, or that is itself a virtual model is demoted rather than
 * fatal. Resolution is pure — it never selects anything — so the virtual model's router
 * can walk the whole chain before it commits to a model for the request.
 */

import type { RouterConfig } from "./config.ts";
import { splitModelRef } from "./classify/llm.ts";
import {
  TIER_SEVERITY_ORDER,
  type ModelRef,
  type ThinkingLevel,
  type Tier,
  type TierTarget,
  tierSeverity,
} from "./types.ts";

/** API id pi gives virtual catalog entries. Requests for it fail unless routed first. */
export const VIRTUAL_MODEL_API = "pi-virtual";

/** Whether `model` is a virtual catalog entry. A virtual model cannot route to another one,
 *  so such an entry is never a usable candidate — this extension's own included. */
export function isVirtualModel(model: unknown): boolean {
  return Boolean(model && typeof model === "object" && (model as { api?: unknown }).api === VIRTUAL_MODEL_API);
}

/** The slice of pi's model registry this module needs. Narrow on purpose: pi's own
 *  `ModelRegistry` satisfies it as-is, and it is trivial to fake in tests. */
export interface ModelLookup {
  find(provider: string, modelId: string): unknown;
  /** False when pi has no credentials for the model's provider. When omitted, every model
   *  the registry knows counts as usable. */
  hasConfiguredAuth?(model: never): boolean;
}

export interface Resolved {
  model: ModelRef;
  thinkingLevel: ThinkingLevel | null;
  /** pi's model object for `model`, ready to hand back from the virtual model's router. */
  target: unknown;
  /** Set when the first choice could not be used. */
  fellBackBecause?: string;
}

const describeError = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/**
 * pi's model object for `ref`, or why there is none that can serve a request.
 *
 * Problems are phrased as `"<ref>" <reason>` so the decision log and the failure toast can
 * name the model that was skipped.
 */
export function resolveModelRef(ref: ModelRef, lookup: ModelLookup): { target: unknown } | { problem: string } {
  const split = splitModelRef(ref);
  if (!split) return { problem: `"${ref}" is not provider/model-id` };
  let model: unknown;
  let authed = true;
  try {
    model = lookup.find(split.provider, split.modelId);
    if (model && !isVirtualModel(model) && lookup.hasConfiguredAuth) {
      authed = lookup.hasConfiguredAuth(model as never);
    }
  } catch (err) {
    return { problem: `"${ref}" failed to resolve: ${describeError(err)}` };
  }
  if (!model) return { problem: `"${ref}" is not in pi's model registry` };
  if (isVirtualModel(model)) return { problem: `"${ref}" is a virtual model` };
  // No credentials is not an error, it just means this candidate is not available to this
  // user right now.
  if (!authed) return { problem: `"${ref}" has no available credentials` };
  return { target: model };
}

/** `defaultModel` as a candidate, thinking level included; null when none is set. */
export function defaultTarget(config: RouterConfig): TierTarget | null {
  if (!config.defaultModel) return null;
  const target: TierTarget = { model: config.defaultModel };
  if (config.defaultModelThinkingLevel) target.thinkingLevel = config.defaultModelThinkingLevel;
  return target;
}

/** Uniform draw in [0, 1). Injected so tests can pin the pool pick. */
export type Rng = () => number;

/** `random.choice`: one entry of a non-empty list, uniformly. */
export function randomChoice<T>(items: readonly T[], rng: Rng): T | undefined {
  if (items.length === 0) return undefined;
  return items[Math.min(items.length - 1, Math.floor(rng() * items.length))];
}

/**
 * Candidates for `tier`, best first.
 *
 * The head of the list is upstream's `get_model_for_tier`: a random pick from the tier's
 * pool (`_pick_from_tier_value` is `random.choice` over a list), else `defaultModel`, else
 * a random pick from MEDIUM. Upstream stops there and raises if nothing is configured.
 *
 * The tail is pi's safety rail, which upstream does not need because a proxy has no notion
 * of a model the user lacks credentials for: the rest of the pool, then progressively lower
 * tiers, then `defaultModel`. Walking *down* rather than up is deliberate — if the tier a
 * request was classified into is unusable, serving it from a cheaper model is a
 * degradation, while silently promoting it to a more expensive one is a bill the user did
 * not ask for.
 */
export function candidatesForTier(tier: Tier, config: RouterConfig, rng: Rng = Math.random): TierTarget[] {
  const candidates: TierTarget[] = [];
  const seen = new Set<string>();

  const push = (target: TierTarget | null | undefined) => {
    if (!target) return;
    const key = `${target.model}::${target.thinkingLevel ?? ""}`;
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(target);
  };

  // ── upstream's pick ──
  const pool = config.tiers[tier];
  if (pool.length > 0) {
    push(randomChoice(pool, rng));
  } else if (config.defaultModel) {
    push(defaultTarget(config));
  } else {
    push(randomChoice(config.tiers.MEDIUM, rng));
  }

  // ── pi's credential rail ──
  for (const target of pool) push(target);
  for (let severity = tierSeverity(tier) - 1; severity >= 0; severity--) {
    const lower = TIER_SEVERITY_ORDER[severity];
    if (!lower) continue;
    for (const target of config.tiers[lower]) push(target);
  }
  push(defaultTarget(config));
  return candidates;
}

/**
 * Resolve the first usable candidate, best first.
 *
 * Nothing is selected here: the caller hands the result back from the virtual model's
 * router, and pi clamps the thinking level to the model the request is routed to.
 */
export function resolveFirstUsable(
  candidates: readonly TierTarget[],
  lookup: ModelLookup,
): { resolved: Resolved | null; problems: string[] } {
  const problems: string[] = [];

  for (const candidate of candidates) {
    const result = resolveModelRef(candidate.model, lookup);
    if ("problem" in result) {
      problems.push(result.problem);
      continue;
    }
    const resolved: Resolved = {
      model: candidate.model,
      thinkingLevel: candidate.thinkingLevel ?? null,
      target: result.target,
    };
    // Problems accumulated before the winner are the reason this was not the first choice.
    if (problems.length > 0) resolved.fellBackBecause = problems.join("; ");
    return { resolved, problems };
  }

  return { resolved: null, problems };
}
