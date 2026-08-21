/**
 * Turning a tier into a model pi will actually run.
 *
 * The safety rail of the whole extension: a candidate that is not in pi's registry, or
 * that pi has no credentials for, is demoted rather than fatal. Routing never fails a
 * prompt — the worst case is that the session keeps the model it already had.
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

/** The slice of pi's API this module needs. Narrow on purpose: it is the whole surface the
 *  extension depends on for applying a decision, so it is easy to fake in tests. */
export interface ModelApplier {
  find(provider: string, modelId: string): unknown;
  /** Returns false when no API key is available for the model. */
  setModel(model: never): Promise<boolean>;
  setThinkingLevel?(level: ThinkingLevel): void;
}

export interface Applied {
  model: ModelRef;
  thinkingLevel: ThinkingLevel | null;
  /** Set when the first choice could not be applied. */
  fellBackBecause?: string;
}

/**
 * Candidates for `tier`, best first.
 *
 * The tier's own targets come first, then progressively lower tiers, then `defaultModel`.
 * Walking *down* rather than up is deliberate: if the tier a request was classified into
 * is unusable, serving it from a cheaper model is a degradation, while silently promoting
 * it to a more expensive one is a bill the user did not ask for.
 */
export function candidatesForTier(tier: Tier, config: RouterConfig): TierTarget[] {
  const candidates: TierTarget[] = [];
  const seen = new Set<string>();

  const push = (target: TierTarget) => {
    const key = `${target.model}::${target.thinkingLevel ?? ""}`;
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(target);
  };

  for (const target of config.tiers[tier]) push(target);

  for (let severity = tierSeverity(tier) - 1; severity >= 0; severity--) {
    const lower = TIER_SEVERITY_ORDER[severity];
    if (!lower) continue;
    for (const target of config.tiers[lower]) push(target);
  }

  if (config.defaultModel) push({ model: config.defaultModel });
  return candidates;
}

/**
 * Apply the first usable candidate.
 *
 * `setThinkingLevel` is called *after* `setModel`, never before: pi clamps the level to
 * the model's capabilities, so setting it against the outgoing model would clamp against
 * the wrong ceiling.
 */
export async function applyFirstUsable(
  candidates: readonly TierTarget[],
  api: ModelApplier,
): Promise<{ applied: Applied | null; problems: string[] }> {
  const problems: string[] = [];

  for (const candidate of candidates) {
    const ref = splitModelRef(candidate.model);
    if (!ref) {
      problems.push(`"${candidate.model}" is not provider/model-id`);
      continue;
    }
    const model = api.find(ref.provider, ref.modelId);
    if (!model) {
      problems.push(`"${candidate.model}" is not in pi's model registry`);
      continue;
    }
    let ok = false;
    try {
      ok = await api.setModel(model as never);
    } catch (err) {
      problems.push(`"${candidate.model}" failed to apply: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }
    if (!ok) {
      // pi returns false when there is no API key for the model. That is not an error, it
      // just means this candidate is not available to this user right now.
      problems.push(`"${candidate.model}" has no available credentials`);
      continue;
    }

    if (candidate.thinkingLevel && api.setThinkingLevel) {
      try {
        api.setThinkingLevel(candidate.thinkingLevel);
      } catch {
        // A thinking level that the model cannot honour is not worth losing the model over.
      }
    }

    const applied: Applied = {
      model: candidate.model,
      thinkingLevel: candidate.thinkingLevel ?? null,
    };
    // Problems accumulated before the winner are the reason this was not the first choice.
    if (problems.length > 0) applied.fellBackBecause = problems.join("; ");
    return { applied, problems };
  }

  return { applied: null, problems };
}
