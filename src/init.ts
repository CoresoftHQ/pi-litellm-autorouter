/**
 * Generating a starting config from the models pi can actually reach.
 *
 * Writing the tier table by hand means knowing which of your models is cheap, which is
 * capable, and how they're named — three things pi already knows. This turns that into a
 * config you can edit, rather than a blank file you have to fill in.
 *
 * The output is a *starting point*, not a recommendation. Price is the only signal
 * available here; it correlates with capability but does not equal it, and nothing in pi's
 * catalogue says which model is good at the work you actually do.
 */

import type { Tier } from "./types.ts";
import { TIER_SEVERITY_ORDER } from "./types.ts";

/** The slice of pi's `Model` this module reads. */
export interface CandidateModel {
  provider: string;
  id: string;
  name?: string;
  reasoning?: boolean;
  input?: readonly string[];
  contextWindow?: number;
  cost?: { input?: number; output?: number };
}

export interface InitOptions {
  /** Only consider models from this provider. */
  provider?: string;
  /** Emit an `llm` classifier block using the cheapest model as the classifier. */
  classifier?: "heuristic" | "llm";
}

export interface InitResult {
  /** The config object, ready to be JSON-serialised. */
  config: Record<string, unknown>;
  /** Which model each tier got, for display. */
  picks: Record<Tier, CandidateModel>;
  /** Human-readable explanation of the choices and their limits. */
  notes: string[];
}

/**
 * Blended price per million tokens, used only for ranking.
 *
 * Weighted towards input because that is what an agent turn is made of: a large context
 * re-read on every turn against a few hundred tokens of reply. Ranking on output rate
 * alone would order the catalogue by a number that barely moves the bill.
 */
export function blendedPrice(model: CandidateModel): number {
  const input = model.cost?.input ?? 0;
  const output = model.cost?.output ?? 0;
  return input * 0.8 + output * 0.2;
}

export function modelRef(model: CandidateModel): string {
  return `${model.provider}/${model.id}`;
}

/** Text-capable, matching the provider filter, and not a duplicate. */
function candidates(models: readonly CandidateModel[], options: InitOptions): CandidateModel[] {
  const seen = new Set<string>();
  const out: CandidateModel[] = [];
  for (const model of models) {
    if (!model.provider || !model.id) continue;
    if (options.provider && model.provider !== options.provider) continue;
    // `input` lists the modalities a model accepts. A model that cannot take text cannot
    // serve a prompt, whatever it costs.
    if (model.input && !model.input.includes("text")) continue;
    const ref = modelRef(model);
    if (seen.has(ref)) continue;
    seen.add(ref);
    out.push(model);
  }
  out.sort((a, b) => {
    const byPrice = blendedPrice(a) - blendedPrice(b);
    if (byPrice !== 0) return byPrice;
    // Same price: prefer the larger context window, then a stable name order so two runs
    // on the same catalogue produce the same config.
    const byContext = (b.contextWindow ?? 0) - (a.contextWindow ?? 0);
    if (byContext !== 0) return byContext;
    return modelRef(a).localeCompare(modelRef(b));
  });
  return out;
}

/**
 * Spread `n` picks across a sorted list, cheapest first, most expensive last.
 *
 * With fewer models than tiers the same model is reused rather than leaving a tier empty:
 * an unconfigured tier would fall through the resolver to a neighbour anyway, and saying so
 * explicitly in the file is easier to read than discovering it at runtime.
 */
function spread<T>(sorted: readonly T[], n: number): T[] {
  if (sorted.length === 0) return [];
  if (sorted.length === 1) return Array.from({ length: n }, () => sorted[0] as T);
  return Array.from({ length: n }, (_unused, i) => {
    const position = Math.round((i * (sorted.length - 1)) / (n - 1));
    return sorted[position] as T;
  });
}

/**
 * Build a starting config from the available models.
 *
 * Throws only when there is nothing to work with; every other degenerate case (one model,
 * no cost data, no reasoning-capable model) produces a usable config plus a note saying
 * what was compromised.
 */
export function buildInitialConfig(models: readonly CandidateModel[], options: InitOptions = {}): InitResult {
  const notes: string[] = [];
  const ranked = candidates(models, options);

  if (ranked.length === 0) {
    throw new Error(
      options.provider
        ? `no usable models from provider "${options.provider}"`
        : "no usable models — pi has no text-capable model with credentials",
    );
  }

  const picked = spread(ranked, TIER_SEVERITY_ORDER.length);
  const picks = {} as Record<Tier, CandidateModel>;
  TIER_SEVERITY_ORDER.forEach((tier, i) => {
    picks[tier] = picked[i] as CandidateModel;
  });

  // The top tier is the one place capability matters more than price: prefer the most
  // expensive model that actually supports extended thinking, when one exists at or above
  // the price point already chosen.
  const topPrice = blendedPrice(picks.REASONING);
  const reasoningCapable = ranked.filter((m) => m.reasoning && blendedPrice(m) >= topPrice);
  const bestReasoning = reasoningCapable[reasoningCapable.length - 1];
  if (bestReasoning && modelRef(bestReasoning) !== modelRef(picks.REASONING)) {
    notes.push(
      `REASONING uses ${modelRef(bestReasoning)} rather than ${modelRef(picks.REASONING)}: ` +
        "it is the priciest model that supports extended thinking.",
    );
    picks.REASONING = bestReasoning;
  }

  const distinct = new Set(TIER_SEVERITY_ORDER.map((t) => modelRef(picks[t])));
  if (distinct.size < TIER_SEVERITY_ORDER.length) {
    notes.push(
      `only ${distinct.size} distinct model(s) across ${TIER_SEVERITY_ORDER.length} tiers — ` +
        "routing can only save money where tiers differ, so add cheaper or pricier models to spread them out.",
    );
  }

  if (ranked.every((m) => blendedPrice(m) === 0)) {
    notes.push(
      "every candidate reports zero cost, so tiers were ordered by name rather than price. " +
        "Check the picks by hand — this ordering is arbitrary.",
    );
  }

  const tiers: Record<string, unknown> = {};
  for (const tier of TIER_SEVERITY_ORDER) {
    const model = picks[tier];
    tiers[tier] =
      tier === "REASONING" && model.reasoning ? { model: modelRef(model), thinkingLevel: "high" } : modelRef(model);
  }

  const config: Record<string, unknown> = {
    // The fallback for a failed classification and for a tier with nothing usable. The
    // cheapest model is the safe choice: a wrong guess costs little.
    defaultModel: modelRef(picks.SIMPLE),
    tiers,
  };

  if (options.classifier === "llm") {
    const classifierModel = ranked[0] as CandidateModel;
    config.classifierType = "llm";
    config.classifierLLMConfig = {
      model: modelRef(classifierModel),
      classificationRubric: "agentic",
      timeoutMs: 3000,
    };
    config.classifierFallback = "heuristic";
    notes.push(
      `classifier uses ${modelRef(classifierModel)}, the cheapest candidate. It runs before every ` +
        "turn, so its latency is added to each prompt — swap it if it is slow.",
    );
  }

  notes.push(
    "Tiers were assigned by price alone, which is a proxy for capability and not the same thing. " +
      "Treat this as a starting point and adjust after seeing what /autoroute explain says about real prompts.",
  );

  return { config, picks, notes };
}

/** A preview table for the terminal. */
export function renderInitPreview(result: InitResult): string {
  const lines: string[] = [];
  for (const tier of TIER_SEVERITY_ORDER) {
    const model = result.picks[tier];
    const price = blendedPrice(model);
    const priceLabel = price > 0 ? `~$${price.toFixed(2)}/Mtok blended` : "no cost data";
    lines.push(`  ${tier.padEnd(10)} ${modelRef(model).padEnd(46)} ${priceLabel}`);
  }
  lines.push("");
  for (const note of result.notes) lines.push(`  · ${note}`);
  return lines.join("\n");
}
