/**
 * Rendering and persisting routing decisions.
 *
 * Decisions are stored with `pi.appendEntry()`, which does not participate in LLM context
 * — the routing log costs no tokens and cannot influence the model it describes.
 */

import type { CellSnapshot } from "./adaptive/router.ts";
import type { RouterConfig } from "./config.ts";
import { REQUEST_TYPES, type RouteDecision } from "./types.ts";

export const DECISION_ENTRY_TYPE = "autoroute-decision";
export const STATE_ENTRY_TYPE = "autoroute-state";

/** Persisted session state: the user's own overrides, which outrank the router. */
export interface AutorouteState {
  /** Routing disabled for this session via `/autoroute off`. */
  disabled?: boolean;
  /** A model the user pinned by hand; the router leaves the session alone while set. */
  pinnedModel?: string | null;
  /** A model forced for the next prompt only, then cleared. Outranks everything else,
   *  including `disabled` and `pinnedModel` — it is a deliberate one-off instruction. */
  nextModel?: string | null;
}

/** Compact footer text: `⏵ COMPLEX · claude-sonnet-5`. */
export function statusLine(decision: RouteDecision): string {
  if (!decision.chosenModel) return "autoroute: no change";
  const model = decision.chosenModel.split("/").slice(1).join("/") || decision.chosenModel;
  const parts = [decision.tier ?? decision.cause, model];
  if (decision.escalated) parts.push("↑");
  if (decision.planFloored) parts.push("plan");
  return parts.join(" · ");
}

/** Multi-line explanation for `/autoroute` and `/autoroute explain`. */
export function explain(decision: RouteDecision, dimensions?: { name: string; score: number; signal?: string }[]): string {
  const lines: string[] = [];
  lines.push(`tier:     ${decision.tier ?? "(none)"}`);
  lines.push(`cause:    ${decision.cause}`);
  lines.push(`model:    ${decision.chosenModel ?? "(unchanged)"}`);
  if (decision.thinkingLevel) lines.push(`thinking: ${decision.thinkingLevel}`);
  if (decision.score !== null) lines.push(`score:    ${decision.score.toFixed(3)}`);
  if (decision.matchedKeyword) lines.push(`keyword:  ${decision.matchedKeyword}`);
  if (decision.escalationKeyword) lines.push(`escalate: ${decision.escalationKeyword}`);
  if (decision.signals.length > 0) lines.push(`signals:  ${decision.signals.join(", ")}`);
  lines.push(`latency:  ${decision.latencyMs}ms`);
  if (decision.fellBackBecause) lines.push(`fallback: ${decision.fellBackBecause}`);

  if (dimensions && dimensions.length > 0) {
    lines.push("");
    lines.push("dimensions:");
    for (const d of dimensions) {
      const signal = d.signal ? `  ${d.signal}` : "";
      lines.push(`  ${d.name.padEnd(18)} ${d.score.toFixed(2).padStart(6)}${signal}`);
    }
  }

  const adaptive = decision.adaptive;
  if (adaptive) {
    lines.push("");
    lines.push(
      `adaptive: ${adaptive.phase} (${adaptive.requestType}, ${adaptive.eligibleMode}, ` +
        `quality ${adaptive.qualityWeight} / cost ${adaptive.costWeight}, penalty ${adaptive.tierDistancePenalty})`,
    );
    for (const c of adaptive.candidates) {
      const marker = c.model === adaptive.chosenModel ? "▸" : " ";
      if (adaptive.phase === "cold_start") {
        lines.push(`  ${marker} ${c.model.padEnd(36)} samples ${c.totalSamples ?? 0}`);
      } else {
        lines.push(
          `  ${marker} ${c.model.padEnd(36)} score ${(c.score ?? 0).toFixed(3).padStart(7)}  ` +
            `q ${(c.qualitySample ?? 0).toFixed(2)}  cost ${(c.costScore ?? 0).toFixed(2)}  dist ${c.tierDistance ?? 0}`,
        );
      }
    }
  }
  return lines.join("\n");
}

/** The bandit's posteriors, one block per request type, for `/autoroute adaptive`. */
export function renderAdaptiveSnapshot(cells: CellSnapshot[], config: RouterConfig, storePath: string): string {
  const lines: string[] = [];
  lines.push(
    `adaptive: ${config.adaptiveEligible}, quality ${config.adaptiveWeights.quality} / cost ${config.adaptiveWeights.cost}, ` +
      `penalty ${config.tierDistancePenalty}`,
  );
  lines.push(`state:    ${storePath}`);
  for (const requestType of REQUEST_TYPES) {
    const rows = cells.filter((c) => c.requestType === requestType);
    if (rows.length === 0) continue;
    lines.push("");
    lines.push(`${requestType}:`);
    for (const row of rows) {
      lines.push(
        `  ${row.model.padEnd(36)} mean ${row.qualityMean.toFixed(2)}  samples ${String(row.samples).padStart(3)}  ` +
          `α ${row.alpha.toFixed(1)} β ${row.beta.toFixed(1)}`,
      );
    }
  }
  return lines.join("\n");
}
