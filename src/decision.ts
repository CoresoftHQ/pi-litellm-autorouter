/**
 * Rendering and persisting routing decisions.
 *
 * Decisions are stored with `pi.appendEntry()`, which does not participate in LLM context
 * — the routing log costs no tokens and cannot influence the model it describes.
 */

import type { RouteDecision } from "./types.ts";

export const DECISION_ENTRY_TYPE = "autoroute-decision";
export const STATE_ENTRY_TYPE = "autoroute-state";

/** Persisted session state: the user's own overrides, which outrank the router. */
export interface AutorouteState {
  /** Routing disabled for this session via `/autoroute off`. */
  disabled?: boolean;
  /** A model the user pinned by hand; the router leaves the session alone while set. */
  pinnedModel?: string | null;
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
  return lines.join("\n");
}
