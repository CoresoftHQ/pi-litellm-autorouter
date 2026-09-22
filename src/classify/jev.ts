/**
 * TypeSafe System One (JEV) classifier.
 *
 * JEV returns a typed Choice result instead of generated text, so it is called directly
 * at TypeSafe's `/v1/systemone` endpoint rather than through pi's model registry.
 */

import type { RouterConfig } from "../config.ts";
import type { Classification, ExtractedTurn } from "../types.ts";
import { isTier } from "../types.ts";
import { classificationSystemPrompt, classifierUserPayload } from "./rubrics.ts";

export type JevOutcome =
  | { ok: true; classification: Classification }
  | { ok: false; reason: string };

export interface JevFetchResponse {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
}

export type JevFetch = (url: string, init: RequestInit) => Promise<JevFetchResponse>;

/** Per-router timeout breaker. It only opens on a deadline, matching LiteLLM's recovery
 * behavior: ordinary HTTP failures fall back but do not suppress the next request. */
export class JevClassifier {
  private openUntil = 0;

  constructor(private readonly fetchImpl: JevFetch = fetch as unknown as JevFetch, private readonly now = Date.now) {}

  async classify(
    turn: ExtractedTurn,
    config: RouterConfig,
    callerSystemPrompt?: string,
  ): Promise<JevOutcome> {
    const jev = config.jevClassifierConfig;
    if (!jev) return { ok: false, reason: "no JEV classifier configured" };
    if (!turn.currentAsk) return { ok: false, reason: "nothing to classify" };

    if (jev.circuitBreakerEnabled && this.openUntil > this.now()) {
      return { ok: false, reason: "JEV circuit breaker is open" };
    }

    const apiKey = process.env[jev.apiKeyEnv];
    if (!apiKey) return { ok: false, reason: `JEV API key is missing (${jev.apiKeyEnv})` };

    const state = classifierUserPayload({
      prompt: turn.currentAsk,
      ...(callerSystemPrompt !== undefined ? { callerSystemPrompt } : {}),
      priorTurns: turn.priorTurns,
      hasPriorConversation: turn.conversationContinuing,
      cumulativeTokens: turn.cumulativeTokens,
      labelRoles: config.classifierContextIncludeAssistantTurns,
    });
    const instructions = jev.instructions ?? classificationSystemPrompt({
      contextWindowSize: config.classifierContextWindowSize,
      rubric: "agentic",
    });
    const base = (jev.apiBase ?? process.env.TYPESAFE_API_BASE ?? "https://api.typesafe.ai").replace(/\/+$/, "");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), jev.timeoutMs);

    try {
      const response = await this.fetchImpl(`${base}/v1/systemone`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: jev.model,
          state,
          questions: {
            tier: {
              type: "choice",
              instructions,
              criteria: {
                SIMPLE: "Quick factual lookups, greetings, or requests with a short known answer.",
                MEDIUM: "Routine work requiring explanation, light reasoning, or minor technical content.",
                COMPLEX: "Non-trivial code, architecture, multi-step technical work, or specialized depth.",
                REASONING: "Proofs, difficult tradeoffs, optimization, or careful extended reasoning.",
              },
            },
          },
        }),
        signal: controller.signal,
      });
      if (!response.ok) return { ok: false, reason: `JEV request failed with HTTP ${response.status}` };
      const body = await response.json();
      const choice = body && typeof body === "object"
        ? (body as { answers?: { tier?: { choice?: unknown } } }).answers?.tier?.choice
        : undefined;
      if (!isTier(choice)) return { ok: false, reason: "JEV response named no known tier" };
      this.openUntil = 0;
      return {
        ok: true,
        classification: { tier: choice, signals: ["jev_classifier"], cause: "jev_classifier" },
      };
    } catch (err) {
      const timedOut = controller.signal.aborted;
      if (timedOut && jev.circuitBreakerEnabled) {
        this.openUntil = this.now() + jev.circuitBreakerCooldownSeconds * 1000;
      }
      return {
        ok: false,
        reason: timedOut
          ? `JEV classifier timed out after ${jev.timeoutMs}ms`
          : `JEV classifier call failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    } finally {
      clearTimeout(timer);
    }
  }
}
