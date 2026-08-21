/**
 * The LLM classifier.
 *
 * Runs through pi's own `modelRegistry.complete()`, so it reuses whatever credentials,
 * base URL and provider plumbing pi already resolved — no second API key, no separate
 * HTTP client.
 *
 * Every failure path here is non-fatal by construction. Choosing a model is a routing
 * decision; a timeout, a context-limit error, or a reply that names no tier must never
 * fail the user's turn.
 */

import type { RouterConfig } from "../config.ts";
import type { Classification, ExtractedTurn, Tier } from "../types.ts";
import { TIER_SEVERITY_ORDER, isTier } from "../types.ts";
import { classificationSystemPrompt, classifierUserPayload } from "./rubrics.ts";

/** The slice of pi's ModelRegistry this module needs, kept narrow so it can be faked in
 *  tests without standing up the real registry. */
export interface ClassifierRegistry {
  find(provider: string, modelId: string): unknown;
  complete(
    model: never,
    context: { systemPrompt?: string; messages: { role: string; content: unknown }[] },
    options?: unknown,
  ): Promise<{ content?: unknown }>;
}

export type ClassifierOutcome =
  | { ok: true; classification: Classification }
  | { ok: false; reason: string };

/** Split `provider/model-id` on the first slash only: model ids can contain slashes
 *  (openrouter's `anthropic/claude-…` becomes `openrouter/anthropic/claude-…`). */
export function splitModelRef(ref: string): { provider: string; modelId: string } | null {
  const idx = ref.indexOf("/");
  if (idx <= 0 || idx === ref.length - 1) return null;
  return { provider: ref.slice(0, idx), modelId: ref.slice(idx + 1) };
}

/**
 * Find the tier a classifier reply names.
 *
 * Deliberately permissive about shape — a model may answer `REASONING`, `"tier":
 * "REASONING"`, or a sentence containing it — but strict about content: only an exact
 * tier name counts, and the *first* one mentioned wins so trailing chatter cannot
 * silently upgrade the answer.
 */
export function parseTierReply(text: string): Tier | null {
  const upper = text.toUpperCase();
  let best: { tier: Tier; at: number } | null = null;
  for (const tier of TIER_SEVERITY_ORDER) {
    const at = upper.search(new RegExp(`\\b${tier}\\b`));
    if (at === -1) continue;
    if (!best || at < best.at) best = { tier, at };
  }
  return best?.tier ?? null;
}

function replyText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object") {
          const b = block as { type?: unknown; text?: unknown };
          if (typeof b.text === "string") return b.text;
        }
        return "";
      })
      .join(" ");
  }
  return "";
}

/**
 * Classify `turn` with the configured classifier model.
 *
 * Returns `{ ok: false }` rather than throwing on any failure, so the caller can apply
 * `classifierFallback` without a try/catch around every call site.
 */
export async function classifyWithLLM(
  turn: ExtractedTurn,
  config: RouterConfig,
  registry: ClassifierRegistry,
  callerSystemPrompt?: string,
): Promise<ClassifierOutcome> {
  const llm = config.classifierLLMConfig;
  if (!llm) return { ok: false, reason: "no classifier model configured" };
  if (!turn.currentAsk) return { ok: false, reason: "nothing to classify" };

  const ref = splitModelRef(llm.model);
  if (!ref) return { ok: false, reason: `classifier model "${llm.model}" is not provider/model-id` };

  const model = registry.find(ref.provider, ref.modelId);
  if (!model) return { ok: false, reason: `classifier model "${llm.model}" is not in pi's registry` };

  const systemPrompt = classificationSystemPrompt({
    contextWindowSize: config.classifierContextWindowSize,
    rubric: llm.classificationRubric,
    ...(llm.systemPrompt !== undefined ? { customPrompt: llm.systemPrompt } : {}),
  });

  const userPayload = classifierUserPayload({
    prompt: turn.currentAsk,
    ...(callerSystemPrompt !== undefined ? { callerSystemPrompt } : {}),
    priorTurns: turn.priorTurns,
    hasPriorConversation: turn.conversationContinuing,
    cumulativeTokens: turn.cumulativeTokens,
    labelRoles: config.classifierContextIncludeAssistantTurns,
  });

  // A slow router is worse than a mediocre one on an interactive prompt, so the timeout is
  // a hard ceiling: the request is abandoned, not merely reported late.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), llm.timeoutMs);

  try {
    const response = await registry.complete(
      model as never,
      { systemPrompt, messages: [{ role: "user", content: userPayload }] },
      { signal: controller.signal, maxTokens: 16 },
    );
    const tier = parseTierReply(replyText(response?.content));
    if (!tier || !isTier(tier)) {
      return { ok: false, reason: "classifier reply named no known tier" };
    }
    return {
      ok: true,
      classification: { tier, signals: ["llm_classifier"], cause: "llm_classifier" },
    };
  } catch (err) {
    const reason = controller.signal.aborted
      ? `classifier timed out after ${llm.timeoutMs}ms`
      : `classifier call failed: ${err instanceof Error ? err.message : String(err)}`;
    return { ok: false, reason };
  } finally {
    clearTimeout(timer);
  }
}
