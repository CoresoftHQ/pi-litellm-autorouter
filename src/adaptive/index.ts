/**
 * Glue between the adaptive router and the rest of the extension: building the bandit
 * from a config and pi's model registry, and turning a finished agent run into a `Turn`.
 */

import { splitModelRef } from "../classify/llm.ts";
import type { RouterConfig } from "../config.ts";
import type { Rng } from "./bandit.ts";
import { AdaptiveRouter, DEFAULT_PREFERENCES } from "./router.ts";
import { poolModels } from "./select.ts";
import { type Turn, type TurnToolCall, type TurnToolResult, mentionsExhaustion } from "./signals.ts";
import { type AdaptivePreferences, type ModelRef, TIER_SEVERITY_ORDER } from "../types.ts";

/** The slice of pi's registry the bandit needs: a model's input price. */
export interface CostRegistry {
  find(provider: string, modelId: string): unknown;
}

function inputCost(registry: CostRegistry | undefined, ref: ModelRef): number {
  const split = splitModelRef(ref);
  if (!split || !registry) return 0;
  const model = registry.find(split.provider, split.modelId) as { cost?: { input?: unknown } } | undefined;
  const cost = model?.cost?.input;
  return typeof cost === "number" && Number.isFinite(cost) ? cost : 0;
}

/** Per-model preferences: the first tier (ascending) that declares them wins. */
function preferencesFor(ref: ModelRef, config: RouterConfig): AdaptivePreferences {
  for (const tier of TIER_SEVERITY_ORDER) {
    for (const target of config.tiers[tier]) {
      if (target.model !== ref) continue;
      if (target.qualityTier === undefined && target.strengths === undefined) continue;
      return {
        qualityTier: target.qualityTier ?? DEFAULT_PREFERENCES.qualityTier,
        strengths: target.strengths ?? [],
      };
    }
  }
  return DEFAULT_PREFERENCES;
}

export interface BuildAdaptiveOptions {
  registry?: CostRegistry;
  now?: () => number;
  rng?: Rng;
}

/** Cold-start bandit for `config`. Returns null when adaptive is off. */
export function buildAdaptiveRouter(config: RouterConfig, options: BuildAdaptiveOptions = {}): AdaptiveRouter | null {
  if (!config.adaptive) return null;
  const models = poolModels(config);
  const modelToPrefs = new Map<ModelRef, AdaptivePreferences>();
  const modelToCost = new Map<ModelRef, number>();
  for (const model of models) {
    modelToPrefs.set(model, preferencesFor(model, config));
    modelToCost.set(model, inputCost(options.registry, model));
  }
  return new AdaptiveRouter({
    availableModels: models,
    modelToPrefs,
    modelToCost,
    ...(options.now ? { now: options.now } : {}),
    ...(options.rng ? { rng: options.rng } : {}),
  });
}

/** The shape of pi's `agent_end` messages this module reads. Structural on purpose. */
export interface RunMessage {
  role: string;
  content?: unknown;
  stopReason?: string;
  errorMessage?: string;
  toolName?: string;
  isError?: boolean;
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => {
      if (block && typeof block === "object" && (block as { type?: unknown }).type === "text") {
        const text = (block as { text?: unknown }).text;
        return typeof text === "string" ? text : "";
      }
      return "";
    })
    .filter((t) => t.length > 0)
    .join("\n");
}

/**
 * One agent run as a `Turn`.
 *
 * Upstream records one turn per LLM call; pi's unit of work is the whole run, so tool
 * calls and results are gathered across it (a loop shows up as repeated calls within a
 * run) and the assistant text is the final reply. A run that ended in error reports 500,
 * or 429 when the error reads as rate limiting or context exhaustion, so upstream's
 * exhaustion detector sees it the way it would see the provider's status.
 */
export function turnFromRun(userText: string | null, messages: readonly RunMessage[]): Turn {
  const toolCalls: TurnToolCall[] = [];
  const toolResults: TurnToolResult[] = [];
  let assistantContent: string | null = null;
  let responseStatus: number | null = 200;

  for (const message of messages) {
    if (message.role === "assistant") {
      const text = textOf(message.content);
      if (text) assistantContent = text;
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (block && typeof block === "object" && (block as { type?: unknown }).type === "toolCall") {
            const call = block as { name?: unknown; arguments?: unknown };
            toolCalls.push({ name: typeof call.name === "string" ? call.name : "", arguments: call.arguments });
          }
        }
      }
      // The run's status is its *final* reply's: an early error that pi retried past is
      // not what the user saw.
      if (message.stopReason === "error") {
        responseStatus = message.errorMessage && mentionsExhaustion(message.errorMessage) ? 429 : 500;
      } else if (message.stopReason !== undefined) {
        responseStatus = 200;
      }
    } else if (message.role === "toolResult") {
      toolResults.push({ content: textOf(message.content), isError: Boolean(message.isError) });
    }
  }

  return { userContent: userText, assistantContent, toolCalls, toolResults, responseStatus };
}
