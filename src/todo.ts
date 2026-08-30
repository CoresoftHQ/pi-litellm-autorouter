import type { Tier } from "./types.ts";

/** The stable result shape published by @juicesharp/rpiv-todo. Kept structural so the
 * router does not need to depend on that optional extension. */
type TodoTask = { status?: unknown };
type TodoResult = { tasks?: unknown };
type SessionEntry = {
  type?: unknown;
  message?: { role?: unknown; toolName?: unknown; details?: unknown; isError?: unknown };
};

/** Read the latest successful `todo` tool snapshot from the active branch. */
export function hasActiveTodos(entries: readonly unknown[], toolName: string): boolean {
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i] as SessionEntry | null;
    const message = entry?.type === "message" ? entry.message : undefined;
    if (!message || message.role !== "toolResult" || message.toolName !== toolName || message.isError === true) continue;
    const details = message.details as TodoResult | null;
    if (!Array.isArray(details?.tasks)) continue;
    return details.tasks.some((task: TodoTask) => task?.status === "pending" || task?.status === "in_progress");
  }
  return false;
}

/** Acknowledgements that approve continuing existing work, but do not add a new ask. */
export function isBriefContinuation(text: string, maxPromptChars: number): boolean {
  const normalized = text.trim().toLowerCase().replace(/[.!?]+$/g, "");
  if (!normalized || normalized.length > maxPromptChars) return false;
  return new Set([
    "ok", "okay", "yes", "yeah", "yep", "sure", "continue", "go ahead", "proceed", "do it",
    "sounds good", "please do", "please proceed",
  ]).has(normalized);
}

export function todoContinuationTier(
  text: string | null,
  active: boolean,
  config: { enabled: boolean; minTier: Tier; maxPromptChars: number },
): Tier | null {
  if (!config.enabled || !active || !text || !isBriefContinuation(text, config.maxPromptChars)) return null;
  return config.minTier;
}
