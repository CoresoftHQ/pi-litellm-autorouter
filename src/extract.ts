/**
 * Input hygiene: turn a session's messages plus the current prompt into the classifiable
 * view of a turn.
 *
 * Ported from LiteLLM's `_strip_reminder_blocks`, `_extract_current_ask_and_system_prompt`,
 * `_newest_turn_ask` and `_extract_prior_turns`.
 *
 * The point of this layer: agent harnesses inject their own context into the conversation
 * as ordinary message text. That text is plumbing, not something a human asked for, so it
 * must not decide which model serves the turn — an injected block full of tool
 * descriptions would push every prompt into the top tier.
 */

import type { ReminderMarkerPair } from "./config.ts";
import { DEFAULT_REMINDER_MARKERS } from "./defaults.ts";
import type { ExtractedTurn } from "./types.ts";

/**
 * pi's own expansion of `/skill:name args`: the skill file wrapped in a `<skill>` block,
 * then the arguments. The block is instructions pi loaded, not something the user typed,
 * so it is stripped like any other harness plumbing, whatever `reminderMarkers` says.
 */
export const SKILL_BLOCK_MARKERS: ReminderMarkerPair = { open: '<skill name="', close: "</skill>" };

/** `markerPairs` plus pi's skill block. */
export function withSkillBlocks(markerPairs: readonly ReminderMarkerPair[]): ReminderMarkerPair[] {
  const hasSkill = markerPairs.some(
    (pair) =>
      pair.open.toLowerCase() === SKILL_BLOCK_MARKERS.open && pair.close.toLowerCase() === SKILL_BLOCK_MARKERS.close,
  );
  return hasSkill ? [...markerPairs] : [...markerPairs, SKILL_BLOCK_MARKERS];
}

/** ~4 characters per token, matching upstream's estimator. */
export function estimateTokens(text: string): number {
  return Math.floor(text.length / 4);
}

/**
 * Spans of complete `open…close` blocks in `lowered`.
 *
 * An *unclosed* opening delimiter yields no span, so prose that merely mentions a
 * delimiter is left alone rather than swallowing the rest of the message.
 */
function* reminderBlockSpans(lowered: string, open: string, close: string): Generator<[number, number]> {
  let searchFrom = 0;
  while (true) {
    const start = lowered.indexOf(open, searchFrom);
    if (start === -1) return;
    const closeAt = lowered.indexOf(close, start + open.length);
    if (closeAt === -1) return; // unclosed: not a block
    const end = closeAt + close.length;
    yield [start, end];
    searchFrom = end;
  }
}

/**
 * Remove every complete reminder block from `text`.
 *
 * Matching is case-insensitive. Blocks that nest or overlap across pairs are removed
 * whole, because the union of their spans is what gets cut.
 */
export function stripReminderBlocks(
  text: string,
  markerPairs: readonly ReminderMarkerPair[] = DEFAULT_REMINDER_MARKERS,
): string {
  if (!text) return text;
  const lowered = text.toLowerCase();

  const spans: [number, number][] = [];
  for (const { open, close } of markerPairs) {
    for (const span of reminderBlockSpans(lowered, open.toLowerCase(), close.toLowerCase())) {
      spans.push(span);
    }
  }
  if (spans.length === 0) return text;

  // Merge overlapping spans so a nested pair is cut once, not twice.
  spans.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [start, end] of spans) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }

  let out = "";
  let cursor = 0;
  for (const [start, end] of merged) {
    out += text.slice(cursor, start);
    cursor = end;
  }
  out += text.slice(cursor);
  return out.trim();
}

/** A message as this module needs to see it, regardless of pi's internal shape. */
export interface SimpleMessage {
  role: "user" | "assistant" | string;
  content: unknown;
}

/** Flatten pi/provider message content down to plain text. */
export function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object") {
          const b = block as { type?: unknown; text?: unknown };
          if (b.type === "text" && typeof b.text === "string") return b.text;
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

/** Message text with reminder blocks removed. */
export function humanText(content: unknown, markerPairs: readonly ReminderMarkerPair[]): string {
  return stripReminderBlocks(messageText(content), markerPairs).trim();
}

/**
 * The newest user ask that still has human content after stripping.
 *
 * A turn that is nothing but a reminder block strips to empty; rather than classify
 * nothing, walk back to the last real ask.
 */
export function newestTurnAsk(
  messages: readonly SimpleMessage[],
  markerPairs: readonly ReminderMarkerPair[],
): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!msg || msg.role !== "user") continue;
    const text = humanText(msg.content, markerPairs);
    if (text) return text;
  }
  return null;
}

/**
 * Split a conversation into the turn being routed and what came before it.
 *
 * The virtual model routes after the prompt has landed in the session, so the current ask
 * is the run of user messages after the newest assistant message: usually exactly one, the
 * prompt. Everything before it is history, in the shape `extractTurn` takes.
 */
export function splitCurrentTurn(messages: readonly SimpleMessage[]): { current: string; prior: SimpleMessage[] } {
  let start = messages.length;
  while (start > 0 && messages[start - 1]?.role === "user") start--;
  const current = messages
    .slice(start)
    .map((message) => messageText(message.content))
    .filter(Boolean)
    .join("\n");
  return { current, prior: messages.slice(0, start) };
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}…`;
}

/**
 * Build the classifiable view of a turn.
 *
 * `currentText` is the prompt being routed and `messages` the conversation before it (see
 * `splitCurrentTurn`). They are passed separately so the current ask can be stripped and
 * judged on its own while history only feeds the classifier's context window.
 */
export function extractTurn(
  currentText: string,
  messages: readonly SimpleMessage[],
  options: {
    markerPairs: readonly ReminderMarkerPair[];
    contextWindowSize: number;
    perTurnChars: number;
    includeAssistantTurns: boolean;
  },
): ExtractedTurn {
  const { markerPairs, contextWindowSize, perTurnChars, includeAssistantTurns } = options;

  const stripped = stripReminderBlocks(currentText, markerPairs).trim();
  // A turn that is nothing but harness plumbing falls back to the last real ask, so a
  // reminder-only turn routes on what the human last actually said.
  const currentAsk = stripped || newestTurnAsk(messages, markerPairs);

  const priorTurns: { role: "user" | "assistant"; text: string }[] = [];
  if (contextWindowSize > 0) {
    // Quoted turns by their position in `messages`, so they can be put back in order.
    const picked: { index: number; role: "user" | "assistant"; text: string }[] = [];

    // The window holds the user's own prior asks: they say what the work in progress is.
    let userTurns = 0;
    for (let i = messages.length - 1; i >= 0 && userTurns < contextWindowSize; i--) {
      const msg = messages[i];
      if (msg?.role !== "user") continue;
      const text = humanText(msg.content, markerPairs);
      if (!text) continue;
      // Drop a prior turn that just repeats the current ask ("continue", "try again"):
      // quoting it back adds nothing the classifier can use.
      if (text === currentAsk) continue;
      picked.push({ index: i, role: "user", text: truncate(text, perTurnChars) });
      userTurns++;
    }

    // At most one assistant turn, and outside the window: the latest text reply, which is
    // what a "yes" or "go ahead" answers. Quoting more would mostly quote an agent's running
    // narration ("let me check the config"), and letting it take window slots pushed the
    // user's own asks out, which is the context that matters.
    if (includeAssistantTurns) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg?.role !== "assistant") continue;
        const text = humanText(msg.content, markerPairs);
        if (!text) continue; // tool calls only: not a reply
        if (text !== currentAsk) picked.push({ index: i, role: "assistant", text: truncate(text, perTurnChars) });
        break;
      }
    }

    // Oldest first, as the classifier payload quotes them.
    picked.sort((a, b) => a.index - b.index);
    for (const { role, text } of picked) priorTurns.push({ role, text });
  }

  // Gated on whether prior conversation exists at all, not on whether any of it survived
  // the window: a long continuation whose every prior ask was redundant is still a
  // continuation, and reporting it as a context-free single-turn request is exactly the
  // misrouting this layer exists to prevent.
  const conversationContinuing = messages.some((m) => m.role === "user" || m.role === "assistant");

  const cumulativeTokens = messages.reduce((sum, m) => sum + estimateTokens(messageText(m.content)), 0);

  return { currentAsk, priorTurns, conversationContinuing, cumulativeTokens };
}
