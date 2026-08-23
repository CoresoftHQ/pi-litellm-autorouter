/**
 * Incremental signal detection for the adaptive router.
 *
 * Port of LiteLLM's `adaptive_router/signals.py`. Each session keeps a small rolling
 * state; every completed turn is compared against it and yields a `SignalDelta` saying
 * which signals fired. O(1) per turn — only the previous turn's text and a bounded tool
 * call history are inspected, never the full session.
 */

/** Which signals fired on a single turn. Each count is 0 or 1. */
export interface SignalDelta {
  misalignment: number;
  stagnation: number;
  disengagement: number;
  satisfaction: number;
  failure: number;
  loop: number;
  exhaustion: number;
}

export function emptyDelta(): SignalDelta {
  return { misalignment: 0, stagnation: 0, disengagement: 0, satisfaction: 0, failure: 0, loop: 0, exhaustion: 0 };
}

export function anyFired(delta: SignalDelta): boolean {
  return Object.values(delta).some((count) => count > 0);
}

/** Rolling state for one (session, model) pair. */
export interface SessionState {
  sessionId: string;
  modelName: string;
  classifiedType: string;
  misalignmentCount: number;
  stagnationCount: number;
  disengagementCount: number;
  satisfactionCount: number;
  failureCount: number;
  loopCount: number;
  exhaustionCount: number;
  lastUserContent: string | null;
  lastAssistantContent: string | null;
  toolCallHistory: string[];
  turnCount: number;
  cleanCreditAwarded: boolean;
  terminalStatus: number | null;
}

export function newSessionState(sessionId: string, modelName: string, classifiedType: string): SessionState {
  return {
    sessionId,
    modelName,
    classifiedType,
    misalignmentCount: 0,
    stagnationCount: 0,
    disengagementCount: 0,
    satisfactionCount: 0,
    failureCount: 0,
    loopCount: 0,
    exhaustionCount: 0,
    lastUserContent: null,
    lastAssistantContent: null,
    toolCallHistory: [],
    turnCount: 0,
    cleanCreditAwarded: false,
    terminalStatus: null,
  };
}

export interface TurnToolCall {
  name: string;
  arguments: unknown;
}

export interface TurnToolResult {
  content: string;
  isError: boolean;
}

/** One completed exchange, assembled by the caller from the request and response. */
export interface Turn {
  userContent: string | null;
  assistantContent: string | null;
  toolCalls: TurnToolCall[];
  toolResults: TurnToolResult[];
  /** HTTP-style status of the response; only the exhaustion set is ever inspected. */
  responseStatus: number | null;
}

// Detector thresholds, from upstream (Plano/Chen 2026).
export const MISALIGNMENT_JACCARD_THRESHOLD = 0.45;
export const STAGNATION_JACCARD_NEAR_DUP = 0.5;
export const LOOP_REPEAT_THRESHOLD = 3;
export const TOOL_CALL_HISTORY_MAX = 20;
/** Turns a session must reach before a "thanks" can credit the model. */
export const MIN_TURNS_FOR_CLEAN_CREDIT = 3;

const TOKEN_RE = /[A-Za-z0-9]+/g;

function tokens(text: string | null): Set<string> {
  if (!text) return new Set();
  return new Set((text.match(TOKEN_RE) ?? []).map((t) => t.toLowerCase()));
}

function jaccard(a: Set<string>, b: Set<string>): number {
  const union = new Set([...a, ...b]);
  if (union.size === 0) return 0;
  let intersection = 0;
  for (const token of a) if (b.has(token)) intersection++;
  return intersection / union.size;
}

const DISENGAGEMENT_PATTERNS = [
  /\b(forget it|never mind|give up|talk to (?:a )?human|cancel)\b/i,
  /\b(this (?:isn'?t|is not) working|stop|abort)\b/i,
  /\bi'?ll do it (?:myself|manually)\b/i,
];

const SATISFACTION_PATTERNS = [
  /\b(that worked|that did it|works now|fixed it|solved it|nice)\b/i,
  /\b(thanks|thank you|thx|appreciated|appreciate it)\b/i,
  /\b(perfect|great|excellent|exactly)\b/i,
];

/** Consecutive user messages that share *some* topic but differ enough to read as a
 *  rephrase — not a topic change (jaccard 0), not a repeat (jaccard high). */
function detectMisalignment(previousUser: string | null, currentUser: string | null): boolean {
  if (!previousUser || !currentUser) return false;
  const j = jaccard(tokens(previousUser), tokens(currentUser));
  return j > 0 && j < MISALIGNMENT_JACCARD_THRESHOLD;
}

/** Consecutive assistant replies that are near-duplicates. */
function detectStagnation(previousAssistant: string | null, currentAssistant: string | null): boolean {
  if (!previousAssistant || !currentAssistant) return false;
  return jaccard(tokens(previousAssistant), tokens(currentAssistant)) >= STAGNATION_JACCARD_NEAR_DUP;
}

function detectDisengagement(currentUser: string | null): boolean {
  if (!currentUser) return false;
  return DISENGAGEMENT_PATTERNS.some((p) => p.test(currentUser));
}

function detectSatisfaction(currentUser: string | null): boolean {
  if (!currentUser) return false;
  return SATISFACTION_PATTERNS.some((p) => p.test(currentUser));
}

/** Any tool result explicitly flagged as an error. Empty output is deliberately *not* a
 *  failure: zero-result searches and silent commands are routine. */
function detectFailure(toolResults: readonly TurnToolResult[]): boolean {
  return toolResults.some((r) => r.isError);
}

/** Stable signature for loop detection: name plus sorted arguments. */
export function toolCallSignature(call: TurnToolCall): string {
  let args: string;
  if (call.arguments && typeof call.arguments === "object" && !Array.isArray(call.arguments)) {
    args = Object.keys(call.arguments as Record<string, unknown>)
      .sort()
      .map((k) => `${k}=${String((call.arguments as Record<string, unknown>)[k])}`)
      .join(",");
  } else {
    args = call.arguments === undefined || call.arguments === null ? "" : String(call.arguments);
  }
  return `${call.name}(${args})`;
}

/** Fires when a new call's signature already appears LOOP_REPEAT_THRESHOLD - 1 times in
 *  recent history, so this call would be the Nth. */
function detectLoop(history: readonly string[], newCalls: readonly TurnToolCall[]): boolean {
  for (const call of newCalls) {
    const signature = toolCallSignature(call);
    const recent = history.filter((s) => s === signature).length;
    if (recent >= LOOP_REPEAT_THRESHOLD - 1) return true;
  }
  return false;
}

export const EXHAUSTION_STATUSES: ReadonlySet<number> = new Set([408, 413, 429, 503, 504]);

export const EXHAUSTION_KEYWORDS: readonly string[] = [
  "context length",
  "context window",
  "token limit",
  "rate limit",
  "too many requests",
  "timeout",
];

export function mentionsExhaustion(text: string): boolean {
  const lowered = text.toLowerCase();
  return EXHAUSTION_KEYWORDS.some((kw) => lowered.includes(kw));
}

function detectExhaustion(status: number | null, toolResults: readonly TurnToolResult[]): boolean {
  if (status !== null && EXHAUSTION_STATUSES.has(status)) return true;
  return toolResults.some((r) => mentionsExhaustion(r.content));
}

/** Signals carried by the *user's* side of a turn — feedback on the previous reply. */
export function detectUserFeedback(
  previousUserContent: string | null,
  currentUserContent: string | null,
  toolResults: readonly TurnToolResult[],
  allowSatisfaction: boolean,
): SignalDelta {
  return {
    ...emptyDelta(),
    misalignment: Number(detectMisalignment(previousUserContent, currentUserContent)),
    disengagement: Number(detectDisengagement(currentUserContent)),
    satisfaction: Number(allowSatisfaction && detectSatisfaction(currentUserContent)),
    failure: Number(detectFailure(toolResults)),
  };
}

/** Signals carried by the *model's* side of a turn. */
export function detectResponseSignals(
  previousAssistantContent: string | null,
  currentAssistantContent: string | null,
  toolCallHistory: readonly string[],
  toolCalls: readonly TurnToolCall[],
  toolResults: readonly TurnToolResult[],
  responseStatus: number | null,
): SignalDelta {
  return {
    ...emptyDelta(),
    stagnation: Number(detectStagnation(previousAssistantContent, currentAssistantContent)),
    loop: Number(detectLoop(toolCallHistory, toolCalls)),
    exhaustion: Number(detectExhaustion(responseStatus, toolResults)),
  };
}

export function mergeSignalDeltas(...deltas: SignalDelta[]): SignalDelta {
  const merged = emptyDelta();
  for (const delta of deltas) {
    for (const key of Object.keys(merged) as (keyof SignalDelta)[]) merged[key] += delta[key];
  }
  return merged;
}

export function applySignalDelta(state: SessionState, delta: SignalDelta): void {
  state.misalignmentCount += delta.misalignment;
  state.stagnationCount += delta.stagnation;
  state.disengagementCount += delta.disengagement;
  state.satisfactionCount += delta.satisfaction;
  state.failureCount += delta.failure;
  state.loopCount += delta.loop;
  state.exhaustionCount += delta.exhaustion;
}

export function advanceSessionState(state: SessionState, turn: Turn): void {
  if (turn.userContent) state.lastUserContent = turn.userContent;
  if (turn.assistantContent) state.lastAssistantContent = turn.assistantContent;
  for (const call of turn.toolCalls) state.toolCallHistory.push(toolCallSignature(call));
  if (state.toolCallHistory.length > TOOL_CALL_HISTORY_MAX) {
    state.toolCallHistory = state.toolCallHistory.slice(-TOOL_CALL_HISTORY_MAX);
  }
  if (turn.responseStatus !== null) state.terminalStatus = turn.responseStatus;
  state.turnCount += 1;
}

/**
 * Translate a turn's signals into a bandit-cell update.
 *
 * Upstream's v0 mapping: satisfaction → +1 alpha; misalignment, stagnation, disengagement
 * and failure → +1 beta each; loop → +0.5 beta (weak: it could be the model or the user);
 * exhaustion → 0 (an uptime problem, not a quality one).
 */
export function banditDelta(delta: SignalDelta): [deltaAlpha: number, deltaBeta: number] {
  const deltaAlpha = delta.satisfaction;
  const deltaBeta = delta.misalignment + delta.stagnation + delta.disengagement + delta.failure + 0.5 * delta.loop;
  return [deltaAlpha, deltaBeta];
}
