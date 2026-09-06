/**
 * "Is this turn the answer to a question the assistant just asked?"
 *
 * Question tools (`ask_user_question`, `ask_question`) are answered the way humans answer
 * questions: "ok", "yes", "option 2", "postgres". Scored as a fresh ask, every one of
 * those lands in SIMPLE and drops the session onto the cheapest model in the middle of
 * work that was classified COMPLEX three seconds earlier — and the model that then has to
 * act on the answer is the weak one.
 *
 * The fix is not to score such a turn at all. Detection is deliberately two-sided:
 *
 *   1. the assistant's last message must actually carry a question tool call, so this can
 *      never fire on an ordinary terse prompt at the start of a session; and
 *   2. the reply must look like an answer rather than a new instruction, so "now refactor
 *      the whole auth layer" still routes on its merits even when it is typed straight
 *      into an open question.
 *
 * Both sides are needed. Tool call alone would swallow real work typed after a question;
 * shape alone would fire on any short prompt anywhere.
 *
 * No pi imports: the whole detector is testable against plain objects.
 */

/** A question the assistant asked and has not moved past. */
export interface PendingQuestion {
  /** The tool that asked, as named in config — carried into the decision log. */
  toolName: string;
  /** Option labels the user was offered, normalized. Empty for free-form questions. */
  optionLabels: string[];
}

/** A message as this module needs to see it. Structural, like `SimpleMessage`. */
export interface QuestionMessage {
  role: string;
  content: unknown;
}

/**
 * Lowercase, collapse whitespace, drop punctuation.
 *
 * Applied to both sides of every comparison, so "Option 2." and "option 2" are one
 * string and an option label carrying "(Recommended)" still matches the words a user
 * would actually type back.
 */
export function normalizeReply(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[^a-z0-9\s'+-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Every `label` string anywhere in a tool call's arguments.
 *
 * Walked structurally rather than against one tool's schema: `ask_user_question` nests
 * labels under `questions[].options[]`, other harnesses' question tools nest them
 * elsewhere, and a detector that hard-codes one shape silently stops working against the
 * next one. Depth is bounded so a pathological argument object cannot spin.
 */
function collectLabels(value: unknown, out: string[], depth = 0): void {
  if (depth > 6 || out.length > 64) return;
  if (Array.isArray(value)) {
    for (const item of value) collectLabels(item, out, depth + 1);
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
    if ((key === "label" || key === "value") && typeof item === "string") {
      const normalized = normalizeReply(item);
      if (normalized) out.push(normalized);
      continue;
    }
    collectLabels(item, out, depth + 1);
  }
}

/** Tool names compare on letters and digits alone, so `ask_user_question`,
 *  `askUserQuestion` and `AskUserQuestion` are one name. Harnesses disagree about casing
 *  and separators for what is plainly the same tool, and a config that has to guess the
 *  spelling is a config that silently does nothing. */
function normalizeToolName(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/**
 * The question the assistant is waiting on, or null.
 *
 * Only the *last* assistant message is inspected, and that single rule covers both shapes
 * a question can end in. A tool that blocked and was answered inline is followed by
 * further assistant messages, so the call is no longer last and this returns null — the
 * agent already acted on the answer, and whatever the user types next is a new turn. A
 * question the run ended on (the questionnaire was dismissed, the tool deferred to the
 * next prompt) leaves the call last, which is exactly the state where the next input is
 * the answer.
 */
export function pendingQuestion(
  messages: readonly QuestionMessage[],
  toolNames: readonly string[],
): PendingQuestion | null {
  const wanted = new Set(toolNames.map(normalizeToolName));
  if (wanted.size === 0) return null;

  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (!message || message.role !== "assistant") continue;
    if (!Array.isArray(message.content)) return null;

    for (const block of message.content) {
      if (!block || typeof block !== "object") continue;
      const call = block as { type?: unknown; name?: unknown; arguments?: unknown };
      if (call.type !== "toolCall" || typeof call.name !== "string") continue;
      if (!wanted.has(normalizeToolName(call.name))) continue;
      const optionLabels: string[] = [];
      collectLabels(call.arguments, optionLabels);
      return { toolName: call.name, optionLabels };
    }
    // The newest assistant message asked nothing: the question, if there was one, has
    // already been answered and acted on.
    return null;
  }
  return null;
}

/** Words that carry no instruction of their own and only ever prefix one. */
const LEADING_FILLERS = new Set([
  "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "sure", "alright", "right", "fine",
  "good", "great", "perfect", "cool", "nice", "no", "nope", "nah", "please", "thanks", "ty",
  "and", "but", "also", "then", "now", "next", "so", "well", "hmm", "actually", "just", "lets",
  "let's", "maybe",
]);

/** A reply that answers and asks for nothing further. */
const AFFIRMATIONS = new Set([
  "", "y", "n", "ye", "yea", "aye", "affirmative", "negative",
  "go", "go ahead", "go for it", "do it", "just do it", "send it", "ship it", "proceed",
  "continue", "carry on", "keep going", "carry on then", "please do", "do that", "that one",
  "agreed", "agree", "correct", "incorrect", "exactly", "indeed", "true", "false",
  "confirm", "confirmed", "sounds good", "looks good", "lgtm", "makes sense", "fair enough",
  "cancel", "stop", "skip", "none", "neither", "both", "all", "all of them", "either",
  "up to you", "your call", "you decide", "you choose", "whatever you think", "either is fine",
  "no preference", "does not matter", "doesnt matter", "dont care", "do not care",
  "done", "noted", "understood", "got it", "thank you", "same", "as before", "default",
]);

/** "2", "option 3", "the first one", "1 and 3", "b". */
const ORDINAL_PATTERNS: readonly RegExp[] = [
  /^(option|choice|answer|number|nr|no|item)?\s*\d{1,2}$/,
  /^[a-d]$/,
  /^(the\s+)?(first|second|third|fourth|fifth|last)(\s+(one|option|choice|answer))?$/,
  /^\d{1,2}(\s*(and|,|\+|&)\s*\d{1,2})+$/,
  /^(option|choice)\s*[a-d]$/,
];

/**
 * Verbs that open a unit of work.
 *
 * Only ever tested against the *first* word after fillers are stripped, so "yes, and add
 * migrations too" stays an answer while "add migrations" is a new instruction. The list is
 * deliberately biased towards over-matching: a false positive here just routes the turn
 * the way it was routed before this feature existed, while a false negative pins the
 * session's model onto work that deserved its own decision.
 */
const TASK_VERBS = new Set([
  "implement", "refactor", "rewrite", "add", "create", "make", "build", "write", "generate",
  "fix", "repair", "debug", "delete", "remove", "drop", "rename", "move", "update", "upgrade",
  "migrate", "port", "convert", "test", "deploy", "release", "publish", "install", "configure",
  "optimize", "optimise", "explain", "analyze", "analyse", "review", "audit", "design",
  "document", "summarize", "summarise", "investigate", "research", "find", "search", "check",
  "read", "open", "run", "execute", "commit", "push", "merge", "rebase", "revert", "split",
  "extract", "inline", "wire", "integrate", "replace", "refactor", "benchmark", "profile",
]);

/** Code, paths and URLs: content, not an answer. */
const CODE_MARKERS =
  /(`|\{|\}|;|=>|::|\bhttps?:\/\/|\/\w+\/|\.(ts|tsx|js|jsx|py|go|rs|java|cs|rb|php|sql|json|ya?ml|toml|md|sh)\b)/i;

/** Drop leading filler words so the first meaningful word can be judged. */
function stripFillers(normalized: string): string {
  let words = normalized.split(" ").filter(Boolean);
  while (words.length > 0 && LEADING_FILLERS.has(words[0] as string)) words = words.slice(1);
  return words.join(" ");
}

/** Does the reply name one of the options the user was shown? */
function matchesOffered(normalized: string, optionLabels: readonly string[]): boolean {
  if (normalized.length < 2) return false;
  for (const label of optionLabels) {
    if (!label) continue;
    if (label === normalized) return true;
    // A clicked option arrives as the whole label; a typed one is usually the first few
    // words of it. Four characters keeps "a" or "to" from matching every label there is.
    if (normalized.length >= 4 && (label.startsWith(normalized) || label.includes(normalized))) return true;
    if (label.length >= 4 && normalized.includes(label)) return true;
  }
  return false;
}

export interface ReplyShapeOptions {
  /** Longest reply still treated as an answer. Beyond it, a turn is its own request. */
  maxChars: number;
  /** Labels offered with the pending question; an exact hit bypasses every other test. */
  optionLabels?: readonly string[];
}

/**
 * Does `text` read as an answer rather than a fresh request?
 *
 * Ordered so the strongest evidence wins: naming an offered option is conclusive and is
 * not subject to the length cap (option labels can be long, and a clicked one arrives
 * verbatim). Everything after that is shape.
 */
export function looksLikeAnswer(text: string, options: ReplyShapeOptions): boolean {
  const normalized = normalizeReply(text);
  if (matchesOffered(normalized, options.optionLabels ?? [])) return true;

  // A multi-paragraph reply is a briefing, not an answer.
  if (text.split("\n").filter((line) => line.trim()).length > 2) return false;
  if (text.length > options.maxChars) return false;
  if (CODE_MARKERS.test(text)) return false;

  const core = stripFillers(normalized);
  if (AFFIRMATIONS.has(core)) return true;
  if (ORDINAL_PATTERNS.some((pattern) => pattern.test(core))) return true;

  const firstWord = core.split(" ")[0] ?? "";
  return !TASK_VERBS.has(firstWord);
}

export interface DetectOptions {
  enabled: boolean;
  toolNames: readonly string[];
  maxChars: number;
}

/** The pending question this turn answers, or null when the turn routes normally. */
export function detectQuestionReply(
  currentText: string,
  messages: readonly QuestionMessage[],
  options: DetectOptions,
): PendingQuestion | null {
  if (!options.enabled) return null;
  const text = currentText.trim();
  if (!text) return null;
  const pending = pendingQuestion(messages, options.toolNames);
  if (!pending) return null;
  return looksLikeAnswer(text, { maxChars: options.maxChars, optionLabels: pending.optionLabels })
    ? pending
    : null;
}
