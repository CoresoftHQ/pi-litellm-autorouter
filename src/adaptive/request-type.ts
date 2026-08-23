/**
 * Rule-based classifier mapping a prompt to a request type.
 *
 * Port of LiteLLM's `adaptive_router/classifier.py`: deterministic regexes, checked in
 * order from most to least specific, falling back to `general`. This is a *taxonomy* for
 * the bandit's cells, not a difficulty score — the complexity tier is decided elsewhere.
 */

import type { RequestType } from "../types.ts";

const RULES: readonly (readonly [RegExp, RequestType])[] = [
  [
    /\b(write|create|generate|implement|build)\s+(?:a |an |the |me )?(?:python|javascript|typescript|java|rust|go|c\+\+|sql|bash|shell)\b/i,
    "code_generation",
  ],
  [
    /\b(write|create|implement|build)\b(?:\s+\w+){0,4}?\s+(function|class|method|script|program|api|endpoint|microservice)\b/i,
    "code_generation",
  ],
  [/\b(explain|describe|understand|walk me through|what does)\b.*\b(code|function|method|class|algorithm|snippet)\b/i, "code_understanding"],
  [
    /\b(debug|fix|why (?:is|does|isn't)|what.s wrong|trace)\b.*\b(error|bug|exception|stacktrace|stack trace|traceback)\b/i,
    "code_understanding",
  ],
  [/\b(review|critique)\s+(?:this |my |the )?(?:code|pr|pull request|diff|patch)\b/i, "code_understanding"],
  [/\b(design|architect|plan|architecture)\b.*\b(system|service|api|database|schema|module|microservice)\b/i, "technical_design"],
  [
    /\b(should i (?:use|choose|pick)|tradeoffs? between|compare)\b.*\b(library|framework|language|database|protocol|postgres|postgresql|mongodb|dynamodb|mysql|redis|kafka|sql|nosql)\b/i,
    "technical_design",
  ],
  [/\bhow (?:should|do) i (?:design|structure|organize|model)\b/i, "technical_design"],
  [/\b(solve|compute|calculate|prove|derive)\b.*\b(equation|integral|derivative|theorem|proof|problem)\b/i, "analytical_reasoning"],
  [/\b(if .+ then|given .+ find|suppose|assume)\b/i, "analytical_reasoning"],
  [/\b(probability|statistics|combinatorics|optimization problem)\b/i, "analytical_reasoning"],
  [
    /\b(write|draft|compose|rewrite|edit|proofread|polish)\b.*\b(email|essay|blog|post|article|letter|memo|copy|paragraph|sentence)\b/i,
    "writing",
  ],
  [/\b(make (?:this|it)|help me)\s+(?:more |less )?(?:concise|formal|casual|professional|persuasive)\b/i, "writing"],
  [/^\s*(who|what|when|where|which)\s+(?:is|was|were|are)\b/i, "factual_lookup"],
  [/^\s*(define|definition of|meaning of)\b/i, "factual_lookup"],
  [/^\s*how (?:do you spell|to spell|many .* are there|tall is)\b/i, "factual_lookup"],
];

/** Only the head of a long prompt is inspected, as upstream does. */
const INSPECTED_CHARS = 2000;

export function classifyRequestType(text: string | null | undefined): RequestType {
  if (!text || !text.trim()) return "general";
  const head = text.slice(0, INSPECTED_CHARS);
  for (const [pattern, requestType] of RULES) {
    if (pattern.test(head)) return requestType;
  }
  return "general";
}
