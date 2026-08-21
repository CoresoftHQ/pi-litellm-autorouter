/**
 * Default keyword lists, weights, boundaries and thresholds.
 *
 * Ported verbatim from LiteLLM's `litellm/router_strategy/complexity_router/config.py`.
 * These are calibrated values, not suggestions: changing one moves tier decisions and
 * therefore spend, so they are kept byte-comparable with upstream and overridden through
 * config rather than edited here.
 */

export const DEFAULT_CODE_KEYWORDS: readonly string[] = [
  "function",
  "class",
  "def",
  "const",
  "let",
  "var",
  "import",
  "export",
  "return",
  "async",
  "await",
  "try",
  "catch",
  "exception",
  "error",
  "debug",
  "api",
  "endpoint",
  "request",
  "response",
  "database",
  "sql",
  "query",
  "schema",
  "algorithm",
  "implement",
  "refactor",
  "optimize",
  "python",
  "javascript",
  "typescript",
  "java",
  "rust",
  "golang",
  "react",
  "vue",
  "angular",
  "node",
  "docker",
  "kubernetes",
  "git",
  "commit",
  "merge",
  "branch",
  "pull request",
];

export const DEFAULT_REASONING_KEYWORDS: readonly string[] = [
  "step by step",
  "think through",
  "let's think",
  "reason through",
  "analyze this",
  "break down",
  "explain your reasoning",
  "show your work",
  "chain of thought",
  "think carefully",
  "consider all",
  "evaluate",
  "pros and cons",
  "compare and contrast",
  "weigh the options",
  "logical",
  "deduce",
  "infer",
  "conclude",
];

export const DEFAULT_TECHNICAL_KEYWORDS: readonly string[] = [
  "architecture",
  "distributed",
  "scalable",
  "microservice",
  "machine learning",
  "neural network",
  "deep learning",
  "encryption",
  "authentication",
  "authorization",
  "performance",
  "latency",
  "throughput",
  "benchmark",
  "concurrency",
  "parallel",
  "threading",
  "memory",
  "cpu",
  "gpu",
  "optimization",
  "protocol",
  "tcp",
  "http",
  "grpc",
  "websocket",
  "container",
  "orchestration",
  // Note: "async", "kubernetes", "docker" are in DEFAULT_CODE_KEYWORDS
];

export const DEFAULT_SIMPLE_KEYWORDS: readonly string[] = [
  "what is",
  "what's",
  "define",
  "definition of",
  "who is",
  "who was",
  "when did",
  "when was",
  "where is",
  "where was",
  "how many",
  "how much",
  "yes or no",
  "true or false",
  "simple",
  "brief",
  "short",
  "quick",
  "hello",
  "hi",
  "hey",
  "thanks",
  "thank you",
  "goodbye",
  "bye",
  "okay",
  // Note: "ok" removed upstream due to false positives (matches "token", "book", etc.)
];

/** Upstream default is ["LITELLM ESCALATE"]; renamed for a router that is not LiteLLM.
 *  Case-sensitive, so it cannot fire on ordinary prose. */
export const DEFAULT_ESCALATION_KEYWORDS: readonly string[] = ["PI ESCALATE"];

/** Non-greedy `.*?` throughout to prevent ReDoS on pathological input. */
export const MULTI_STEP_PATTERNS: readonly RegExp[] = [
  /first.*?then/i,
  /step\s*\d/i,
  /\d+\.\s/,
  /[a-z]\)\s/i,
];

export const DEFAULT_DIMENSION_WEIGHTS: Readonly<Record<string, number>> = {
  tokenCount: 0.1, // length matters less than content
  codePresence: 0.3, // code requests need capable models
  reasoningMarkers: 0.25, // explicit reasoning requests
  technicalTerms: 0.25, // technical content matters
  simpleIndicators: 0.05, // don't over-penalize simple patterns
  multiStepPatterns: 0.03,
  questionComplexity: 0.02,
};

/** Canonical key names. These name the gap *between* two tiers, not a tier, and stay
 *  canonical even when tiers are renamed — a decision log has to stay comparable. */
export const DEFAULT_TIER_BOUNDARIES: Readonly<Record<string, number>> = {
  simple_medium: 0.15,
  medium_complex: 0.35,
  complex_reasoning: 0.6,
};

export const DEFAULT_TOKEN_THRESHOLDS: Readonly<Record<string, number>> = {
  simple: 15, // below this, a prompt is "short"
  complex: 400, // above this, a prompt is "long"
};

/** Default context window for the LLM classifier, in turns. */
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
/** Per-turn truncation for quoted context, in characters. */
export const DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS = 200;
/** Timeout budget for one classification call. */
export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
/** TTL for a session-affinity pin, refreshed on every hit. */
export const DEFAULT_SESSION_AFFINITY_TTL_SECONDS = 3600;

/** The reminder-block delimiters pi itself emits. Replaced wholesale, never extended,
 *  when `reminderMarkers` is configured. */
export const DEFAULT_REMINDER_MARKERS: readonly { open: string; close: string }[] = [
  { open: "<system-reminder>", close: "</system-reminder>" },
];
