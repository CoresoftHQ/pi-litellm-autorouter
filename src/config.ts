/**
 * Config loading, validation and defaulting.
 *
 * Two files are read, project over global:
 *   ~/.pi/agent/autorouter.json   (global)
 *   .pi/autorouter.json           (project-local, shallow-merged on top)
 *
 * An invalid config never crashes startup. It disables routing and reports why, because
 * a coding agent that won't start is worse than one that doesn't route.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  DEFAULT_DIMENSION_WEIGHTS,
  DEFAULT_ESCALATION_KEYWORDS,
  DEFAULT_REMINDER_MARKERS,
  DEFAULT_SESSION_AFFINITY_TTL_SECONDS,
  DEFAULT_TIER_BOUNDARIES,
  DEFAULT_TOKEN_THRESHOLDS,
} from "./defaults.ts";
import { type ModelRef, type Tier, type TierTarget, TIER_SEVERITY_ORDER, isTier } from "./types.ts";

export type ClassifierType = "heuristic" | "llm";
export type ClassifierFallback = "heuristic" | "default_model";
export type ClassificationRubric = "legacy" | "agentic" | "chat" | "business";

export interface KeywordTierRule {
  keywords: string[];
  tier: Tier;
}

export interface ReminderMarkerPair {
  open: string;
  close: string;
}

export interface ClassifierLLMConfig {
  model: ModelRef;
  timeoutMs: number;
  classificationRubric: ClassificationRubric;
  /** Replaces the built-in rubric entirely. See the warning in `validate`. */
  systemPrompt?: string;
}

export interface RouterConfig {
  enabled: boolean;
  strategy: "local" | "proxy";
  defaultModel: ModelRef | null;
  tiers: Record<Tier, TierTarget[]>;
  tierBoundaries: Record<string, number>;
  tokenThresholds: Record<string, number>;
  dimensionWeights: Record<string, number>;
  reasoningOverrideMinScore: number | null;
  codeKeywords?: string[];
  reasoningKeywords?: string[];
  technicalKeywords?: string[];
  simpleKeywords?: string[];
  classifierType: ClassifierType;
  classifierLLMConfig: ClassifierLLMConfig | null;
  classifierFallback: ClassifierFallback;
  classifierContextWindowSize: number;
  classifierContextPerTurnChars: number;
  classifierContextIncludeAssistantTurns: boolean;
  keywordTierRules: KeywordTierRule[];
  escalationKeywords: string[];
  planModeMinTier: Tier | null;
  planModePatterns: string[];
  sessionAffinity: boolean;
  sessionAffinityTtlSeconds: number;
  reminderMarkers: ReminderMarkerPair[];
  proxy: { baseUrl: string; apiKey?: string; model: string } | null;
}

export interface LoadedConfig {
  config: RouterConfig;
  /** Non-fatal problems. Routing continues. */
  warnings: string[];
  /** Fatal problems. Routing is disabled and `config.enabled` is false. */
  errors: string[];
  /** Files that contributed, for `/autoroute`. */
  sources: string[];
}

const EMPTY_TIERS: Record<Tier, TierTarget[]> = {
  SIMPLE: [],
  MEDIUM: [],
  COMPLEX: [],
  REASONING: [],
};

export function defaultConfig(): RouterConfig {
  return {
    enabled: true,
    strategy: "local",
    defaultModel: null,
    tiers: { ...EMPTY_TIERS },
    tierBoundaries: { ...DEFAULT_TIER_BOUNDARIES },
    tokenThresholds: { ...DEFAULT_TOKEN_THRESHOLDS },
    dimensionWeights: { ...DEFAULT_DIMENSION_WEIGHTS },
    reasoningOverrideMinScore: null,
    classifierType: "heuristic",
    classifierLLMConfig: null,
    classifierFallback: "heuristic",
    classifierContextWindowSize: DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
    classifierContextPerTurnChars: DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS,
    classifierContextIncludeAssistantTurns: false,
    keywordTierRules: [],
    escalationKeywords: [...DEFAULT_ESCALATION_KEYWORDS],
    planModeMinTier: null,
    planModePatterns: [],
    sessionAffinity: false,
    sessionAffinityTtlSeconds: DEFAULT_SESSION_AFFINITY_TTL_SECONDS,
    reminderMarkers: DEFAULT_REMINDER_MARKERS.map((m) => ({ ...m })),
    proxy: null,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Normalize one tier's config value: a string, an object, or a list mixing both. */
function parseTierTargets(raw: unknown, tier: string, errors: string[]): TierTarget[] {
  const entries = Array.isArray(raw) ? raw : [raw];
  const targets: TierTarget[] = [];
  for (const entry of entries) {
    if (typeof entry === "string") {
      const model = entry.trim();
      if (model) targets.push({ model });
      continue;
    }
    if (isRecord(entry) && typeof entry.model === "string" && entry.model.trim()) {
      const target: TierTarget = { model: entry.model.trim() };
      if (typeof entry.thinkingLevel === "string") {
        target.thinkingLevel = entry.thinkingLevel as TierTarget["thinkingLevel"];
      }
      targets.push(target);
      continue;
    }
    errors.push(`tiers.${tier}: entries must be a model string or { model, thinkingLevel }`);
  }
  return targets;
}

/**
 * Merge and validate raw config objects, project last.
 *
 * Validation mirrors upstream's, including the rules that exist for a specific reason
 * rather than for tidiness — those are called out inline.
 */
export function buildConfig(layers: unknown[]): { config: RouterConfig; warnings: string[]; errors: string[] } {
  const config = defaultConfig();
  const warnings: string[] = [];
  const errors: string[] = [];

  const raw: Record<string, unknown> = {};
  for (const layer of layers) {
    if (layer === undefined) continue;
    if (!isRecord(layer)) {
      errors.push("config must be a JSON object");
      continue;
    }
    Object.assign(raw, layer);
  }

  if (typeof raw.enabled === "boolean") config.enabled = raw.enabled;

  if (raw.strategy !== undefined) {
    if (raw.strategy === "local" || raw.strategy === "proxy") {
      config.strategy = raw.strategy;
    } else {
      errors.push(`strategy must be "local" or "proxy", got ${JSON.stringify(raw.strategy)}`);
    }
  }

  if (typeof raw.defaultModel === "string" && raw.defaultModel.trim()) {
    config.defaultModel = raw.defaultModel.trim();
  }

  if (raw.tiers !== undefined) {
    if (!isRecord(raw.tiers)) {
      errors.push("tiers must be an object mapping tier names to models");
    } else {
      for (const [key, value] of Object.entries(raw.tiers)) {
        if (!isTier(key)) {
          errors.push(`tiers.${key} is not a known tier (${TIER_SEVERITY_ORDER.join(", ")})`);
          continue;
        }
        config.tiers[key] = parseTierTargets(value, key, errors);
      }
    }
  }

  for (const [key, target] of [
    ["tierBoundaries", config.tierBoundaries],
    ["tokenThresholds", config.tokenThresholds],
    ["dimensionWeights", config.dimensionWeights],
  ] as const) {
    const value = raw[key];
    if (value === undefined) continue;
    if (!isRecord(value)) {
      errors.push(`${key} must be an object`);
      continue;
    }
    for (const [k, v] of Object.entries(value)) {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        errors.push(`${key}.${k} must be a finite number`);
        continue;
      }
      target[k] = v;
    }
  }

  if (typeof raw.reasoningOverrideMinScore === "number") {
    config.reasoningOverrideMinScore = raw.reasoningOverrideMinScore;
  }

  for (const key of ["codeKeywords", "reasoningKeywords", "technicalKeywords", "simpleKeywords"] as const) {
    const value = raw[key];
    if (value === undefined) continue;
    if (!Array.isArray(value) || value.some((k) => typeof k !== "string")) {
      errors.push(`${key} must be an array of strings`);
      continue;
    }
    config[key] = value as string[];
  }

  if (raw.classifierType !== undefined) {
    if (raw.classifierType === "heuristic" || raw.classifierType === "llm") {
      config.classifierType = raw.classifierType;
    } else {
      errors.push(`classifierType must be "heuristic" or "llm"`);
    }
  }

  if (raw.classifierLLMConfig !== undefined) {
    const value = raw.classifierLLMConfig;
    if (!isRecord(value) || typeof value.model !== "string" || !value.model.trim()) {
      errors.push("classifierLLMConfig.model is required and must be a model string");
    } else {
      const rubric = value.classificationRubric;
      const systemPrompt = value.systemPrompt;

      // Mutually exclusive upstream: a custom system prompt IS the whole system role, so a
      // preset set alongside it would never reach the wire. Rejecting beats honouring one of
      // two settings the user asked for.
      if (rubric !== undefined && systemPrompt !== undefined) {
        errors.push(
          "classifierLLMConfig.classificationRubric and systemPrompt are mutually exclusive: " +
            "systemPrompt replaces the rubric the preset would select. Drop one.",
        );
      }
      if (systemPrompt !== undefined && (typeof systemPrompt !== "string" || !systemPrompt.trim())) {
        // A blank string would send an empty system role, leaving the classifier no rubric
        // at all. Omit the field to get the default.
        errors.push("classifierLLMConfig.systemPrompt must be non-empty; omit it to use the built-in rubric");
      }
      if (
        rubric !== undefined &&
        !["legacy", "agentic", "chat", "business"].includes(rubric as string)
      ) {
        errors.push(`classifierLLMConfig.classificationRubric must be one of legacy, agentic, chat, business`);
      }

      const llm: ClassifierLLMConfig = {
        model: value.model.trim(),
        timeoutMs:
          typeof value.timeoutMs === "number" && value.timeoutMs > 0
            ? value.timeoutMs
            : DEFAULT_CLASSIFIER_TIMEOUT_MS,
        // Upstream defaults to "legacy" so an existing deployment's spend does not move on
        // upgrade. A new project has no such constraint, and this one is agent traffic by
        // definition, so "agentic" is the default here.
        classificationRubric: (rubric as ClassificationRubric) ?? "agentic",
      };
      if (typeof systemPrompt === "string" && systemPrompt.trim()) {
        llm.systemPrompt = systemPrompt;
        // The built-in rubric's closing paragraph is the prompt-injection defence. A full
        // replacement drops it, and the heuristic fallback still scores *complexity*, which a
        // router on some other taxonomy does not want.
        warnings.push(
          "classifierLLMConfig.systemPrompt replaces the built-in rubric, including its " +
            "prompt-injection defence. Restate that instruction yourself, and consider " +
            'classifierFallback: "default_model" if your taxonomy is not complexity.',
        );
      }
      config.classifierLLMConfig = llm;
    }
  }

  if (raw.classifierFallback !== undefined) {
    if (raw.classifierFallback === "heuristic" || raw.classifierFallback === "default_model") {
      config.classifierFallback = raw.classifierFallback;
    } else {
      errors.push(`classifierFallback must be "heuristic" or "default_model"`);
    }
  }

  if (typeof raw.classifierContextWindowSize === "number" && raw.classifierContextWindowSize >= 0) {
    config.classifierContextWindowSize = Math.floor(raw.classifierContextWindowSize);
  }
  if (typeof raw.classifierContextPerTurnChars === "number" && raw.classifierContextPerTurnChars > 0) {
    config.classifierContextPerTurnChars = Math.floor(raw.classifierContextPerTurnChars);
  }
  if (typeof raw.classifierContextIncludeAssistantTurns === "boolean") {
    config.classifierContextIncludeAssistantTurns = raw.classifierContextIncludeAssistantTurns;
  }

  if (raw.keywordTierRules !== undefined) {
    if (!Array.isArray(raw.keywordTierRules)) {
      errors.push("keywordTierRules must be an array");
    } else {
      for (const [i, entry] of raw.keywordTierRules.entries()) {
        if (!isRecord(entry) || !Array.isArray(entry.keywords) || !isTier(entry.tier)) {
          errors.push(`keywordTierRules[${i}] must be { keywords: string[], tier: <tier name> }`);
          continue;
        }
        // Blank keywords are a routing foot-gun, not a harmless typo: "" substring-matches
        // essentially every prompt, so one stray blank silently forces this rule's tier for
        // all traffic. Drop blanks, and reject a rule that has nothing left.
        const keywords = entry.keywords
          .filter((k): k is string => typeof k === "string")
          .map((k) => k.trim())
          .filter((k) => k.length > 0);
        if (keywords.length === 0) {
          errors.push(`keywordTierRules[${i}] must contain at least one non-empty keyword`);
          continue;
        }
        config.keywordTierRules.push({ keywords, tier: entry.tier });
      }
    }
  }

  if (raw.escalationKeywords !== undefined) {
    if (!Array.isArray(raw.escalationKeywords) || raw.escalationKeywords.some((k) => typeof k !== "string")) {
      errors.push("escalationKeywords must be an array of strings");
    } else {
      // An empty array disables escalation, which is a legitimate choice.
      config.escalationKeywords = (raw.escalationKeywords as string[])
        .map((k) => k.trim())
        .filter((k) => k.length > 0);
    }
  }

  if (raw.planMode !== undefined) {
    if (!isRecord(raw.planMode)) {
      errors.push("planMode must be an object");
    } else {
      if (raw.planMode.minTier !== undefined) {
        if (isTier(raw.planMode.minTier)) {
          config.planModeMinTier = raw.planMode.minTier;
        } else {
          errors.push(`planMode.minTier must be one of ${TIER_SEVERITY_ORDER.join(", ")}`);
        }
      }
      if (Array.isArray(raw.planMode.patterns)) {
        config.planModePatterns = raw.planMode.patterns
          .filter((p): p is string => typeof p === "string")
          .map((p) => p.trim())
          .filter((p) => p.length > 0);
      }
    }
  }

  if (raw.sessionAffinity !== undefined) {
    if (typeof raw.sessionAffinity === "boolean") {
      config.sessionAffinity = raw.sessionAffinity;
    } else if (isRecord(raw.sessionAffinity)) {
      if (typeof raw.sessionAffinity.enabled === "boolean") {
        config.sessionAffinity = raw.sessionAffinity.enabled;
      }
      if (typeof raw.sessionAffinity.ttlSeconds === "number" && raw.sessionAffinity.ttlSeconds > 0) {
        config.sessionAffinityTtlSeconds = Math.floor(raw.sessionAffinity.ttlSeconds);
      }
    } else {
      errors.push("sessionAffinity must be a boolean or { enabled, ttlSeconds }");
    }
  }

  if (raw.reminderMarkers !== undefined) {
    if (!Array.isArray(raw.reminderMarkers) || raw.reminderMarkers.length === 0) {
      errors.push("reminderMarkers must be a non-empty array of { open, close } pairs");
    } else {
      const pairs: ReminderMarkerPair[] = [];
      for (const [i, entry] of raw.reminderMarkers.entries()) {
        if (!isRecord(entry) || typeof entry.open !== "string" || typeof entry.close !== "string") {
          errors.push(`reminderMarkers[${i}] must be { open: string, close: string }`);
          continue;
        }
        if (!entry.open.trim() || !entry.close.trim()) {
          errors.push(`reminderMarkers[${i}] open and close must both be non-empty`);
          continue;
        }
        pairs.push({ open: entry.open, close: entry.close });
      }
      // Setting this *replaces* the built-in pair rather than adding to it, matching upstream:
      // list the built-in pair too if your harness still emits it.
      if (pairs.length > 0) config.reminderMarkers = pairs;
    }
  }

  if (raw.proxy !== undefined) {
    if (!isRecord(raw.proxy) || typeof raw.proxy.baseUrl !== "string" || typeof raw.proxy.model !== "string") {
      errors.push("proxy must be { baseUrl, model, apiKey? }");
    } else {
      config.proxy = {
        baseUrl: raw.proxy.baseUrl,
        model: raw.proxy.model,
        ...(typeof raw.proxy.apiKey === "string" ? { apiKey: raw.proxy.apiKey } : {}),
      };
    }
  }

  // Cross-field checks.
  if (config.strategy === "proxy" && !config.proxy) {
    errors.push('strategy is "proxy" but no proxy config was given');
  }
  if (config.classifierType === "llm" && !config.classifierLLMConfig) {
    errors.push('classifierType is "llm" but classifierLLMConfig is missing');
  }
  if (config.strategy === "local") {
    const configuredTiers = TIER_SEVERITY_ORDER.filter((t) => config.tiers[t].length > 0);
    if (configuredTiers.length === 0 && !config.defaultModel) {
      errors.push("no tiers and no defaultModel are configured, so there is nothing to route to");
    }
    for (const tier of TIER_SEVERITY_ORDER) {
      if (config.tiers[tier].length === 0 && configuredTiers.length > 0) {
        warnings.push(`tiers.${tier} has no models; requests classified there fall back to the next candidate`);
      }
    }
  }

  if (errors.length > 0) config.enabled = false;

  return { config, warnings, errors };
}

function readJsonIfPresent(path: string, errors: string[]): { value: unknown; present: boolean } {
  try {
    const text = readFileSync(path, "utf8");
    return { value: JSON.parse(text), present: true };
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return { value: undefined, present: false };
    // A malformed config is worth reporting loudly; a missing one is not an error at all.
    errors.push(`${path}: ${err instanceof Error ? err.message : String(err)}`);
    return { value: undefined, present: false };
  }
}

/** Load global then project config. Never throws. */
export function loadConfig(cwd: string, home: string = homedir()): LoadedConfig {
  const errors: string[] = [];
  const sources: string[] = [];

  const globalPath = join(home, ".pi", "agent", "autorouter.json");
  const projectPath = join(cwd, ".pi", "autorouter.json");

  const globalLayer = readJsonIfPresent(globalPath, errors);
  if (globalLayer.present) sources.push(globalPath);
  const projectLayer = readJsonIfPresent(projectPath, errors);
  if (projectLayer.present) sources.push(projectPath);

  const built = buildConfig([globalLayer.value, projectLayer.value]);
  const allErrors = [...errors, ...built.errors];
  if (allErrors.length > 0) built.config.enabled = false;

  if (sources.length === 0) {
    // No config at all is not an error — it just means there is nothing to route with yet.
    built.config.enabled = false;
    built.warnings.push(`no autorouter config found (looked in ${globalPath} and ${projectPath})`);
  }

  return { config: built.config, warnings: built.warnings, errors: allErrors, sources };
}
