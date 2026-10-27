/**
 * pi-litellm-autorouter — LiteLLM's Auto Router v2, running inside pi.
 *
 * Wiring only: the routing decision itself lives in `router.ts`, which has no pi imports
 * so it can be tested without an agent.
 *
 * The seam is a pi virtual model, `autoroute/auto`. Selecting it in `/model` (or with
 * `--model autoroute/auto`) is what turns routing on; pi then asks this extension for a
 * physical model before every request. The selection stays `autoroute/auto`, while each
 * assistant message records the model that actually answered, so nothing here ever calls
 * `pi.setModel()` to route a prompt. Picking any other model leaves the router alone.
 *
 * Only the first request of a prompt is classified. Tool-call continuations, retries,
 * mid-run steers and follow-ups, and requests outside the agent loop (compaction
 * summaries) stay on the model already serving the conversation, which keeps prompt
 * caches and thinking signatures valid.
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionContext, ModelRoute, ModelRouteRequest } from "@earendil-works/pi-coding-agent";
import { buildConfig, type RouterConfig, loadConfig } from "./config.ts";
import { buildAdaptiveRouter, turnFromRun } from "./adaptive/index.ts";
import { classifyRequestType } from "./adaptive/request-type.ts";
import type { AdaptiveRouter } from "./adaptive/router.ts";
import { defaultAdaptiveStorePath, readPersistedCells, writePersistedCells } from "./adaptive/store.ts";
import { classifyHeuristic } from "./classify/heuristic.ts";
import { splitModelRef } from "./classify/llm.ts";
import { type SemanticMatcher, buildSemanticMatcher } from "./classify/semantic.ts";
import { ClassifierCircuitBreaker } from "./classify/decision-model.ts";
import {
  type AutorouteState,
  DECISION_ENTRY_TYPE,
  STATE_ENTRY_TYPE,
  decisionLogLine,
  explain,
  renderAdaptiveSnapshot,
  routingFailureNotice,
  classifierFallbackNotice,
  renderDecisionEntry,
  statusLine,
} from "./decision.ts";
import {
  extractTurn,
  splitCurrentTurn,
  stripReminderBlocks,
  type SimpleMessage,
  withSkillBlocks,
} from "./extract.ts";
import { type CandidateModel, buildInitialConfig, renderInitPreview } from "./init.ts";
import { detectQuestionReply } from "./question-reply.ts";
import {
  type ModelLookup,
  type Resolved,
  candidatesForTier,
  defaultTarget,
  isVirtualModel,
  resolveFirstUsable,
  resolveModelRef,
} from "./resolve.ts";
import { type RouteOutput, type SessionPin, route } from "./router.ts";
import {
  THINKING_LEVELS,
  isThinkingLevel,
  isTier,
  type ExtractedTurn,
  type ModelRef,
  type RouteDecision,
  type ThinkingLevel,
  type Tier,
  type TierTarget,
} from "./types.ts";

const STATUS_KEY = "autoroute";

/** The router's place in pi's model catalog: `autoroute/auto`. The provider id is used by
 *  nothing else, so the model needs no credentials of its own and is always selectable. */
export const VIRTUAL_PROVIDER = "autoroute";
export const VIRTUAL_MODEL_ID = "auto";
export const VIRTUAL_MODEL_REF = `${VIRTUAL_PROVIDER}/${VIRTUAL_MODEL_ID}`;

/**
 * Router state pi keeps on the session branch. It follows forks and `/tree` navigation and
 * survives compaction, which an in-memory variable would not.
 */
export interface BranchState {
  /** Tier of the last routed prompt: what a question reply escalates or floors from. */
  tier: Tier | null;
  /** Model and thinking level the last prompt was routed to. */
  model: ModelRef | null;
  thinkingLevel: ThinkingLevel | null;
  /** The session-affinity pin, when `sessionAffinity` is on. */
  pin: SessionPin | null;
}

type Route = ModelRoute<BranchState>;
type RouteRequest = ModelRouteRequest<BranchState>;
type PiModel = Route["model"];

function isRouterModel(model: { provider?: unknown; id?: unknown } | null | undefined): boolean {
  return model?.provider === VIRTUAL_PROVIDER && model?.id === VIRTUAL_MODEL_ID;
}

function refOf(model: { provider: string; id: string }): ModelRef {
  return `${model.provider}/${model.id}`;
}

function sameState(a: BranchState | undefined, b: BranchState): boolean {
  return a !== undefined && JSON.stringify(a) === JSON.stringify(b);
}

/** `provider/model` of the newest assistant message that a physical model answered. */
function servedModel(messages: readonly unknown[]): ModelRef | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as
      | { role?: unknown; provider?: unknown; model?: unknown; api?: unknown; stopReason?: unknown }
      | undefined;
    if (message?.role !== "assistant") continue;
    // A failed routing attempt leaves the virtual model on its message.
    if (isVirtualModel(message) || message.stopReason === "error" || message.stopReason === "aborted") continue;
    if (typeof message.provider === "string" && typeof message.model === "string") {
      return `${message.provider}/${message.model}`;
    }
  }
  return null;
}

/** Convert validated config back to the documented JSON shape before command edits.
 * Writing a complete project layer avoids shallow config merging dropping global nested
 * fields (for example, three tiers when only `tiers.SIMPLE` was changed). */
function configToJson(config: RouterConfig): Record<string, unknown> {
  const tierValue = (targets: TierTarget[]): unknown =>
    targets.length === 1 && Object.keys(targets[0]!).length === 1
      ? targets[0]!.model
      : targets.map((target) => ({ ...target }));
  return {
    enabled: config.enabled,
    ...(config.defaultModel
      ? {
          defaultModel: config.defaultModelThinkingLevel
            ? { model: config.defaultModel, thinkingLevel: config.defaultModelThinkingLevel }
            : config.defaultModel,
        }
      : {}),
    tiers: Object.fromEntries(Object.entries(config.tiers).map(([tier, targets]) => [tier, tierValue(targets)])),
    tierBoundaries: config.tierBoundaries,
    tokenThresholds: config.tokenThresholds,
    dimensionWeights: config.dimensionWeights,
    ...(config.reasoningOverrideMinScore !== null ? { reasoningOverrideMinScore: config.reasoningOverrideMinScore } : {}),
    ...(config.codeKeywords ? { codeKeywords: config.codeKeywords } : {}),
    ...(config.reasoningKeywords ? { reasoningKeywords: config.reasoningKeywords } : {}),
    ...(config.simpleKeywords ? { simpleKeywords: config.simpleKeywords } : {}),
    customTechnicalKeywords: config.customTechnicalKeywords,
    classifierType: config.classifierType,
    ...(config.classifierLLMConfig ? { classifierLLMConfig: config.classifierLLMConfig } : {}),
    ...(config.decisionClassifierConfig
      ? {
          decisionClassifierConfig: Object.fromEntries(
            Object.entries(config.decisionClassifierConfig).filter(([, value]) => value !== null && value !== undefined),
          ),
        }
      : {}),
    classifierFallback: config.classifierFallback,
    classifierContextWindowSize: config.classifierContextWindowSize,
    classifierContextPerTurnChars: config.classifierContextPerTurnChars,
    classifierContextIncludeAssistantTurns: config.classifierContextIncludeAssistantTurns,
    keywordTierRules: config.keywordTierRules,
    escalationKeywords: config.escalationKeywords,
    planMode: {
      ...(config.planModeMinTier ? { minTier: config.planModeMinTier } : {}),
      patterns: config.planModePatterns,
    },
    sessionAffinity: { enabled: config.sessionAffinity, ttlSeconds: config.sessionAffinityTtlSeconds },
    questionReply: {
      enabled: config.questionReply,
      toolNames: config.questionReplyToolNames,
      maxReplyChars: config.questionReplyMaxChars,
    },
    reminderMarkers: config.reminderMarkers,
    adaptive: config.adaptive,
    adaptiveWeights: config.adaptiveWeights,
    tierDistancePenalty: config.tierDistancePenalty,
    adaptiveEligible: config.adaptiveEligible,
    semanticKeywordMatching: config.semanticKeywordMatching,
    ...(config.embeddingModel ? { embeddingModel: config.embeddingModel } : {}),
    matchThreshold: config.matchThreshold,
    embeddingEndpoint: config.embeddingEndpoint,
    decisionLog: config.decisionLog,
  };
}

function setPath(object: Record<string, unknown>, path: string, value: unknown): boolean {
  const parts = path.split(".").filter(Boolean);
  if (parts.length === 0 || parts.some((part) => !/^[A-Za-z][A-Za-z0-9_]*$/.test(part))) return false;
  let cursor: Record<string, unknown> = object;
  for (const part of parts.slice(0, -1)) {
    const current = cursor[part];
    if (current === undefined) cursor[part] = {};
    else if (typeof current !== "object" || current === null || Array.isArray(current)) return false;
    cursor = cursor[part] as Record<string, unknown>;
  }
  cursor[parts.at(-1)!] = value;
  return true;
}

function readSettingValue(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export default function autorouter(pi: ExtensionAPI): void {
  let config: RouterConfig | null = null;
  let configWarnings: string[] = [];
  let configErrors: string[] = [];
  let configSources: string[] = [];
  /** No config file at all, as opposed to one that failed to load. */
  let configMissing = false;

  let state: AutorouteState = {};
  let lastDecision: RouteDecision | null = null;
  /** The ask the last decision was made on; what the bandit credits feedback against.
   *  A held turn deliberately leaves this alone: "ok" describes no work, so feedback from
   *  the run it triggers belongs to the request that was already in flight. */
  let lastRoutedAsk: string | null = null;
  let adaptive: AdaptiveRouter | null = null;
  const adaptiveStorePath = defaultAdaptiveStorePath();
  let semantic: SemanticMatcher | null = null;
  /** The decision classifier's timeout breaker; rebuilt, and so closed, on every config load. */
  let decisionBreaker: ClassifierCircuitBreaker | null = null;
  let planModeActive = false;
  /** Model refs for command completion; the completion callback gets no context. */
  let availableRefs: string[] = [];
  /** Set once the current agent run's prompt has been routed, cleared at `agent_end`. Pi
   *  routes a steer or a follow-up as a user request too; this is what keeps such a
   *  message on the model the run was routed to rather than reclassifying it. */
  let routedThisRun = false;
  /** Set when this run's prompt went through the classifier and left a decision, which is
   *  what the bandit may learn from. A paused or pinned router chose nothing. */
  let decidedThisRun = false;
  /** Whether the prompt that started the run came from another extension
   *  (`pi.sendUserMessage`) rather than from the user. */
  let promptFromExtension = false;

  const supportsVirtualModels =
    typeof (pi as { registerVirtualModel?: unknown }).registerVirtualModel === "function";

  pi.registerFlag("no-autoroute", {
    description: "Start with automatic model routing paused (autoroute/auto holds its current model)",
    type: "boolean",
    default: false,
  });

  // The decision log. Every decision is already persisted as a custom entry, which pi can
  // draw in the chat without it ever reaching the model; this is the drawing. It renders
  // live as each entry is appended and again when a session is resumed, and ctrl+o
  // (expand tool output) swaps the one-liner for the full explanation.
  pi.registerEntryRenderer<RouteDecision>(DECISION_ENTRY_TYPE, (entry, { expanded }, theme) => {
    if (config?.decisionLog === false || !entry.data) return undefined;
    return renderDecisionEntry(entry.data, expanded, theme) as never;
  });

  /** The same line for runs with no chat to draw in (`pi -p`, RPC): stderr, so it never
   *  mixes with the agent's own output on stdout. */
  const logDecisionToConsole = (decision: RouteDecision, ctx: ExtensionContext): void => {
    if (!config?.decisionLog || ctx.hasUI) return;
    process.stderr.write(`${decisionLogLine(decision)}\n`);
  };

  /** Read the session's message history in the shape `extractTurn` wants. */
  const sessionMessages = (ctx: ExtensionContext): SimpleMessage[] => {
    try {
      const entries = ctx.sessionManager.getBranch();
      const messages: SimpleMessage[] = [];
      for (const entry of entries) {
        if (entry.type !== "message") continue;
        const message = (entry as { message?: { role?: unknown; content?: unknown } }).message;
        if (!message || typeof message.role !== "string") continue;
        if (message.role !== "user" && message.role !== "assistant") continue;
        messages.push({ role: message.role, content: message.content });
      }
      return messages;
    } catch {
      // History is an optimisation for the classifier's context window, never a
      // requirement. A session that cannot produce it still routes on the current ask.
      return [];
    }
  };

  /**
   * The conversation a request is routed on, user and assistant messages only.
   *
   * Read from the session branch when it has one: there, messages other extensions inject
   * (`before_agent_start` context, say) are custom entries rather than user messages, so
   * they cannot pass for the user's ask. The request's own messages are the fallback.
   */
  const conversation = (request: RouteRequest, ctx: ExtensionContext): SimpleMessage[] => {
    const fromBranch = sessionMessages(ctx);
    if (fromBranch.some((message) => message.role === "user")) return fromBranch;
    return request.messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map((message) => ({ role: message.role, content: message.content }));
  };

  const restoreState = (ctx: ExtensionContext): void => {
    state = {};
    lastDecision = null;
    try {
      for (const entry of ctx.sessionManager.getEntries()) {
        if (entry.type !== "custom") continue;
        const custom = entry as { customType?: string; data?: unknown };
        if (custom.customType === STATE_ENTRY_TYPE && custom.data) {
          state = { ...state, ...(custom.data as AutorouteState) };
        }
        // Decisions are replayed too, so `/autoroute explain` and a question reply in a
        // session from before router state existed still know the last tier.
        if (custom.customType === DECISION_ENTRY_TYPE && custom.data) {
          lastDecision = custom.data as RouteDecision;
        }
      }
    } catch {
      // A session whose entries cannot be read starts with default state rather than
      // refusing to load.
    }
  };

  const persistState = (): void => {
    pi.appendEntry<AutorouteState>(STATE_ENTRY_TYPE, state);
  };

  /**
   * (Re)build the bandit for the current config, overlaying whatever it learned before.
   * Posteriors for models no longer in any pool are dropped on load, so a config change
   * never carries stale beliefs across.
   */
  const rebuildAdaptive = (ctx: ExtensionContext): void => {
    adaptive = config ? buildAdaptiveRouter(config, { registry: ctx.modelRegistry }) : null;
    if (adaptive) adaptive.load(readPersistedCells(adaptiveStorePath));
  };

  /** The semantic matcher embeds the rule keywords lazily, on the first prompt that
   *  needs them, so building it here costs nothing until routing does. */
  const rebuildSemantic = (ctx: ExtensionContext): void => {
    semantic = config ? buildSemanticMatcher(config, { registry: ctx.modelRegistry }) : null;
  };

  const reloadConfig = (ctx: ExtensionContext): void => {
    const loaded = loadConfig(ctx.cwd);
    config = loaded.config;
    configWarnings = loaded.warnings;
    configErrors = loaded.errors;
    configSources = loaded.sources;
    configMissing = loaded.missing;
    rebuildAdaptive(ctx);
    rebuildSemantic(ctx);
    const decision = config.classifierType === "decision" ? config.decisionClassifierConfig : null;
    decisionBreaker = decision?.circuitBreakerEnabled
      ? new ClassifierCircuitBreaker(decision.circuitBreakerCooldownSeconds)
      : null;
  };

  /** Persist one complete effective config layer and activate it immediately. */
  const writeConfig = (scope: "global" | "project", updated: Record<string, unknown>, ctx: ExtensionContext): string | null => {
    const target =
      scope === "global" ? join(homedir(), ".pi", "agent", "autorouter.json") : join(ctx.cwd, ".pi", "autorouter.json");
    try {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, `${JSON.stringify(updated, null, 2)}\n`, "utf8");
      reloadConfig(ctx);
      return target;
    } catch (err) {
      ctx.ui.notify(`autoroute: could not write settings: ${err instanceof Error ? err.message : String(err)}`, "error");
      return null;
    }
  };

  /** Why prompts are not being classified, or null when they are. */
  const routingDisabled = (): string | null => {
    if (!config?.enabled) return "no usable config";
    if (pi.getFlag("no-autoroute") === true) return "--no-autoroute";
    if (state.disabled) return "/autoroute off";
    if (state.pinnedModel) return `pinned to ${state.pinnedModel}`;
    return null;
  };

  /** Physical models only: tiers, pins and overrides cannot name a virtual model. */
  const physicalModels = (models: readonly unknown[]): CandidateModel[] =>
    models.filter((model) => !isVirtualModel(model)) as CandidateModel[];

  /** The model serving this session, for `/autoroute pin` without an argument. */
  const currentServedModel = (ctx: ExtensionContext): ModelRef | null => {
    try {
      const messages = ctx.sessionManager
        .getBranch()
        .flatMap((entry) => (entry.type === "message" ? [(entry as { message?: unknown }).message] : []));
      const served = servedModel(messages);
      if (served) return served;
    } catch {
      // Fall through to what this process last routed to.
    }
    if (lastDecision?.chosenModel) return lastDecision.chosenModel;
    return ctx.model && !isVirtualModel(ctx.model) ? refOf(ctx.model) : null;
  };

  /** A model reference a command was given, checked against pi's catalog. */
  const checkCommandModel = (model: string, ctx: ExtensionContext): string | null => {
    const ref = splitModelRef(model);
    const found = ref ? ctx.modelRegistry.find(ref.provider, ref.modelId) : undefined;
    if (!found) return `autoroute: "${model}" is not a model pi knows. Use provider/model-id, e.g. anthropic/claude-opus-5.`;
    if (isVirtualModel(found)) return `autoroute: "${model}" is a virtual model; route to a physical model instead.`;
    return null;
  };

  /** Make `autoroute/auto` the session's model. Never persists it as the global default. */
  const selectRouterModel = async (ctx: ExtensionContext): Promise<boolean> => {
    if (isRouterModel(ctx.model)) return true;
    try {
      const model = ctx.modelRegistry.find(VIRTUAL_PROVIDER, VIRTUAL_MODEL_ID);
      return model ? await pi.setModel(model) : false;
    } catch {
      return false;
    }
  };

  // ── Routing ────────────────────────────────────────────────────────────────

  /**
   * The model already serving this conversation, when pi can still route to it: the one
   * that failed (for a retry), else the one that answered last, else the one router state
   * says the last prompt went to.
   */
  const servingRoute = (request: RouteRequest, lookup: ModelLookup): Route | null => {
    const fallbackLevel = request.state?.thinkingLevel ?? request.thinkingLevel;
    for (const served of [request.failed, request.previous]) {
      if (!served) continue;
      const resolved = resolveModelRef(refOf(served.model), lookup);
      if ("target" in resolved) {
        return { model: resolved.target as PiModel, thinkingLevel: served.thinkingLevel ?? fallbackLevel };
      }
    }
    const routed = request.state?.model;
    if (routed) {
      const resolved = resolveModelRef(routed, lookup);
      if ("target" in resolved) return { model: resolved.target as PiModel, thinkingLevel: fallbackLevel };
    }
    return null;
  };

  /** No classification, just a usable model: `defaultModel`, then the MEDIUM chain. */
  const configFallback = (lookup: ModelLookup): Resolved | null => {
    if (!config) return null;
    const candidates = [defaultTarget(config), ...candidatesForTier("MEDIUM", config, () => 0)].filter(
      (target): target is TierTarget => target !== null,
    );
    return resolveFirstUsable(candidates, lookup).resolved;
  };

  const routeRequest = async (request: RouteRequest, ctx: ExtensionContext): Promise<Route> => {
    const lookup: ModelLookup = ctx.modelRegistry;
    // A tier entry without a thinking level takes the level selected on autoroute/auto;
    // pi clamps whatever is returned to what the routed model supports.
    const toRoute = (target: unknown, level: ThinkingLevel | null | undefined): Route => ({
      model: target as PiModel,
      thinkingLevel: (level ?? request.thinkingLevel) as Route["thinkingLevel"],
    });
    const hold = (allowConfigFallback: boolean): Route | null => {
      const serving = servingRoute(request, lookup);
      if (serving) return serving;
      const fallback = allowConfigFallback && config?.enabled ? configFallback(lookup) : null;
      return fallback ? toRoute(fallback.target, fallback.thinkingLevel) : null;
    };
    const unroutable = (why: string): Error =>
      new Error(`autoroute: ${why}. Run /autoroute for details, or pick another model with /model.`);

    // Only a new prompt is classified. Tool-call continuations and retries stay on the
    // model serving the turn, which keeps prompt caches and thinking signatures valid; a
    // compaction summary or an extension's own call is not a prompt; and a steer or
    // follow-up arriving mid-run, or a prompt another extension sent, rides on the model
    // the run already has, as it always has.
    const newPrompt = request.reason === "user" && !routedThisRun && !promptFromExtension;
    if (!newPrompt) {
      const serving = servingRoute(request, lookup);
      if (serving) return serving;
      if (request.reason !== "user") {
        const fallback = config?.enabled ? configFallback(lookup) : null;
        if (fallback) return toRoute(fallback.target, fallback.thinkingLevel);
        throw unroutable("no model to route this request to");
      }
      // A user message with nothing to hold onto is routed like a new prompt.
    }
    routedThisRun = true;

    if (!config?.enabled) {
      const held = hold(false);
      if (held) return held;
      const why = configMissing
        ? "no autorouter config (run /autoroute init)"
        : (configErrors[0] ?? "routing is disabled in autorouter.json");
      throw unroutable(why);
    }

    // A one-shot override is an explicit instruction for this prompt, so it is honoured
    // even when routing is otherwise off or pinned. It is checked before those, and always
    // cleared afterwards, so "off" still means off from the next prompt onwards.
    const oneShot = state.nextTarget ?? (state.nextModel ? { model: state.nextModel } : null);

    const paused = pi.getFlag("no-autoroute") === true ? "--no-autoroute" : state.disabled ? "/autoroute off" : null;
    if (paused && !oneShot) {
      const held = hold(true);
      if (held) return held;
      throw unroutable(`routing is off (${paused}) and there is no model to hold`);
    }

    if (state.pinnedModel && !oneShot) {
      const pinned = resolveModelRef(state.pinnedModel, lookup);
      if ("target" in pinned) return toRoute(pinned.target, null);
      if (ctx.hasUI) ctx.ui.notify(`autoroute: cannot use the pinned model: ${pinned.problem}`, "error");
      const held = hold(false);
      if (held) return held;
      throw unroutable(`cannot use the pinned model: ${pinned.problem}`);
    }

    const branchState = request.state;
    let turn: ExtractedTurn;
    let result: RouteOutput;
    try {
      const { current, prior } = splitCurrentTurn(conversation(request, ctx));
      const markerPairs = withSkillBlocks(config.reminderMarkers);
      turn = extractTurn(current, prior, {
        markerPairs,
        contextWindowSize: config.classifierContextWindowSize,
        perTurnChars: config.classifierContextPerTurnChars,
        includeAssistantTurns: config.classifierContextIncludeAssistantTurns,
      });

      // Detection reads *this* turn's text, not `turn.currentAsk`: that field falls back to
      // the newest prior ask when the prompt is reminder-only, and answering a question
      // with the previous request's text would hold on the wrong grounds.
      const pendingReply = detectQuestionReply(stripReminderBlocks(current, markerPairs), prior, {
        enabled: config.questionReply,
        toolNames: config.questionReplyToolNames,
        maxChars: config.questionReplyMaxChars,
      });

      result = await route({
        turn,
        config,
        models: lookup,
        registry: ctx.modelRegistry as never,
        planModeActive,
        pin: branchState?.pin ?? null,
        oneShot,
        questionReply: pendingReply
          ? { toolName: pendingReply.toolName, lastTier: branchState?.tier ?? lastDecision?.tier ?? null }
          : null,
        callerSystemPrompt: undefined,
        adaptive,
        semantic,
        decisionBreaker,
        ...(request.signal ? { signal: request.signal } : {}),
      });
    } catch (err) {
      // Choosing a model is a routing decision; a failure in it falls back to the model
      // already serving the session rather than failing the turn, whenever there is one.
      const message = err instanceof Error ? err.message : String(err);
      if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, `autoroute error: ${message}`);
      const held = hold(true);
      if (held) return held;
      throw unroutable(message);
    }

    const { decision } = result;
    lastDecision = decision;
    if (decision.cause !== "question_reply") lastRoutedAsk = turn.currentAsk;
    if (result.consumedOneShot) {
      state.nextTarget = null;
      state.nextModel = null;
      persistState();
      if (ctx.hasUI && !decision.chosenModel) {
        ctx.ui.notify(`autoroute: ${decision.fellBackBecause ?? "one-shot override failed"}`, "error");
      }
    }

    // A held reply goes to the model that asked the question (or, with none, to the config
    // fallback). A chain with nothing usable in it keeps the model already serving the
    // session and nothing else: the MEDIUM fallback could promote a request past the tier
    // it was classified into, which is a bill the user did not ask for.
    const routed: Route | null = result.target
      ? toRoute(result.target, decision.thinkingLevel)
      : hold(decision.cause === "question_reply");

    pi.appendEntry<RouteDecision>(DECISION_ENTRY_TYPE, decision);
    decidedThisRun = true;
    logDecisionToConsole(decision, ctx);
    const heldModel = !result.target && routed ? refOf(routed.model) : null;
    const failureNotice = !decision.chosenModel ? routingFailureNotice(decision, heldModel) : null;
    if (ctx.hasUI) {
      ctx.ui.setStatus(STATUS_KEY, statusLine(decision));
      // The full candidate failures remain in the decision entry and `/autoroute explain`;
      // keep the toast actionable.
      if (failureNotice) ctx.ui.notify(failureNotice, "error");
      const classifierNotice = classifierFallbackNotice(decision);
      if (classifierNotice) ctx.ui.notify(classifierNotice, "warning");
    }
    if (!routed) throw failureNotice ? new Error(failureNotice) : unroutable("no configured model is usable");

    const nextState: BranchState = {
      tier: decision.tier,
      model: decision.chosenModel ?? refOf(routed.model),
      thinkingLevel: routed.thinkingLevel as ThinkingLevel,
      pin: result.pinToWrite ?? branchState?.pin ?? null,
    };
    // Returning the current state object keeps it; a new one is stored even when equal.
    return { ...routed, state: sameState(branchState, nextState) ? branchState : nextState };
  };

  if (supportsVirtualModels) {
    pi.registerVirtualModel<BranchState>({
      provider: VIRTUAL_PROVIDER,
      id: VIRTUAL_MODEL_ID,
      name: "Auto router",
      // The selected level is the default for tier entries that do not set their own.
      thinkingLevels: THINKING_LEVELS,
      route: routeRequest,
    });
  }

  // ── Events ─────────────────────────────────────────────────────────────────

  pi.on("session_start", async (_event, ctx) => {
    routedThisRun = false;
    decidedThisRun = false;
    promptFromExtension = false;
    reloadConfig(ctx);
    restoreState(ctx);
    try {
      availableRefs = physicalModels(ctx.modelRegistry.getAvailable()).map((m) => `${m.provider}/${m.id}`);
    } catch {
      availableRefs = [];
    }

    if (!supportsVirtualModels && ctx.hasUI) {
      ctx.ui.notify("autoroute needs a pi version with virtual models (pi.registerVirtualModel); please update pi", "error");
      return;
    }
    if (!ctx.hasUI) return;
    if (configMissing) {
      // Nothing to report until the router is actually selected without a config.
      if (isRouterModel(ctx.model)) ctx.ui.notify(`autoroute: no config yet; run /autoroute init`, "warning");
    } else if (configErrors.length > 0) {
      // Loud, but not fatal: a broken router config must not stop the agent from starting.
      ctx.ui.notify(`autoroute disabled: ${configErrors[0]}`, "error");
    }
  });

  // Plan mode is not a built-in pi concept — pi ships it as an extension — so there is no
  // native state to query. Instead this is an integration contract: a plan-mode extension
  // announces its state on the shared bus and the router treats it as a tier floor.
  //
  //   pi.events.emit("autoroute:plan-mode", { active: true })
  //
  // Failing that, `planMode.patterns` still matches sentinels carried in prompt text,
  // which is the only mechanism the LiteLLM proxy has. Text sentinels are spoofable by
  // anyone who pastes one, which is why the floor can only raise a tier, never lower it.
  pi.events.on("autoroute:plan-mode", (payload: unknown) => {
    planModeActive = Boolean(
      payload && typeof payload === "object" && (payload as { active?: unknown }).active,
    );
  });

  // Who sent the prompt that is about to start a run. Pi routes every user message the
  // same way, so this is the only place an extension's `sendUserMessage` can be told apart
  // from the user typing. Steers and follow-ups never start a run, so they are not counted.
  pi.on("input", async (event) => {
    if (!event.streamingBehavior) promptFromExtension = event.source === "extension";
    return { action: "continue" as const };
  });

  // The footer status describes the router's last decision; it means nothing once another
  // model is selected.
  pi.on("model_select", async (event, ctx) => {
    if (ctx.hasUI && !isRouterModel(event.model)) ctx.ui.setStatus(STATUS_KEY, undefined);
  });

  // The bandit learns from what happened after its pick: a rephrase or a "forget it" in
  // the next prompt counts against the model that produced the previous reply, a tool-call
  // loop or a near-duplicate reply counts against the model that produced this one.
  // Upstream does this from the proxy's post-call hook; `agent_end` is pi's equivalent.
  pi.on("agent_end", async (event, ctx) => {
    const decided = decidedThisRun;
    routedThisRun = false;
    decidedThisRun = false;
    promptFromExtension = false;
    // Only runs the router made a decision for teach it anything.
    if (!decided || !config?.adaptive || !adaptive) return;
    // Credit the model that actually answered, which on a held turn is the one that was
    // already serving the session.
    const model = servedModel(event.messages as readonly unknown[]) ?? lastDecision?.chosenModel;
    if (!model) return;
    try {
      const turn = turnFromRun(lastRoutedAsk, event.messages as never);
      adaptive.recordTurn(ctx.sessionManager.getSessionId(), model, classifyRequestType(lastRoutedAsk), turn);
      writePersistedCells(adaptiveStorePath, adaptive.serialize());
    } catch {
      // Learning is a bonus on top of routing; a failure here must not surface as one.
    }
  });

  // ── Commands ───────────────────────────────────────────────────────────────

  /**
   * Write a starting config built from the models pi can reach.
   *
   * Never silently overwrites: an existing file needs confirmation, and there is no
   * confirmation to give in a non-interactive run, so it refuses instead.
   */
  const runInit = async (rest: string[], ctx: ExtensionContext): Promise<void> => {
    const toProject = rest.includes("project");
    const useLLM = rest.includes("llm");
    const provider = rest.find((token) => !["project", "global", "llm", "heuristic"].includes(token));

    // Session scoping (`--models` / `enabledModels`) is the set the user actually intends
    // to use, so it wins over the full catalogue when it is set. Virtual models, this one
    // included, cannot be routed to and are left out either way.
    const scoped = physicalModels((ctx.scopedModels ?? []).map((entry) => entry.model));
    const source: CandidateModel[] = scoped.length > 0 ? scoped : physicalModels(ctx.modelRegistry.getAvailable());

    let result;
    try {
      result = buildInitialConfig(source, {
        ...(provider ? { provider } : {}),
        classifier: useLLM ? "llm" : "heuristic",
      });
    } catch (err) {
      ctx.ui.notify(`autoroute init: ${err instanceof Error ? err.message : String(err)}`, "error");
      return;
    }

    const target = toProject
      ? join(ctx.cwd, ".pi", "autorouter.json")
      : join(homedir(), ".pi", "agent", "autorouter.json");

    const preview = [
      `autoroute init — ${scoped.length > 0 ? "session-scoped models" : "all available models"}`,
      "",
      renderInitPreview(result),
      "",
      `  → ${target}`,
    ].join("\n");

    if (existsSync(target)) {
      if (!ctx.hasUI) {
        ctx.ui.notify(`autoroute init: ${target} already exists (no UI available to confirm)`, "error");
        return;
      }
      const ok = await ctx.ui.confirm("Overwrite existing autoroute config?", `${target}\n\n${preview}`);
      if (!ok) {
        ctx.ui.notify("autoroute init: cancelled", "info");
        return;
      }
    } else if (ctx.hasUI) {
      const ok = await ctx.ui.confirm("Write this autoroute config?", preview);
      if (!ok) {
        ctx.ui.notify("autoroute init: cancelled", "info");
        return;
      }
    }

    try {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, `${JSON.stringify(result.config, null, 2)}\n`, "utf8");
    } catch (err) {
      ctx.ui.notify(`autoroute init: could not write ${target}: ${err instanceof Error ? err.message : err}`, "error");
      return;
    }

    // Load it straight away and select the router, so the session routes without a restart.
    reloadConfig(ctx);
    const selected = config?.enabled ? await selectRouterModel(ctx) : false;
    const outcome = !config?.enabled
      ? "routing is still disabled"
      : selected
        ? `routing is active on ${VIRTUAL_MODEL_REF}`
        : `select ${VIRTUAL_MODEL_REF} in /model to start routing`;
    ctx.ui.notify(`${preview}\n\n  written. ${outcome}.`, "info");
  };

  pi.registerCommand("autoroute", {
    description: `Show or control automatic model routing (select ${VIRTUAL_MODEL_REF} to route)`,
    getArgumentCompletions: (prefix: string) => {
      // `next` and `pin` both take a model, so complete against the catalogue once the
      // verb is typed. Cached at session_start, since this callback gets no context.
      const modelVerb = /^(next|pin)\s+(.*)$/.exec(prefix);
      if (modelVerb) {
        const [, verb = "", typed = ""] = modelVerb;
        const matches = availableRefs
          .filter((ref) => ref.includes(typed))
          .slice(0, 25)
          .map((ref) => ({ value: `${verb} ${ref}`, label: ref }));
        return matches.length > 0 ? matches : null;
      }
      const verbs = ["init", "set", "tier", "next", "on", "off", "pin", "unpin", "explain", "log", "adaptive", "status"];
      const items = verbs.filter((v) => v.startsWith(prefix)).map((v) => ({ value: v, label: v }));
      return items.length > 0 ? items : null;
    },
    handler: async (args, ctx) => {
      const [verb = "status", ...rest] = args.trim().split(/\s+/).filter(Boolean);

      switch (verb) {
        case "init":
          await runInit(rest, ctx);
          return;

        case "set": {
          const explicitScope = rest[0] === "global" || rest[0] === "project";
          const scope: "global" | "project" = explicitScope ? rest[0]! as "global" | "project" : "project";
          const path = rest[explicitScope ? 1 : 0];
          const valueParts = rest.slice(explicitScope ? 2 : 1);
          if (!path || valueParts.length === 0) {
            ctx.ui.notify('usage: /autoroute set [global|project] <setting.path> <JSON value>', "info");
            return;
          }
          const updated = configToJson(config ?? loadConfig(ctx.cwd).config);
          // A missing or previously-invalid config is disabled by the loader. A successful
          // settings edit is an explicit recovery action, except when it itself sets enabled.
          if (path !== "enabled") updated.enabled = true;
          if (!setPath(updated, path, readSettingValue(valueParts.join(" ")))) {
            ctx.ui.notify(`autoroute: invalid setting path "${path}"`, "error");
            return;
          }
          const checked = buildConfig([updated]);
          if (checked.errors.length > 0) {
            ctx.ui.notify(`autoroute: setting was not saved: ${checked.errors.join("; ")}`, "error");
            return;
          }
          const target = writeConfig(scope, configToJson(checked.config), ctx);
          if (target) ctx.ui.notify(`autoroute: set ${path} in ${scope} settings (${target})`, "info");
          return;
        }

        case "tier": {
          const [tierName, model, thinking] = rest;
          if (!tierName || !model || rest.length > 3 || !isTier(tierName)) {
            ctx.ui.notify("usage: /autoroute tier <SIMPLE|MEDIUM|COMPLEX|REASONING> <provider/model> [thinking-level]", "info");
            return;
          }
          if (thinking !== undefined && !isThinkingLevel(thinking)) {
            ctx.ui.notify("autoroute: thinking level must be one of off, minimal, low, medium, high, xhigh, max", "error");
            return;
          }
          const problem = checkCommandModel(model, ctx);
          if (problem) {
            ctx.ui.notify(problem, "error");
            return;
          }
          const updated = configToJson(config ?? loadConfig(ctx.cwd).config);
          updated.enabled = true;
          setPath(updated, `tiers.${tierName}`, thinking ? { model, thinkingLevel: thinking } : model);
          const checked = buildConfig([updated]);
          if (checked.errors.length > 0) {
            ctx.ui.notify(`autoroute: tier was not saved: ${checked.errors.join("; ")}`, "error");
            return;
          }
          const target = writeConfig("project", configToJson(checked.config), ctx);
          if (target) ctx.ui.notify(`autoroute: ${tierName} → ${model}${thinking ? ` (${thinking})` : ""}`, "info");
          return;
        }

        case "next": {
          if (rest.length === 0) {
            const had = state.nextTarget;
            state.nextTarget = null;
            state.nextModel = null;
            persistState();
            ctx.ui.notify(had ? `autoroute: cleared one-shot override (${had.model})` : "autoroute: no override set", "info");
            return;
          }
          const [model, thinking] = rest;
          if (!model || rest.length > 2 || (thinking !== undefined && !isThinkingLevel(thinking))) {
            ctx.ui.notify("usage: /autoroute next <provider/model> [thinking-level]", "info");
            return;
          }
          // Validate now rather than at the next prompt: a typo should fail while the user
          // is still looking at the command, not silently route the prompt they cared about.
          const problem = checkCommandModel(model, ctx);
          if (problem) {
            ctx.ui.notify(problem, "error");
            return;
          }
          state.nextTarget = thinking ? { model, thinkingLevel: thinking } : { model };
          persistState();
          const hint = isRouterModel(ctx.model) ? "" : ` (applies once ${VIRTUAL_MODEL_REF} is selected)`;
          ctx.ui.notify(`autoroute: next prompt only → ${model}${thinking ? ` (${thinking})` : ""}${hint}`, "info");
          return;
        }

        case "on": {
          state.disabled = false;
          state.pinnedModel = null;
          state.nextTarget = null;
          state.nextModel = null;
          persistState();
          const selected = await selectRouterModel(ctx);
          ctx.ui.notify(
            selected
              ? `autoroute: on (${VIRTUAL_MODEL_REF})`
              : `autoroute: on, but ${VIRTUAL_MODEL_REF} could not be selected; pick it in /model`,
            selected ? "info" : "error",
          );
          return;
        }

        case "off":
          state.disabled = true;
          persistState();
          ctx.ui.setStatus(STATUS_KEY, undefined);
          ctx.ui.notify(
            `autoroute: off for this session; ${VIRTUAL_MODEL_REF} keeps the model it is on. Pick another model with /model to leave the router.`,
            "info",
          );
          return;

        case "pin": {
          const model = rest.join(" ") || currentServedModel(ctx);
          if (!model) {
            ctx.ui.notify("usage: /autoroute pin <provider/model>", "info");
            return;
          }
          const problem = checkCommandModel(model, ctx);
          if (problem) {
            ctx.ui.notify(problem, "error");
            return;
          }
          state.pinnedModel = model;
          persistState();
          ctx.ui.notify(`autoroute: pinned to ${model}`, "info");
          return;
        }

        case "unpin":
          state.pinnedModel = null;
          persistState();
          ctx.ui.notify("autoroute: unpinned", "info");
          return;

        case "explain": {
          if (!lastDecision) {
            ctx.ui.notify("autoroute: no decision yet this session", "info");
            return;
          }
          // Re-score the last ask so the per-dimension breakdown reflects the same input
          // the decision was made on.
          const dimensions =
            config && lastDecision.tier
              ? classifyHeuristic(
                  extractTurn("", sessionMessages(ctx), {
                    markerPairs: withSkillBlocks(config.reminderMarkers),
                    contextWindowSize: 0,
                    perTurnChars: config.classifierContextPerTurnChars,
                    includeAssistantTurns: false,
                  }).currentAsk ?? "",
                  config,
                ).dimensions
              : undefined;
          ctx.ui.notify(explain(lastDecision, dimensions), "info");
          return;
        }

        case "log": {
          // The session's decisions, oldest first, as log lines; `/autoroute log 5` for the
          // last five. Read back from the session so it works after a resume too.
          const limit = Number.parseInt(rest[0] ?? "", 10);
          const decisions: RouteDecision[] = [];
          try {
            for (const entry of ctx.sessionManager.getEntries()) {
              if (entry.type !== "custom") continue;
              const custom = entry as { customType?: string; data?: unknown };
              if (custom.customType === DECISION_ENTRY_TYPE && custom.data) decisions.push(custom.data as RouteDecision);
            }
          } catch {
            // Unreadable history: fall through to whatever this process saw.
          }
          if (decisions.length === 0 && lastDecision) decisions.push(lastDecision);
          if (decisions.length === 0) {
            ctx.ui.notify("autoroute: no decisions yet this session", "info");
            return;
          }
          const shown = Number.isFinite(limit) && limit > 0 ? decisions.slice(-limit) : decisions;
          ctx.ui.notify(shown.map(decisionLogLine).join("\n"), "info");
          return;
        }

        case "adaptive": {
          if (!config?.adaptive || !adaptive) {
            ctx.ui.notify("autoroute: adaptive selection is off (set \"adaptive\": true in autorouter.json)", "info");
            return;
          }
          ctx.ui.notify(renderAdaptiveSnapshot(adaptive.snapshot(), config, adaptiveStorePath), "info");
          return;
        }

        default: {
          const lines: string[] = [];
          const why = routingDisabled();
          lines.push(`status:   ${why ? `disabled (${why})` : "on"}`);
          lines.push(
            `model:    ${VIRTUAL_MODEL_REF} ${
              isRouterModel(ctx.model) ? "(selected)" : "(not selected: /autoroute on, or pick it in /model)"
            }`,
          );
          if (config) {
            const llm = config.classifierType === "llm" ? config.classifierLLMConfig : null;
            const decisionModel = config.classifierType === "decision" ? config.decisionClassifierConfig : null;
            lines.push(
              `classifier: ${config.classifierType}${
                llm
                  ? ` (${llm.model}, ${llm.classificationRubric})`
                  : decisionModel
                    ? ` (${decisionModel.model}, timeout ${decisionModel.timeoutMs}ms${
                        decisionBreaker?.isOpen() ? ", circuit open" : ""
                      })`
                    : ""
              }`,
            );
            if (config.adaptive) {
              lines.push(
                `adaptive: on (${config.adaptiveEligible}, quality ${config.adaptiveWeights.quality} / cost ${config.adaptiveWeights.cost}, penalty ${config.tierDistancePenalty})`,
              );
            }
            if (config.semanticKeywordMatching) {
              lines.push(`keywords: semantic (${config.embeddingModel}, threshold ${config.matchThreshold})`);
            }
          }
          if (configSources.length > 0) lines.push(`config:   ${configSources.join(", ")}`);
          for (const error of configErrors) lines.push(`error:    ${error}`);
          for (const warning of configWarnings) lines.push(`warning:  ${warning}`);
          if (lastDecision) {
            lines.push("");
            lines.push(explain(lastDecision));
          }
          ctx.ui.notify(lines.join("\n"), "info");
        }
      }
    },
  });
}
