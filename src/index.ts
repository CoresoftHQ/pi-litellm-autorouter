/**
 * pi-litellm-autorouter — LiteLLM's Auto Router v2, running inside pi.
 *
 * Wiring only: the routing decision itself lives in `router.ts`, which has no pi imports
 * so it can be tested without an agent.
 *
 * The seam is the `input` event. It fires after extension commands are dispatched and
 * before skill/template expansion and `before_agent_start`, so it is the last point at
 * which the model for the turn about to run can still be changed.
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type RouterConfig, loadConfig } from "./config.ts";
import { classifyHeuristic } from "./classify/heuristic.ts";
import { splitModelRef } from "./classify/llm.ts";
import {
  type AutorouteState,
  DECISION_ENTRY_TYPE,
  STATE_ENTRY_TYPE,
  explain,
  statusLine,
} from "./decision.ts";
import { extractTurn, type SimpleMessage } from "./extract.ts";
import { type CandidateModel, buildInitialConfig, renderInitPreview } from "./init.ts";
import { type SessionPin, route } from "./router.ts";
import type { RouteDecision } from "./types.ts";

const STATUS_KEY = "autoroute";

export default function autorouter(pi: ExtensionAPI): void {
  let config: RouterConfig | null = null;
  let configWarnings: string[] = [];
  let configErrors: string[] = [];
  let configSources: string[] = [];

  let state: AutorouteState = {};
  let pin: SessionPin | null = null;
  let lastDecision: RouteDecision | null = null;
  let planModeActive = false;
  /** Model refs for command completion; the completion callback gets no context. */
  let availableRefs: string[] = [];
  /** True while our own setModel calls are in flight, so their model_select echoes are
   *  not mistaken for the user picking a model by hand. */
  let applyingOwnModel = false;

  pi.registerFlag("no-autoroute", {
    description: "Start with automatic model routing disabled",
    type: "boolean",
    default: false,
  });

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

  const restoreState = (ctx: ExtensionContext): void => {
    state = {};
    pin = null;
    lastDecision = null;
    try {
      for (const entry of ctx.sessionManager.getEntries()) {
        if (entry.type !== "custom") continue;
        const custom = entry as { customType?: string; data?: unknown };
        if (custom.customType === STATE_ENTRY_TYPE && custom.data) {
          state = { ...state, ...(custom.data as AutorouteState) };
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

  const routingDisabled = (): string | null => {
    if (!config?.enabled) return "no usable config";
    if (pi.getFlag("no-autoroute") === true) return "--no-autoroute";
    if (state.disabled) return "/autoroute off";
    if (state.pinnedModel) return `pinned to ${state.pinnedModel}`;
    return null;
  };

  pi.on("session_start", async (_event, ctx) => {
    const loaded = loadConfig(ctx.cwd);
    config = loaded.config;
    configWarnings = loaded.warnings;
    configErrors = loaded.errors;
    configSources = loaded.sources;
    restoreState(ctx);
    try {
      availableRefs = ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`);
    } catch {
      availableRefs = [];
    }

    if (configErrors.length > 0 && ctx.hasUI) {
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

  // A model the user chose by hand outranks the router until they clear it.
  pi.on("model_select", async (event) => {
    if (event.source !== "set") return;
    // Our own pi.setModel() also raises model_select. Mistaking that echo for a hand pick
    // would have the router disable itself the moment it made its first decision, so the
    // echo is suppressed by a flag held across the routing call rather than by comparing
    // against lastDecision — which is not assigned until route() has already returned, and
    // so is still the *previous* turn's model while setModel is running.
    if (applyingOwnModel) return;
    const model = event.model;
    if (!model) return;
    const ref = `${model.provider}/${model.id}`;
    // Belt and braces for an echo delivered after route() returned, when lastDecision is
    // current.
    if (lastDecision?.chosenModel === ref) return;
    state.pinnedModel = ref;
    persistState();
  });

  pi.on("input", async (event, ctx) => {
    // Messages this extension injected are not user asks; routing them would classify our
    // own text.
    if (event.source === "extension") return { action: "continue" as const };
    // Mid-stream steers and queued follow-ups arrive while a run is in flight, where
    // switching models would break the conversation's provider-specific message shapes.
    if (event.streamingBehavior) return { action: "continue" as const };

    // A one-shot override is an explicit instruction for this prompt, so it is honoured
    // even when routing is otherwise off or pinned. It is checked before those, and always
    // cleared afterwards, so "off" still means off from the next prompt onwards.
    const oneShot = state.nextModel ?? null;

    const disabled = routingDisabled();
    if ((disabled && !oneShot) || !config) return { action: "continue" as const };

    try {
      const turn = extractTurn(event.text, sessionMessages(ctx), {
        markerPairs: config.reminderMarkers,
        contextWindowSize: config.classifierContextWindowSize,
        perTurnChars: config.classifierContextPerTurnChars,
        includeAssistantTurns: config.classifierContextIncludeAssistantTurns,
      });

      applyingOwnModel = true;
      const result = await route({
        turn,
        config,
        api: {
          find: (provider, modelId) => ctx.modelRegistry.find(provider, modelId),
          setModel: (model) => pi.setModel(model),
          setThinkingLevel: (level) => pi.setThinkingLevel(level as never),
        },
        registry: ctx.modelRegistry as never,
        planModeActive,
        pin,
        oneShot,
        callerSystemPrompt: undefined,
      });

      applyingOwnModel = false;
      lastDecision = result.decision;
      if (result.pinToWrite) pin = result.pinToWrite;
      if (result.consumedOneShot) {
        state.nextModel = null;
        persistState();
        if (ctx.hasUI && !result.decision.chosenModel) {
          ctx.ui.notify(`autoroute: ${result.decision.fellBackBecause ?? "one-shot override failed"}`, "error");
        }
      }

      pi.appendEntry<RouteDecision>(DECISION_ENTRY_TYPE, result.decision);
      if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, statusLine(result.decision));
    } catch (err) {
      // Choosing a model is a routing decision; no failure in it may fail the user's turn.
      applyingOwnModel = false;
      if (ctx.hasUI) {
        ctx.ui.setStatus(STATUS_KEY, `autoroute error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    return { action: "continue" as const };
  });

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
    // to use, so it wins over the full catalogue when it is set.
    const scoped = ctx.scopedModels ?? [];
    const source: CandidateModel[] =
      scoped.length > 0
        ? scoped.map((entry) => entry.model as unknown as CandidateModel)
        : (ctx.modelRegistry.getAvailable() as unknown as CandidateModel[]);

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

    // Load it straight away so the session routes without a restart.
    const loaded = loadConfig(ctx.cwd);
    config = loaded.config;
    configWarnings = loaded.warnings;
    configErrors = loaded.errors;
    configSources = loaded.sources;

    ctx.ui.notify(`${preview}\n\n  written. routing is ${config.enabled ? "active" : "still disabled"}.`, "info");
  };

  pi.registerCommand("autoroute", {
    description: "Show or control automatic model routing",
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
      const verbs = ["init", "next", "on", "off", "pin", "unpin", "explain", "status"];
      const items = verbs.filter((v) => v.startsWith(prefix)).map((v) => ({ value: v, label: v }));
      return items.length > 0 ? items : null;
    },
    handler: async (args, ctx) => {
      const [verb = "status", ...rest] = args.trim().split(/\s+/).filter(Boolean);

      switch (verb) {
        case "init":
          await runInit(rest, ctx);
          return;

        case "next": {
          if (rest.length === 0) {
            const had = state.nextModel;
            state.nextModel = null;
            persistState();
            ctx.ui.notify(had ? `autoroute: cleared one-shot override (${had})` : "autoroute: no override set", "info");
            return;
          }
          const target = rest.join(" ");
          const ref = splitModelRef(target);
          // Validate now rather than at the next prompt: a typo should fail while the user
          // is still looking at the command, not silently route the prompt they cared about.
          if (!ref || !ctx.modelRegistry.find(ref.provider, ref.modelId)) {
            ctx.ui.notify(
              `autoroute: "${target}" is not a model pi knows. Use provider/model-id, e.g. anthropic/claude-opus-5.`,
              "error",
            );
            return;
          }
          state.nextModel = target;
          persistState();
          ctx.ui.notify(`autoroute: next prompt only → ${target}`, "info");
          return;
        }

        case "on":
          state.disabled = false;
          state.pinnedModel = null;
          state.nextModel = null;
          persistState();
          ctx.ui.notify("autoroute: on", "info");
          return;

        case "off":
          state.disabled = true;
          persistState();
          ctx.ui.setStatus(STATUS_KEY, undefined);
          ctx.ui.notify("autoroute: off for this session", "info");
          return;

        case "pin": {
          const target = rest.join(" ") || ctx.model ? `${ctx.model?.provider}/${ctx.model?.id}` : null;
          state.pinnedModel = rest.join(" ") || target;
          persistState();
          ctx.ui.notify(`autoroute: pinned to ${state.pinnedModel}`, "info");
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
                    markerPairs: config.reminderMarkers,
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

        default: {
          const lines: string[] = [];
          const why = routingDisabled();
          lines.push(`status:   ${why ? `disabled (${why})` : "on"}`);
          if (config) {
            lines.push(
              `classifier: ${config.classifierType}${
                config.classifierLLMConfig
                  ? ` (${config.classifierLLMConfig.model}, ${config.classifierLLMConfig.classificationRubric})`
                  : ""
              }`,
            );
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
