/**
 * Wiring tests for `src/index.ts`.
 *
 * These exist because pi is not a safety net: a `pi -e` run exits 0 even when an
 * extension's factory throws or the file has a syntax error, with no message on stdout or
 * stderr. A broken extension is silently ignored, so nothing at the CLI level can tell you
 * the wiring works. This does, by importing the real module and driving it with a mock
 * ExtensionAPI.
 */

import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import autorouter from "../src/index.ts";
import { DECISION_ENTRY_TYPE } from "../src/decision.ts";
import type { RouteDecision } from "../src/types.ts";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown> | unknown;

function mockPi() {
  const handlers = new Map<string, Handler[]>();
  /** Set by a test to make setModel raise model_select, the way real pi does. */
  let modelSelectEcho = false;
  let echoCtx: unknown = {};
  const busHandlers = new Map<string, ((payload: unknown) => void)[]>();
  const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
  const flags = new Map<string, unknown>();
  const entries: { type: string; data: unknown }[] = [];
  const setModelCalls: string[] = [];
  const thinkingCalls: string[] = [];

  const pi = {
    registerFlag: vi.fn((name: string, opts: { default?: unknown }) => flags.set(name, opts.default)),
    getFlag: vi.fn((name: string) => flags.get(name)),
    on: vi.fn((event: string, handler: Handler) => {
      handlers.set(event, [...(handlers.get(event) ?? []), handler]);
    }),
    events: {
      on: vi.fn((name: string, handler: (payload: unknown) => void) => {
        busHandlers.set(name, [...(busHandlers.get(name) ?? []), handler]);
      }),
      emit: vi.fn(),
    },
    registerCommand: vi.fn((name: string, opts: { handler: (args: string, ctx: unknown) => Promise<void> }) =>
      commands.set(name, opts),
    ),
    appendEntry: vi.fn((type: string, data: unknown) => entries.push({ type, data })),
    setModel: vi.fn(async (model: { provider: string; id: string }) => {
      setModelCalls.push(`${model.provider}/${model.id}`);
      // Real pi raises model_select from inside setModel. Firing it here rather than from
      // the test body is what makes the echo-suppression path testable at all: at this
      // moment the extension has not yet recorded the decision, which is exactly the
      // window in which a naive guard mistakes our own call for the user's.
      if (modelSelectEcho) {
        for (const handler of handlers.get("model_select") ?? []) {
          await handler({ source: "set", model }, echoCtx);
        }
      }
      return true;
    }),
    setThinkingLevel: vi.fn((level: string) => thinkingCalls.push(level)),
    registerProvider: vi.fn(),
  };

  const fire = async (event: string, payload: unknown, ctx: unknown): Promise<unknown> => {
    let last: unknown;
    for (const handler of handlers.get(event) ?? []) last = await handler(payload, ctx);
    return last;
  };

  const emit = (name: string, payload: unknown): void => {
    for (const handler of busHandlers.get(name) ?? []) handler(payload);
  };

  const enableModelSelectEcho = (ctx: unknown): void => {
    modelSelectEcho = true;
    echoCtx = ctx;
  };

  return {
    pi,
    fire,
    emit,
    commands,
    entries,
    setModelCalls,
    thinkingCalls,
    flags,
    handlers,
    enableModelSelectEcho,
  };
}

const CATALOGUE = [
  { provider: "openai", id: "gpt-mini", cost: { input: 0.15, output: 0.6 }, input: ["text"] },
  { provider: "anthropic", id: "haiku", cost: { input: 0.8, output: 4 }, input: ["text"] },
  { provider: "anthropic", id: "sonnet", cost: { input: 3, output: 15 }, input: ["text"] },
  { provider: "anthropic", id: "opus", cost: { input: 15, output: 75 }, input: ["text"], reasoning: true },
];

function mockCtx(cwd: string, overrides: Record<string, unknown> = {}) {
  return {
    cwd,
    hasUI: true,
    ui: { notify: vi.fn(), setStatus: vi.fn(), confirm: vi.fn(async () => true) },
    sessionManager: { getBranch: () => [], getEntries: () => [], getSessionFile: () => "/tmp/session.jsonl" },
    scopedModels: [],
    modelRegistry: {
      find: (provider: string, modelId: string) => ({ provider, id: modelId }),
      getAvailable: () => CATALOGUE,
      complete: vi.fn(),
    },
    model: undefined,
    ...overrides,
  };
}

const CONFIG = {
  defaultModel: "anthropic/haiku",
  tiers: {
    SIMPLE: "anthropic/haiku",
    MEDIUM: "anthropic/sonnet",
    COMPLEX: "anthropic/sonnet",
    REASONING: { model: "anthropic/opus", thinkingLevel: "high" },
  },
};

describe("extension wiring", () => {
  let dir: string;
  let home: string;
  let originalHome: string | undefined;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "autoroute-cwd-"));
    home = mkdtempSync(join(tmpdir(), "autoroute-home-"));
    mkdirSync(join(dir, ".pi"), { recursive: true });
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify(CONFIG));
    // loadConfig defaults to os.homedir(); point it at an empty dir so a developer's own
    // global config cannot influence the test.
    originalHome = process.env.HOME;
    process.env.HOME = home;
  });

  afterEach(() => {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    rmSync(dir, { recursive: true, force: true });
    rmSync(home, { recursive: true, force: true });
  });

  it("registers its flag, command, and handlers", () => {
    const m = mockPi();
    autorouter(m.pi as never);

    expect(m.pi.registerFlag).toHaveBeenCalledWith("no-autoroute", expect.anything());
    expect(m.commands.has("autoroute")).toBe(true);
    for (const event of ["session_start", "input", "model_select"]) {
      expect(m.handlers.has(event), `expected a handler for ${event}`).toBe(true);
    }
    expect(m.pi.events.on).toHaveBeenCalledWith("autoroute:plan-mode", expect.anything());
  });

  it("routes a prompt end to end and records the decision", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    const result = await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(result).toEqual({ action: "continue" });
    expect(m.setModelCalls).toEqual(["anthropic/haiku"]);

    const decision = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)?.data as RouteDecision;
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.chosenModel).toBe("anthropic/haiku");
    expect(ctx.ui.setStatus).toHaveBeenCalled();
  });

  it("routes a hard prompt to the top tier and applies its thinking level", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire(
      "input",
      { text: "think step by step and analyze this: weigh the options for our schema", source: "interactive" },
      ctx,
    );

    expect(m.setModelCalls).toEqual(["anthropic/opus"]);
    expect(m.thinkingCalls).toEqual(["high"]);
  });

  it("does not route when --no-autoroute is set", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    m.flags.set("no-autoroute", true);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual([]);
  });

  it("does not route its own injected messages", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "extension" }, ctx);

    expect(m.setModelCalls).toEqual([]);
  });

  it("does not switch models mid-stream", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive", streamingBehavior: "steer" }, ctx);

    expect(m.setModelCalls).toEqual([]);
  });

  it("stops routing once the user picks a model by hand", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("model_select", { source: "set", model: { provider: "anthropic", id: "opus" } }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual([]);
  });

  it("keeps routing when setModel raises model_select from inside the routing call", async () => {
    // The ordering that matters: real pi fires model_select *during* setModel, before the
    // extension has recorded the decision. A guard that compares against the last decision
    // sees the previous turn's model, treats our own call as a hand pick, and disables the
    // router after its very first decision.
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);
    m.enableModelSelectEcho(ctx);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    await m.fire("input", { text: "hi again", source: "interactive" }, ctx);
    await m.fire("input", { text: "and again", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/haiku", "anthropic/haiku", "anthropic/haiku"]);
  });

  it("still yields to a genuine hand pick while the echo guard is armed", async () => {
    // The suppression must be scoped to our own call, not a blanket "ignore model_select".
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);
    m.enableModelSelectEcho(ctx);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    await m.fire("model_select", { source: "set", model: { provider: "anthropic", id: "opus" } }, ctx);
    await m.fire("input", { text: "hi again", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/haiku"]);
  });

  it("keeps routing when the model changed because we changed it", async () => {
    // Our own setModel comes back as a model_select event; treating that as a hand pick
    // would disable the router after its first decision.
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    await m.fire("model_select", { source: "set", model: { provider: "anthropic", id: "haiku" } }, ctx);
    await m.fire("input", { text: "hi again", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/haiku", "anthropic/haiku"]);
  });

  it("survives a session with no config at all", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const empty = mkdtempSync(join(tmpdir(), "autoroute-empty-"));
    const ctx = mockCtx(empty);

    await m.fire("session_start", { reason: "startup" }, ctx);
    const result = await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(result).toEqual({ action: "continue" });
    expect(m.setModelCalls).toEqual([]);
    rmSync(empty, { recursive: true, force: true });
  });

  it("survives a malformed config without disabling pi", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), "{ not json");
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    const result = await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(result).toEqual({ action: "continue" });
    expect(ctx.ui.notify).toHaveBeenCalled();
  });

  it("never fails a turn when the registry throws", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir, {
      modelRegistry: {
        find: () => {
          throw new Error("registry exploded");
        },
        complete: vi.fn(),
      },
    });

    await m.fire("session_start", { reason: "startup" }, ctx);
    const result = await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(result).toEqual({ action: "continue" });
  });

  it("takes the plan-mode floor from the shared event bus", async () => {
    writeFileSync(
      join(dir, ".pi", "autorouter.json"),
      JSON.stringify({ ...CONFIG, planMode: { minTier: "COMPLEX" } }),
    );
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    m.emit("autoroute:plan-mode", { active: true });
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/sonnet"]);
    const decision = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)?.data as RouteDecision;
    expect(decision.planFloored).toBe(true);
  });

  it("forces a model for exactly one prompt, then reverts", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/opus", ctx);

    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    // First prompt overridden; second classified normally.
    expect(m.setModelCalls).toEqual(["anthropic/opus", "anthropic/haiku"]);

    const decisions = m.entries
      .filter((e) => e.type === DECISION_ENTRY_TYPE)
      .map((e) => e.data as RouteDecision);
    expect(decisions[0]?.cause).toBe("one_shot_override");
    expect(decisions[1]?.cause).toBe("heuristic_scorer");
  });

  it("rejects an unknown model at command time rather than at the next prompt", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir, {
      modelRegistry: { find: () => undefined, getAvailable: () => CATALOGUE, complete: vi.fn() },
    });

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/nope", ctx);

    expect(ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("not a model pi knows"), "error");
  });

  it("clears a pending override with a bare `next`", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/opus", ctx);
    await m.commands.get("autoroute")!.handler("next", ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/haiku"]);
  });

  it("honours an override even while routing is switched off", async () => {
    // An explicit one-off instruction outranks "off"; "off" still holds from the next
    // prompt onwards.
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("off", ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/opus", ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/opus"]);
  });

  it("does not spend the override on a mid-stream steer", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/opus", ctx);
    await m.fire("input", { text: "hi", source: "interactive", streamingBehavior: "steer" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/opus"]);
  });

  it("writes a config from the available models and routes with it immediately", async () => {
    const fresh = mkdtempSync(join(tmpdir(), "autoroute-init-"));
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(fresh);

    await m.fire("session_start", { reason: "startup" }, ctx);
    // No config yet, so nothing routes.
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    expect(m.setModelCalls).toEqual([]);

    await m.commands.get("autoroute")!.handler("init project", ctx);

    const written = JSON.parse(readFileSync(join(fresh, ".pi", "autorouter.json"), "utf8"));
    expect(written.defaultModel).toBe("openai/gpt-mini");
    expect(written.tiers.REASONING).toEqual({ model: "anthropic/opus", thinkingLevel: "high" });

    // The new config takes effect without a restart.
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    expect(m.setModelCalls).toEqual(["openai/gpt-mini"]);
    rmSync(fresh, { recursive: true, force: true });
  });

  it("does not overwrite an existing config when the user declines", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir, {
      ui: { notify: vi.fn(), setStatus: vi.fn(), confirm: vi.fn(async () => false) },
    });

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("init project", ctx);

    const onDisk = JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"));
    expect(onDisk).toEqual(CONFIG);
    expect(ctx.ui.confirm).toHaveBeenCalled();
  });

  it("refuses to overwrite with no UI to confirm through", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir, { hasUI: false });

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("init project", ctx);

    expect(JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"))).toEqual(CONFIG);
  });

  it("reports a provider filter that matches nothing instead of writing", async () => {
    const fresh = mkdtempSync(join(tmpdir(), "autoroute-init-"));
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(fresh);

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("init nosuchprovider project", ctx);

    expect(existsSync(join(fresh, ".pi", "autorouter.json"))).toBe(false);
    expect(ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("no usable models"), "error");
    rmSync(fresh, { recursive: true, force: true });
  });

  it("prefers session-scoped models over the whole catalogue", async () => {
    // `--models` / `enabledModels` is the set the user intends to use.
    const fresh = mkdtempSync(join(tmpdir(), "autoroute-init-"));
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(fresh, {
      scopedModels: [{ model: CATALOGUE[1] }, { model: CATALOGUE[2] }],
    });

    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("init project", ctx);

    const written = JSON.parse(readFileSync(join(fresh, ".pi", "autorouter.json"), "utf8"));
    expect(written.defaultModel).toBe("anthropic/haiku");
    expect(JSON.stringify(written)).not.toContain("gpt-mini");
    rmSync(fresh, { recursive: true, force: true });
  });

  it("stops flooring once plan mode ends", async () => {
    writeFileSync(
      join(dir, ".pi", "autorouter.json"),
      JSON.stringify({ ...CONFIG, planMode: { minTier: "COMPLEX" } }),
    );
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);

    await m.fire("session_start", { reason: "startup" }, ctx);
    m.emit("autoroute:plan-mode", { active: true });
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    m.emit("autoroute:plan-mode", { active: false });
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);

    expect(m.setModelCalls).toEqual(["anthropic/sonnet", "anthropic/haiku"]);
  });
});

describe("adaptive wiring", () => {
  let dir: string;
  let home: string;
  let originalHome: string | undefined;

  const ADAPTIVE_CONFIG = {
    ...CONFIG,
    adaptive: true,
    tiers: { SIMPLE: ["anthropic/haiku", "openai/gpt-mini"], MEDIUM: "anthropic/sonnet", COMPLEX: "anthropic/sonnet", REASONING: "anthropic/opus" },
  };

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "autoroute-cwd-"));
    home = mkdtempSync(join(tmpdir(), "autoroute-home-"));
    mkdirSync(join(dir, ".pi"), { recursive: true });
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify(ADAPTIVE_CONFIG));
    originalHome = process.env.HOME;
    process.env.HOME = home;
  });

  afterEach(() => {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    rmSync(dir, { recursive: true, force: true });
    rmSync(home, { recursive: true, force: true });
  });

  function adaptiveCtx() {
    return mockCtx(dir, {
      sessionManager: {
        getBranch: () => [],
        getEntries: () => [],
        getSessionFile: () => "/tmp/session.jsonl",
        getSessionId: () => "session-1",
      },
      modelRegistry: {
        find: (provider: string, modelId: string) => CATALOGUE.find((m) => m.provider === provider && m.id === modelId),
        getAvailable: () => CATALOGUE,
        complete: vi.fn(),
      },
    });
  }

  it("picks from the pool, learns from the run, and persists what it learned", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = adaptiveCtx();
    await m.fire("session_start", { reason: "startup" }, ctx);

    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    const decision = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)?.data as RouteDecision;
    expect(decision.adaptive?.phase).toBe("cold_start");
    expect(["anthropic/haiku", "openai/gpt-mini"]).toContain(decision.chosenModel);
    expect(m.setModelCalls).toEqual([decision.chosenModel]);

    // A run whose tool result errored is a failure signal against the chosen model.
    await m.fire(
      "agent_end",
      {
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: [{ type: "toolCall", name: "bash", arguments: { command: "x" } }], stopReason: "toolUse" },
          { role: "toolResult", content: [{ type: "text", text: "command not found" }], isError: true },
          { role: "assistant", content: [{ type: "text", text: "that failed" }], stopReason: "stop" },
        ],
      },
      ctx,
    );

    const storePath = join(home, ".pi", "agent", "autorouter-adaptive.json");
    expect(existsSync(storePath)).toBe(true);
    const persisted = JSON.parse(readFileSync(storePath, "utf8")) as { cells: { model: string; requestType: string; beta: number }[] };
    const cell = persisted.cells.find((c) => c.model === decision.chosenModel && c.requestType === "general")!;
    expect(cell.beta).toBeCloseTo(6); // prior 5 + 1 failure

    // A fresh session overlays the persisted posterior on its priors.
    const again = mockPi();
    autorouter(again.pi as never);
    const ctx2 = adaptiveCtx();
    await again.fire("session_start", { reason: "startup" }, ctx2);
    await again.commands.get("autoroute")!.handler("adaptive", ctx2);
    const shown = (ctx2.ui.notify as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as string;
    expect(shown).toContain("general:");
    expect(shown).toContain("samples   1");
  });

  it("reports adaptive as off when the config does not enable it", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify(CONFIG));
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = adaptiveCtx();
    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.commands.get("autoroute")!.handler("adaptive", ctx);
    expect(ctx.ui.notify).toHaveBeenCalledWith(expect.stringMatching(/adaptive selection is off/), "info");
    await m.fire("agent_end", { messages: [] }, ctx);
    expect(existsSync(join(home, ".pi", "agent", "autorouter-adaptive.json"))).toBe(false);
  });
});

describe("semantic keyword matching wiring", () => {
  let dir: string;
  let home: string;
  let originalHome: string | undefined;
  let originalFetch: typeof fetch;

  const SEMANTIC_CONFIG = {
    ...CONFIG,
    keywordTierRules: [{ keywords: ["kubernetes deployment"], tier: "REASONING" }],
    semanticKeywordMatching: true,
    embeddingModel: "voyage/voyage-3-5",
    embeddingEndpoint: { apiKeyEnv: "TEST_EMBED_KEY" },
  };

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "autoroute-cwd-"));
    home = mkdtempSync(join(tmpdir(), "autoroute-home-"));
    mkdirSync(join(dir, ".pi"), { recursive: true });
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify(SEMANTIC_CONFIG));
    originalHome = process.env.HOME;
    process.env.HOME = home;
    process.env.TEST_EMBED_KEY = "k";
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    delete process.env.TEST_EMBED_KEY;
    globalThis.fetch = originalFetch;
    rmSync(dir, { recursive: true, force: true });
    rmSync(home, { recursive: true, force: true });
  });

  it("embeds the rule keywords and the prompt, and routes on similarity", async () => {
    const calls: { url: string; body: { model: string; input: string[] } }[] = [];
    globalThis.fetch = (async (url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { model: string; input: string[] };
      calls.push({ url, body });
      // Every input maps to the same direction: the prompt is a perfect match.
      return new Response(JSON.stringify({ data: body.input.map((_, index) => ({ index, embedding: [1, 0] })) }), {
        status: 200,
      });
    }) as unknown as typeof fetch;

    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);
    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "help me roll out my k8s cluster", source: "interactive" }, ctx);

    expect(calls.map((c) => c.url)).toEqual([
      "https://api.voyageai.com/v1/embeddings",
      "https://api.voyageai.com/v1/embeddings",
    ]);
    expect(calls[0]?.body).toEqual({ model: "voyage-3-5", input: ["kubernetes deployment"] });
    expect(calls[1]?.body.input).toEqual(["help me roll out my k8s cluster"]);
    const decision = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)?.data as RouteDecision;
    expect(decision.cause).toBe("semantic_keyword_match");
    expect(decision.chosenModel).toBe("anthropic/opus");
    expect(m.setModelCalls).toEqual(["anthropic/opus"]);

    await m.commands.get("autoroute")!.handler("", ctx);
    expect(ctx.ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("keywords: semantic (voyage/voyage-3-5"), "info");
  });

  it("still routes when the embeddings endpoint is down", async () => {
    globalThis.fetch = (async () => new Response("down", { status: 503 })) as unknown as typeof fetch;
    const m = mockPi();
    autorouter(m.pi as never);
    const ctx = mockCtx(dir);
    await m.fire("session_start", { reason: "startup" }, ctx);
    await m.fire("input", { text: "hi", source: "interactive" }, ctx);
    const decision = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)?.data as RouteDecision;
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals).toContainEqual(expect.stringMatching(/semantic_keyword_match_failed .*503/));
    expect(m.setModelCalls).toEqual(["anthropic/haiku"]);
  });
});
