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

  return { pi, fire, emit, commands, entries, setModelCalls, thinkingCalls, flags, handlers };
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
