/**
 * Wiring tests for `src/index.ts`.
 *
 * These exist because pi is not a safety net: a `pi -e` run exits 0 even when an
 * extension's factory throws or the file has a syntax error, with no message on stdout or
 * stderr. A broken extension is silently ignored, so nothing at the CLI level can tell you
 * the wiring works. This does, by importing the real module and driving it with a mock
 * ExtensionAPI — including pi's side of the virtual model: a prompt lands in the session,
 * then pi asks the router for a physical model, then the response names that model.
 */

import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import autorouter, { type BranchState, VIRTUAL_MODEL_ID, VIRTUAL_PROVIDER } from "../src/index.ts";
import { DECISION_ENTRY_TYPE } from "../src/decision.ts";
import type { RouteDecision } from "../src/types.ts";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown> | unknown;

interface TestModel {
  provider: string;
  id: string;
  api?: string;
  cost?: { input: number; output: number };
  input?: string[];
  reasoning?: boolean;
}

interface RouteRequest {
  model: TestModel;
  thinkingLevel: string;
  reason: "user" | "continuation" | "retry" | "direct";
  previous?: { model: TestModel; thinkingLevel?: string };
  failed?: { model: TestModel; thinkingLevel?: string; message: unknown };
  state?: BranchState;
  messages: { role: string; content: unknown }[];
  signal?: AbortSignal;
}

interface Routed {
  model: TestModel;
  thinkingLevel: string;
  state?: BranchState;
}

interface VirtualModelDefinition {
  provider: string;
  id: string;
  name: string;
  thinkingLevels?: readonly string[];
  route(request: RouteRequest, ctx: unknown): Promise<Routed> | Routed;
}

const VIRTUAL: TestModel = {
  provider: VIRTUAL_PROVIDER,
  id: VIRTUAL_MODEL_ID,
  api: "pi-virtual",
  cost: { input: 0, output: 0 },
  input: ["text", "image"],
};

const CATALOGUE: TestModel[] = [
  { provider: "openai", id: "gpt-mini", cost: { input: 0.15, output: 0.6 }, input: ["text"] },
  { provider: "anthropic", id: "haiku", cost: { input: 0.8, output: 4 }, input: ["text"] },
  { provider: "anthropic", id: "sonnet", cost: { input: 3, output: 15 }, input: ["text"] },
  { provider: "anthropic", id: "opus", cost: { input: 15, output: 75 }, input: ["text"], reasoning: true },
];

const THEME = { fg: (_color: string, text: string) => `<${text}>` };

const CONFIG = {
  defaultModel: "anthropic/haiku",
  tiers: {
    SIMPLE: "anthropic/haiku",
    MEDIUM: "anthropic/sonnet",
    COMPLEX: "anthropic/sonnet",
    REASONING: { model: "anthropic/opus", thinkingLevel: "high" },
  },
};

const HARD_PROMPT = "think step by step and analyze this: weigh the options for our schema";

const refOf = (model: { provider: string; id: string }) => `${model.provider}/${model.id}`;

function mockPi(options: { virtualModels?: boolean } = {}) {
  const handlers = new Map<string, Handler[]>();
  const busHandlers = new Map<string, ((payload: unknown) => void)[]>();
  const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
  const flags = new Map<string, unknown>();
  const entries: { type: string; data: unknown }[] = [];
  const entryRenderers = new Map<string, (entry: unknown, opts: { expanded: boolean }, theme: unknown) => unknown>();
  const virtualModels: VirtualModelDefinition[] = [];
  const setModelCalls: string[] = [];

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
    registerEntryRenderer: vi.fn(
      (type: string, renderer: (entry: unknown, opts: { expanded: boolean }, theme: unknown) => unknown) =>
        entryRenderers.set(type, renderer),
    ),
    // Only ever used to select the virtual model; routing never calls it.
    setModel: vi.fn(async (model: TestModel) => {
      setModelCalls.push(refOf(model));
      return true;
    }),
    setThinkingLevel: vi.fn(),
    registerVirtualModel: options.virtualModels === false
      ? undefined
      : vi.fn((definition: VirtualModelDefinition) => virtualModels.push(definition)),
  };

  const fire = async (event: string, payload: unknown, ctx: unknown): Promise<unknown> => {
    let last: unknown;
    for (const handler of handlers.get(event) ?? []) last = await handler(payload, ctx);
    return last;
  };

  const emit = (name: string, payload: unknown): void => {
    for (const handler of busHandlers.get(name) ?? []) handler(payload);
  };

  const decisions = (): RouteDecision[] =>
    entries.filter((e) => e.type === DECISION_ENTRY_TYPE).map((e) => e.data as RouteDecision);

  return { pi, fire, emit, commands, entries, entryRenderers, virtualModels, setModelCalls, flags, handlers, decisions };
}

type MockPi = ReturnType<typeof mockPi>;

/**
 * A session with `autorouter/tiered` selected, driven the way pi drives it: the prompt is
 * persisted, then the router is asked for a model, then the response is persisted under
 * the model that answered. Router state is carried between requests like pi's
 * `pi.virtual-model-state` entries.
 */
function harness(m: MockPi, cwd: string, overrides: Record<string, unknown> = {}) {
  const branch: { type: string; message?: Record<string, unknown>; [key: string]: unknown }[] = [];
  const h = {
    branch,
    state: undefined as BranchState | undefined,
    ctx: {
      cwd,
      hasUI: true,
      ui: { notify: vi.fn(), setStatus: vi.fn(), confirm: vi.fn(async () => true) },
      sessionManager: {
        getBranch: () => branch,
        getEntries: () => [...branch, ...m.entries.map((e) => ({ type: "custom", customType: e.type, data: e.data }))],
        getSessionFile: () => "/tmp/session.jsonl",
        getSessionId: () => "session-1",
      },
      scopedModels: [] as { model: TestModel }[],
      modelRegistry: {
        find: (provider: string, id: string): TestModel | undefined =>
          provider === VIRTUAL.provider && id === VIRTUAL.id
            ? VIRTUAL
            : (CATALOGUE.find((model) => model.provider === provider && model.id === id) ?? { provider, id }),
        getAvailable: () => [...CATALOGUE, VIRTUAL],
        hasConfiguredAuth: (_model: TestModel) => true,
        complete: vi.fn(),
      } as Record<string, unknown>,
      model: VIRTUAL as TestModel | undefined,
      ...overrides,
    },

    get router(): VirtualModelDefinition {
      const router = m.virtualModels.at(-1);
      if (!router) throw new Error("no virtual model registered");
      return router;
    },

    /** The newest response a physical model gave, as pi reports it in `previous`. */
    previous(): RouteRequest["previous"] {
      for (let i = branch.length - 1; i >= 0; i--) {
        const message = branch[i]?.message;
        if (message?.role !== "assistant" || message.stopReason === "error") continue;
        const find = h.ctx.modelRegistry.find as (p: string, id: string) => TestModel;
        return {
          model: find(message.provider as string, message.model as string),
          thinkingLevel: message.thinkingLevel as string,
        };
      }
      return undefined;
    },

    async request(
      reason: RouteRequest["reason"],
      options: { thinkingLevel?: string; failed?: RouteRequest["failed"] } = {},
    ): Promise<Routed> {
      const previous = h.previous();
      const routed = await h.router.route(
        {
          model: VIRTUAL,
          thinkingLevel: options.thinkingLevel ?? "medium",
          reason,
          ...(previous ? { previous } : {}),
          ...(options.failed ? { failed: options.failed } : {}),
          ...(reason !== "direct" && h.state ? { state: h.state } : {}),
          messages: branch.flatMap((entry) => (entry.type === "message" ? [entry.message as never] : [])),
        },
        h.ctx,
      );
      if (reason !== "direct" && routed.state !== undefined) h.state = routed.state;
      return routed;
    },

    /** An assistant response from `model`, as pi persists it. */
    respond(model: string, content: unknown[] = [{ type: "text", text: "done" }], thinkingLevel = "medium") {
      const [provider, ...id] = model.split("/");
      const message = { role: "assistant", content, provider, model: id.join("/"), api: "test", thinkingLevel, stopReason: "stop" };
      branch.push({ type: "message", message });
      return message;
    },

    /** One prompt, start to finish: input event, routing, response, `agent_end`. */
    async prompt(
      text: string,
      options: { source?: string; thinkingLevel?: string; reply?: unknown[]; keepRunning?: boolean } = {},
    ): Promise<Routed> {
      await m.fire("input", { text, source: options.source ?? "interactive" }, h.ctx);
      const user = { role: "user", content: text };
      branch.push({ type: "message", message: user });
      let routed: Routed;
      try {
        routed = await h.request("user", options);
      } catch (err) {
        // A routing failure ends the run with an error response that names the virtual model.
        const message = { role: "assistant", content: [], provider: VIRTUAL.provider, model: VIRTUAL.id, api: "pi-virtual", stopReason: "error" };
        branch.push({ type: "message", message });
        await m.fire("agent_end", { messages: [user, message] }, h.ctx);
        throw err;
      }
      const response = h.respond(refOf(routed.model), options.reply, routed.thinkingLevel);
      if (!options.keepRunning) await m.fire("agent_end", { messages: [user, response] }, h.ctx);
      return routed;
    },

    async start() {
      await m.fire("session_start", { reason: "startup" }, h.ctx);
    },
  };
  return h;
}

let home: string;
let originalHome: string | undefined;
let originalProfile: string | undefined;

/** loadConfig and the adaptive store default to os.homedir(): point it at an empty dir so
 *  a developer's own global config (and learned posteriors) cannot leak in, or be
 *  overwritten. os.homedir() reads HOME on POSIX and USERPROFILE on Windows. */
function isolateHome(): void {
  home = mkdtempSync(join(tmpdir(), "autoroute-home-"));
  originalHome = process.env.HOME;
  originalProfile = process.env.USERPROFILE;
  process.env.HOME = home;
  process.env.USERPROFILE = home;
}

function restoreHome(): void {
  if (originalHome === undefined) delete process.env.HOME;
  else process.env.HOME = originalHome;
  if (originalProfile === undefined) delete process.env.USERPROFILE;
  else process.env.USERPROFILE = originalProfile;
  for (const dir of [home, ...created.splice(0)]) rmSync(dir, { recursive: true, force: true });
}

/** Every project directory a test made, removed after it even when the test fails. */
const created: string[] = [];

/** A project directory with `config` as its autorouter.json; null for none at all. */
function projectDir(config: unknown = CONFIG): string {
  const dir = mkdtempSync(join(tmpdir(), "autoroute-cwd-"));
  created.push(dir);
  mkdirSync(join(dir, ".pi"), { recursive: true });
  if (config !== null) writeFileSync(join(dir, ".pi", "autorouter.json"), typeof config === "string" ? config : JSON.stringify(config));
  return dir;
}

describe("extension wiring", () => {
  let dir: string;

  beforeEach(() => {
    isolateHome();
    dir = projectDir();
  });

  afterEach(() => {
    restoreHome();
  });

  it("registers the virtual model, its flag, command, and handlers", () => {
    const m = mockPi();
    autorouter(m.pi as never);

    expect(m.virtualModels).toHaveLength(1);
    const [router] = m.virtualModels;
    expect(router?.provider).toBe("autorouter");
    expect(router?.id).toBe("tiered");
    expect(router?.thinkingLevels).toEqual(["off", "minimal", "low", "medium", "high", "xhigh", "max"]);
    expect(m.pi.registerFlag).toHaveBeenCalledWith("no-autoroute", expect.anything());
    expect(m.commands.has("autoroute")).toBe(true);
    for (const event of ["session_start", "input", "model_select", "agent_end"]) {
      expect(m.handlers.has(event), `expected a handler for ${event}`).toBe(true);
    }
    expect(m.pi.events.on).toHaveBeenCalledWith("autoroute:plan-mode", expect.anything());
  });

  it("says so, rather than failing silently, on a pi without virtual models", async () => {
    const m = mockPi({ virtualModels: false });
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("virtual models"), "error");
  });

  it("routes a prompt end to end and records the decision", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);

    await h.start();
    const routed = await h.prompt("hi");

    expect(refOf(routed.model)).toBe("anthropic/haiku");
    // Routing is a route, never a model switch: the selection stays autorouter/tiered.
    expect(m.setModelCalls).toEqual([]);
    const decision = m.decisions()[0]!;
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.chosenModel).toBe("anthropic/haiku");
    expect(routed.state).toEqual({ tier: "SIMPLE", model: "anthropic/haiku", thinkingLevel: "medium", pin: null });
    expect(h.ctx.ui.setStatus).toHaveBeenCalledWith("autoroute", "SIMPLE · haiku");
  });

  it("passes the selected thinking level through when the tier sets none", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    expect((await h.prompt("hi", { thinkingLevel: "low" })).thinkingLevel).toBe("low");
  });

  it("routes a hard prompt to the top tier with its own thinking level", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    const routed = await h.prompt(HARD_PROMPT, { thinkingLevel: "low" });

    expect(refOf(routed.model)).toBe("anthropic/opus");
    expect(routed.thinkingLevel).toBe("high");
  });

  it("keeps tool-call continuations and retries on the model serving the turn", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await h.prompt(HARD_PROMPT, { keepRunning: true });
    const continuation = await h.request("continuation");
    expect(refOf(continuation.model)).toBe("anthropic/opus");
    expect(continuation.thinkingLevel).toBe("high");
    expect(continuation.state).toBeUndefined();

    const opus = CATALOGUE[3]!;
    const retry = await h.request("retry", { failed: { model: opus, thinkingLevel: "high", message: {} } });
    expect(refOf(retry.model)).toBe("anthropic/opus");

    // Neither request was a prompt, so neither was classified or logged.
    expect(m.decisions()).toHaveLength(1);
  });

  it("keeps a steer that arrives mid-run on the run's model", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await h.prompt("hi", { keepRunning: true });
    // A steer is a user message too, so pi routes it with reason "user".
    h.branch.push({ type: "message", message: { role: "user", content: HARD_PROMPT } });
    const steer = await h.request("user");
    expect(refOf(steer.model)).toBe("anthropic/haiku");
    expect(m.decisions()).toHaveLength(1);

    // Once the run is over, the next prompt is classified on its own merits.
    await m.fire("agent_end", { messages: [] }, h.ctx);
    expect(refOf((await h.prompt(HARD_PROMPT)).model)).toBe("anthropic/opus");
    expect(m.decisions()).toHaveLength(2);
  });

  it("sends requests outside the agent loop to the serving model, else to defaultModel", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    // A compaction summary before anything has answered.
    expect(refOf((await h.request("direct")).model)).toBe("anthropic/haiku");

    await h.prompt(HARD_PROMPT);
    expect(refOf((await h.request("direct")).model)).toBe("anthropic/opus");
    expect(m.decisions()).toHaveLength(1);
  });

  it("keeps the serving model when no configured model has credentials", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.respond("anthropic/sonnet");
    h.ctx.modelRegistry.hasConfiguredAuth = (model: TestModel) => model.id === "sonnet";

    const routed = await h.prompt("hi");

    expect(refOf(routed.model)).toBe("anthropic/sonnet");
    const decision = m.decisions()[0]!;
    expect(decision.chosenModel).toBeNull();
    expect(decision.fellBackBecause).toContain("has no available credentials");
    expect(h.ctx.ui.setStatus).toHaveBeenCalledWith("autoroute", "autoroute error: anthropic/haiku unavailable");
    expect(h.ctx.ui.notify).toHaveBeenCalledWith(
      "autoroute: SIMPLE tier could not use anthropic/haiku (no available credentials). Keeping current model (anthropic/sonnet). Run /autoroute explain for all attempted models.",
      "error",
    );
  });

  it("fails the request with an actionable error when nothing is usable", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.ctx.modelRegistry.hasConfiguredAuth = () => false;

    await expect(h.prompt("hi")).rejects.toThrow(
      "autoroute: SIMPLE tier could not use anthropic/haiku (no available credentials). No configured model is usable.",
    );
    expect(m.decisions()).toHaveLength(1);
  });

  it("does not promote a failed chain past its tier", async () => {
    // SIMPLE and defaultModel have no key, but MEDIUM does. Holding is fine; quietly
    // paying for a bigger model the classifier did not ask for is not.
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.ctx.modelRegistry.hasConfiguredAuth = (model: TestModel) => model.id !== "haiku";

    await expect(h.prompt("hi")).rejects.toThrow(/No configured model is usable/);
  });

  it("holds the model when the turn only answers an open question", async () => {
    // The full path: a hard prompt puts the session on opus, the assistant asks a
    // question, and "ok" arrives. Without the hold, "ok" scores SIMPLE and the answer is
    // handed to haiku — the model the user was mid-task with is gone.
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    const routed = await h.prompt(HARD_PROMPT, {
      reply: [
        {
          type: "toolCall",
          name: "ask_user_question",
          arguments: { questions: [{ options: [{ label: "Postgres" }, { label: "SQLite" }] }] },
        },
      ],
    });
    expect(refOf(routed.model)).toBe("anthropic/opus");

    const reply = await h.prompt("ok");

    expect(refOf(reply.model)).toBe("anthropic/opus");
    expect(reply.thinkingLevel).toBe("high");
    const decision = m.decisions().at(-1)!;
    expect(decision.cause).toBe("question_reply");
    expect(decision.chosenModel).toBeNull();
    expect(decision.tier).toBe("REASONING");
  });

  it("holds a question reply from router state after a resume", async () => {
    // The tier the reply holds at lives on the session branch, not in this process.
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.state = { tier: "COMPLEX", model: "anthropic/sonnet", thinkingLevel: "medium", pin: null };
    h.branch.push({ type: "message", message: { role: "user", content: "design the cache" } });
    h.respond("anthropic/sonnet", [{ type: "toolCall", name: "ask_user_question", arguments: {} }]);

    const reply = await h.prompt("go ahead PI ESCALATE");

    expect(refOf(reply.model)).toBe("anthropic/opus");
    expect(m.decisions().at(-1)?.cause).toBe("question_reply_escalation");
    expect(m.decisions().at(-1)?.tier).toBe("REASONING");
  });

  it("still routes a new instruction typed into an open question", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.respond("anthropic/opus", [{ type: "toolCall", name: "ask_user_question", arguments: {} }]);

    await h.prompt("now refactor the whole auth layer to use JWT");

    // An open question does not turn the next real request into an answer: this one is
    // classified and routed like any other.
    const decision = m.decisions().at(-1)!;
    expect(decision.cause).not.toBe("question_reply");
    expect(decision.chosenModel).not.toBeNull();
  });

  it("classifies the user's own words, not a skill pi expanded around them", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    const skill = `<skill name="deep-review" location="/skills/deep-review/SKILL.md">\nReferences are relative to /skills.\n\n${HARD_PROMPT}\n</skill>\n\nhi`;
    expect(refOf((await h.prompt(skill)).model)).toBe("anthropic/haiku");
  });

  it("classifies the prompt, not context another extension injected after it", async () => {
    // pi turns custom messages into user messages for the provider; on the branch they
    // are custom entries, which is where the router reads the prompt.
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.fire("input", { text: "hi", source: "interactive" }, h.ctx);
    h.branch.push({ type: "message", message: { role: "user", content: "hi" } });
    h.branch.push({ type: "custom_message", customType: "context", content: HARD_PROMPT });

    const routed = await h.router.route(
      {
        model: VIRTUAL,
        thinkingLevel: "medium",
        reason: "user",
        messages: [
          { role: "user", content: "hi" },
          { role: "user", content: HARD_PROMPT },
        ],
      },
      h.ctx,
    );
    expect(refOf(routed.model)).toBe("anthropic/haiku");
  });

  it("holds the current model when --no-autoroute is set", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    m.flags.set("no-autoroute", true);
    const h = harness(m, dir);
    await h.start();

    // Nothing answered yet: the default model, unclassified.
    expect(refOf((await h.prompt(HARD_PROMPT)).model)).toBe("anthropic/haiku");
    expect(refOf((await h.prompt(HARD_PROMPT)).model)).toBe("anthropic/haiku");
    expect(m.decisions()).toEqual([]);
  });

  it("keeps a prompt another extension sent on the serving model", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await h.prompt(HARD_PROMPT);
    const injected = await h.prompt("hi", { source: "extension" });

    expect(refOf(injected.model)).toBe("anthropic/opus");
    expect(m.decisions()).toHaveLength(1);
  });

  it("explains how to fix a session with no config at all", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const empty = projectDir(null);
    const h = harness(m, empty);
    await h.start();

    expect(h.ctx.ui.notify).toHaveBeenCalledWith("autoroute: no config yet; run /autoroute init", "warning");
    await expect(h.prompt("hi")).rejects.toThrow(/no autorouter config \(run \/autoroute init\)/);

    // Not a word while another model is selected: the router is simply not in use.
    const other = mockPi();
    autorouter(other.pi as never);
    const quiet = harness(other, empty, { model: CATALOGUE[2] });
    await quiet.start();
    expect(quiet.ctx.ui.notify).not.toHaveBeenCalled();

    // A model that already answered is held rather than failing every prompt.
    h.respond("anthropic/sonnet");
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/sonnet");
  });

  it("survives a malformed config without disabling pi", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), "{ not json");
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("autoroute disabled"), "error");
    await expect(h.prompt("hi")).rejects.toThrow(/autorouter\.json/);
  });

  it("fails the request, not the process, when the registry throws", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.ctx.modelRegistry.find = () => {
      throw new Error("registry exploded");
    };

    await expect(h.prompt("hi")).rejects.toThrow(/registry exploded/);
  });

  it("takes the plan-mode floor from the shared event bus", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify({ ...CONFIG, planMode: { minTier: "COMPLEX" } }));
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    m.emit("autoroute:plan-mode", { active: true });
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/sonnet");
    expect(m.decisions()[0]?.planFloored).toBe(true);

    m.emit("autoroute:plan-mode", { active: false });
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");
  });

  it("forces a model for exactly one prompt, then reverts", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("next anthropic/opus", h.ctx);

    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/opus");
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");
    expect(m.decisions().map((d) => d.cause)).toEqual(["one_shot_override", "heuristic_scorer"]);
  });

  it("applies the thinking level supplied with a one-shot override", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("next anthropic/opus max", h.ctx);

    const routed = await h.prompt("hi");

    expect(refOf(routed.model)).toBe("anthropic/opus");
    expect(routed.thinkingLevel).toBe("max");
    expect(m.decisions()[0]?.thinkingLevel).toBe("max");
  });

  it("does not spend the override on a mid-run steer", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await h.prompt("hi", { keepRunning: true });
    await m.commands.get("autoroute")!.handler("next anthropic/opus", h.ctx);
    h.branch.push({ type: "message", message: { role: "user", content: "also this" } });
    expect(refOf((await h.request("user")).model)).toBe("anthropic/haiku");
    await m.fire("agent_end", { messages: [] }, h.ctx);

    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/opus");
  });

  it("clears a pending override with a bare `next`", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("next anthropic/opus", h.ctx);
    await m.commands.get("autoroute")!.handler("next", h.ctx);

    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");
  });

  it("honours an override even while routing is switched off", async () => {
    // An explicit one-off instruction outranks "off"; "off" still holds from the next
    // prompt onwards, on whatever model is serving the session.
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("off", h.ctx);
    await m.commands.get("autoroute")!.handler("next anthropic/opus", h.ctx);

    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/opus");
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/opus");
    expect(m.decisions()).toHaveLength(1);
  });

  it("pins every prompt to one model until unpinned", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await m.commands.get("autoroute")!.handler("pin anthropic/opus", h.ctx);
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/opus");
    await m.commands.get("autoroute")!.handler("unpin", h.ctx);
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");

    // Without an argument, pin the model serving the session.
    await m.commands.get("autoroute")!.handler("pin", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith("autoroute: pinned to anthropic/haiku", "info");
  });

  it("selects autorouter/tiered with `on`", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir, { model: CATALOGUE[2] });
    await h.start();

    await m.commands.get("autoroute")!.handler("on", h.ctx);

    expect(m.setModelCalls).toEqual(["autorouter/tiered"]);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith("autoroute: on (autorouter/tiered)", "info");
  });

  it("clears its footer status when another model is selected", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await h.prompt("hi");

    await m.fire("model_select", { source: "set", model: CATALOGUE[3] }, h.ctx);
    expect(h.ctx.ui.setStatus).toHaveBeenLastCalledWith("autoroute", undefined);
  });

  it("updates a tier and its thinking level from the slash command", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("tier SIMPLE anthropic/opus high", h.ctx);

    const routed = await h.prompt("hi");

    expect(refOf(routed.model)).toBe("anthropic/opus");
    expect(routed.thinkingLevel).toBe("high");
    const written = JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"));
    expect(written.tiers.SIMPLE).toEqual([{ model: "anthropic/opus", thinkingLevel: "high" }]);
  });

  it("updates arbitrary settings with the generic slash command", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("set decisionLog false", h.ctx);

    const written = JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"));
    expect(written.decisionLog).toBe(false);
    await m.commands.get("autoroute")!.handler("", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("model:    autorouter/tiered (selected)"), "info");
  });

  it("rejects an unknown model at command time rather than at the next prompt", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    h.ctx.modelRegistry.find = () => undefined;

    await m.commands.get("autoroute")!.handler("next anthropic/nope", h.ctx);

    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("not a model pi knows"), "error");
  });

  it("refuses to route a tier to a virtual model", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    await m.commands.get("autoroute")!.handler("tier SIMPLE autorouter/tiered", h.ctx);

    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("is a virtual model"), "error");
    expect(JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"))).toEqual(CONFIG);
  });

  it("writes a config from the available models, then selects the router", async () => {
    const fresh = projectDir(null);
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, fresh, { model: CATALOGUE[2] });
    await h.start();

    await m.commands.get("autoroute")!.handler("init project", h.ctx);

    const written = JSON.parse(readFileSync(join(fresh, ".pi", "autorouter.json"), "utf8"));
    // The virtual model costs nothing, and would be the "cheapest" model without the filter.
    expect(written.defaultModel).toBe("openai/gpt-mini");
    expect(JSON.stringify(written)).not.toContain("autorouter/tiered");
    expect(written.tiers.REASONING).toEqual({ model: "anthropic/opus", thinkingLevel: "high" });
    expect(m.setModelCalls).toEqual(["autorouter/tiered"]);

    // The new config takes effect without a restart.
    expect(refOf((await h.prompt("hi")).model)).toBe("openai/gpt-mini");
  });

  it("does not overwrite an existing config when the user declines", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir, { ui: { notify: vi.fn(), setStatus: vi.fn(), confirm: vi.fn(async () => false) } });
    await h.start();
    await m.commands.get("autoroute")!.handler("init project", h.ctx);

    expect(JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"))).toEqual(CONFIG);
    expect(h.ctx.ui.confirm).toHaveBeenCalled();
  });

  it("refuses to overwrite with no UI to confirm through", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir, { hasUI: false });
    await h.start();
    await m.commands.get("autoroute")!.handler("init project", h.ctx);

    expect(JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"))).toEqual(CONFIG);
  });

  it("reports a provider filter that matches nothing instead of writing", async () => {
    const fresh = projectDir(null);
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, fresh);
    await h.start();
    await m.commands.get("autoroute")!.handler("init nosuchprovider project", h.ctx);

    expect(existsSync(join(fresh, ".pi", "autorouter.json"))).toBe(false);
    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringContaining("no usable models"), "error");
  });

  it("prefers session-scoped models over the whole catalogue", async () => {
    // `--models` / `enabledModels` is the set the user intends to use.
    const fresh = projectDir(null);
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, fresh, { scopedModels: [{ model: VIRTUAL }, { model: CATALOGUE[1] }, { model: CATALOGUE[2] }] });
    await h.start();
    await m.commands.get("autoroute")!.handler("init project", h.ctx);

    const written = JSON.parse(readFileSync(join(fresh, ".pi", "autorouter.json"), "utf8"));
    expect(written.defaultModel).toBe("anthropic/haiku");
    expect(JSON.stringify(written)).not.toContain("gpt-mini");
  });
});

describe("decision-model classifier wiring", () => {
  const DECISION_CONFIG = {
    ...CONFIG,
    classifierType: "decision",
    decisionClassifierConfig: { model: "openrouter/cloudflare/clef", timeoutMs: 20 },
  };

  beforeEach(() => isolateHome());
  afterEach(() => restoreHome());

  /** Clef in pi's classifier catalog, answering with `answer`. */
  function withClef(h: ReturnType<typeof harness>, answer: () => Promise<unknown> | unknown) {
    const classify = vi.fn(async (_model: unknown, _context: unknown, options?: { signal?: AbortSignal }) => {
      const result = answer();
      return result instanceof Promise
        ? Promise.race([
            result,
            new Promise((resolve) => options?.signal?.addEventListener("abort", () => resolve({ stopReason: "aborted" }))),
          ])
        : result;
    });
    h.ctx.modelRegistry.findOfType = (type: string, provider: string, id: string) =>
      type === "classifier" && provider === "openrouter" && id === "cloudflare/clef" ? { provider, id } : undefined;
    h.ctx.modelRegistry.classify = classify;
    return classify;
  }

  it("routes on Clef's tier and shows it in /autoroute", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, projectDir(DECISION_CONFIG));
    const classify = withClef(h, () => ({
      provider: "openrouter",
      model: "cloudflare/clef",
      stopReason: "stop",
      answers: { tier: { type: "choice", choice: "REASONING", probabilities: { REASONING: 0.9 }, confidence: 0.9 } },
    }));
    await h.start();

    const routed = await h.prompt("hi");

    expect(classify).toHaveBeenCalledTimes(1);
    expect(refOf(routed.model)).toBe("anthropic/opus");
    expect(routed.thinkingLevel).toBe("high");
    const decision = m.decisions()[0]!;
    expect(decision.cause).toBe("decision_classifier");
    expect(decision.classifierModel).toBe("openrouter/cloudflare/clef");

    await m.commands.get("autoroute")!.handler("", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith(
      expect.stringContaining("classifier: decision (openrouter/cloudflare/clef, timeout 20ms)"),
      "info",
    );
  });

  it("warns once on a timeout, then skips the classifier while its circuit is open", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, projectDir(DECISION_CONFIG));
    const classify = withClef(h, () => new Promise(() => {}));
    await h.start();

    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");
    expect(refOf((await h.prompt("hi")).model)).toBe("anthropic/haiku");

    // The breaker outlives a single prompt: the second one never reached Clef.
    expect(classify).toHaveBeenCalledTimes(1);
    expect(m.decisions().map((d) => d.cause)).toEqual(["heuristic_scorer", "heuristic_scorer"]);
    expect(m.decisions()[1]?.signals).toContain("classifier-circuit-open");
    const warnings = (h.ctx.ui.notify as ReturnType<typeof vi.fn>).mock.calls.filter(([, level]) => level === "warning");
    expect(warnings).toEqual([
      [
        "autoroute: decision classifier unavailable; used heuristic classification instead (decision classifier timed out after 20ms).",
        "warning",
      ],
    ]);
    await m.commands.get("autoroute")!.handler("", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("circuit open"), "info");
  });

  it("keeps the block when settings are edited from the slash command", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const dir = projectDir(DECISION_CONFIG);
    const h = harness(m, dir);
    await h.start();

    await m.commands.get("autoroute")!.handler("set decisionClassifierConfig.timeoutMs 15000", h.ctx);

    const written = JSON.parse(readFileSync(join(dir, ".pi", "autorouter.json"), "utf8"));
    expect(written.classifierType).toBe("decision");
    expect(written.decisionClassifierConfig).toEqual({
      model: "openrouter/cloudflare/clef",
      timeoutMs: 15000,
      circuitBreakerEnabled: true,
      circuitBreakerCooldownSeconds: 30,
    });
  });
});

describe("adaptive wiring", () => {
  let dir: string;

  const ADAPTIVE_CONFIG = {
    ...CONFIG,
    adaptive: true,
    tiers: { SIMPLE: ["anthropic/haiku", "openai/gpt-mini"], MEDIUM: "anthropic/sonnet", COMPLEX: "anthropic/sonnet", REASONING: "anthropic/opus" },
  };

  beforeEach(() => {
    isolateHome();
    dir = projectDir(ADAPTIVE_CONFIG);
  });

  afterEach(() => {
    restoreHome();
  });

  it("picks from the pool, learns from the run, and persists what it learned", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();

    const routed = await h.prompt("hi", { keepRunning: true });
    const decision = m.decisions()[0]!;
    expect(decision.adaptive?.phase).toBe("cold_start");
    expect(["anthropic/haiku", "openai/gpt-mini"]).toContain(decision.chosenModel);
    expect(refOf(routed.model)).toBe(decision.chosenModel);

    // A run whose tool result errored is a failure signal against the model that answered.
    const [provider, model] = decision.chosenModel!.split("/");
    await m.fire(
      "agent_end",
      {
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", provider, model, content: [{ type: "toolCall", name: "bash", arguments: { command: "x" } }], stopReason: "toolUse" },
          { role: "toolResult", content: [{ type: "text", text: "command not found" }], isError: true },
          { role: "assistant", provider, model, content: [{ type: "text", text: "that failed" }], stopReason: "stop" },
        ],
      },
      h.ctx,
    );

    const storePath = join(home, ".pi", "agent", "autorouter-adaptive.json");
    expect(existsSync(storePath)).toBe(true);
    const persisted = JSON.parse(readFileSync(storePath, "utf8")) as { cells: { model: string; requestType: string; beta: number }[] };
    const cell = persisted.cells.find((c) => c.model === decision.chosenModel && c.requestType === "general")!;
    expect(cell.beta).toBeCloseTo(6); // prior 5 + 1 failure

    // A fresh session overlays the persisted posterior on its priors.
    const again = mockPi();
    autorouter(again.pi as never);
    const h2 = harness(again, dir);
    await h2.start();
    await again.commands.get("autoroute")!.handler("adaptive", h2.ctx);
    const shown = (h2.ctx.ui.notify as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as string;
    expect(shown).toContain("general:");
    expect(shown).toContain("samples   1");
  });

  it("learns nothing from a run it did not route", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.fire("agent_end", { messages: [{ role: "assistant", provider: "anthropic", model: "haiku", content: [], stopReason: "stop" }] }, h.ctx);
    expect(existsSync(join(home, ".pi", "agent", "autorouter-adaptive.json"))).toBe(false);
  });

  it("reports adaptive as off when the config does not enable it", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify(CONFIG));
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("adaptive", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenCalledWith(expect.stringMatching(/adaptive selection is off/), "info");
    await h.prompt("hi");
    expect(existsSync(join(home, ".pi", "agent", "autorouter-adaptive.json"))).toBe(false);
  });
});

describe("semantic keyword matching wiring", () => {
  let dir: string;
  let originalFetch: typeof fetch;

  const SEMANTIC_CONFIG = {
    ...CONFIG,
    keywordTierRules: [{ keywords: ["kubernetes deployment"], tier: "REASONING" }],
    semanticKeywordMatching: true,
    embeddingModel: "voyage/voyage-3-5",
    embeddingEndpoint: { apiKeyEnv: "TEST_EMBED_KEY" },
  };

  beforeEach(() => {
    isolateHome();
    dir = projectDir(SEMANTIC_CONFIG);
    process.env.TEST_EMBED_KEY = "k";
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    restoreHome();
    delete process.env.TEST_EMBED_KEY;
    globalThis.fetch = originalFetch;
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
    const h = harness(m, dir);
    await h.start();
    const routed = await h.prompt("help me roll out my k8s cluster");

    expect(calls.map((c) => c.url)).toEqual([
      "https://api.voyageai.com/v1/embeddings",
      "https://api.voyageai.com/v1/embeddings",
    ]);
    expect(calls[0]?.body).toEqual({ model: "voyage-3-5", input: ["kubernetes deployment"] });
    expect(calls[1]?.body.input).toEqual(["help me roll out my k8s cluster"]);
    const decision = m.decisions()[0]!;
    expect(decision.cause).toBe("semantic_keyword_match");
    expect(decision.chosenModel).toBe("anthropic/opus");
    expect(refOf(routed.model)).toBe("anthropic/opus");

    await m.commands.get("autoroute")!.handler("", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("keywords: semantic (voyage/voyage-3-5"), "info");
  });

  it("still routes when the embeddings endpoint is down", async () => {
    globalThis.fetch = (async () => new Response("down", { status: 503 })) as unknown as typeof fetch;
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    const routed = await h.prompt("hi");
    const decision = m.decisions()[0]!;
    expect(decision.cause).toBe("heuristic_scorer");
    expect(decision.signals).toContainEqual(expect.stringMatching(/semantic_keyword_match_failed .*503/));
    expect(refOf(routed.model)).toBe("anthropic/haiku");
  });
});

describe("decision log", () => {
  let dir: string;

  beforeEach(() => {
    isolateHome();
    dir = projectDir();
  });

  afterEach(() => {
    restoreHome();
  });

  it("renders each decision entry in the chat as an upstream-shaped log line", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await h.prompt("hi");

    const entry = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)!;
    const render = m.entryRenderers.get(DECISION_ENTRY_TYPE)!;
    const component = render({ type: "custom", customType: DECISION_ENTRY_TYPE, data: entry.data }, { expanded: false }, THEME) as {
      render(width: number): string[];
    };
    const lines = component.render(400);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatch(
      /^<autoroute: routing decision cause=heuristic_scorer, tier=SIMPLE, score=-?\d\.\d{3}, signals=\[.*\], routed_model=anthropic\/haiku>$/,
    );

    const expanded = render({ type: "custom", customType: DECISION_ENTRY_TYPE, data: entry.data }, { expanded: true }, THEME) as {
      render(width: number): string[];
    };
    const expandedLines = expanded.render(200);
    expect(expandedLines.length).toBeGreaterThan(3);
    expect(expandedLines.some((l) => l.includes("model:    anthropic/haiku"))).toBe(true);

    // Long lines wrap to the viewport instead of overflowing it.
    expect(component.render(40).every((l) => l.length <= 42)).toBe(true);
  });

  it("writes the line to stderr when there is no chat to draw in", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir, { hasUI: false });
    const written: string[] = [];
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(((chunk: unknown) => {
      written.push(String(chunk));
      return true;
    }) as never);
    try {
      await h.start();
      await h.prompt("hi");
    } finally {
      stderr.mockRestore();
    }
    expect(written).toHaveLength(1);
    expect(written[0]).toMatch(/^autoroute: routing decision cause=heuristic_scorer, .*routed_model=anthropic\/haiku\n$/);
  });

  it("can be switched off, leaving the entries recorded but undrawn", async () => {
    writeFileSync(join(dir, ".pi", "autorouter.json"), JSON.stringify({ ...CONFIG, decisionLog: false }));
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir, { hasUI: false });
    const written: string[] = [];
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(((chunk: unknown) => {
      written.push(String(chunk));
      return true;
    }) as never);
    try {
      await h.start();
      await h.prompt("hi");
    } finally {
      stderr.mockRestore();
    }
    expect(written).toEqual([]);
    const entry = m.entries.find((e) => e.type === DECISION_ENTRY_TYPE)!;
    expect(entry).toBeDefined();
    const render = m.entryRenderers.get(DECISION_ENTRY_TYPE)!;
    expect(render({ type: "custom", customType: DECISION_ENTRY_TYPE, data: entry.data }, { expanded: false }, THEME)).toBeUndefined();
  });

  it("replays the session's decisions with /autoroute log", async () => {
    const m = mockPi();
    autorouter(m.pi as never);
    const h = harness(m, dir);
    await h.start();
    await m.commands.get("autoroute")!.handler("log", h.ctx);
    expect(h.ctx.ui.notify).toHaveBeenLastCalledWith("autoroute: no decisions yet this session", "info");

    await h.prompt("hi");
    await h.prompt("think step by step and analyze this: weigh the options");
    await m.commands.get("autoroute")!.handler("log", h.ctx);
    const all = (h.ctx.ui.notify as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as string;
    expect(all.split("\n")).toHaveLength(2);
    expect(all).toMatch(/tier=SIMPLE.*\n.*tier=REASONING/s);

    await m.commands.get("autoroute")!.handler("log 1", h.ctx);
    const last = (h.ctx.ui.notify as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as string;
    expect(last.split("\n")).toHaveLength(1);
    expect(last).toContain("tier=REASONING");
  });
});
