/**
 * Question-reply detection and the router's hold path.
 *
 * The invariant under test is narrow and load-bearing: answering a question the assistant
 * asked must never move the session's model. Everything else here exists to keep that
 * from becoming "short prompts never route".
 */

import { describe, expect, it } from "vitest";
import { buildConfig } from "../src/config.ts";
import { detectQuestionReply, looksLikeAnswer, pendingQuestion } from "../src/question-reply.ts";
import { route } from "../src/router.ts";
import type { ExtractedTurn } from "../src/types.ts";
import type { ModelApplier } from "../src/resolve.ts";

const TIERS = {
  SIMPLE: "anthropic/haiku",
  MEDIUM: "anthropic/sonnet",
  COMPLEX: "anthropic/sonnet",
  REASONING: "anthropic/opus",
};

function config(overrides: Record<string, unknown> = {}) {
  const built = buildConfig([{ defaultModel: "anthropic/haiku", tiers: TIERS, ...overrides }]);
  expect(built.errors).toEqual([]);
  return built.config;
}

function turn(currentAsk: string | null, extra: Partial<ExtractedTurn> = {}): ExtractedTurn {
  return { currentAsk, priorTurns: [], conversationContinuing: true, cumulativeTokens: 0, ...extra };
}

function applier(): ModelApplier & { applied: string[] } {
  const applied: string[] = [];
  return {
    applied,
    find: (provider, modelId) => ({ provider, id: modelId }),
    setModel: async (model) => {
      const m = model as unknown as { provider: string; id: string };
      applied.push(`${m.provider}/${m.id}`);
      return true;
    },
    setThinkingLevel: () => {},
  } as ModelApplier & { applied: string[] };
}

const now = () => 1_000_000;

/** An assistant message carrying a question tool call, as pi stores it. */
function asked(options: string[] = ["Use PostgreSQL", "Use SQLite"]) {
  return {
    role: "assistant",
    content: [
      { type: "text", text: "Which database should we use?" },
      {
        type: "toolCall",
        name: "ask_user_question",
        arguments: {
          questions: [
            {
              question: "Which database?",
              header: "Database",
              options: options.map((label) => ({ label, description: "..." })),
            },
          ],
        },
      },
    ],
  };
}

const TOOL_NAMES = ["ask_user_question", "ask_question"];

describe("pendingQuestion", () => {
  it("finds a question tool call in the newest assistant message", () => {
    const found = pendingQuestion([{ role: "user", content: "set up the db" }, asked()], TOOL_NAMES);
    expect(found?.toolName).toBe("ask_user_question");
    expect(found?.optionLabels).toContain("use postgresql");
  });

  it("ignores a question the assistant has already moved past", () => {
    // The tool answered inline, the run carried on: a later assistant message exists, so
    // whatever the user types next is a new turn rather than an answer.
    const messages = [asked(), { role: "assistant", content: [{ type: "text", text: "Done, postgres it is." }] }];
    expect(pendingQuestion(messages, TOOL_NAMES)).toBeNull();
  });

  it("ignores tool calls that are not question tools", () => {
    const messages = [{ role: "assistant", content: [{ type: "toolCall", name: "bash", arguments: {} }] }];
    expect(pendingQuestion(messages, TOOL_NAMES)).toBeNull();
  });

  it("matches tool names case-insensitively", () => {
    const messages = [{ role: "assistant", content: [{ type: "toolCall", name: "AskUserQuestion", arguments: {} }] }];
    expect(pendingQuestion(messages, ["ask_user_question"])).not.toBeNull();
  });

  it("returns null for a session with no assistant messages at all", () => {
    expect(pendingQuestion([{ role: "user", content: "hi" }], TOOL_NAMES)).toBeNull();
  });
});

describe("looksLikeAnswer", () => {
  const opts = { maxChars: 120, optionLabels: ["use postgresql", "use sqlite"] };

  it.each([
    "ok",
    "OK.",
    "okay",
    "yes",
    "yes please",
    "sure, go ahead",
    "do it",
    "2",
    "option 2",
    "the first one",
    "1 and 3",
    "b",
    "no",
    "up to you",
    "postgres",
    "use postgresql",
    "Use PostgreSQL",
    "yeah, and keep the existing migrations",
  ])("treats %j as an answer", (text) => {
    expect(looksLikeAnswer(text, opts)).toBe(true);
  });

  it.each([
    "now refactor the whole auth layer to use JWT",
    "add a migration for the users table",
    "implement it",
    "fix the failing test",
    "review src/router.ts and tell me what breaks",
    "run `npm test` and report back",
    "write the docs for this",
  ])("treats %j as a new instruction", (text) => {
    expect(looksLikeAnswer(text, opts)).toBe(false);
  });

  it("rejects a reply longer than maxChars", () => {
    expect(looksLikeAnswer("yes ".repeat(60), opts)).toBe(false);
  });

  it("rejects a multi-paragraph reply", () => {
    expect(looksLikeAnswer("yes\n\nalso:\nsome other thing\nand another", opts)).toBe(false);
  });

  it("accepts an offered label even when it is longer than maxChars", () => {
    // A clicked option arrives verbatim, and option labels can be long. Naming one of the
    // choices the user was shown is conclusive regardless of length.
    const label = "pending question plus short reply which is a rather long label indeed for a single option here";
    expect(looksLikeAnswer(label, { maxChars: 20, optionLabels: [label.toLowerCase()] })).toBe(true);
  });
});

describe("detectQuestionReply", () => {
  const options = { enabled: true, toolNames: TOOL_NAMES, maxChars: 120 };

  it("fires when a question is open and the reply is answer-shaped", () => {
    expect(detectQuestionReply("ok", [asked()], options)?.toolName).toBe("ask_user_question");
  });

  it("does not fire on a new instruction typed into an open question", () => {
    expect(detectQuestionReply("now refactor the whole auth layer to use JWT", [asked()], options)).toBeNull();
  });

  it("does not fire on a terse prompt when nothing was asked", () => {
    expect(detectQuestionReply("ok", [{ role: "user", content: "hi" }], options)).toBeNull();
  });

  it("does not fire when disabled", () => {
    expect(detectQuestionReply("ok", [asked()], { ...options, enabled: false })).toBeNull();
  });

  it("does not fire on empty text", () => {
    expect(detectQuestionReply("   ", [asked()], options)).toBeNull();
  });
});

describe("route — question reply", () => {
  const signal = { toolName: "ask_user_question", lastTier: "COMPLEX" as const };

  it("applies no model at all", async () => {
    const api = applier();
    const { decision } = await route({ turn: turn("ok"), config: config(), api, questionReply: signal, now });

    expect(api.applied).toEqual([]);
    expect(decision.cause).toBe("question_reply");
    expect(decision.chosenModel).toBeNull();
    expect(decision.tier).toBe("COMPLEX");
    expect(decision.signals).toContain("question_reply (ask_user_question)");
  });

  it("would otherwise have dropped the session to SIMPLE", async () => {
    // The control: same turn, no signal. This is the regression the hold exists to stop.
    const api = applier();
    const { decision } = await route({ turn: turn("ok"), config: config(), api, now });
    expect(decision.tier).toBe("SIMPLE");
    expect(decision.chosenModel).toBe("anthropic/haiku");
  });

  it("holds even when the last tier is unknown", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok"),
      config: config(),
      api,
      questionReply: { toolName: "ask_user_question", lastTier: null },
      now,
    });
    expect(api.applied).toEqual([]);
    expect(decision.cause).toBe("question_reply");
  });

  it("still escalates on an escalation keyword, from the tier the session was on", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok PI ESCALATE"),
      config: config(),
      api,
      questionReply: signal,
      now,
    });
    expect(decision.cause).toBe("question_reply_escalation");
    expect(decision.escalated).toBe(true);
    expect(decision.tier).toBe("REASONING");
    expect(api.applied).toEqual(["anthropic/opus"]);
  });

  it("holds rather than escalating from nothing when the last tier is unknown", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok PI ESCALATE"),
      config: config(),
      api,
      questionReply: { toolName: "ask_user_question", lastTier: null },
      now,
    });
    expect(decision.cause).toBe("question_reply");
    expect(api.applied).toEqual([]);
  });

  it("honours a plan-mode floor that would raise the held tier", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok"),
      config: config({ planMode: { minTier: "REASONING" } }),
      api,
      planModeActive: true,
      questionReply: { toolName: "ask_user_question", lastTier: "SIMPLE" },
      now,
    });
    expect(decision.cause).toBe("plan_mode");
    expect(decision.tier).toBe("REASONING");
  });

  it("does not let a plan-mode floor lower the held tier", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok"),
      config: config({ planMode: { minTier: "MEDIUM" } }),
      api,
      planModeActive: true,
      questionReply: signal,
      now,
    });
    expect(decision.cause).toBe("question_reply");
    expect(api.applied).toEqual([]);
  });

  it("is outranked by a one-shot override", async () => {
    const api = applier();
    const { decision, consumedOneShot } = await route({
      turn: turn("ok"),
      config: config(),
      api,
      oneShot: "anthropic/opus",
      questionReply: signal,
      now,
    });
    expect(consumedOneShot).toBe(true);
    expect(decision.cause).toBe("one_shot_override");
    expect(api.applied).toEqual(["anthropic/opus"]);
  });

  it("outranks a session-affinity pin", async () => {
    const api = applier();
    const { decision } = await route({
      turn: turn("ok"),
      config: config({ sessionAffinity: true }),
      api,
      pin: { model: "anthropic/sonnet", tier: "MEDIUM", expiresAt: now() + 60_000 },
      questionReply: signal,
      now,
    });
    expect(decision.cause).toBe("question_reply");
    expect(api.applied).toEqual([]);
  });
});

describe("questionReply config", () => {
  it("is on by default with the built-in tool names", () => {
    const { config: c } = buildConfig([{}]);
    expect(c.questionReply).toBe(true);
    expect(c.questionReplyToolNames).toContain("ask_user_question");
    expect(c.questionReplyMaxChars).toBeGreaterThan(0);
  });

  it("accepts a boolean", () => {
    expect(buildConfig([{ questionReply: false }]).config.questionReply).toBe(false);
  });

  it("accepts an object and replaces the tool names wholesale", () => {
    const { config: c, errors } = buildConfig([
      { tiers: TIERS, questionReply: { enabled: true, toolNames: ["my_question_tool"], maxReplyChars: 40 } },
    ]);
    expect(errors).toEqual([]);
    expect(c.questionReplyToolNames).toEqual(["my_question_tool"]);
    expect(c.questionReplyMaxChars).toBe(40);
  });

  it("rejects a bad shape", () => {
    expect(buildConfig([{ questionReply: 3 }]).errors[0]).toMatch(/questionReply must be/);
    expect(buildConfig([{ questionReply: { toolNames: "nope" } }]).errors[0]).toMatch(/toolNames/);
    expect(buildConfig([{ questionReply: { maxReplyChars: 0 } }]).errors[0]).toMatch(/maxReplyChars/);
  });

  it("warns when the tool name list is emptied", () => {
    expect(buildConfig([{ questionReply: { toolNames: [] } }]).warnings[0]).toMatch(/no turn can be recognised/);
  });
});
