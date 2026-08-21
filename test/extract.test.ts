import { describe, expect, it } from "vitest";
import { DEFAULT_REMINDER_MARKERS } from "../src/defaults.ts";
import { extractTurn, newestTurnAsk, stripReminderBlocks } from "../src/extract.ts";

const MARKERS = DEFAULT_REMINDER_MARKERS;

describe("stripReminderBlocks", () => {
  it("removes a complete block", () => {
    const text = "before <system-reminder>plumbing</system-reminder> after";
    expect(stripReminderBlocks(text, MARKERS)).toBe("before  after");
  });

  it("matches delimiters case-insensitively", () => {
    const text = "a <SYSTEM-REMINDER>x</SYSTEM-REMINDER> b";
    expect(stripReminderBlocks(text, MARKERS)).toBe("a  b");
  });

  it("leaves an unclosed delimiter in place", () => {
    // Prose that merely mentions a delimiter must not swallow the rest of the message.
    const text = "what does <system-reminder> mean in this codebase?";
    expect(stripReminderBlocks(text, MARKERS)).toBe(text);
  });

  it("removes several blocks", () => {
    const text = "a <system-reminder>x</system-reminder> b <system-reminder>y</system-reminder> c";
    expect(stripReminderBlocks(text, MARKERS)).toBe("a  b  c");
  });

  it("removes overlapping pairs whole", () => {
    // Upstream: "blocks that nest or overlap across pairs are stripped whole". The <a> and
    // <b> spans overlap, so their union goes — not just the first one.
    const markers = [
      { open: "<a>", close: "</a>" },
      { open: "<b>", close: "</b>" },
    ];
    const text = "keep <a>one <b>two</a> three</b> keep";
    expect(stripReminderBlocks(text, markers)).toBe("keep  keep");
  });

  it("removes a nested pair whole", () => {
    const markers = [
      { open: "<a>", close: "</a>" },
      { open: "<b>", close: "</b>" },
    ];
    expect(stripReminderBlocks("keep <a>one <b>two</b> three</a> keep", markers)).toBe("keep  keep");
  });

  it("strips a reminder-only turn to empty", () => {
    expect(stripReminderBlocks("<system-reminder>only plumbing</system-reminder>", MARKERS)).toBe("");
  });

  it("honours custom markers and drops the built-in pair", () => {
    const markers = [{ open: "<<<CTX>>>", close: "<<<END>>>" }];
    const text = "a <<<CTX>>>x<<<END>>> b <system-reminder>y</system-reminder>";
    // Setting reminderMarkers replaces the built-in pair rather than adding to it.
    expect(stripReminderBlocks(text, markers)).toBe("a  b <system-reminder>y</system-reminder>");
  });
});

describe("newestTurnAsk", () => {
  it("walks back past a reminder-only turn", () => {
    const messages = [
      { role: "user", content: "the real ask" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "<system-reminder>plumbing</system-reminder>" },
    ];
    expect(newestTurnAsk(messages, MARKERS)).toBe("the real ask");
  });

  it("flattens multimodal content to its text blocks", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "image", data: "…" },
          { type: "text", text: "describe this" },
        ],
      },
    ];
    expect(newestTurnAsk(messages, MARKERS)).toBe("describe this");
  });

  it("returns null when there is no user message", () => {
    expect(newestTurnAsk([{ role: "assistant", content: "hi" }], MARKERS)).toBeNull();
  });
});

describe("extractTurn", () => {
  const options = {
    markerPairs: MARKERS,
    contextWindowSize: 3,
    perTurnChars: 200,
    includeAssistantTurns: false,
  };

  it("uses the current prompt as the ask", () => {
    const turn = extractTurn("fix the tests", [], options);
    expect(turn.currentAsk).toBe("fix the tests");
    expect(turn.conversationContinuing).toBe(false);
  });

  it("falls back to the last real ask when the turn is all plumbing", () => {
    const messages = [{ role: "user", content: "why is p99 latency high?" }];
    const turn = extractTurn("<system-reminder>ctx</system-reminder>", messages, options);
    expect(turn.currentAsk).toBe("why is p99 latency high?");
  });

  it("quotes prior user turns oldest-first, bounded by the window", () => {
    const messages = [
      { role: "user", content: "one" },
      { role: "user", content: "two" },
      { role: "user", content: "three" },
      { role: "user", content: "four" },
    ];
    const turn = extractTurn("five", messages, options);
    expect(turn.priorTurns.map((t) => t.text)).toEqual(["two", "three", "four"]);
  });

  it("omits assistant turns unless asked for them", () => {
    const messages = [
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
    ];
    expect(extractTurn("x", messages, options).priorTurns.map((t) => t.text)).toEqual(["u1"]);
    expect(
      extractTurn("x", messages, { ...options, includeAssistantTurns: true }).priorTurns.map((t) => t.role),
    ).toEqual(["user", "assistant"]);
  });

  it("drops a prior turn that repeats the current ask", () => {
    const messages = [
      { role: "user", content: "real question" },
      { role: "user", content: "continue" },
    ];
    const turn = extractTurn("continue", messages, options);
    expect(turn.priorTurns.map((t) => t.text)).toEqual(["real question"]);
  });

  it("reports a continuation even when the window kept nothing", () => {
    // Gating depth on the window's output would report a long continuation as a
    // context-free single-turn request.
    const messages = [{ role: "user", content: "continue" }];
    const turn = extractTurn("continue", messages, options);
    expect(turn.priorTurns).toHaveLength(0);
    expect(turn.conversationContinuing).toBe(true);
  });

  it("truncates quoted turns", () => {
    const messages = [{ role: "user", content: "x".repeat(500) }];
    const turn = extractTurn("now", messages, { ...options, perTurnChars: 10 });
    expect(turn.priorTurns[0]?.text).toBe(`${"x".repeat(10)}…`);
  });

  it("sends no context at all when the window is 0", () => {
    const messages = [{ role: "user", content: "one" }];
    expect(extractTurn("two", messages, { ...options, contextWindowSize: 0 }).priorTurns).toHaveLength(0);
  });
});
