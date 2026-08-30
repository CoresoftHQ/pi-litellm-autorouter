import { describe, expect, it } from "vitest";
import { hasActiveTodos, isBriefContinuation, todoContinuationTier } from "../src/todo.ts";

describe("todo continuation state", () => {
  it("uses the latest successful todo snapshot", () => {
    const entries = [
      { type: "message", message: { role: "toolResult", toolName: "todo", details: { tasks: [{ status: "in_progress" }] } } },
      { type: "message", message: { role: "toolResult", toolName: "todo", details: { tasks: [{ status: "completed" }] } } },
    ];
    expect(hasActiveTodos(entries, "todo")).toBe(false);
  });

  it("ignores unrelated and failed tool results", () => {
    const entries = [
      { type: "message", message: { role: "toolResult", toolName: "todo", isError: true, details: { tasks: [{ status: "in_progress" }] } } },
      { type: "message", message: { role: "toolResult", toolName: "other", details: { tasks: [{ status: "in_progress" }] } } },
    ];
    expect(hasActiveTodos(entries, "todo")).toBe(false);
  });

  it("recognizes only brief acknowledgements", () => {
    expect(isBriefContinuation("Go ahead!", 100)).toBe(true);
    expect(isBriefContinuation("Please implement the other task too", 100)).toBe(false);
    expect(isBriefContinuation("ok", 1)).toBe(false);
  });

  it("applies the configured floor only to active acknowledgements", () => {
    const config = { enabled: true, minTier: "COMPLEX" as const, maxPromptChars: 100 };
    expect(todoContinuationTier("ok", true, config)).toBe("COMPLEX");
    expect(todoContinuationTier("What time is it?", true, config)).toBeNull();
    expect(todoContinuationTier("ok", false, config)).toBeNull();
  });
});
