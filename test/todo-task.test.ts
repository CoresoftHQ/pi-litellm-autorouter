import { describe, expect, it } from "vitest";
import { cacheTodoTasks, taskStartingFromTodoInput, todoTaskText, withTodoRoutingMetadata } from "../src/todo-task.ts";

describe("optional todo task integration", () => {
  it("routes an in-progress update using the task snapshot", () => {
    const tasks = new Map();
    cacheTodoTasks(
      { tasks: [{ id: 4, subject: "Implement routing", description: "Add tests", status: "pending" }] },
      tasks,
    );

    const task = taskStartingFromTodoInput({ action: "update", id: 4, status: "in_progress" }, tasks);
    expect(task).toMatchObject({ id: 4, subject: "Implement routing" });
    expect(todoTaskText(task!)).toBe("Implement routing\n\nAdd tests");
  });

  it("does not route ordinary task creation or unrelated todo inputs", () => {
    const tasks = new Map();
    expect(taskStartingFromTodoInput({ action: "create", subject: "Plan work", status: "pending" }, tasks)).toBeNull();
    expect(taskStartingFromTodoInput({ action: "list" }, tasks)).toBeNull();
  });

  it("supports creation that starts work immediately and preserves user metadata", () => {
    const input: Record<string, unknown> = {
      action: "create",
      subject: "Fix bug",
      status: "in_progress",
      metadata: { owner: "agent" },
    };
    expect(taskStartingFromTodoInput(input, new Map())).toMatchObject({ subject: "Fix bug" });

    withTodoRoutingMetadata(input, { model: "anthropic/sonnet", tier: "COMPLEX", cause: "heuristic_scorer" });
    expect(input.metadata).toEqual({
      owner: "agent",
      autorouter: { model: "anthropic/sonnet", tier: "COMPLEX", cause: "heuristic_scorer" },
    });
  });
});
