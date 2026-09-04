/**
 * Optional integration with @juicesharp/rpiv-todo.
 *
 * This deliberately uses structural types instead of importing the todo package: the
 * autorouter must still load when that separately-installed extension is absent.
 */
export interface TodoTask {
  id: number;
  subject: string;
  description?: string;
  status?: string;
  metadata?: Record<string, unknown>;
}

export type TodoTaskCache = Map<number, TodoTask>;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Read the task snapshot in rpiv-todo's result details, ignoring unrelated todo tools. */
export function cacheTodoTasks(details: unknown, cache: TodoTaskCache): void {
  const record = asRecord(details);
  const tasks = record?.tasks;
  if (!Array.isArray(tasks)) return;
  cache.clear();
  for (const value of tasks) {
    const task = asRecord(value);
    if (!task || typeof task.id !== "number" || typeof task.subject !== "string") continue;
    cache.set(task.id, {
      id: task.id,
      subject: task.subject,
      ...(typeof task.description === "string" ? { description: task.description } : {}),
      ...(typeof task.status === "string" ? { status: task.status } : {}),
      ...(asRecord(task.metadata) ? { metadata: asRecord(task.metadata)! } : {}),
    });
  }
}

/**
 * Return the task whose work is about to begin. Creation with an initial
 * in-progress status is included; a normal creation remains only planning.
 */
export function taskStartingFromTodoInput(input: unknown, cache: TodoTaskCache): TodoTask | null {
  const record = asRecord(input);
  if (!record || record.status !== "in_progress") return null;

  if (record.action === "create" && typeof record.subject === "string") {
    return {
      id: typeof record.id === "number" ? record.id : -1,
      subject: record.subject,
      ...(typeof record.description === "string" ? { description: record.description } : {}),
      status: "in_progress",
    };
  }

  if (record.action === "update" && typeof record.id === "number") {
    const task = cache.get(record.id);
    return task ? { ...task, status: "in_progress" } : null;
  }
  return null;
}

/** Build the classifiable text from the fields the todo extension exposes. */
export function todoTaskText(task: TodoTask): string {
  return task.description ? `${task.subject}\n\n${task.description}` : task.subject;
}

/** Attach routing details without discarding metadata owned by the todo extension/user. */
export function withTodoRoutingMetadata(
  input: unknown,
  routing: { model: string | null; tier: string | null; cause: string },
): void {
  const record = asRecord(input);
  if (!record) return;
  const metadata = asRecord(record.metadata) ?? {};
  record.metadata = { ...metadata, autorouter: routing };
}
