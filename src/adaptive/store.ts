/**
 * Where learned posteriors live between sessions.
 *
 * Upstream flushes to Postgres; here a JSON file next to the config is enough, since one
 * pi process is one router. Reads and writes never throw: a missing or corrupt file means
 * "start from the priors", and a write that fails costs at most one turn of learning.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { PersistedCells } from "./router.ts";

export function defaultAdaptiveStorePath(): string {
  return join(homedir(), ".pi", "agent", "autorouter-adaptive.json");
}

export function readPersistedCells(path: string): PersistedCells | null {
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as PersistedCells;
  } catch {
    return null;
  }
}

export function writePersistedCells(path: string, cells: PersistedCells): boolean {
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(cells, null, 2)}\n`, "utf8");
    return true;
  } catch {
    return false;
  }
}
