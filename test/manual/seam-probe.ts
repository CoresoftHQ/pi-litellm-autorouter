/**
 * Phase 0 probe: does `pi.setModel()` affect the turn already in flight?
 *
 * Not a unit test — it needs a real pi process and at least two working models. See
 * ../../docs/seam.md for how to read the output.
 *
 *   pi -e ./test/manual/seam-probe.ts
 *
 * Then send two prompts in one session. The second is the one that matters: if routing is
 * a turn behind, the model observed on prompt 2 is the one chosen during prompt 1.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const log = (stage: string, detail: string): void => {
  // stderr so the probe's output cannot be confused with the agent's answer.
  process.stderr.write(`[seam] ${stage.padEnd(24)} ${detail}\n`);
};

const modelOf = (ctx: ExtensionContext): string =>
  ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "(none)";

export default function seamProbe(pi: ExtensionAPI): void {
  let turnCounter = 0;
  /** What we asked for during this turn's input handler. */
  let intended: string | null = null;
  let inputHandlerResolvedAt: number | null = null;

  pi.on("session_start", async (_event, ctx) => {
    const available = ctx.modelRegistry.getAvailable();
    log("session_start", `${available.length} models available, current=${modelOf(ctx)}`);
    for (const model of available.slice(0, 10)) {
      log("  candidate", `${model.provider}/${model.id}`);
    }
    if (available.length < 2) {
      log("WARNING", "need at least 2 usable models for this probe to mean anything");
    }
  });

  pi.on("input", async (event, ctx) => {
    turnCounter += 1;
    const before = modelOf(ctx);

    // Always pick a model that differs from the one currently active. If `before` already
    // equalled the target, a "matches intent" reading downstream would prove nothing —
    // the model would be right whether or not setModel had any effect on this turn.
    const available = ctx.modelRegistry.getAvailable();
    const target = available.find((m) => `${m.provider}/${m.id}` !== before);
    if (!target) {
      log("input", "need a second usable model to force a switch; nothing to probe");
      return { action: "continue" as const };
    }

    intended = `${target.provider}/${target.id}`;
    log(`input (turn ${turnCounter})`, `text=${JSON.stringify(event.text.slice(0, 40))} before=${before}`);
    log("  setModel ->", intended);

    const started = Date.now();
    const ok = await pi.setModel(target);
    inputHandlerResolvedAt = Date.now();
    log("  setModel returned", `${ok} after ${inputHandlerResolvedAt - started}ms, ctx.model=${modelOf(ctx)}`);

    return { action: "continue" as const };
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    log("before_agent_start", `model=${modelOf(ctx)} intended=${intended ?? "(none)"}`);
    return undefined;
  });

  pi.on("turn_start", async (_event, ctx) => {
    const observed = modelOf(ctx);
    const verdict = observed === intended ? "MATCHES intent" : `MISMATCH (intended ${intended})`;
    // If this line ever prints before "setModel returned", pi did not await the input
    // handler, and an async classifier cannot run there.
    log("turn_start", `model=${observed} — ${verdict}`);
    if (inputHandlerResolvedAt === null) {
      log("  ORDERING", "turn_start ran BEFORE the input handler resolved: pi does not await input");
    }
  });

  pi.on("before_provider_request", async (_event, ctx) => {
    // The last observable point before the request leaves. This is the authoritative
    // answer: whatever model is active here is the one that served the turn.
    log("before_provider_request", `model=${modelOf(ctx)} intended=${intended ?? "(none)"}`);
    return undefined;
  });

  pi.on("agent_end", async () => {
    inputHandlerResolvedAt = null;
  });
}
