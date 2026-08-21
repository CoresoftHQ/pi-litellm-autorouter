# The seam: does `pi.setModel()` affect the turn already in flight?

**Status: VERIFIED — yes**, on pi 0.84.2. The `input` event is a usable routing hook, and
pi awaits the handler, so an async classifier can run there.

This was the blocking question for the whole design: if the model set during a turn only
took effect on the *next* turn, routing would be permanently one prompt behind and the
in-process approach would not work as specified.

## What was run

`test/manual/seam-probe.ts` loaded into a real pi process, in a container with a stale
bedrock token (114 models catalogued, so the registry and `setModel` were exercised even
though the eventual provider call failed auth):

```bash
pi -e ./test/manual/seam-probe.ts -p "x"
```

The probe picks a target model that **differs from the currently active one**, calls
`await pi.setModel(target)` inside the `input` handler, and logs which model each later
hook observes.

## Result

```
[seam] session_start            114 models available, current=amazon-bedrock/amazon.nova-2-lite-v1:0
[seam] input (turn 1)           text="x" before=amazon-bedrock/amazon.nova-2-lite-v1:0
[seam]   setModel ->            amazon-bedrock/amazon.nova-lite-v1:0
[seam]   setModel returned      true after 2ms, ctx.model=amazon-bedrock/amazon.nova-lite-v1:0
[seam] before_agent_start       model=amazon-bedrock/amazon.nova-lite-v1:0 intended=amazon-bedrock/amazon.nova-lite-v1:0
[seam] turn_start               model=amazon-bedrock/amazon.nova-lite-v1:0 — MATCHES intent
[seam] before_provider_request  model=amazon-bedrock/amazon.nova-lite-v1:0 intended=amazon-bedrock/amazon.nova-lite-v1:0
```

Three things this establishes:

1. **The switch takes effect on the current turn.** `before` was `nova-2-lite` and the
   model reaching `before_provider_request` — the last observable point before the request
   leaves — was `nova-lite`. The model genuinely changed, and the change reached the wire.
2. **pi awaits the `input` handler.** `setModel returned` logged before `turn_start`, and
   the probe's ordering warning (which fires when `turn_start` runs before the handler
   resolves) never triggered. An `await`ed LLM classification inside `input` is therefore
   viable.
3. **`setModel` resolved `true` in 2ms**, so the call itself adds nothing measurable; any
   latency in this design comes from the classifier, not from applying the decision.

### An earlier run that proved nothing

The first attempt picked target index 0 unconditionally, and pi had persisted the model
from a previous run — so `before` already *equalled* the target. "Matches intent"
downstream was consistent with the switch working and with it doing nothing at all. The
probe now always picks a model different from the active one, which is what makes the
reading above meaningful. Worth keeping in mind if this is ever re-run.

## Still unverified

The result above is one turn, in print mode (`-p`), against one provider. These remain
open and want an interactive session with working credentials:

- **Multi-turn.** Does turn *N* consistently get turn *N*'s classification across a long
  session? The probe alternates targets per turn and is written for this; it just needs a
  session where more than one prompt is sent.
- **Streaming behaviours.** `src/index.ts` deliberately skips routing when
  `event.streamingBehavior` is set (`"steer"` / `"followUp"`), on the assumption that
  switching models mid-run would break the conversation's provider-specific message
  shapes. That assumption is untested — it may be safe, in which case the guard is
  unnecessarily conservative.
- **`ctx.modelRegistry.complete()` from inside an `input` handler**, which is how
  `src/classify/llm.ts` runs the LLM classifier with pi's own resolved credentials. The
  types support it; no live call has been made.
- **Classifier latency.** The README claims the heuristic classifier should stay the
  default until p50/p95 for a small-model classification is measured. Still unmeasured.
- **`pi.setThinkingLevel()` clamping.** `src/resolve.ts` calls it *after* `setModel` on the
  grounds that pi clamps to the model's capabilities. Ordering is right by construction;
  the clamping behaviour itself has not been observed.

## What was also confirmed in passing

- The extension loads clean: `pi -e ./src/index.ts -p "hello"` reaches the provider call,
  so the factory ran and every handler registered.
- `registerFlag` took effect: `--no-autoroute` is accepted, while
  `--definitely-not-a-flag` is rejected with `Unknown option`.
- **pi has no built-in plan mode.** It ships as an example extension, so there is no
  native state to query. `src/index.ts` therefore listens on the shared event bus for
  `autoroute:plan-mode` (`{ active: boolean }`) as an integration contract, and falls back
  to `planMode.patterns` for sentinels carried in prompt text.
