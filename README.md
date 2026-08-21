# pi-litellm-autorouter

A [pi](https://pi.dev) extension that picks the cheapest model able to handle each prompt, automatically —
in-process, no proxy required.

> **Status:** implemented and unit-tested, but not yet run against a real session with real credentials.
> Treat the shipped defaults as untuned.

## What it does

pi normally uses one model for a whole session, switched by hand with `/model`. That's wasteful: a trivial
question ("what does this env var do?") doesn't need the same model as a hard debugging session. This
extension classifies every prompt before it runs and calls `pi.setModel()` to route it to the right tier —
so cheap prompts go to cheap models and hard prompts go to strong ones, automatically.

It's a TypeScript port of [LiteLLM's Auto Router v2](https://docs.litellm.ai/docs/proxy/auto_routing)
(`auto_router/complexity_router`), adapted to run inside pi's extension API instead of behind a separate
LiteLLM proxy process. That means no extra network hop, and pi's UI/thinking-level/credential logic can see
the model that was actually picked.

## How it works

Each user prompt goes through: **extract → classify → override → select → apply**.

1. **Extract** — strip `<system-reminder>` blocks (harness plumbing, not something the user asked), pull out
   the current ask plus a few prior turns for context.
2. **Classify** — score the prompt into one of four tiers, `SIMPLE` → `MEDIUM` → `COMPLEX` → `REASONING`.
   Two classifiers are supported:
   - `heuristic` (default) — a local, sub-millisecond weighted scorer (code presence, reasoning markers,
     technical terms, length, etc.), no API call.
   - `llm` — a small model classifies the prompt against a rubric. The `agentic` rubric preset is the
     default here, calibrated so routine engineering work (installs, multi-file edits, standard debugging)
     lands at `MEDIUM` instead of being over-classified as top-tier, which is what a chat-tuned rubric does
     to agent traffic.
3. **Override** — a few signals outrank the classifier, in order: an explicit `/model` pin or `--no-autoroute`
   escape hatch, a session affinity pin (reuse the first turn's model for the whole session, if enabled),
   `keyword_tier_rules` (deterministic keyword → tier), and a plan-mode floor (routes at least to a
   configured tier while a plan-mode extension or sentinel is active). `escalation_keywords` can bump the
   result up exactly one tier — never down, never a caller-chosen model.
4. **Select** — a tier maps to one model, a pool of models, or (with `adaptive: true`) a Thompson-sampled
   pick across a pool weighted by quality/cost.
5. **Apply** — resolve the chosen model against pi's own model registry and credentials, call
   `pi.setModel()` (falling back down the chain, then to `defaultModel`, if a model has no credentials), then
   `pi.setThinkingLevel()`, and record the decision (visible via `/autoroute explain` and the footer status).

Routing never fails a prompt: any classifier error or unresolved model falls back to `defaultModel`, and the
model is chosen once per prompt and held for the whole agent turn (including all its tool calls) — it never
switches mid-turn.

An optional **proxy mode** exists for people already running a LiteLLM proxy: set `"strategy": "proxy"` and
the extension delegates to LiteLLM's own router instead of deciding locally.

## Installation

```bash
git clone https://github.com/CoresoftHQ/pi-litellm-autorouter
cd pi-litellm-autorouter && npm install
pi -e ./src/index.ts
# then, inside pi:  /autoroute init
```

`/autoroute init` reads the models pi can reach, ranks them by price, and writes a starter config to
`~/.pi/agent/autorouter.json` (or `.pi/autorouter.json` for a project-local override). Edit the tiers from
there — price is only a proxy for capability, not the same thing.

Once it behaves, drop `-e` and point `settings.json` at the clone instead, so `/reload` picks up changes:

```json
{ "extensions": ["/path/to/pi-litellm-autorouter/src/index.ts"] }
```

### The minimum config

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "tiers": {
    "SIMPLE": "anthropic/claude-haiku-4-5",
    "MEDIUM": "anthropic/claude-sonnet-5",
    "COMPLEX": "anthropic/claude-sonnet-5",
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
  }
}
```

Model names are `provider/model-id`, exactly as pi's `/model` picker shows them.

### Example: heuristic classifier (default)

Local, sub-millisecond, no API calls. Good starting point — this is what `/autoroute init` writes.
Full file: [`examples/autorouter.heuristic.json`](examples/autorouter.heuristic.json).

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "tiers": {
    "SIMPLE": "anthropic/claude-haiku-4-5",
    "MEDIUM": "anthropic/claude-sonnet-5",
    "COMPLEX": "anthropic/claude-sonnet-5",
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
  },
  "keywordTierRules": [
    { "keywords": ["migration", "schema change", "security review"], "tier": "REASONING" },
    { "keywords": ["typo", "rename", "formatting"], "tier": "SIMPLE" }
  ],
  "escalationKeywords": ["PI ESCALATE"],
  "planMode": { "minTier": "COMPLEX" },
  "sessionAffinity": { "enabled": false, "ttlSeconds": 3600 }
}
```

### Example: LLM classifier

Asks a small model to classify each prompt instead of scoring it locally — costs an extra round-trip per
turn, but reads intent rather than keywords. `classificationRubric: "agentic"` is the coding-agent-calibrated
rubric described above; `classifierFallback: "heuristic"` means a timeout degrades to local
scoring instead of stalling the turn. Generate this shape with the `llm` flag:

```
/autoroute init llm
```

Full file: [`examples/autorouter.llm.json`](examples/autorouter.llm.json).

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "classifierType": "llm",
  "classifierLLMConfig": {
    "model": "anthropic/claude-haiku-4-5",
    "classificationRubric": "agentic",
    "timeoutMs": 3000
  },
  "classifierFallback": "heuristic",
  "classifierContextWindowSize": 3,
  "classifierContextPerTurnChars": 200,
  "classifierContextIncludeAssistantTurns": false,
  "tiers": {
    "SIMPLE": "anthropic/claude-haiku-4-5",
    "MEDIUM": "anthropic/claude-sonnet-5",
    "COMPLEX": "anthropic/claude-sonnet-5",
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
  }
}
```

Use the heuristic classifier by default; switch to `llm` if the scorer keeps misjudging your traffic and you
can tolerate the extra latency. `/autoroute init [provider] [project] [llm]` accepts all three flags together
— e.g. `/autoroute init anthropic project llm` scopes to one provider, writes to `.pi/autorouter.json`, and
configures the LLM classifier in one go. There's no separate command to flip classifier mode afterwards;
re-run `init`, or edit `classifierType` (and `classifierLLMConfig`) directly in the config file.

## Useful commands

| Command | Purpose |
|---|---|
| `/autoroute` | Show current classifier, last decision, tier, and score |
| `/autoroute init [provider] [project] [llm]` | Generate a config from the models pi can reach |
| `/autoroute explain` | Per-dimension breakdown of why a prompt got its tier |
| `/autoroute next <model>` | Force a model for the next prompt only |
| `/autoroute pin <model>` | Freeze on one model until unpinned |
| `/autoroute escalate` | Re-run the last prompt one tier up |
| `/autoroute off` / `on` | Toggle routing for the session |
| `--no-autoroute` | CLI flag to start a session with routing disabled |

## Development

```bash
npm run check      # tsc --noEmit && vitest run
npm test           # vitest run
```

## Prior art

Ported from [BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT, outside `enterprise/`) — the v2
router lives in `litellm/router_strategy/complexity_router/`. This is a port of the algorithms, rubrics, and
config shapes, not a redistribution of LiteLLM code. 

## License

MIT — see [LICENSE](LICENSE).
