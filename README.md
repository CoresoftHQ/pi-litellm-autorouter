# pi-litellm-autorouter

A [pi](https://pi.dev) extension that picks the cheapest model able to handle each prompt, automatically -
in-process, no proxy required.

## What it does

pi normally uses one model for a whole session, switched by hand with `/model`. That's wasteful: a trivial
question ("what does this env var do?") doesn't need the same model as a hard debugging session. This
extension classifies every prompt before it runs and calls `pi.setModel()` to route it to the right tier -
so cheap prompts go to cheap models and hard prompts go to strong ones, automatically.

It's a TypeScript port of [LiteLLM's Auto Router v2](https://docs.litellm.ai/docs/proxy/auto_routing)
(`auto_router/complexity_router`), adapted to run inside pi's extension API instead of behind a separate
LiteLLM proxy process. That means no extra network hop, and pi's UI/thinking-level/credential logic can see
the model that was actually picked.

## How it works

Each user prompt goes through: **extract → classify → override → select → apply**.

1. **Extract** - strip `<system-reminder>` blocks (harness plumbing, not something the user asked), pull out
   the current ask plus a few prior turns for context.
2. **Classify** - score the prompt into one of four tiers, `SIMPLE` → `MEDIUM` → `COMPLEX` → `REASONING`.
   Two classifiers are supported:
   - `heuristic` (default) - a local, sub-millisecond weighted scorer (code presence, reasoning markers,
     technical terms, length, etc.), no API call.
   - `llm` - a small model classifies the prompt against a rubric. The `agentic` rubric preset is the
     default here, calibrated so routine engineering work (installs, multi-file edits, standard debugging)
     lands at `MEDIUM` instead of being over-classified as top-tier, which is what a chat-tuned rubric does
     to agent traffic.
3. **Override** - a few signals outrank the classifier, in order: an explicit `/model` pin or `--no-autoroute`
   escape hatch, a session affinity pin (reuse the first turn's model for the whole session, if enabled),
   `keywordTierRules` (keyword → tier, literal or [semantic](#semantic-keyword-matching)), and a plan-mode floor (routes at least to a
   configured tier while a plan-mode extension or sentinel is active). `escalation_keywords` can bump the
   result up exactly one tier - never down, never a caller-chosen model.
4. **Select** - a tier maps to one model or a pool of models (a uniformly random pick, as upstream), or,
   with `adaptive: true`,
   a Thompson-sampled pick across the pools weighted by learned quality, price, and distance from the
   classified tier. See [Adaptive selection](#adaptive-selection).
5. **Apply** - resolve the chosen model against pi's own model registry and credentials, call
   `pi.setModel()` (falling back down the chain, then to `defaultModel`, if a model has no credentials), then
   `pi.setThinkingLevel()`, and record the decision (visible via `/autoroute explain` and the footer status).

Routing never fails a prompt: any classifier error or unresolved model falls back to `defaultModel`, and the
model is chosen once per prompt and held for the whole agent turn (including all its tool calls) - it never
switches mid-turn.

## Installation

```bash
pi install npm:@coresofthq/pi-litellm-autorouter
# then, inside pi:  /autoroute init
```

`/autoroute init` reads the models pi can reach, ranks them by price, and writes a starter config to
`~/.pi/agent/autorouter.json` (or `.pi/autorouter.json` for a project-local override). Edit the tiers from
there - price is only a proxy for capability, not the same thing.

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

Model names are `provider/model-id`, exactly as pi's `/model` picker shows them. `defaultModel` takes the same
shape as a tier entry, so it can carry a thinking level too:

```json
{ "defaultModel": { "model": "anthropic/claude-haiku-4-5", "thinkingLevel": "low" } }
```

### Example: heuristic classifier (default)

Local, sub-millisecond, no API calls. Good starting point - this is what `/autoroute init` writes.
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

Asks a small model to classify each prompt instead of scoring it locally - costs an extra round-trip per
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
- e.g. `/autoroute init anthropic project llm` scopes to one provider, writes to `.pi/autorouter.json`, and
configures the LLM classifier in one go. There's no separate command to flip classifier mode afterwards;
re-run `init`, or edit `classifierType` (and `classifierLLMConfig`) directly in the config file.

## Options

All keys go in the same `autorouter.json`. Names are the camelCase spelling of LiteLLM's
`complexity_router_config` keys, so a config can move between the two systems.

### Domain keywords for the heuristic scorer

The scorer's `technicalTerms` dimension counts hits against a built-in list of ~80 terms. That list is
calibrated against the dimension's thresholds and **cannot be replaced** (a `technicalKeywords` key is
rejected), but it can be extended:

```json
{
  "customTechnicalKeywords": ["kafka", "redis", "postgresql", "udp", "dns"]
}
```

Entries are appended in order and deduplicated case-insensitively against the built-in list, so listing
`"TCP"` when `"tcp"` is already built in changes nothing. Mirrors upstream's `custom_technical_keywords`.

### Semantic keyword matching

By default `keywordTierRules` match literally (word-bounded, case-insensitive). With
`semanticKeywordMatching: true` they match by embedding similarity instead, so a paraphrase with no keyword
in it ("help me roll out my k8s cluster") still hits a rule for `"kubernetes deployment"`. Port of upstream's
`semantic_keyword_matching` / `embedding_model` / `match_threshold`.

```json
{
  "keywordTierRules": [
    { "keywords": ["kubernetes deployment", "container orchestration"], "tier": "REASONING" },
    { "keywords": ["hello", "thanks"], "tier": "SIMPLE" }
  ],
  "semanticKeywordMatching": true,
  "embeddingModel": "voyage/voyage-3-5",
  "matchThreshold": 0.5,
  "embeddingEndpoint": { "apiKeyEnv": "VOYAGE_API_KEY", "timeoutMs": 3000 }
}
```

- One route per tier, that tier's keywords as its utterances, `max` aggregation: a prompt matches a tier when
  it is close to *any* of the tier's keywords, and the closest tier wins if its similarity is at least
  `matchThreshold`. Keywords are embedded once per session; only the prompt is embedded per turn.
- With semantic matching on, literal matching is **not** consulted (as upstream). An embedding failure -
  timeout, bad key, endpoint down - yields no override and the prompt falls through to the classifier; the
  decision records why under `signals`.
- pi has no embeddings API, so the call goes straight to an OpenAI-compatible `/embeddings` endpoint.
  `embeddingModel` is `provider/model-id`; the base URL comes from `embeddingEndpoint.baseUrl`, else the provider
  pi knows by that name, else a built-in table (`voyage`, `openai`, `mistral`, `openrouter`, `together`,
  `fireworks`, `google`). The key comes from `embeddingEndpoint.apiKeyEnv`, else pi's key for the provider,
  else `<PROVIDER>_API_KEY`.
- Requires `embeddingModel` and at least one rule; `matchThreshold` is in `[0, 1]`.

### Adaptive selection

Off by default. With `adaptive: true`, a tier's pool is no longer sampled uniformly: every model gets a
Beta posterior per request type (`code_generation`, `code_understanding`, `technical_design`,
`analytical_reasoning`, `writing`, `factual_lookup`, `general`), and each prompt draws one Thompson sample
per candidate and scores it as

```
quality_weight · sample + cost_weight · normalised_price − tier_distance_penalty · |tier − classified tier|
```

Port of upstream's `adaptive` / `adaptive_weights` / `tier_distance_penalty` / `adaptive_eligible`, including
its cold-start phase (unobserved models in the classified tier are tried uniformly first) and its defaults.

```json
{
  "adaptive": true,
  "adaptiveWeights": { "quality": 0.3, "cost": 0.7 },
  "tierDistancePenalty": 0.5,
  "adaptiveEligible": "all",
  "tiers": {
    "SIMPLE": [{ "model": "anthropic/claude-haiku-4-5", "qualityTier": 1, "strengths": ["factual_lookup"] }],
    "MEDIUM": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
    "COMPLEX": [{ "model": "anthropic/claude-sonnet-5", "qualityTier": 2, "strengths": ["code_generation"] }],
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high", "qualityTier": 3 }
  }
}
```

- `adaptiveWeights` must sum to 1. The upstream complexity-router default leans on cost (0.3 / 0.7).
- `adaptiveEligible: "all"` scores every pool model with the distance penalty, so a cheap model with a strong
  posterior can win a `COMPLEX` prompt (a *soft* floor). `"classified_tier"` samples only inside the classified
  tier's pool. The plan-mode floor is always hard: candidates below it are excluded outright.
- `qualityTier` (1–3, default 2) and `strengths` on a tier entry set the cold-start prior, mirroring upstream's
  `model_info.adaptive_router_preferences`. Prices come from pi's model registry.
- Only the classifier path is adaptive, as upstream: `keywordTierRules`, a session-affinity pin and the plan-mode
  shortcut still take the plain uniform pool pick.

The bandit learns from what happens after each pick, using upstream's signal detectors on pi's `agent_end`
event: a rephrase or a "forget it" in the next prompt counts against the model that produced the previous
reply; a tool-call loop, a near-duplicate reply, or an errored tool result counts against the model that
produced this one; a "thanks" (after the third turn) counts for it. Posteriors persist in
`~/.pi/agent/autorouter-adaptive.json`; rows for models no longer in any pool are dropped on load. Inspect
them with `/autoroute adaptive`, and see how the last pick was scored with `/autoroute explain`.

## Useful commands

| Command | Purpose |
|---|---|
| `/autoroute` | Show current classifier, last decision, tier, and score |
| `/autoroute init [provider] [project] [llm]` | Generate a config from the models pi can reach |
| `/autoroute explain` | Per-dimension breakdown of why a prompt got its tier (and how the bandit scored it) |
| `/autoroute adaptive` | The bandit's learned posteriors per request type and model |
| `/autoroute next <model>` | Force a model for the next prompt only |
| `/autoroute pin <model>` | Freeze on one model until unpinned |
| `/autoroute escalate` | Re-run the last prompt one tier up |
| `/autoroute off` / `on` | Toggle routing for the session |
| `--no-autoroute` | CLI flag to start a session with routing disabled |

## Disabling the extension

`/autoroute off` only pauses routing for the current session, and `--no-autoroute` only for one launch. To
stop the extension from loading at all - for instance because you'd rather let an external LiteLLM proxy
do the routing - use pi's own mechanisms:

- **Toggle it interactively**: run `pi config`, find the extension, and disable it. Tab switches between global
  (`~/.pi/agent/settings.json`) and project-local (`.pi/settings.json`) scope; `pi config -l` starts in the
  project scope.
- **Keep the package installed but load nothing from it**: use the object form in `settings.json`, which
  filters what a package contributes:

  ```json
  {
    "packages": [
      { "source": "npm:@coresofthq/pi-litellm-autorouter", "extensions": [] }
    ]
  }
  ```

  A project `.pi/settings.json` entry overrides the global one, so you can disable it for a single repo and
  keep it everywhere else.
- **Remove it entirely**: `pi remove npm:@coresofthq/pi-litellm-autorouter`.

Your `autorouter.json` is left untouched by all of these, so re-enabling picks up where you left off.

## Development

```bash
npm run check      # tsc --noEmit && vitest run
npm test           # vitest run
```

## Prior art

Ported from [BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT, outside `enterprise/`) - the v2
router lives in `litellm/router_strategy/complexity_router/`. This is a port of the algorithms, rubrics, and
config shapes, not a redistribution of LiteLLM code. 

## License

MIT - see [LICENSE](LICENSE).
