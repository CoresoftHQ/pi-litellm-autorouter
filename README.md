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
   escape hatch, a [question reply](#question-replies) (a turn that only answers a question the assistant
   asked keeps the model it was asked with), a session affinity pin (reuse the first turn's model for the
   whole session, if enabled),
   `keywordTierRules` (keyword → tier, literal or [semantic](#semantic-keyword-matching)), and a plan-mode floor (routes at least to a
   configured tier while a plan-mode extension or sentinel is active). `escalation_keywords` can bump the
   result up exactly one tier - never down, never a caller-chosen model.
4. **Select** - a tier maps to one model or a pool of models (a uniformly random pick), or,
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
  "defaultModel": { "model": "anthropic/claude-haiku-4-5", "thinkingLevel": "low" },
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
  "sessionAffinity": { "enabled": false, "ttlSeconds": 3600 },
  "questionReply": { "enabled": true }
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
  "defaultModel": { "model": "anthropic/claude-haiku-4-5", "thinkingLevel": "low" },
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

### Example: mixing providers

Tiers are independent, so each can name whichever provider does that job best - and the classifier can be a
third. Here OpenAI covers the cheap tiers, Anthropic the hard ones, and a small Mistral model classifies.
Full file: [`examples/autorouter.multillm.json`](examples/autorouter.multillm.json).

```json
{
  "defaultModel": { "model": "openai/gpt-5.6-terra", "thinkingLevel": "medium" },
  "classifierType": "llm",
  "classifierLLMConfig": {
    "model": "mistral/mistral-small-2603",
    "classificationRubric": "agentic",
    "timeoutMs": 2000
  },
  "classifierFallback": "heuristic",
  "classifierContextWindowSize": 3,
  "classifierContextPerTurnChars": 200,
  "classifierContextIncludeAssistantTurns": false,
  "tiers": {
    "SIMPLE": { "model": "openai/gpt-5.6-luna", "thinkingLevel": "low" },
    "MEDIUM": { "model": "openai/gpt-5.6-terra", "thinkingLevel": "medium" },
    "COMPLEX": { "model": "anthropic/claude-sonnet-5", "thinkingLevel": "high" },
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "xhigh" }
  }
}
```

Every model named must be one pi can reach with its own credentials; a tier whose provider has no key falls
down the chain to the next usable candidate, and the decision log says so under `fallback=`.

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

### Question replies

On by default. When the assistant asks something with a question tool (`ask_user_question`, `ask_question`,
`ask_followup_question`) and the next turn only *answers* it, the router holds the session's model instead of
classifying the answer.

Without this, a mid-task question is a downgrade trap. The prompt that started the work classified `COMPLEX`
and put the session on Sonnet; the assistant asks "Postgres or SQLite?"; the user types `ok`. Scored on its
own, `ok` is `SIMPLE` - so the session drops to Haiku, and Haiku is the model that then has to act on the
answer. The turn carries no information about the work, so the honest thing is not to score it at all:

```
autoroute: routing decision cause=heuristic_scorer, tier=COMPLEX, routed_model=anthropic/claude-sonnet-5
autoroute: routing decision cause=question_reply, tier=COMPLEX, signals=[question_reply (ask_user_question)], routed_model=(unchanged)
```

`routed_model=(unchanged)` is literal: on this path no candidate is resolved and `pi.setModel()` is never
called, so it is the one decision that cannot move the model even by falling back. The footer reads
`autoroute: held (question reply)`.

A turn has to pass **both** tests to be held:

1. **A question is open** - the newest assistant message carries a question tool call. A tool that was
   answered inline and acted on is followed by further assistant messages, so it no longer qualifies; this
   can never fire on a terse prompt at the start of a session.
2. **The reply reads as an answer** - it names one of the offered options (conclusive, at any length), or it
   is short and does not open a unit of work. `ok`, `yes please`, `option 2`, `the first one`, `up to you`
   and `use postgres` are answers. `now refactor the whole auth layer to use JWT` is not: it leads with a
   task verb, so it is classified and routed like any other prompt even though a question was open.

Escalation still works from the held tier - `PI ESCALATE` in a reply raises one tier above the tier the
session was already on, not one tier above the reply's own trivial classification. A plan-mode floor entered
while the question was open also still applies, since a floor can only raise.

```json
{
  "questionReply": {
    "enabled": true,
    "toolNames": ["ask_user_question", "ask_question", "ask_followup_question"],
    "maxReplyChars": 120
  }
}
```

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `true` | `"questionReply": false` is shorthand for turning the whole thing off |
| `toolNames` | the three above | **Replaces** the list, so a custom question tool can be named on its own. Compared on letters and digits only, so `askUserQuestion` matches `ask_user_question` |
| `maxReplyChars` | `120` | Longest reply still treated as an answer. Deliberately tight: a new instruction is often shorter than it looks, and a generous cap would swallow exactly the turns that need their own decision. Naming an offered option bypasses it |

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

### Decision log

On by default. Every routing decision prints one line in the shape of upstream's
[decision log](https://docs.litellm.ai/docs/proxy/auto_routing#decision-log), so `cause=` searches work the same
against pi's console and a LiteLLM proxy log:

```
autoroute: routing decision cause=heuristic_scorer, tier=SIMPLE, score=-0.150, signals=[short (7 tokens), simple (what is)], routed_model=anthropic/claude-haiku-4-5
autoroute: routing decision cause=literal_keyword_match, tier=REASONING, routed_model=anthropic/claude-opus-5, thinking=high, keyword="migration"
autoroute: routing decision cause=semantic_keyword_match, tier=REASONING, signals=[semantic_match (0.81 ≈ "kubernetes deployment")], routed_model=anthropic/claude-opus-5, thinking=high
autoroute: routing decision cause=llm_classifier, tier=COMPLEX, signals=[llm_classifier], routed_model=anthropic/claude-sonnet-5
autoroute: routing decision cause=session_affinity_pin, tier=MEDIUM, routed_model=anthropic/claude-sonnet-5
autoroute: routing decision cause=question_reply, tier=COMPLEX, signals=[question_reply (ask_user_question)], routed_model=(unchanged)
autoroute: routing decision cause=default_model_fallback, signals=[classifier timed out after 3000ms], routed_model=anthropic/claude-haiku-4-5, thinking=low
```

pi-specific fields (`thinking`, `keyword`, `escalated`, `plan_floored`, `fallback`, the adaptive phase under
`signals`) appear only when set, after upstream's fields.

- **Interactive mode**: the line is drawn dimly in the chat, just above the prompt it routed. It is a rendering
  of the decision entry the extension already records in the session, so it costs no tokens, never reaches the
  model, and is redrawn when a session is resumed. Press **ctrl+o** (expand tool output) to swap every line for
  the full breakdown `/autoroute explain` would show.
- **`pi -p` / RPC**: the line goes to **stderr**, keeping stdout clean for the agent's output.
- `/autoroute log [n]` replays the session's decisions (or the last `n`) on demand.
- `"decisionLog": false` silences both; decisions are still recorded and `/autoroute log` still works.

## Useful commands

| Command | Purpose |
|---|---|
| `/autoroute` | Show current classifier, last decision, tier, and score |
| `/autoroute init [provider] [project] [llm]` | Generate a config from the models pi can reach |
| `/autoroute explain` | Per-dimension breakdown of why a prompt got its tier (and how the bandit scored it) |
| `/autoroute log [n]` | Replay the session's routing decisions as log lines |
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

## Changelog
### Unreleased
* Question replies: answering a question the assistant asked with `ask_user_question` ("ok", "option 2")
  no longer re-routes the turn, so a mid-task question can't downgrade the model that has to act on the
  answer. See [Question replies](#question-replies).

### 1.1.0
This version is all about upstream feature parity.

* Decision log: The extension now writes what model was selected and why
* Custom Technical Keywords: Append to the built in technical keywords list
* Adaptive Mode: Pick tier/model from a configured pool
* Semantic Keyword Matching: Use an embedding model to determine keyword matches for tier selection.

### 1.0.2
* Removed proxy strategy, only local makes sense. if using external LiteLLM, just disable the extension.

### 1.0.1
* Initial NPM release

## Prior art

Ported from [BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT, outside `enterprise/`) - the v2
router lives in `litellm/router_strategy/complexity_router/`. This is a port of the algorithms, rubrics, and
config shapes, not a redistribution of LiteLLM code. 

## License

MIT - see [LICENSE](LICENSE).
