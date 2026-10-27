# pi-litellm-autorouter

A [pi](https://pi.dev) extension that picks the cheapest model able to handle each prompt, automatically -
in-process, no proxy required.

## What it does

pi normally uses one model for a whole session, switched by hand with `/model`. That's wasteful: a trivial
question ("what does this env var do?") doesn't need the same model as a hard debugging session. This
extension adds a [virtual model](https://pi.dev/docs/latest/virtual-models), **`autoroute/auto`**. Select it
like any other model, and every prompt is classified before it runs and routed to the right tier - so cheap
prompts go to cheap models and hard prompts go to strong ones, automatically.

It's a TypeScript port of [LiteLLM's Auto Router v2](https://docs.litellm.ai/docs/proxy/auto_routing)
(`auto_router/complexity_router`), adapted to run inside pi instead of behind a separate LiteLLM proxy
process. That means no extra network hop, and pi itself knows which model answered: the footer shows the
routed model next to the selection (`auto • medium → claude-sonnet-5 • high`), each assistant message
records it, and `/session` lists the cost per physical model.

## How it works

While `autoroute/auto` is selected, pi asks the router for a physical model before every request. Each new
prompt goes through: **extract → classify → override → select → route**.

1. **Extract** - strip `<system-reminder>` blocks and pi's expanded `/skill:` blocks (harness plumbing, not
 something the user asked), pull out the current ask plus a few prior turns for context.
2. **Classify** - score the prompt into one of four tiers, `SIMPLE` → `MEDIUM` → `COMPLEX` → `REASONING`. Three classifiers are supported:
  - `heuristic` (default) - a local, sub-millisecond weighted scorer (code presence, reasoning markers,
   technical terms, length, etc.), no API call.
  - `llm` - a small model classifies the prompt against a rubric. The `agentic` rubric preset is the
  default here, calibrated so routine engineering work (installs, multi-file edits, standard debugging)
  lands at `MEDIUM` instead of being over-classified as top-tier, which is what a chat-tuned rubric does
  to agent traffic.
  - `decision` - a [decision model](#example-decision-model-classifier-jev-clef) such as TypeSafe's
  Jev or Cloudflare's Clef answers one typed `tier` question with probabilities, instead of generating
  text.
3. **Override** - a few signals outrank the classifier, in order: a one-shot `/autoroute next`, an
   `/autoroute pin` or `/autoroute off` / `--no-autoroute` escape hatch, a
   [question reply](#question-replies) (a turn that only answers a
   question the assistant asked keeps the model it was asked with), a session-affinity pin (reuse the
   first turn's model for the whole session, if enabled), `keywordTierRules` (keyword → tier, literal or
   [semantic](#semantic-keyword-matching)), and a plan-mode floor (routes at least to a configured tier
   while a plan-mode extension or sentinel is active). `escalationKeywords` can bump the result up exactly
   one tier — never down or to a caller-chosen model.
4. **Select** - a tier maps to one model or a pool of models (a uniformly random pick), or, with
   `adaptive: true`, a Thompson-sampled pick across the pools weighted by learned quality, price, and
   distance from the classified tier. See [Adaptive selection](#adaptive-selection).
5. **Route** - resolve the chosen model against pi's own model registry and credentials (falling back down
   the chain, then to `defaultModel`, if a model has no credentials), hand it to pi with its thinking level,
   and record the decision (visible via `/autoroute explain` and the footer status). The selection stays
   `autoroute/auto`; nothing calls `pi.setModel()`.

The model is chosen once per prompt and held for the whole agent run - it never switches mid-run. Only the
first request of a prompt is classified; everything else stays on the model already serving the
conversation, which keeps prompt caches and thinking signatures valid:

| Request | Routed to |
| --- | --- |
| A prompt you typed (or sent over RPC, or with `pi -p`) | Classified and routed |
| Tool-call continuations within the run | The model serving the run |
| Automatic retries, including after compaction for a context overflow | The model that failed |
| Steering and follow-up messages queued while the run is in flight | The model serving the run |
| Prompts other extensions send with `pi.sendUserMessage()` | The model serving the session |
| Compaction summaries and other requests outside the agent loop | The model serving the session, else `defaultModel` |

Routing rarely fails a prompt: a classifier error falls back per `classifierFallback`, an unusable model
falls down the chain, and when nothing in the chain is usable the request stays on the model already
serving the session. Only when there is no such model either does the request end with an error that
names the models it tried.

### Thinking levels

`autoroute/auto` offers every thinking level, from `off` to `max`. The level you select is the default for
tier entries that don't set their own: with `autoroute/auto • low`, a plain `"anthropic/claude-sonnet-5"`
entry runs at `low`, while `{ "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }` always runs at
`high`. pi clamps the result to what the routed model supports.

## Installation

Requires pi 0.99 or later (virtual models).

```bash
pi install npm:@coresofthq/pi-litellm-autorouter
# then, inside pi:  /autoroute init
```

`/autoroute init` reads the models pi can reach, ranks them by price, and writes a starter config to
`~/.pi/agent/autorouter.json` (or `.pi/autorouter.json` for a project-local override), then selects
`autoroute/auto` for the session. Edit the tiers from there - price is only a proxy for capability, not the
same thing.

To route every session, pick `autoroute/auto` in `/model` (or start pi with `--model autoroute/auto`), or
make it the default in `~/.pi/agent/settings.json`:

```json
{ "defaultProvider": "autoroute", "defaultModel": "auto" }
```

Selecting any other model takes the router out of the loop until you select `autoroute/auto` again (or run
`/autoroute on`). With `enabledModels` / `--models` scoping, add `autoroute/auto` to the list.

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

Model names are `provider/model-id`, exactly as pi's `/model` picker shows them. They must be physical models:
a virtual model, `autoroute/auto` included, cannot be routed to and is skipped. `defaultModel` takes the same
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

### Example: decision-model classifier (Jev, Clef)

A decision model doesn't chat: it answers typed questions about a request, with a probability for each
answer. With `classifierType: "decision"`, the router asks one System One `choice` question named `tier`
(each tier's meaning as the criteria) and sends the same request payload the `llm` classifier gets
(current ask, prior turns, conversation depth) as the state. The call goes through one of pi's own
classifier models, straight to the service that hosts it, with the credentials pi already has.

`model` names the classifier model as `provider/model-id`, like every other model in this config. Hosted
Jev, with `TYPESAFE_API_KEY` set (full file: [`examples/autorouter.jev.json`](examples/autorouter.jev.json)):

```json
{
  "classifierType": "decision",
  "decisionClassifierConfig": { "model": "typesafe/jev-latest", "timeoutMs": 3000 },
  "classifierFallback": "heuristic"
}
```

Cloudflare's Clef on Workers AI, with `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` set (full file:
[`examples/autorouter.clef.json`](examples/autorouter.clef.json)):

```json
{
  "classifierType": "decision",
  "decisionClassifierConfig": { "model": "cloudflare-workers-ai/@cf/cloudflare/clef", "timeoutMs": 15000 }
}
```

Any classifier model pi supports works, including ones a pi update or `models.json` adds later:

| Decision model | `model` |
| --- | --- |
| Jev, TypeSafe | `typesafe/jev-latest` |
| Jev, OpenRouter / Workers AI / Vercel AI Gateway / OpenCode | `openrouter/typesafe/jev-1.14`, `cloudflare-workers-ai/typesafe/jev`, `vercel-ai-gateway/typesafe-ai/jev`, `opencode/jev-1.13` |
| Clef, Workers AI | `cloudflare-workers-ai/@cf/cloudflare/clef`, `cloudflare-workers-ai/@cf/cloudflare/clef-flash`, `cloudflare-workers-ai/@cf/cloudflare/clef-omni` |
| Clef, OpenRouter | `openrouter/cloudflare/clef`, `openrouter/cloudflare/clef-flash`, `openrouter/cloudflare/clef-omni` |
| OpenAI's Decisions API | `openai/gpt-6-luna` (needs `OPENAI_API_KEY`, not a ChatGPT login) |

pi lists more; see pi's [classifier models](https://pi.dev/docs/latest/models#use-classifier-models).

| Key (`decisionClassifierConfig`) | Default | Purpose |
| --- | --- | --- |
| `model` | `typesafe/jev-latest` | The classifier model, `provider/model-id` |
| `timeoutMs` | `3000` | A hard deadline: the call is abandoned and `classifierFallback` applies |
| `instructions` | built in | Replaces the question's instructions; the tier criteria stay |
| `circuitBreakerEnabled` | `true` | One timeout skips the classifier, straight to `classifierFallback`, for the cooldown |
| `circuitBreakerCooldownSeconds` | `30` | After it, one prompt probes; success closes the circuit, failure restarts the cooldown |

There are no endpoint or key settings: pi owns the connection, so set keys with `/login` or the
provider's environment variable, and endpoints in `models.json`. Decisions are logged as
`cause=decision_classifier`, with the choice, its confidence and every tier's probability under `signals`;
`/autoroute explain` also names the model and the call's cost. A tier with no models counts as a
classifier failure, so `classifierFallback` decides instead. A `decisionClassifierConfig` left in place
under another `classifierType` is ignored with a warning.

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
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
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

`routed_model=(unchanged)` is literal: on this path no candidate is resolved and the request goes to the
model that asked the question, so it is the one decision that cannot move the model even by falling back.
The footer reads `autoroute: held (question reply)`. The tier it holds at is kept as router state on the
session branch, so it survives a resume, a fork, and compaction.

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


| Key             | Default         | Purpose                                                                                                                                                                                                                               |
| --------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`       | `true`          | `"questionReply": false` is shorthand for turning the whole thing off                                                                                                                                                                 |
| `toolNames`     | the three above | **Replaces** the list, so a custom question tool can be named on its own. Compared on letters and digits only, so `askUserQuestion` matches `ask_user_question`                                                                       |
| `maxReplyChars` | `120`           | Longest reply still treated as an answer. Deliberately tight: a new instruction is often shorter than it looks, and a generous cap would swallow exactly the turns that need their own decision. Naming an offered option bypasses it |


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
autoroute: routing decision cause=decision_classifier, tier=COMPLEX, signals=[typesafe-classifier:COMPLEX, typesafe-confidence=0.812000, tier-probability:SIMPLE=0.021000, tier-probability:MEDIUM=0.142000, tier-probability:COMPLEX=0.812000, tier-probability:REASONING=0.025000], routed_model=anthropic/claude-sonnet-5
autoroute: routing decision cause=session_affinity_pin, tier=MEDIUM, routed_model=anthropic/claude-sonnet-5
autoroute: routing decision cause=question_reply, tier=COMPLEX, signals=[question_reply (ask_user_question)], routed_model=(unchanged)
autoroute: routing decision cause=default_model_fallback, signals=[classifier timed out after 3000ms], routed_model=anthropic/claude-haiku-4-5, thinking=low
```

pi-specific fields (`thinking`, `keyword`, `escalated`, `plan_floored`, `fallback`, the adaptive phase under
`signals`) appear only when set, after upstream's fields.

- **Interactive mode**: the line is drawn dimly in the chat, just below the prompt it routed. It is a rendering
of the decision entry the extension already records in the session, so it costs no tokens, never reaches the
model, and is redrawn when a session is resumed. Press **ctrl+o** (expand tool output) to swap every line for
the full breakdown `/autoroute explain` would show.
- `**pi -p` / RPC**: the line goes to **stderr**, keeping stdout clean for the agent's output.
- `/autoroute log [n]` replays the session's decisions (or the last `n`) on demand.
- `"decisionLog": false` silences both; decisions are still recorded and `/autoroute log` still works.

## Useful commands


| Command                                                                                     | Purpose                                                                             |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `/autoroute`                                                                                | Show current classifier, last decision, tier, and score                             |
| `/autoroute init [provider] [project] [llm]`                                                | Generate a config from the models pi can reach                                      |
| `/autoroute explain`                                                                        | Per-dimension breakdown of why a prompt got its tier (and how the bandit scored it) |
| `/autoroute log [n]`                                                                        | Replay the session's routing decisions as log lines                                 |
| `/autoroute adaptive`                                                                       | The bandit's learned posteriors per request type and model                          |
| `/autoroute set [global\|project] <setting.path> <JSON value>`                             | Persist any router setting (project scope by default)                              |
| `/autoroute tier <tier> <model> [thinking-level]`                                            | Set a tier's model and optional thinking level                                     |
| `/autoroute next <model> [thinking-level]`                                                   | Force a model and optional thinking level for the next prompt only                 |
| `/autoroute pin [model]`                                                                     | Route every prompt to one model (default: the one serving the session) until `unpin` |
| `/autoroute off`                                                                            | Pause routing for the session: `autoroute/auto` keeps the model it is on            |
| `/autoroute on`                                                                             | Resume routing (clears `off`, `pin`, and `next`) and select `autoroute/auto`        |
| `--no-autoroute`                                                                            | CLI flag to start a session with routing paused                                     |


`set` accepts dotted config paths and JSON values, so it covers every `autorouter.json`
setting without restarting Pi. For example:

```text
/autoroute tier REASONING anthropic/claude-opus-5 high
/autoroute next anthropic/claude-sonnet-5 medium
/autoroute set tierBoundaries.medium 0.35
/autoroute set questionReply.enabled false
/autoroute set keywordTierRules [{"keywords":["migration"],"tier":"REASONING"}]
```

`set` writes `.pi/autorouter.json` by default; prefix it with `global` to write
`~/.pi/agent/autorouter.json`. Each save validates the entire configuration and takes effect immediately.

## Disabling the extension

Routing only happens while `autoroute/auto` is selected, so picking another model in `/model` is enough to
stop it. `/autoroute off` pauses routing for the current session without leaving `autoroute/auto`, and
`--no-autoroute` does the same for one launch. To stop the extension from loading at all - for instance
because you'd rather let an external LiteLLM proxy do the routing - use pi's own mechanisms:

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

### 1.3.0

- Allow configuring the extension without editing the json file

### 1.2.1

- Notify when one or several of the configured models are unavailable

### 1.2.0

- Better support for todo tools. I.e. use the best model based on the todo item content, instead of using the current one
- Improved handling off user question tools. Now it will choose the model based on the context of the task, instead of reclassifying based on an "ok" or "option 2" response.

### 1.1.0

This version is all about upstream feature parity.

- Decision log: The extension now writes what model was selected and why
- Custom Technical Keywords: Append to the built in technical keywords list
- Adaptive Mode: Pick tier/model from a configured pool
- Semantic Keyword Matching: Use an embedding model to determine keyword matches for tier selection.

### 1.0.2

- Removed proxy strategy, only local makes sense. if using external LiteLLM, just disable the extension.

### 1.0.1

- Initial NPM release

## Prior art

Ported from [BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT, outside `enterprise/`) - the v2
router lives in `litellm/router_strategy/complexity_router/`. This is a port of the algorithms, rubrics, and
config shapes, not a redistribution of LiteLLM code. 

## License

MIT - see [LICENSE](LICENSE).