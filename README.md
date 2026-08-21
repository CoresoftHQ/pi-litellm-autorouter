# pi-litellm-autorouter

A [pi](https://pi.dev) extension that lifts LiteLLM's **auto-router** out of the LiteLLM proxy and runs it
inside the pi coding agent, so each prompt is answered by the cheapest model that can actually handle it.

LiteLLM already solves "which model should serve this request?" behind an OpenAI-compatible proxy. pi already
solves "run an agent loop against any provider". This extension is the missing seam: it ports LiteLLM's routing
strategies to TypeScript and wires them to `pi.setModel()`, so routing happens in the agent — with pi's own
model registry, credentials, and thinking levels — instead of behind a separate proxy hop.

> **Status: design stage.** This repository currently contains the specification (this file) and the
> implementation plan ([`PLAN.md`](PLAN.md)). No extension code has been written yet. Everything below marked
> "planned" describes the intended behaviour, not shipped behaviour.

---

## Table of contents

- [Why](#why)
- [How LiteLLM's auto-router works](#how-litellms-auto-router-works)
- [What this extension does](#what-this-extension-does)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Commands and flags](#commands-and-flags)
- [Proxy mode](#proxy-mode)
- [Design constraints](#design-constraints)
- [Prior art and attribution](#prior-art-and-attribution)

---

## Why

A coding-agent session is not uniform. Inside one session you get:

| Prompt | Model you actually need |
|---|---|
| "what does this env var do?" | a small, fast model |
| "rename `foo` to `bar` across the repo" | a mid-tier model |
| "why does this race condition only fire under load?" | a frontier reasoning model |

Today pi picks one model per session and you change it by hand with `/model` or `Ctrl+P`. That means you either
overpay on trivia or underpower the hard turns. The auto-router makes that choice per prompt.

The alternative — running a LiteLLM proxy in front of pi and pointing pi's provider at it — works, and this
extension supports it as [proxy mode](#proxy-mode). But it costs you a process to run, a network hop per turn,
and it hides the decision: pi's UI still shows the proxy's single logical model, spend attribution collapses,
and pi's thinking-level and model-capability logic can't see the model that was actually chosen. Running the
router in-process fixes all four.

## How LiteLLM's auto-router works

Reference: [LiteLLM auto-routing docs](https://docs.litellm.ai/docs/proxy/auto_routing) ·
[source](https://github.com/BerriAI/litellm) (`litellm/router_strategy/`).

In LiteLLM, an auto-router is a *virtual deployment* in `model_list`. Clients call it by name like any other
model; a **pre-routing hook** (`async_pre_routing_hook`) runs before deployment selection, inspects the
messages, and swaps in the real target model. If the hook returns nothing usable, the request falls through to
a configured default model.

LiteLLM ships four strategies, distinguished by the `auto_router/` model prefix:

### 1. Semantic (`auto_router/<name>`)

The original strategy, built on [`semantic-router`](https://github.com/aurelio-labs/semantic-router). You
define named routes, each with example *utterances*; the incoming prompt is embedded and cosine-compared
against them, and the best route above its `score_threshold` wins. **The route name is the model name.**

```yaml
model_list:
  - model_name: auto-router
    litellm_params:
      model: auto_router/my-router
      auto_router_config_path: router.json
      auto_router_default_model: gpt-4o-mini
      auto_router_embedding_model: text-embedding-3-large
```

```json
{
  "encoder_type": "openai",
  "encoder_name": "text-embedding-3-large",
  "routes": [
    {
      "name": "litellm-claude-35",
      "utterances": ["how to code a program in [language]"],
      "description": "coding assistant",
      "score_threshold": 0.5,
      "metadata": {}
    }
  ]
}
```

Notable implementation details, taken from `litellm/router_strategy/auto_router/auto_router.py`:

- Only the **last user message** is embedded (`_extract_text_from_messages`), walking backwards past
  assistant/tool messages and flattening multimodal content blocks down to their `text` parts.
- The prompt is truncated to `auto_router_max_input_chars` before embedding, so a small embedding context
  window can't fail a request destined for a large model.
- Embedding failures are **swallowed**, not raised: a timeout, context-limit, or provider error logs a warning
  and falls back to `auto_router_default_model`. Choosing a model is a routing decision and must never fail the
  user's request.
- The `SemanticRouter` layer is built lazily on first request and cached.

### 2. Complexity (`auto_router/complexity_router`)

Rule-based, zero API calls, sub-millisecond. It scores the prompt across seven weighted dimensions —
`tokenCount` (0.10), `codePresence` (0.30), `reasoningMarkers` (0.25), `technicalTerms` (0.25),
`simpleIndicators` (0.05, negative), `multiStepPatterns` (0.03), `questionComplexity` (0.02) — and maps the
weighted sum onto four tiers via configurable boundaries:

| Tier | Score | Boundary key below it |
|---|---|---|
| `SIMPLE` | < 0.15 | — |
| `MEDIUM` | 0.15 – 0.35 | `simple_medium` |
| `COMPLEX` | 0.35 – 0.60 | `medium_complex` |
| `REASONING` | > 0.60 | `complex_reasoning` |

```yaml
litellm_params:
  model: auto_router/complexity_router
  complexity_router_config:
    tiers:
      SIMPLE: gpt-4o-mini
      MEDIUM: gpt-4o
      COMPLEX: claude-sonnet-4
      REASONING: o1-preview
```

Tiers may also carry per-tier request-parameter overrides (e.g. pinning `reasoning_effort` on `REASONING`),
and `tier_labels` renames tiers for display while config keys stay canonical.

### 3. Quality (`auto_router/quality_router`)

Decouples "how hard is this?" from "which model?". Deployments declare a `quality_tier` in
`model_info.litellm_routing_preferences`; the router classifies complexity, maps the tier to a required
quality level via `complexity_to_quality`, and picks among models that clear it — ties broken by highest
quality tier, then cheapest `input_cost_per_token`. Per-deployment `keywords` can force a route.

### 4. Adaptive (`auto_router/adaptive_router`)

A contextual bandit. Each candidate model carries a Beta posterior seeded from its declared quality tier
(cold-start prior mass 10.0, base weights `{1: 0.3, 2: 0.5, 3: 0.7}`), scored against a
quality/cost trade-off (default weights 0.7 / 0.3). Post-call signal detectors — misalignment, stagnation,
tool-call looping — update the posterior, and a 24h owner cache keeps one conversation from flooding the
bandit with correlated updates. LiteLLM's own constants are marked `UNVALIDATED`; treat it as experimental.

## What this extension does

It re-implements the routing *decision* in TypeScript and applies it through pi's public extension API. The
mapping:

| LiteLLM concept | pi equivalent in this extension |
|---|---|
| Virtual deployment in `model_list` | the extension itself — no fake model appears in pi's picker |
| `async_pre_routing_hook` | an `input` / `before_agent_start` handler |
| Route name → deployment lookup | `ctx.modelRegistry.find(provider, id)` + `pi.setModel(model)` |
| `auto_router_default_model` | `defaultModel` in extension config; also the fallback on any router error |
| `auto_router_embedding_model` | an embedding provider configured in extension config (or a local model) |
| Model catalogue (`model_list`) | pi's model registry, narrowed by `ctx.scopedModels` |
| Spend logs / `routing_decision` | `pi.appendEntry("autoroute-decision", …)` + a footer status via `ctx.ui.setStatus()` |
| Per-tier `reasoning_effort` override | `pi.setThinkingLevel()` |

Planned behaviour per prompt:

1. The user submits a prompt. The extension's input handler sees the raw text.
2. Escape hatches run first: an explicit `/model` pin, a `pi.registerFlag()` opt-out, or a per-prompt override
   prefix all bypass routing.
3. The configured strategy classifies the prompt.
4. The winning route resolves to a model in pi's registry. If it doesn't resolve, or has no credentials
   (`pi.setModel()` returns `false`), the router falls back to the next candidate and then to `defaultModel`.
5. `pi.setModel()` — and optionally `pi.setThinkingLevel()` — is applied, the decision is recorded, and the
   turn proceeds normally.

Strategies planned for v1: **complexity** (no API calls, the natural default for an agent) and **semantic**.
Quality is a thin layer on complexity and follows; adaptive is explicitly out of scope for v1.

### What it deliberately does not do

- **Route mid-turn.** The model is chosen once per user prompt and held for that whole agent run, including
  every tool call in it. Swapping models between tool calls breaks prompt caching and confuses the
  conversation's provider-specific message shapes.
- **Run a LiteLLM proxy for you.** Local mode has no Python dependency at all.
- **Override an explicit choice.** If you picked a model by hand this session, that wins until you clear it.

## Architecture

```
user prompt
   │
   ├─► pi checks extension commands (/autoroute … handled here, routing skipped)
   │
   ├─► input event ───────────────────────────────────────────┐
   │      │                                                   │
   │      ├─ escape hatches (pinned model? --no-autoroute?)   │
   │      │                                                   │
   │      ├─ strategy.classify(text) ──► RouteDecision        │
   │      │     complexity: local weighted scoring (<1ms)     │
   │      │     semantic:   embed + cosine vs. utterances     │
   │      │                                                   │
   │      ├─ resolve → ctx.modelRegistry.find(provider, id)   │
   │      ├─ apply   → pi.setModel() / pi.setThinkingLevel()  │
   │      └─ record  → appendEntry + ctx.ui.setStatus()       │
   │                                                          │
   │      any throw ─► warn + keep current/default model ─────┤
   │                                                          │
   └─► before_agent_start ──► agent loop runs on chosen model ┘
```

Planned layout — a directory extension with a `package.json`, since the semantic strategy needs a dependency
or two:

```
pi-litellm-autorouter/
├── package.json            # "pi": { "extensions": ["./src/index.ts"] }
├── src/
│   ├── index.ts            # default export: (pi: ExtensionAPI) => void
│   ├── config.ts           # load + validate config, typebox schemas
│   ├── resolve.ts          # route name/tier → pi model, with fallback chain
│   ├── decision.ts         # RouteDecision record, persistence, status line
│   ├── commands.ts         # /autoroute …
│   └── strategies/
│       ├── types.ts        # Strategy interface
│       ├── complexity.ts   # port of complexity_router
│       ├── semantic.ts     # port of auto_router (embeddings + cosine)
│       └── proxy.ts        # delegate to a running LiteLLM proxy
└── test/
```

## Installation

*(planned)*

As a pi package, via `settings.json`:

```json
{
  "packages": ["npm:@coresoft/pi-litellm-autorouter@1"]
}
```

Or from a clone, for development:

```bash
git clone https://github.com/CoresoftHQ/pi-litellm-autorouter
cd pi-litellm-autorouter && npm install
pi -e ./src/index.ts
```

Extensions in `~/.pi/agent/extensions/` (global) or `.pi/extensions/` (project-local) are auto-discovered and
hot-reloadable with `/reload`; `pi -e` is for quick tests. Note that extensions run with your full system
permissions.

## Configuration

*(planned)* — `~/.pi/agent/autorouter.json`, overridable per project at `.pi/autorouter.json`.

### Complexity strategy

```json
{
  "strategy": "complexity",
  "defaultModel": "anthropic/claude-haiku-4-5",
  "complexity": {
    "tiers": {
      "SIMPLE":    "anthropic/claude-haiku-4-5",
      "MEDIUM":    "anthropic/claude-sonnet-5",
      "COMPLEX":   "anthropic/claude-sonnet-5",
      "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
    },
    "tierBoundaries": { "simple_medium": 0.15, "medium_complex": 0.35, "complex_reasoning": 0.60 }
  }
}
```

Weights and boundaries keep LiteLLM's defaults and its canonical key names (`simple_medium`,
`medium_complex`, `complex_reasoning`), so a config can be moved between the two systems without translation.

### Semantic strategy

```json
{
  "strategy": "semantic",
  "defaultModel": "anthropic/claude-haiku-4-5",
  "semantic": {
    "encoder": { "provider": "openai", "model": "text-embedding-3-small" },
    "maxInputChars": 2000,
    "routes": [
      {
        "name": "anthropic/claude-opus-5",
        "description": "hard debugging and architecture",
        "utterances": ["why does this deadlock under load", "design a migration path for [system]"],
        "score_threshold": 0.5
      },
      {
        "name": "anthropic/claude-haiku-4-5",
        "description": "quick lookups",
        "utterances": ["what does [flag] do", "where is [symbol] defined"],
        "score_threshold": 0.5
      }
    ]
  }
}
```

The route file is intentionally shape-compatible with LiteLLM's `auto_router_config.json` — same `routes`
array, same `name`/`utterances`/`description`/`score_threshold` fields — so an existing router config drops in,
with route names read as `provider/model` identifiers instead of LiteLLM deployment names.

## Commands and flags

*(planned)*

| | Purpose |
|---|---|
| `/autoroute` | Show the current strategy, the last decision, and its score |
| `/autoroute off` / `on` | Toggle routing for the session |
| `/autoroute pin <model>` | Freeze on one model until unpinned |
| `/autoroute explain` | Per-dimension score breakdown for the last prompt |
| `--no-autoroute` | CLI flag (`pi.registerFlag()`) to start with routing disabled |

The active route is surfaced in the footer via `ctx.ui.setStatus()`, and every decision is persisted with
`pi.appendEntry()` — custom entries don't enter LLM context, so the routing log costs no tokens.

## Proxy mode

If you already run a LiteLLM proxy, set `"strategy": "proxy"` and the extension stops deciding anything
locally: it registers the proxy through `pi.registerProvider()` and lets LiteLLM's own auto-router — including
the quality and adaptive strategies this extension does not port — pick the model.

```json
{
  "strategy": "proxy",
  "proxy": { "baseUrl": "http://localhost:4000", "apiKey": "$LITELLM_MASTER_KEY", "model": "auto-router" }
}
```

The trade-off is the one described in [Why](#why): you get every LiteLLM strategy and centralised spend
tracking, but pi sees one logical model and the chosen model is only visible in LiteLLM's response headers
(e.g. `x-litellm-adaptive-router-model`), which this mode surfaces in the footer where the provider exposes
response headers.

## Design constraints

These fall out of the two systems and shape the whole implementation:

1. **Routing must never fail a prompt.** Every classification path is wrapped; any error logs and falls back to
   `defaultModel`. This is LiteLLM's own rule and it matters more in an interactive agent.
2. **Latency is on the critical path.** Complexity routing is local and sub-millisecond. Semantic routing costs
   an embedding round-trip per prompt, so it caches embeddings by prompt hash and precomputes utterance
   vectors at load.
3. **A model is only usable if pi has credentials for it.** `pi.setModel()` returns `false` when there's no API
   key; the resolver treats that as "not a candidate" and moves down the fallback chain.
4. **The user always outranks the router.** An explicit `/model` selection or a pin wins.
5. **One model per agent run.** See [what it does not do](#what-it-deliberately-does-not-do).

## Prior art and attribution

- **LiteLLM** — [BerriAI/litellm](https://github.com/BerriAI/litellm), MIT (outside `enterprise/`). The routing
  strategies ported here live in `litellm/router_strategy/`. This project is a clean-room-in-spirit port of the
  *algorithms and config shapes*, not a redistribution of LiteLLM code.
- **semantic-router** — [aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router), the
  embedding-and-threshold library behind LiteLLM's semantic strategy.
- **pi** — [pi.dev](https://pi.dev), [extension docs](https://pi.dev/docs/latest/extensions).

Neither LiteLLM nor pi endorses this project.

## License

MIT — see [LICENSE](LICENSE).
