# pi-litellm-autorouter

A [pi](https://pi.dev) extension that lifts LiteLLM's **Auto Router v2** out of the LiteLLM proxy and runs it
inside the pi coding agent, so each prompt is answered by the cheapest model that can actually handle it.

LiteLLM already solves "which model should serve this request?" behind an OpenAI-compatible proxy. pi already
solves "run an agent loop against any provider". This extension is the missing seam: it ports LiteLLM's v2
routing engine to TypeScript and wires it to `pi.setModel()`, so routing happens in the agent — with pi's own
model registry, credentials, and thinking levels — instead of behind a separate proxy hop.

> **Status: design stage.** This repository currently contains the specification (this file) and the
> implementation plan ([`PLAN.md`](PLAN.md)). No extension code has been written yet. Everything below marked
> "planned" describes the intended behaviour, not shipped behaviour.

> **v2 only.** This port targets LiteLLM's v2 auto-router (`auto_router/complexity_router`). The v1 semantic
> auto-router is deprecated upstream and is **not** ported — see [What we deliberately don't
> port](#what-we-deliberately-dont-port).

---

## Table of contents

- [Why](#why)
- [How LiteLLM's Auto Router v2 works](#how-litellms-auto-router-v2-works)
  - [Classification](#classification)
  - [Overrides and precedence](#overrides-and-precedence)
  - [Selection: pools and adaptive sampling](#selection-pools-and-adaptive-sampling)
  - [Affinity](#affinity)
  - [Input hygiene and the trust boundary](#input-hygiene-and-the-trust-boundary)
- [What we deliberately don't port](#what-we-deliberately-dont-port)
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
| "set up a Jupyter server with token auth on 8888" | a mid-tier model |
| "why does p99 latency triple when we double replicas?" | a frontier reasoning model |

Today pi picks one model per session and you change it by hand with `/model` or `Ctrl+P`. That means you either
overpay on trivia or underpower the hard turns. The auto-router makes that choice per prompt.

The alternative — running a LiteLLM proxy in front of pi and pointing pi's provider at it — works, and this
extension supports it as [proxy mode](#proxy-mode). But it costs you a process to run, a network hop per turn,
and it hides the decision: pi's UI still shows the proxy's single logical model, spend attribution collapses,
and pi's thinking-level and model-capability logic can't see the model that was actually chosen. Running the
router in-process fixes all four.

## How LiteLLM's Auto Router v2 works

Reference: [LiteLLM auto-routing docs](https://docs.litellm.ai/docs/proxy/auto_routing) ·
[v2 announcement](https://docs.litellm.ai/blog/autorouter-v2) ·
[source](https://github.com/BerriAI/litellm) (`litellm/router_strategy/complexity_router/`).

v1 shipped four separate routers you had to choose between up front — semantic, complexity, quality, adaptive.
v2 folds them into **one** router with configurable signals, so classification and selection become independent
knobs rather than a mode you pick at config time. In LiteLLM it's a virtual deployment:

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config: { ... }
```

Clients call `smart-router` like any model; a **pre-routing hook** (`async_pre_routing_hook`) runs before
deployment selection, classifies the request, and substitutes the real target.

The pipeline is: **extract → classify → override → select**.

### Classification

Every request lands in one of four tiers — `SIMPLE`, `MEDIUM`, `COMPLEX`, `REASONING` — in ascending severity.
`tier_definitions` can replace that set entirely with up to 8 operator-defined tiers, and `tier_labels` renames
the built-ins for display while config keys stay canonical.

Three classifier types (`classifier_type`):

**`heuristic`** (default) — rule-based, zero API calls, sub-millisecond. Seven weighted dimensions:

| Dimension | Weight |
|---|---|
| `codePresence` | 0.30 |
| `reasoningMarkers` | 0.25 |
| `technicalTerms` | 0.25 |
| `tokenCount` | 0.10 |
| `simpleIndicators` | 0.05 (negative) |
| `multiStepPatterns` | 0.03 |
| `questionComplexity` | 0.02 |

The weighted sum maps to tiers on `tier_boundaries`: `simple_medium` 0.15, `medium_complex` 0.35,
`complex_reasoning` 0.60. Two or more reasoning markers promote to `REASONING`, but only if the score also
clears `reasoning_override_min_score` (which tracks `simple_medium` unless set) — so stock phrases on a trivial
prompt can't buy the top tier. Reasoning markers *in the system prompt* never trigger the promotion.

**`llm`** — a small model classifies the prompt against a rubric and returns a tier via structured output
(`classifier_llm_config.model`, `timeout_ms` default 3000). This is where v2's most useful feature for us
lives: **`classification_rubric` presets**.

| Preset | Calibrated for |
|---|---|
| `legacy` (default) | the pre-calibration rubric; the default so upgrading never moves an existing router's spend |
| `agentic` | **agent, terminal, and coding-assistant traffic** |
| `chat` | conversational traffic only |
| `business` | sales/support/GTM traffic; also swaps the tier criteria |

The `agentic` preset exists because of exactly the failure mode this project would otherwise have hit. From
upstream's own notes: a rubric written for consumer chat puts "non-trivial code, multi-step technical work" at
the *top* of the scale — which is the **median** request in agent traffic, so ordinary engineering reads as
top-tier and the router pays frontier prices for it. The preset anchors routine installs, builds, multi-file
edits, and standard debugging at `MEDIUM` with worked examples:

```
- "write /app/ode_solve.py, a small RK4 initial value problem solver..." -> MEDIUM
- "set up a Jupyter server with token auth on port 8888 and confirm it serves" -> MEDIUM
- "solve this 5x4 Huarong Dao sliding block puzzle in the fewest moves" -> COMPLEX, it needs a real search formulation
- "separability_matrix computes the wrong result for nested CompoundModels; find the root cause" -> COMPLEX
```

**For this extension, `agentic` is the default.** We are, definitionally, agent traffic.

The classifier can also see prior turns: `classifier_context_window_size` (default 3) and
`classifier_context_per_turn_chars` (default 200), with `classifier_context_include_assistant_turns` opting
assistant replies in. This is what makes a bare `"yes"` classify as the work it approves rather than as the
word — the rubric's closing line changes to say so when a window is configured.

**`custom`** — a classifier plugin, with `classifier_plugin_timeout_ms`.

On classifier failure, `classifier_fallback` decides: `heuristic` (re-score locally) or `default_model`.

### Overrides and precedence

Several signals outrank the classifier. The order the hook actually applies them:

1. **Session affinity pin** — if enabled and a session id resolves, reuse the pinned model and skip
   classification entirely. Suppressed when `plugins` are configured, since a stale pin would bypass a policy
   plugin whose answer can change between turns.
2. **Plan-mode floor at top tier** — if `plan_mode_min_tier` is the highest configured tier, route there
   immediately and skip the classifier call entirely, since nothing could outrank it.
3. **`keyword_tier_rules`** — deterministic keyword → tier overrides, matched either lexically or by embedding
   similarity (`semantic_keyword_matching` + `embedding_model` + `match_threshold`, default 0.5). Multiple
   matches escalate to the highest tier, so rule order never silently changes behaviour.
4. **Classifier** — heuristic, LLM, or plugin.

Then two modifiers apply on top of whatever won:

- **`escalation_keywords`** (default `["LITELLM ESCALATE"]`, case-sensitive) — a user phrase that bumps the
  result one tier. Users can force a *stronger* model, never pick which one.
- **`plan_mode_min_tier`** — requests carrying a coding-agent plan-mode sentinel (Claude Code plan mode, VS
  Code Copilot Plan mode, Copilot CLI's `exit_plan_mode`) get a tier *floor*; the classified tier still wins
  when higher. Notably the floor also overrides a session pin — but only for the turns carrying the sentinel,
  and without rewriting the pin, so the first turn after plan mode exits routes as if it never happened.
  `plan_mode_patterns` adds your own sentinels.

Both are floors, never ceilings: a caller who pastes a sentinel can spend up to that tier's models — never
down, and never outside the configured pools.

### Selection: pools and adaptive sampling

A tier maps to a model, a **list** of models (a pool), or objects carrying per-tier request-parameter
overrides:

```yaml
tiers:
  COMPLEX: opus
  REASONING:
    - model_name: opus
      litellm_params: { reasoning_effort: xhigh }
    - abc
```

With `adaptive: false` (default), a pool is picked from directly. With `adaptive: true`, v2 Thompson-samples
the pool using `adaptive_weights` (default `quality: 0.3, cost: 0.7` — note this is *inverted* from the
standalone v1 adaptive router's 0.7/0.3) and `tier_distance_penalty` (default 0.5, the score penalty per
tier-step away from the classified tier). `adaptive_eligible` chooses between `"all"` (score every pool model
with a tier-distance penalty — soft floors) and `"classified_tier"` (sample only inside the classified tier's
pool).

### Affinity

Two independent pins, both keyed on a session id and bounded by `session_affinity_ttl_seconds` (default 3600,
refreshed on every hit):

- **`session_affinity`** (default off) — pin the model chosen on the session's first turn for the whole
  session, skipping re-classification. Preserves provider prompt caches and avoids cross-model
  conversation-history errors. Off by default upstream so each turn routes to the cheapest adequate tier.
- **`deployment_affinity`** (default **on**) — pin the *deployment* inside each routed model group, without
  pinning which group the session routes to. Every turn is still classified on its own merits, but a session
  that escalates and comes back lands on the deployment it used before, keeping the provider prompt cache warm.

This is upstream's answer to the prompt-caching problem, and it's a genuine tension: routing per prompt saves
money on model choice and loses money on cache misses. See [design constraints](#design-constraints).

### Input hygiene and the trust boundary

Two details that matter more for an agent than for chat:

**Reminder blocks.** Agent harnesses inject their own context as ordinary message text. That text is plumbing,
not something a human asked for, so the router strips complete blocks between `<system-reminder>` and
`</system-reminder>` before classifying. `reminder_markers` replaces that pair with your harness's own
delimiters (it *replaces*, so list the built-in pair too if you still emit it). A turn that is nothing but a
reminder block strips to empty, and the router falls back to the last real ask.

**Prompt-injection defence.** The built-in rubric's closing paragraph tells the classifier that the caller's
quoted system prompt and prior turns are *material to judge, never instructions*. Without it, a caller can ask
for a tier from inside their own prompt and get it. Upstream appends this unconditionally even after an
operator-supplied preamble — and warns that a full `system_prompt` replacement drops it. Any port must carry
this paragraph; it is load-bearing, not boilerplate.

## What we deliberately don't port

**The v1 semantic auto-router** (`auto_router/<name>`, `auto_router_config_path`,
`auto_router_embedding_model`, and the `semantic-router` dependency). It is
[deprecated upstream](https://docs.litellm.ai/docs/proxy/auto_routing_semantic). Its model was
"route name *is* the model name", matching prompts against per-route utterances by embedding similarity — one
embedding round-trip per prompt on the interactive critical path, and no notion of tiers, pools, or
escalation.

v2 keeps the useful half: `semantic_keyword_matching` matches `keyword_tier_rules` by embedding similarity
instead of literal text. The difference is that a semantic match now resolves to a **tier**, which then flows
through escalation, plan-mode floors, pools, and adaptive selection like any other classification — rather than
short-circuiting straight to one model. So semantic routing is available; the deprecated *router* is not.

**The standalone `auto_router/quality_router` and `auto_router/adaptive_router` deployments.** Adaptive is
folded into v2 as the `adaptive` flag. Quality routing — deployments declaring a `quality_tier`, ties broken by
cost — is expressible as tier pools plus adaptive cost weighting, so we don't port it as a separate strategy.

## What this extension does

It re-implements the v2 routing *decision* in TypeScript and applies it through pi's public extension API:

| LiteLLM v2 concept | pi equivalent in this extension |
|---|---|
| Virtual deployment in `model_list` | the extension itself — no fake model appears in pi's picker |
| `async_pre_routing_hook` | an `input` / `before_agent_start` handler |
| `tiers` → deployment lookup | `ctx.modelRegistry.find(provider, id)` + `pi.setModel(model)` |
| Per-tier `litellm_params.reasoning_effort` | `pi.setThinkingLevel()` (called *after* `setModel`, since it clamps to model capability) |
| `default_model` | `defaultModel`; also the fallback on any router error |
| Model catalogue (`model_list`) | pi's model registry, narrowed by `ctx.scopedModels` |
| `classifier_llm_config.model` | a pi registry model, using credentials pi already resolved |
| `session_affinity` / session id | the pi session — `ctx.sessionManager.getSessionFile()` |
| `plan_mode_min_tier` sentinels | pi's own plan mode, read directly instead of sniffed from prompt text |
| `reminder_markers` | pi's `<system-reminder>` context injections |
| `escalation_keywords` | both the keyword and an `/autoroute escalate` command |
| Spend logs / `routing_decision` | `pi.appendEntry("autoroute-decision", …)` + `ctx.ui.setStatus()` |

Planned behaviour per prompt:

1. The user submits a prompt. The extension's input handler sees the raw text.
2. Escape hatches run first: an explicit `/model` pin, `--no-autoroute`, or `/autoroute off` bypass routing.
3. Session affinity pin, if enabled and set → done.
4. Reminder blocks stripped; the current ask and prior-turn context extracted.
5. Plan-mode floor → keyword rules → classifier, in that precedence, then escalation and floors applied.
6. The tier resolves to a model (pool pick or adaptive sample). If it doesn't resolve, or has no credentials
   (`pi.setModel()` returns `false`), fall to the next candidate, then `defaultModel`.
7. `pi.setModel()` and optionally `pi.setThinkingLevel()` applied; decision recorded; turn proceeds.

**Sentinel detection is where pi beats the proxy.** Upstream has to sniff plan mode out of client-injected
prompt text — brittle strings that drift with every client release, and spoofable by anyone who pastes one.
In-process, pi's plan mode is a fact we can read directly. Same for the session id and the reminder markers.

### What it deliberately does not do

- **Route mid-turn.** The model is chosen once per user prompt and held for the whole agent run, including
  every tool call in it. Swapping models between tool calls breaks prompt caching and confuses
  provider-specific message shapes.
- **Run a LiteLLM proxy for you.** Local mode has no Python dependency.
- **Override an explicit choice.** If you picked a model by hand this session, that wins until you clear it.
- **Let a prompt choose its own tier.** Escalation moves one tier up and nothing else; the trust-boundary
  paragraph stays in the rubric.

## Architecture

```
user prompt
   │
   ├─► pi checks extension commands (/autoroute … handled here, routing skipped)
   │
   ├─► input event ───────────────────────────────────────────────┐
   │      │                                                       │
   │      ├─ escape hatches (pinned? --no-autoroute? off?)        │
   │      ├─ session affinity pin ──────────────► done            │
   │      │                                                       │
   │      ├─ extract: strip reminder blocks, current ask + N turns│
   │      │                                                       │
   │      ├─ classify (first match wins)                          │
   │      │    1. plan-mode floor at top tier ─► skip classifier  │
   │      │    2. keyword_tier_rules (lexical | embedding)        │
   │      │    3. classifier: heuristic <1ms | llm (agentic)      │
   │      │                                                       │
   │      ├─ modify: escalation keyword ▲1, plan-mode floor       │
   │      │                                                       │
   │      ├─ select: tier pool pick, or Thompson sample (adaptive)│
   │      ├─ resolve → ctx.modelRegistry.find(provider, id)       │
   │      ├─ apply   → pi.setModel() / pi.setThinkingLevel()      │
   │      └─ record  → appendEntry + ctx.ui.setStatus()           │
   │                                                              │
   │      any throw ─► warn + keep current/default model ─────────┤
   │                                                              │
   └─► before_agent_start ──► agent loop runs on chosen model ────┘
```

Planned layout — a directory extension with a `package.json`:

```
pi-litellm-autorouter/
├── package.json            # "pi": { "extensions": ["./src/index.ts"] }
├── src/
│   ├── index.ts            # default export: (pi: ExtensionAPI) => void
│   ├── config.ts           # load + validate config, typebox schemas
│   ├── extract.ts          # reminder stripping, current ask, prior-turn context
│   ├── classify/
│   │   ├── heuristic.ts    # 7-dimension weighted scorer
│   │   ├── llm.ts          # structured-output classifier
│   │   ├── rubrics.ts      # tier criteria + calibration presets (agentic default)
│   │   └── keywords.ts     # keyword_tier_rules, lexical + embedding
│   ├── select.ts           # tier → pool pick → adaptive sample
│   ├── adaptive.ts         # Thompson sampling, posteriors, persistence
│   ├── affinity.ts         # session pin, TTL
│   ├── resolve.ts          # tier model → pi model, fallback chain
│   ├── decision.ts         # RouteDecision record, persistence, status line
│   ├── commands.ts         # /autoroute …
│   └── proxy.ts            # delegate to a running LiteLLM proxy
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

*(planned)* — `~/.pi/agent/autorouter.json`, overridable per project at `.pi/autorouter.json`. Key names track
LiteLLM's `complexity_router_config` so a config ports between the two systems.

### Minimal

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "tiers": {
    "SIMPLE":    "anthropic/claude-haiku-4-5",
    "MEDIUM":    "anthropic/claude-sonnet-5",
    "COMPLEX":   "anthropic/claude-sonnet-5",
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
  }
}
```

That's the heuristic classifier with upstream's default weights and boundaries.

### LLM classifier with the agentic rubric

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "classifier_type": "llm",
  "classifier_llm_config": {
    "model": "anthropic/claude-haiku-4-5",
    "classification_rubric": "agentic",
    "timeout_ms": 3000
  },
  "classifier_fallback": "heuristic",
  "classifier_context_window_size": 3,
  "classifier_context_per_turn_chars": 200,
  "tiers": { "...": "..." }
}
```

`classification_rubric` defaults to `agentic` here, not `legacy` — upstream defaults to `legacy` only to avoid
moving existing deployments' spend on upgrade, which is not a constraint a new project has.

### Full surface

```json
{
  "defaultModel": "anthropic/claude-haiku-4-5",
  "tiers": {
    "SIMPLE":    "anthropic/claude-haiku-4-5",
    "MEDIUM":    ["anthropic/claude-sonnet-5", "openai/gpt-5-mini"],
    "COMPLEX":   "anthropic/claude-sonnet-5",
    "REASONING": { "model": "anthropic/claude-opus-5", "thinkingLevel": "high" }
  },
  "tier_boundaries": { "simple_medium": 0.15, "medium_complex": 0.35, "complex_reasoning": 0.60 },
  "keyword_tier_rules": [
    { "keywords": ["migration", "schema change", "security"], "tier": "REASONING" },
    { "keywords": ["typo", "rename", "format"], "tier": "SIMPLE" }
  ],
  "semantic_keyword_matching": false,
  "escalation_keywords": ["ESCALATE"],
  "planMode": { "minTier": "COMPLEX" },
  "adaptive": false,
  "adaptive_weights": { "quality": 0.3, "cost": 0.7 },
  "tier_distance_penalty": 0.5,
  "adaptive_eligible": "all",
  "sessionAffinity": { "enabled": false, "ttlSeconds": 3600 }
}
```

## Commands and flags

*(planned)*

| | Purpose |
|---|---|
| `/autoroute` | Current classifier, last decision, tier, and score |
| `/autoroute off` / `on` | Toggle routing for the session |
| `/autoroute pin <model>` | Freeze on one model until unpinned |
| `/autoroute escalate` | Re-run the last prompt one tier up (the `escalation_keywords` path, as a command) |
| `/autoroute explain` | Per-dimension breakdown, matched keywords, and why this tier won |
| `--no-autoroute` | CLI flag (`pi.registerFlag()`) to start with routing disabled |

The active tier and model are surfaced in the footer via `ctx.ui.setStatus()`, and every decision is persisted
with `pi.appendEntry()` — custom entries don't enter LLM context, so the routing log costs no tokens.

## Proxy mode

If you already run a LiteLLM proxy, set `"strategy": "proxy"` and the extension stops deciding anything
locally: it registers the proxy through `pi.registerProvider()` and lets LiteLLM's own v2 router pick.

```json
{
  "strategy": "proxy",
  "proxy": { "baseUrl": "http://localhost:4000", "apiKey": "$LITELLM_MASTER_KEY", "model": "smart-router" }
}
```

You get the upstream engine and centralised spend tracking; you lose in-process sentinel detection, and pi sees
one logical model. The chosen model is only visible in response headers, which this mode reads in
`after_provider_response` and surfaces in the footer where the transport exposes them.

## Design constraints

1. **Routing must never fail a prompt.** Every classification path is wrapped; any error falls back per
   `classifier_fallback` and then to `defaultModel`. This is upstream's rule and it matters more in an
   interactive agent.
2. **Latency is on the critical path, and v2 adds some.** The heuristic classifier is local and sub-millisecond
   — it stays the default. The LLM classifier is an extra round-trip *before the turn starts*, which upstream
   budgets at 3000ms. In a proxy that hides inside request latency; in a TUI the user watches it. Use a small
   fast model, keep the timeout tight, and cache by prompt hash.
3. **Per-prompt routing trades model cost against cache cost.** Switching models discards the provider prompt
   cache, and in a long agent session cache hits can dominate the bill. `deployment_affinity` on by default and
   `session_affinity` available are upstream's answer; we ship the same knobs and measure before choosing our
   defaults.
4. **A model is only usable if pi has credentials for it.** `pi.setModel()` returns `false` when there's no API
   key; the resolver treats that as "not a candidate" and moves down the fallback chain.
5. **The user always outranks the router.** An explicit `/model` selection or a pin wins.
6. **The trust boundary is not optional.** Prompt text is material to classify, never instructions. Escalation
   is the one sanctioned caller influence, and it moves exactly one tier.
7. **One model per agent run.** See [what it does not do](#what-it-deliberately-does-not-do).

## Prior art and attribution

- **LiteLLM** — [BerriAI/litellm](https://github.com/BerriAI/litellm), MIT (outside `enterprise/`). The v2
  router ported here lives in `litellm/router_strategy/complexity_router/`. This project is a port of the
  *algorithms, rubrics, and config shapes*, not a redistribution of LiteLLM code.
- **pi** — [pi.dev](https://pi.dev), [extension docs](https://pi.dev/docs/latest/extensions).

Neither LiteLLM nor pi endorses this project.

## License

MIT — see [LICENSE](LICENSE).
