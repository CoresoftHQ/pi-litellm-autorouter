# Implementation plan

Plan of record for building `pi-litellm-autorouter` as described in [README.md](README.md).
Scope for v1: **complexity** and **semantic** strategies, plus **proxy** passthrough. Quality is a follow-up;
adaptive is out of scope.

---

## Phase 0 — Verify the seam (blocking, ~half a day)

Everything downstream depends on one question the pi docs do not answer outright:

> **Does `pi.setModel()` affect the turn that is already in flight?**

The `input` event fires before agent processing begins, and `before_agent_start` fires after the prompt is
submitted but before the agent loop. It is not documented whether a model set inside either handler is picked
up by the run that follows, or only by the next one. If it's the latter, routing is always one prompt behind
and the design changes materially.

Tasks:

- [ ] Spike an extension that sets a distinct model in `input` and logs `ctx.model` from `turn_start` and
      `before_provider_request`. Confirm which handler wins.
- [ ] Confirm `await pi.setModel(m)` resolves before the agent starts (it is async; the input handler must
      await it, and `input` handlers must be able to hold up the turn).
- [ ] Confirm behaviour under `streamingBehavior: "steer"` and `"followUp"` — a queued follow-up message may
      arrive mid-run, where switching models is not safe.
- [ ] Confirm `ctx.modelRegistry.find(provider, id)` accepts the ids in pi's catalogue, and check what
      `ctx.modelRegistry.getAvailable()` / `ctx.scopedModels` return for filtering candidates.
- [ ] Confirm `pi.setThinkingLevel()` clamping: it is clamped to model capability, so it must be called
      **after** `setModel()`, not before.

**Exit criteria:** a one-file spike that provably routes prompt N to model N, not N+1. Record the answer in
`docs/seam.md`; if `input` can't do it, fall back to `before_agent_start`, and if neither works for the current
turn, escalate before building further.

## Phase 1 — Skeleton and config

- [ ] `package.json` with `"pi": { "extensions": ["./src/index.ts"] }`, deps in `dependencies` (not
      `devDependencies` — pi package installs are production-only). `typebox` for schemas.
- [ ] TypeScript setup, `vitest`, lint. No build step needed — pi loads TS via jiti — but keep `tsc --noEmit`
      in CI.
- [ ] `src/config.ts`: load `~/.pi/agent/autorouter.json`, merge `.pi/autorouter.json` over it, validate with
      typebox, return a fully-defaulted config. Invalid config = loud warning + routing disabled, never a crash
      at startup.
- [ ] `src/index.ts`: default export factory. Register the flag and commands in the factory; do **not** start
      any background work there (factories run in invocations that never open a session). Defer strategy
      construction to `session_start`.
- [ ] `src/decision.ts`: the `RouteDecision` record — `{ strategy, input hash, chosen model, score, tier or
      route name, latency ms, fellBackBecause? }` — persisted with `pi.appendEntry("autoroute-decision", …)`
      and rendered to the footer with `ctx.ui.setStatus()`.

**Exit criteria:** extension loads, `/autoroute` reports "no strategy configured", nothing else happens.

## Phase 2 — Resolver and safety rails

This is the part that must be bulletproof; both strategies plug into it.

- [ ] `src/resolve.ts`: given an ordered candidate list of `provider/model` strings, return the first that
      exists in the registry **and** for which `pi.setModel()` succeeds. `setModel()` returning `false` (no API
      key) demotes a candidate rather than failing.
- [ ] Fallback chain: `strategy result → configured tier/route fallback → defaultModel → leave model
      unchanged`. Never throw out of the handler.
- [ ] Escape hatches, checked before any classification: session pin (`/autoroute pin`), `--no-autoroute`,
      `/autoroute off`, and "user changed model by hand this session" — track the latter via `model_select`
      with `event.source === "set"` and suppress routing until unpinned.
- [ ] `session_start` restores pin/off state from prior `appendEntry` records.
- [ ] Tests: every failure mode returns the default model and logs, none throws.

**Exit criteria:** a stub strategy that always names a nonexistent model still leaves the session usable.

## Phase 3 — Complexity strategy (the v1 default)

Port of `litellm/router_strategy/complexity_router/`.

- [ ] `src/strategies/complexity.ts`: the seven scorers — `tokenCount` 0.10, `codePresence` 0.30,
      `reasoningMarkers` 0.25, `technicalTerms` 0.25, `simpleIndicators` 0.05 (negative), `multiStepPatterns`
      0.03, `questionComplexity` 0.02 — weighted sum, mapped to `SIMPLE`/`MEDIUM`/`COMPLEX`/`REASONING` on
      boundaries `simple_medium` 0.15, `medium_complex` 0.35, `complex_reasoning` 0.60.
- [ ] Keep LiteLLM's canonical config key names so configs port both ways. Support `tier_labels` for display.
- [ ] Support the per-tier object form (`{ model, thinkingLevel }`) as well as a bare model string.
- [ ] Keyword tier rules (`keyword_tier_rules`), which override the score.
- [ ] **Agent-specific calibration.** LiteLLM's weights are tuned for general chat traffic; a coding agent's
      prompts are code-heavy almost by definition, so `codePresence` at 0.30 will drag nearly everything into
      `COMPLEX`. Build a corpus of ~100 real prompts, label them by hand, and retune boundaries against it
      before shipping defaults. Ship the corpus as a fixture so retuning is repeatable.
- [ ] `/autoroute explain`: per-dimension breakdown for the last prompt.

**Exit criteria:** on the labelled corpus, tier assignment matches hand labels well enough to be worth using;
routing adds <1ms.

## Phase 4 — Semantic strategy

Port of `litellm/router_strategy/auto_router/auto_router.py` plus the bits of `semantic-router` it uses.

- [ ] `src/strategies/semantic.ts`: embed the prompt, cosine-compare against precomputed utterance vectors,
      take the best route above its `score_threshold`, else `null` (→ default model).
- [ ] Text extraction mirroring `_extract_text_from_messages`: last user message only, walking backwards past
      assistant/tool messages, flattening multimodal blocks to their `text` parts.
- [ ] Truncate to `maxInputChars` before embedding.
- [ ] Precompute utterance embeddings at `session_start`; cache to disk keyed by a hash of
      (route set + encoder model) so startup doesn't pay for it every time.
- [ ] Cache prompt embeddings by content hash within a session.
- [ ] Embedding calls: use the credentials pi already resolved via
      `ctx.modelRegistry.getProviderAuth(providerId)` rather than asking the user for a second key. Verify
      that this yields a usable base URL + key for embedding endpoints; if not, fall back to an explicit
      `semantic.encoder.apiKey`.
- [ ] Every embedding failure → warn + default model, per LiteLLM's own behaviour. Timeout must be short
      (~2s): a slow router is worse than a mediocre one on an interactive prompt.
- [ ] Accept LiteLLM's `auto_router_config.json` shape verbatim as an input format.

**Exit criteria:** an existing LiteLLM router config file routes correctly with only the model names remapped;
p95 added latency stays inside the timeout budget.

## Phase 5 — Proxy mode

- [ ] `src/strategies/proxy.ts`: `pi.registerProvider()` for the LiteLLM proxy (OpenAI-completions API,
      `baseUrl` + `apiKey`), exposing the configured auto-router model.
- [ ] Local classification disabled entirely in this mode.
- [ ] Surface the chosen model from response headers where the transport exposes them — `after_provider_response`
      gives status and headers before the stream is consumed; read `x-litellm-adaptive-router-model` and
      friends into the footer. Document that some providers abstract headers away and this is best-effort.

**Exit criteria:** with a local LiteLLM proxy running an auto-router deployment, pi routes through it and the
footer shows what LiteLLM picked.

## Phase 6 — Polish and release

- [ ] `/autoroute`, `/autoroute on|off`, `/autoroute pin <model>`, `/autoroute explain`, with
      `getArgumentCompletions` for model names.
- [ ] `--no-autoroute` flag.
- [ ] Cost accounting: record per-decision estimated spend and expose a session summary, so the extension can
      show what routing actually saved.
- [ ] README updated from "planned" to shipped; add a real example config and a short recording.
- [ ] CI: `tsc --noEmit`, vitest, lint. Publish to npm as `@coresoft/pi-litellm-autorouter`.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `setModel()` doesn't apply to the in-flight turn | Design-breaking | Phase 0 spike is blocking |
| LiteLLM's weights misclassify agent prompts | Router is useless or actively costly | Labelled corpus + retuned defaults in Phase 3 |
| Semantic embedding latency on every prompt | Interactive feel degrades | Complexity is the default; caching, short timeout, precomputation |
| Mid-run model switching breaks prompt caching | Cost goes *up*, not down | One model per agent run, by design |
| pi's extension API changes | Extension breaks on pi upgrade | Pin a pi version range; keep API surface used as small as possible |
| Routing hides a bad model choice from the user | Confusing sessions | Footer status on every turn + `/autoroute explain` |

## Open questions

1. Phase 0's question — which handler can change the model for the current turn.
2. Can `getProviderAuth()` produce credentials usable for an embeddings endpoint, or does semantic mode need
   its own key?
3. Should routing consider conversation history (a session that has been hard so far probably stays hard),
   or only the current prompt as LiteLLM does? History-aware routing is a v2 idea worth measuring.
4. Should a tier be allowed to route *down* mid-session — e.g. a long session pinned to Opus dropping to Haiku
   for one trivia question — given the prompt-cache cost of switching? Possibly gate downgrades on cache-miss
   cost.
