# Implementation plan

Plan of record for building `pi-litellm-autorouter` as described in [README.md](README.md).

**Target: LiteLLM Auto Router v2** (`auto_router/complexity_router`). The v1 semantic auto-router is deprecated
upstream and is not in scope at any phase — see [What we deliberately don't
port](README.md#what-we-deliberately-dont-port). Nothing in this plan may introduce `semantic-router`,
`auto_router_config_path`, `auto_router_embedding_model`, or route-name-as-model-name.

v1 scope: heuristic classifier, LLM classifier with the `agentic` rubric, keyword tier rules, tier pools,
escalation, plan-mode floors, affinity. Adaptive sampling and semantic keyword matching are v2-of-this-project.

---

## Phase 0 — Verify the seam (blocking, ~half a day)

Everything downstream depends on one question the pi docs do not answer outright:

> **Does `pi.setModel()` affect the turn that is already in flight?**

The `input` event fires before agent processing begins, and `before_agent_start` fires after the prompt is
submitted but before the agent loop. It is not documented whether a model set inside either handler is picked
up by the run that follows, or only by the next one. If it's the latter, routing is always one prompt behind
and the design changes materially.

- [ ] Spike an extension that sets a distinct model in `input` and logs `ctx.model` from `turn_start` and
      `before_provider_request`. Confirm which handler wins.
- [ ] Confirm `await pi.setModel(m)` resolves before the agent starts, and that an `input` handler can hold up
      the turn while an async classifier runs (this gates the LLM classifier entirely).
- [ ] Confirm behaviour under `streamingBehavior: "steer"` and `"followUp"` — a queued follow-up may arrive
      mid-run, where switching models is not safe.
- [ ] Confirm `ctx.modelRegistry.find(provider, id)` accepts the ids in pi's catalogue; check
      `getAvailable()` / `ctx.scopedModels` for candidate filtering.
- [ ] Confirm `pi.setThinkingLevel()` clamping happens against the *new* model, i.e. call it after
      `setModel()`.

**Exit criteria:** a one-file spike that provably routes prompt N to model N, not N+1. Record the answer in
`docs/seam.md`. If neither handler can change the current turn, escalate before building further.

## Phase 1 — Skeleton, config, and extraction

- [ ] `package.json` with `"pi": { "extensions": ["./src/index.ts"] }`, runtime deps in `dependencies` (not
      `devDependencies` — pi package installs are production-only). `typebox` for schemas.
- [ ] TypeScript + vitest + lint. No build step (pi loads TS via jiti), but keep `tsc --noEmit` in CI.
- [ ] `src/config.ts`: load `~/.pi/agent/autorouter.json`, merge `.pi/autorouter.json` over it, validate,
      return a fully-defaulted config. Invalid config = loud warning + routing disabled, never a startup crash.
      Mirror upstream's validation rules, including the ones that exist for a reason: reject blank keywords in
      `keyword_tier_rules` (a blank substring-matches every prompt and would silently force one tier for all
      traffic), and reject `classification_rubric` alongside a custom `system_prompt` (mutually exclusive
      upstream, since the custom prompt replaces the rubric the preset would select).
- [ ] `src/index.ts`: default export factory. Register flag and commands there; start no background work (the
      factory runs in invocations that never open a session). Defer classifier construction to `session_start`.
- [ ] `src/extract.ts` — the input hygiene layer, ported from upstream's `_strip_reminder_blocks`,
      `_extract_current_ask_and_system_prompt`, `_newest_turn_ask`, `_extract_prior_turns`:
  - [ ] Strip complete reminder blocks (default `<system-reminder>`/`</system-reminder>`, case-insensitive).
        Nested/overlapping blocks strip whole; an *unclosed* delimiter is not a block and stays, so prose that
        merely mentions a delimiter isn't eaten.
  - [ ] A turn that strips to empty falls back to the last real ask.
  - [ ] Prior-turn context: last N turns, truncated to `classifier_context_per_turn_chars`.
  - [ ] Configurable marker pairs, *replacing* the built-in pair rather than adding to it.
- [ ] `src/decision.ts`: the `RouteDecision` record — `{ cause, tier, score, signals, matchedKeyword,
      escalationKeyword, escalated, chosenModel, latencyMs }`, mirroring upstream's `routing_decision` causes
      (`plan_mode`, `literal_keyword_match`, `semantic_keyword_match`, `session_affinity_pin`,
      `session_affinity_escalation`, `default_fallback`, `default_model_fallback`). Persisted with
      `pi.appendEntry()`, surfaced via `ctx.ui.setStatus()`.

**Exit criteria:** extension loads, `/autoroute` reports its config, extraction is unit-tested against
pi-shaped transcripts including reminder blocks.

## Phase 2 — Resolver and safety rails

Bulletproof first; every classifier plugs into this.

- [ ] `src/resolve.ts`: given an ordered candidate list, return the first that exists in the registry **and**
      for which `pi.setModel()` succeeds. A `false` return (no API key) demotes rather than fails.
- [ ] Fallback chain: `tier pick → tier fallback → defaultModel → leave model unchanged`. Never throw out of
      the handler.
- [ ] Escape hatches, before any classification: session pin, `--no-autoroute`, `/autoroute off`, and "user
      changed model by hand" — track via `model_select` with `event.source === "set"` and suppress until
      unpinned.
- [ ] `session_start` restores pin/off state from prior `appendEntry` records.
- [ ] Tests: every failure mode returns a usable model and logs; none throws.

**Exit criteria:** a stub classifier that always names a nonexistent model still leaves the session usable.

## Phase 3 — Heuristic classifier

Port of upstream's `_score_and_classify`. This is the default path and must stay sub-millisecond.

- [ ] Seven scorers with upstream weights: `codePresence` 0.30, `reasoningMarkers` 0.25, `technicalTerms` 0.25,
      `tokenCount` 0.10, `simpleIndicators` 0.05 (negative), `multiStepPatterns` 0.03, `questionComplexity`
      0.02.
- [ ] Tier boundaries `simple_medium` 0.15, `medium_complex` 0.35, `complex_reasoning` 0.60; canonical key
      names kept so configs port both ways.
- [ ] Token thresholds (`simple` 15, `complex` 400), overridable keyword lists, word-boundary matching for
      single-word keywords.
- [ ] Reasoning override: 2+ markers promote to `REASONING`, gated on `reasoning_override_min_score` (tracks
      `simple_medium` unless set). Markers in the **system prompt** must not trigger it.
- [ ] `tier_labels` for display; config keys stay canonical.
- [ ] Per-tier object form (`{ model, thinkingLevel }`) and bare strings both accepted.
- [ ] `/autoroute explain` renders the per-dimension breakdown.

**Exit criteria:** scoring matches a fixture set ported from upstream's tests; <1ms per classification.

## Phase 4 — LLM classifier with the agentic rubric

The reason to target v2 at all. Upstream's own analysis is that a chat-calibrated rubric puts non-trivial code
at the top of the scale, which is the *median* agent request — so ordinary engineering routes to the most
expensive tier. The `agentic` preset is the fix, and it is a measured artifact: the accuracy reported for a
preset describes that exact text, so it must be ported byte-for-byte, not paraphrased.

- [ ] `src/classify/rubrics.ts`: port the tier criteria, the preamble, the calibration example blocks, and the
      trust-boundary paragraph verbatim. Presets: `agentic` (our default), `chat`, `business`, `legacy`.
- [ ] Tier names render as placeholders so `tier_labels` substitute correctly and the enum the classifier may
      return always matches the rubric's vocabulary.
- [ ] Closing line switches on context window: with a window, "rate the work a short reply approves"; without,
      "classify only the current message".
- [ ] **Trust boundary appended unconditionally** after any operator preamble. A full `system_prompt`
      replacement drops it — document that loudly and default `classifier_fallback` to `default_model` in that
      case, per upstream.
- [ ] Structured output for the tier; classifier model resolved from pi's registry using
      `ctx.modelRegistry.getProviderAuth()`.
- [ ] `timeout_ms` (default 3000) with a hard abort; on timeout or error, `classifier_fallback` decides
      (`heuristic` re-scores locally, or `default_model`).
- [ ] Cache decisions by hash of (extracted ask + context turns + rubric) within a session.
- [ ] **Latency review before making this the default.** 3s is fine behind a proxy and long in a TUI. Measure
      p50/p95 with a small model; if it's not comfortably sub-second, heuristic stays the default and this is
      opt-in.

**Exit criteria:** agentic-rubric classification reproduces upstream's worked examples; measured added latency
documented in the README.

## Phase 5 — Overrides: keywords, escalation, plan mode

Precedence must match upstream exactly; it is load-bearing, not incidental.

- [ ] `keyword_tier_rules`, lexical matching first. Multiple matches escalate to the **highest** tier, so rule
      order never silently changes behaviour.
- [ ] `escalation_keywords` (default `["LITELLM ESCALATE"]`, case-sensitive) — bump one tier, never pick a
      model. Plus `/autoroute escalate` as the ergonomic front door.
- [ ] Plan mode: read **pi's own plan mode state directly** rather than sniffing sentinel strings out of prompt
      text. Upstream has to pattern-match client-injected strings that drift with client releases and are
      spoofable by anyone who pastes one; in-process this is a fact we can query. Keep `planMode.patterns` as
      an escape hatch for text-carried sentinels.
- [ ] Floor semantics: the classified tier still wins when higher; the floor overrides a session pin only for
      the turns in plan mode, without rewriting the pin, so the first turn after exit routes as if plan mode
      never happened.
- [ ] Short-circuit: when the floor is the top configured tier, skip the classifier call entirely.
- [ ] Order the whole pipeline as upstream does: session pin → top-tier plan floor → keyword rules →
      classifier, then escalation and floor applied on top.

**Exit criteria:** a precedence test matrix covering every pair of signals, asserted against upstream's
documented behaviour.

## Phase 6 — Affinity and cache-cost measurement

The honest open question of the whole project: per-prompt routing saves on model choice and loses on prompt
cache misses.

- [ ] `session_affinity` (pin model for the session, skip re-classification) and `deployment_affinity` — pi's
      analogue is holding a provider/model pair steady across turns — both TTL-bounded
      (`session_affinity_ttl_seconds` 3600, refreshed on hit).
- [ ] Escalation re-pins higher; the plan-mode floor does not rewrite the pin.
- [ ] **Instrument before choosing defaults.** Record per-decision estimated spend *including* cache-miss cost
      on every model switch. Run a real multi-day session both ways and pick defaults from the numbers, not
      from upstream's (which are tuned for proxy traffic, not long agent sessions).

**Exit criteria:** a measured recommendation for default affinity settings, written into the README.

## Phase 7 — Pools, adaptive sampling, semantic keywords

Deferred deliberately: each is only worth it once the base router is proven.

- [ ] Tier pools (`tiers` as a list) with a plain pick.
- [ ] `adaptive: true` — Thompson sampling over the pool with `adaptive_weights` (upstream default
      `quality: 0.3, cost: 0.7` in v2, *inverted* from the standalone v1 adaptive router — port the v2 values),
      `tier_distance_penalty` 0.5, `adaptive_eligible` `all` | `classified_tier`. Posteriors persist via
      `appendEntry`.
- [ ] `semantic_keyword_matching` — embedding-matched `keyword_tier_rules` with `match_threshold` 0.5 and MAX
      aggregation, so one strong keyword match isn't diluted by others on the tier. Precompute keyword
      embeddings at load; cache prompt embeddings by hash. This is v2's semantic path — resolving to a *tier*,
      not straight to a model.

## Phase 8 — Proxy mode, polish, release

- [ ] `src/proxy.ts`: `pi.registerProvider()` for a LiteLLM proxy; local classification disabled entirely.
- [ ] Surface the chosen model from response headers in `after_provider_response` where the transport exposes
      them; document that this is best-effort.
- [ ] Commands complete, with `getArgumentCompletions` for model names; `--no-autoroute` flag.
- [ ] Session cost summary: what routing actually saved, net of cache misses.
- [ ] README updated from "planned" to shipped. CI green. Publish as `@coresoft/pi-litellm-autorouter`.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `setModel()` doesn't apply to the in-flight turn | Design-breaking | Phase 0 spike is blocking |
| LLM classifier latency is visible in the TUI | Every prompt feels slow | Heuristic is the default; Phase 4 gates promotion on measured latency; tight timeout + cache |
| Model switching costs more in cache misses than it saves | The whole project is net-negative | Phase 6 measures it explicitly before defaults are set |
| Rubric ported by paraphrase | Silently different tier decisions and spend | Presets are measured artifacts; port verbatim, fixture-test against upstream examples |
| Trust-boundary paragraph dropped | A prompt can buy itself the top tier | Appended unconditionally; custom-prompt path documented and defaulted to `default_model` fallback |
| Upstream v2 config keeps evolving | Port drifts from source | Track `complexity_router/config.py`; keep canonical key names so configs stay portable |
| pi's extension API changes | Breaks on pi upgrade | Pin a pi version range; keep the API surface used small |

## Resolved by targeting v2

Two risks from the previous (v1-targeted) plan are gone:

- *"LiteLLM's weights misclassify agent prompts"* — upstream hit this and fixed it with the `agentic`
  calibration preset. We adopt it as our default instead of building our own labelled corpus.
- *"Should routing consider conversation history?"* — v2 answers yes, via
  `classifier_context_window_size` and `classifier_context_include_assistant_turns`, so a bare "yes" is rated
  on the work it approves.

## Open questions

1. Phase 0's question — which handler can change the model for the current turn.
2. Does `getProviderAuth()` yield credentials usable for the classifier model, and for an embeddings endpoint
   in Phase 7, or do those need their own keys?
3. Can pi's plan mode be queried directly from an extension, or only observed via a registered flag? Phase 5's
   headline improvement over upstream depends on the answer.
4. Should `classifier_context_include_assistant_turns` default on for us? Upstream defaults it off only to
   avoid shifting an already-deployed router's spend. In an agent, the assistant turn often carries the
   difficulty — measure it.
5. What is the right default for `session_affinity` in a coding agent? Upstream's default (off) is tuned for
   proxy traffic; Phase 6 decides ours from data.
