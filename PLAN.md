# Implementation plan

Plan of record for building `pi-litellm-autorouter` as described in [README.md](README.md).

**Target: LiteLLM Auto Router v2** (`auto_router/complexity_router`). The v1 semantic auto-router is deprecated
upstream and is not in scope at any phase — see [What we deliberately don't
port](README.md#what-we-deliberately-dont-port). Nothing in this plan may introduce `semantic-router`,
`auto_router_config_path`, `auto_router_embedding_model`, or route-name-as-model-name.

v1 scope: heuristic classifier, LLM classifier with the `agentic` rubric, keyword tier rules, tier pools,
escalation, plan-mode floors, affinity. Adaptive sampling and semantic keyword matching are v2-of-this-project.

---

## Phase 0 — Verify the seam ✅ DONE

> **Answered: yes.** `pi.setModel()` inside the `input` handler changes the model for the turn already in
> flight, and pi awaits the handler, so an async classifier can run there. Evidence and the exact trace are in
> [`docs/seam.md`](docs/seam.md); the probe is `test/manual/seam-probe.ts`.

Remaining from this phase, needing an interactive session with credentials:

- [ ] Multi-turn: confirm turn *N* gets turn *N*'s classification across a long session (the probe alternates
      targets per turn and is ready for this).
- [ ] `streamingBehavior` `"steer"` / `"followUp"`: routing is currently skipped there on the assumption that
      switching models mid-run is unsafe. Possibly over-cautious.
- [ ] Confirm `pi.setThinkingLevel()` clamps against the model set moments earlier in the same handler.

## Phase 1 — Skeleton, config, and extraction ✅ DONE

- [x] `package.json` with `"pi": { "extensions": ["./src/index.ts"] }`; deps in `dependencies`, not
      `devDependencies`, since pi package installs are production-only.
- [x] TypeScript + vitest; `tsc --noEmit` clean. No build step — pi loads TS via jiti.
- [x] `src/config.ts`: global + project layers, full defaulting, never crashes startup. Upstream's
      load-bearing validations are carried over: blank keywords rejected, `classificationRubric` and
      `systemPrompt` mutually exclusive, blank `systemPrompt` rejected.
- [x] `src/index.ts`: factory registers flag and commands only; classifier construction deferred to
      `session_start`.
- [x] `src/extract.ts`: reminder-block stripping (unclosed delimiters left alone, nested/overlapping pairs cut
      whole), fallback to the last real ask, prior-turn window with truncation.
- [x] `src/decision.ts`: `RouteDecision` with upstream's cause vocabulary, persisted via `appendEntry`,
      rendered to the footer.

## Phase 2 — Resolver and safety rails ✅ DONE

- [x] `src/resolve.ts`: first usable candidate wins; `setModel() === false` (no API key) demotes rather than
      fails.
- [x] Fallback chain walks *down* the tiers, then `defaultModel`, then leaves the model untouched. Walking
      down is deliberate: serving a request from a cheaper model is a degradation, silently promoting it is a
      bill the user did not ask for.
- [x] Escape hatches before classification: `--no-autoroute`, `/autoroute off`, a hand-picked model (tracked
      via `model_select` with `source === "set"`), and `/autoroute pin`.
- [x] State restored from `appendEntry` records on `session_start`.
- [x] Tests: every failure mode returns a usable model or leaves the session alone; none throws.

## Phase 3 — Heuristic classifier ✅ DONE

- [x] Seven scorers with upstream weights, boundaries, token thresholds and keyword lists, ported verbatim.
- [x] Reasoning override: 2+ markers, gated on `reasoningOverrideMinScore` (tracks `simple_medium`).
- [x] System prompt not scored at all, per upstream's reasoning about dynamic range.
- [x] Word-boundary matching with the CJK carve-out.
- [x] Per-tier `{ model, thinkingLevel }` and bare strings both accepted.
- [x] `/autoroute explain` renders the per-dimension breakdown.

Deferred: **calibration against real traffic**. The scorer is faithful to upstream, which is not the same as
correct for this router. One quirk already found and pinned in tests: `\blet\b` matches the "let" in "let's",
so "let's think about it" scores `codePresence` — enough on its own to push a short prompt over the
reasoning-override floor.

## Phase 4 — LLM classifier with the agentic rubric ✅ DONE (unproven)

- [x] `src/classify/rubrics.ts`: tier criteria, preamble, calibration examples and trust boundary ported
      verbatim. Presets `agentic` (default here), `chat`, `business`, `legacy`.
- [x] Trust boundary appended unconditionally; a custom `systemPrompt` warns that it drops the injection
      defence.
- [x] Closing line switches on the context window.
- [x] Runs through `ctx.modelRegistry.complete()`, reusing pi's resolved credentials — no second API key.
- [x] Hard timeout via `AbortController`; on failure, `classifierFallback` decides.
- [x] Tier parsing takes the *first* tier named, so trailing chatter cannot upgrade the answer.

- [ ] **Never called against a live model.** Typed to work, untested end to end.
- [ ] **Latency unmeasured.** Heuristic stays the default until p50/p95 exists.
- [ ] Cache classifications by hash within a session.

## Phase 5 — Overrides: keywords, escalation, plan mode ✅ DONE

Precedence matches upstream exactly; it is load-bearing, not incidental.

- [x] `keywordTierRules`, lexical matching. Multiple matches escalate to the **highest** tier, so rule order
      never silently changes behaviour.
- [x] `escalationKeywords` (default `["PI ESCALATE"]`, case-sensitive) — bump one tier, never pick a model.
- [x] Floor semantics: the classified tier still wins when higher; the floor overrides a session pin only for
      the turns in plan mode, without rewriting the pin, so the first turn after exit routes as if plan mode
      never happened.
- [x] Short-circuit: when the floor is the top configured tier, the classifier call is skipped entirely.
- [x] Pipeline ordered as upstream: session pin → top-tier plan floor → keyword rules → classifier, then
      escalation and floor on top. 29 router tests cover the pairs.

**Corrected from the previous plan.** This phase promised to "read pi's own plan mode state directly" and
called it the headline win over upstream. That was based on a wrong assumption: **pi has no built-in plan
mode** — it ships as an example extension, so there is no native state to query. What shipped instead is an
integration contract — a plan-mode extension emits `autoroute:plan-mode` `{ active: boolean }` on pi's shared
event bus — with `planMode.patterns` as the text-sentinel fallback, which is the mechanism upstream is stuck
with.

- [ ] `/autoroute escalate` as an ergonomic front door for the keyword.

## Phase 6 — Affinity and cache-cost measurement

The honest open question of the whole project: per-prompt routing saves on model choice and loses on prompt
cache misses.

- [x] `sessionAffinity` (pin the model for the session, skip re-classification), TTL-bounded
      (`ttlSeconds` 3600, refreshed on hit). Off by default, as upstream.
- [x] Escalation re-pins higher; the plan-mode floor does not rewrite the pin.
- [ ] `deployment_affinity` — pi's analogue is holding a provider/model pair steady across turns. Not yet
      implemented; pi's registry may make it moot, since a `provider/modelId` already names one endpoint.
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

## Phase 8a — Config ergonomics ✅ DONE

- [x] `/autoroute init` builds a config from `modelRegistry.getAvailable()` (or `ctx.scopedModels` when the
      session is scoped), ranking by a blended `input × 0.8 + output × 0.2` price per million tokens — weighted
      to input because that is what an agent turn is made of.
- [x] Top tier prefers the priciest model that supports extended thinking, and gets `thinkingLevel: "high"`.
      A *cheap* reasoning model must not be promoted past the price ordering; tested.
- [x] Degenerate catalogues (one model, no cost data, provider filter matching nothing) produce a usable
      config plus a note saying what was compromised, or a clear error.
- [x] Deterministic: same catalogue in any order produces the same file.
- [x] Never overwrites without confirmation, and refuses when there is no UI to confirm through.
- [x] Generated configs are round-tripped through the loader in tests, so `init` cannot emit something
      `loadConfig` would reject.
- [x] README configuration guide: init, layering, tuning by `/autoroute explain`, the LLM classifier and its
      latency trade-off, escape hatches, affinity, and a full key reference table.

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
| LLM classifier latency is visible in the TUI | Every prompt feels slow | Heuristic is the default; Phase 4 gates promotion on measured latency; tight timeout + cache |
| Model switching costs more in cache misses than it saves | The whole project is net-negative | Phase 6 measures it explicitly before defaults are set |
| Rubric ported by paraphrase | Silently different tier decisions and spend | Presets are measured artifacts; port verbatim, fixture-test against upstream examples |
| Trust-boundary paragraph dropped | A prompt can buy itself the top tier | Appended unconditionally; custom-prompt path documented and defaulted to `default_model` fallback |
| Upstream v2 config keeps evolving | Port drifts from source | Track `complexity_router/config.py`; keep canonical key names so configs stay portable |
| pi's extension API changes | Breaks on pi upgrade | Pin a pi version range; keep the API surface used small |

## Resolved

The blocking risk is gone: **`pi.setModel()` affects the turn already in flight**, verified against pi 0.84.2
([`docs/seam.md`](docs/seam.md)). pi also awaits the `input` handler, so the LLM classifier can run there.

Two more were resolved by targeting v2 rather than v1:

- *"LiteLLM's weights misclassify agent prompts"* — upstream hit this and fixed it with the `agentic`
  calibration preset. We adopt it as our default instead of building our own labelled corpus.
- *"Should routing consider conversation history?"* — v2 answers yes, via
  `classifier_context_window_size` and `classifier_context_include_assistant_turns`, so a bare "yes" is rated
  on the work it approves.

## Open questions

1. ~~Which handler can change the model for the current turn.~~ Answered: `input`, and pi awaits it.
2. ~~Can pi's plan mode be queried directly?~~ Answered: no — pi has no built-in plan mode. Hence the
   `autoroute:plan-mode` event contract.
3. Does `modelRegistry.complete()` actually work for a classifier call from inside an `input` handler? Typed
   to; never called for real.
4. Should `classifier_context_include_assistant_turns` default on for us? Upstream defaults it off only to
   avoid shifting an already-deployed router's spend. In an agent, the assistant turn often carries the
   difficulty — measure it.
5. What is the right default for `session_affinity` in a coding agent? Upstream's default (off) is tuned for
   proxy traffic; Phase 6 decides ours from data.
