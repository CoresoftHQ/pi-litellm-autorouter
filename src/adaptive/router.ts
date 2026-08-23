/**
 * The adaptive router's learned state.
 *
 * Port of LiteLLM's `AdaptiveRouter` (`adaptive_router/adaptive_router.py`), minus the
 * Postgres flusher: posteriors are serialised to a JSON file by the caller instead.
 *
 *   cells            Beta(alpha, beta) per (request type, model); what the bandit samples
 *   sessionStates    rolling per-(session, model) state for the incremental detectors
 *   feedbackContexts per session, the previous turn — so feedback in *this* user message
 *                    is credited to the model that produced the *previous* reply
 */

import { type BanditCell, type Rng, applyDelta, cellMean, cellTotalSamples, initialCell } from "./bandit.ts";
import {
  type SessionState,
  type SignalDelta,
  type Turn,
  advanceSessionState,
  anyFired,
  applySignalDelta,
  banditDelta,
  detectResponseSignals,
  detectUserFeedback,
  emptyDelta,
  mergeSignalDeltas,
  MIN_TURNS_FOR_CLEAN_CREDIT,
  newSessionState,
} from "./signals.ts";
import { type AdaptivePreferences, type ModelRef, REQUEST_TYPES, type RequestType, isRequestType } from "../types.ts";

/** A conversation's state is held this long after its last turn. */
export const OWNER_CACHE_TTL_MS = 24 * 3600 * 1000;
const SESSION_STATE_SWEEP_THRESHOLD = 1024;
const FEEDBACK_CONTEXT_MAX_ENTRIES = 1024;

export const DEFAULT_PREFERENCES: AdaptivePreferences = { qualityTier: 2, strengths: [] };

export interface AdaptiveRouterOptions {
  /** Every model the bandit may choose, in first-seen order. */
  availableModels: readonly ModelRef[];
  modelToPrefs: ReadonlyMap<ModelRef, AdaptivePreferences>;
  /** Any consistent unit: costs are normalised against each other before use. */
  modelToCost: ReadonlyMap<ModelRef, number>;
  now?: () => number;
  rng?: Rng;
}

/** On-disk shape of the learned posteriors. */
export interface PersistedCells {
  version: 1;
  cells: { requestType: RequestType; model: ModelRef; alpha: number; beta: number }[];
}

export interface CellSnapshot {
  requestType: RequestType;
  model: ModelRef;
  alpha: number;
  beta: number;
  samples: number;
  qualityMean: number;
}

interface FeedbackContext {
  modelName: ModelRef;
  requestType: RequestType;
  userContent: string | null;
  assistantContent: string | null;
  turnCount: number;
  cleanCreditAwarded: boolean;
  expiresAt: number;
}

function cellKey(requestType: RequestType, model: ModelRef): string {
  return `${requestType}::${model}`;
}

export class AdaptiveRouter {
  readonly availableModels: readonly ModelRef[];
  readonly modelToPrefs: ReadonlyMap<ModelRef, AdaptivePreferences>;
  readonly modelToCost: ReadonlyMap<ModelRef, number>;
  readonly rng: Rng;
  private readonly now: () => number;

  private readonly cells = new Map<string, BanditCell>();
  private readonly sessionStates = new Map<string, SessionState>();
  private readonly sessionStateExpiry = new Map<string, number>();
  private readonly feedbackContexts = new Map<string, FeedbackContext>();

  /** Counters mirrored from upstream's introspection endpoint. */
  feedbackAttributedTotal = 0;
  feedbackWithoutContextTotal = 0;
  crossModelFeedbackTotal = 0;
  responseSignalUpdatesTotal = 0;

  constructor(options: AdaptiveRouterOptions) {
    this.availableModels = [...new Set(options.availableModels)];
    this.modelToPrefs = options.modelToPrefs;
    this.modelToCost = options.modelToCost;
    this.now = options.now ?? Date.now;
    this.rng = options.rng ?? Math.random;
    this.initColdStartCells();
  }

  private prefsFor(model: ModelRef): AdaptivePreferences {
    return this.modelToPrefs.get(model) ?? DEFAULT_PREFERENCES;
  }

  private initColdStartCells(): void {
    for (const requestType of REQUEST_TYPES) {
      for (const model of this.availableModels) {
        this.cells.set(cellKey(requestType, model), initialCell(this.prefsFor(model), requestType));
      }
    }
  }

  cell(requestType: RequestType, model: ModelRef): BanditCell {
    const cell = this.cells.get(cellKey(requestType, model));
    if (cell) return cell;
    // A model the caller names that is not in the pool gets a fresh prior rather than a
    // crash; the pick logic only ever asks about pool models, so this is defensive.
    const fresh = initialCell(this.prefsFor(model), requestType);
    this.cells.set(cellKey(requestType, model), fresh);
    return fresh;
  }

  /** Test/seeding hook: replace a posterior outright. */
  setCell(requestType: RequestType, model: ModelRef, cell: BanditCell): void {
    this.cells.set(cellKey(requestType, model), cell);
  }

  cost(model: ModelRef): number {
    return this.modelToCost.get(model) ?? 0;
  }

  // ── Persistence ────────────────────────────────────────────────────────────

  /** Overlay persisted posteriors on the cold-start priors. Unknown models and request
   *  types are skipped, so a stale file after a config change is harmless. */
  load(persisted: PersistedCells | null | undefined): number {
    if (!persisted || persisted.version !== 1 || !Array.isArray(persisted.cells)) return 0;
    let loaded = 0;
    for (const row of persisted.cells) {
      if (!isRequestType(row.requestType)) continue;
      if (!this.availableModels.includes(row.model)) continue;
      if (!Number.isFinite(row.alpha) || !Number.isFinite(row.beta) || row.alpha <= 0 || row.beta <= 0) continue;
      this.cells.set(cellKey(row.requestType, row.model), { alpha: row.alpha, beta: row.beta });
      loaded++;
    }
    return loaded;
  }

  serialize(): PersistedCells {
    const cells: PersistedCells["cells"] = [];
    for (const requestType of REQUEST_TYPES) {
      for (const model of this.availableModels) {
        const cell = this.cell(requestType, model);
        cells.push({ requestType, model, alpha: cell.alpha, beta: cell.beta });
      }
    }
    return { version: 1, cells };
  }

  /** Human-readable state for `/autoroute adaptive`. */
  snapshot(): CellSnapshot[] {
    const rows: CellSnapshot[] = [];
    for (const requestType of REQUEST_TYPES) {
      for (const model of this.availableModels) {
        const cell = this.cell(requestType, model);
        rows.push({
          requestType,
          model,
          alpha: cell.alpha,
          beta: cell.beta,
          samples: cellTotalSamples(cell),
          qualityMean: cellMean(cell),
        });
      }
    }
    return rows;
  }

  // ── Session state ──────────────────────────────────────────────────────────

  private sessionKey(sessionId: string, model: ModelRef): string {
    return `${sessionId}::${model}`;
  }

  getOrCreateSessionState(sessionId: string, model: ModelRef, requestType: RequestType): SessionState {
    const key = this.sessionKey(sessionId, model);
    const now = this.now();
    if (this.sessionStates.size >= SESSION_STATE_SWEEP_THRESHOLD) this.evictExpiredSessionStates(now);
    let state = this.sessionStates.get(key);
    if (!state) {
      state = newSessionState(sessionId, model, requestType);
      this.sessionStates.set(key, state);
    }
    this.sessionStateExpiry.set(key, now + OWNER_CACHE_TTL_MS);
    return state;
  }

  private evictExpiredSessionStates(now: number): void {
    for (const [key, expiresAt] of this.sessionStateExpiry) {
      if (expiresAt > now) continue;
      this.sessionStates.delete(key);
      this.sessionStateExpiry.delete(key);
    }
  }

  private evictExpiredFeedbackContexts(now: number): void {
    for (const [key, context] of this.feedbackContexts) {
      // Insertion order is age order, so the first live entry ends the sweep.
      if (context.expiresAt > now) break;
      this.feedbackContexts.delete(key);
    }
  }

  /**
   * Record one completed turn and update the posteriors it speaks to.
   *
   * Feedback carried by the user message (a rephrase, a "thanks", a "forget it") is about
   * the *previous* reply, so it is credited to the model that produced it; signals in the
   * response itself (a near-duplicate reply, a tool-call loop) go to the current model.
   * Returns the merged delta, mostly for tests and logging.
   */
  recordTurn(sessionId: string, modelName: ModelRef, requestType: RequestType, turn: Turn): SignalDelta {
    const now = this.now();
    this.evictExpiredFeedbackContexts(now);
    const previous = this.feedbackContexts.get(sessionId) ?? null;
    this.feedbackContexts.delete(sessionId);

    // A follow-up that reads as "general" usually continues the previous ask rather than
    // starting a new kind of work, so it inherits the previous turn's type.
    const effectiveRequestType = previous && requestType === "general" ? previous.requestType : requestType;
    const currentState = this.getOrCreateSessionState(sessionId, modelName, effectiveRequestType);

    const feedbackDelta = detectUserFeedback(
      previous?.userContent ?? null,
      turn.userContent,
      turn.toolResults,
      previous !== null && !previous.cleanCreditAwarded && previous.turnCount + 1 >= MIN_TURNS_FOR_CLEAN_CREDIT,
    );
    const responseDelta = detectResponseSignals(
      previous?.assistantContent ?? null,
      turn.assistantContent,
      currentState.toolCallHistory,
      turn.toolCalls,
      turn.toolResults,
      turn.responseStatus,
    );

    const banditDeltas = new Map<string, { requestType: RequestType; model: ModelRef; delta: SignalDelta }>();
    const addDelta = (type: RequestType, model: ModelRef, delta: SignalDelta) => {
      const key = cellKey(type, model);
      const existing = banditDeltas.get(key);
      banditDeltas.set(key, {
        requestType: type,
        model,
        delta: existing ? mergeSignalDeltas(existing.delta, delta) : delta,
      });
    };

    if (previous) {
      const feedbackState = this.getOrCreateSessionState(sessionId, previous.modelName, previous.requestType);
      applySignalDelta(feedbackState, feedbackDelta);
      if (feedbackDelta.satisfaction) feedbackState.cleanCreditAwarded = true;
      if (anyFired(feedbackDelta)) {
        this.feedbackAttributedTotal++;
        if (previous.modelName !== modelName) this.crossModelFeedbackTotal++;
      }
      addDelta(previous.requestType, previous.modelName, feedbackDelta);
    } else {
      if (anyFired(feedbackDelta)) this.feedbackWithoutContextTotal++;
      // With nothing to attribute feedback to, only a hard failure in this turn's own tool
      // results says anything about the current model.
      const initialFailure: SignalDelta = { ...emptyDelta(), failure: feedbackDelta.failure };
      applySignalDelta(currentState, initialFailure);
      addDelta(effectiveRequestType, modelName, initialFailure);
    }

    applySignalDelta(currentState, responseDelta);
    const [responseAlpha, responseBeta] = banditDelta(responseDelta);
    if (responseAlpha !== 0 || responseBeta !== 0) this.responseSignalUpdatesTotal++;
    addDelta(effectiveRequestType, modelName, responseDelta);
    advanceSessionState(currentState, turn);

    const nextTurnCount = (previous?.turnCount ?? 0) + 1;
    const cleanCreditAwarded = Boolean(previous?.cleanCreditAwarded || feedbackDelta.satisfaction);
    if (this.feedbackContexts.size >= FEEDBACK_CONTEXT_MAX_ENTRIES) {
      const oldest = this.feedbackContexts.keys().next().value;
      if (oldest !== undefined) this.feedbackContexts.delete(oldest);
    }
    this.feedbackContexts.set(sessionId, {
      modelName,
      requestType: effectiveRequestType,
      userContent: turn.userContent,
      assistantContent: turn.assistantContent,
      turnCount: nextTurnCount,
      cleanCreditAwarded,
      expiresAt: now + OWNER_CACHE_TTL_MS,
    });

    let combined = emptyDelta();
    for (const { requestType: type, model, delta } of banditDeltas.values()) {
      combined = mergeSignalDeltas(combined, delta);
      const [deltaAlpha, deltaBeta] = banditDelta(delta);
      if (deltaAlpha === 0 && deltaBeta === 0) continue;
      this.cells.set(cellKey(type, model), applyDelta(this.cell(type, model), deltaAlpha, deltaBeta));
    }
    return combined;
  }
}
