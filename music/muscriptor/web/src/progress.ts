/**
 * Smooths the backend's coarse chunk-completion anchors into a continuous
 * progress fraction + ETA.
 *
 * The backend emits one `{completed, total}` anchor per fixed-size audio chunk
 * (plus a `completed: 0` anchor up front). Chunks cover equal *input* time but
 * take wildly uneven *wall-clock* time, so we can't predict pace from the audio
 * — we have to measure it here. We keep an EMA of per-chunk wall-clock duration
 * and use it both to interpolate the bar *within* the current chunk and to
 * estimate the time remaining. Before the first chunk finishes there's no EMA
 * yet, so the bar creeps along an asymptotic curve that never reaches the next
 * anchor.
 */

const EMA_ALPHA = 0.4; // weight on the most recent chunk's duration
const WITHIN_CAP = 0.95; // within-chunk interpolation never reaches the next anchor
const CREEP_TAU_MS = 4000; // time constant for the pre-EMA asymptotic creep

export class ProgressEstimator {
  private total = 0;
  private completed = 0;
  private emaChunkMs: number | null = null;
  private lastTickMs = 0; // perf time of the last anchor (incl. the t0 anchor)
  private active = false;

  /** Start of a fresh transcription — forget everything. */
  reset(): void {
    this.total = 0;
    this.completed = 0;
    this.emaChunkMs = null;
    this.lastTickMs = 0;
    this.active = false;
  }

  /** Feed a chunk-completion anchor. `now` should be `performance.now()`. */
  onAnchor(completed: number, total: number, now: number): void {
    this.total = total;
    if (this.active && completed > this.completed) {
      // Wall-clock spent on the chunk(s) that just finished (usually 1, since
      // the web path runs batch_size=1).
      const perChunk = (now - this.lastTickMs) / (completed - this.completed);
      this.emaChunkMs =
        this.emaChunkMs == null
          ? perChunk
          : EMA_ALPHA * perChunk + (1 - EMA_ALPHA) * this.emaChunkMs;
    }
    this.completed = completed;
    this.lastTickMs = now;
    this.active = true;
  }

  /**
   * Exact fraction of chunks completed, with no within-chunk smoothing. Unlike
   * `fraction`, this only advances when an anchor lands — so it stays in lockstep
   * with the notes the backend emits alongside that same anchor.
   */
  completedFraction(): number {
    if (!this.active || this.total === 0) return 0;
    return Math.min(this.completed / this.total, 1);
  }

  /** Smoothed progress fraction in [0, 1] at time `now`. */
  fraction(now: number): number {
    if (!this.active || this.total === 0) return 0;
    if (this.completed >= this.total) return 1;
    const base = this.completed / this.total;
    const elapsed = now - this.lastTickMs;
    const within =
      this.emaChunkMs && this.emaChunkMs > 0
        ? Math.min(elapsed / this.emaChunkMs, WITHIN_CAP)
        : (1 - Math.exp(-elapsed / CREEP_TAU_MS)) * WITHIN_CAP;
    return Math.min(base + within / this.total, 0.999);
  }

  /** Estimated time remaining in ms, or null while still unknown. */
  etaMs(now: number): number | null {
    if (!this.active) return null;
    if (this.completed >= this.total) return 0;
    if (!this.emaChunkMs) return null; // no measured pace yet
    const remaining = this.total - this.completed;
    return Math.max(0, this.emaChunkMs * remaining - (now - this.lastTickMs));
  }
}

/** Seconds as a m:ss clock, e.g. 15 → "0:15", 62 → "1:02". */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
