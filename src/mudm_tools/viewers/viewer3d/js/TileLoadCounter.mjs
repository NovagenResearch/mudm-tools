// Pure per-burst tile-load gauge. No DOM, no timers — unit-tested with `node --test`.
// Fed a single "outstanding" number (in-flight + queued tiles) by each viewer's adapter.
// Per-burst semantics: total = max cumulative work seen since idle; loaded = total - outstanding.
// A navigation (zoom/pan) re-baselines the burst so an abandoned view's progress can't inflate
// the new view's denominator — see setOutstanding's viewChanged argument.
export class TileLoadCounter {
  constructor() { this._loaded = 0; this._total = 0; this._active = false; this._navigating = false; }

  get state() { return { loaded: this._loaded, total: this._total, active: this._active }; }

  /**
   * Report the current number of outstanding tiles. Returns the new state.
   * @param {number} k             outstanding tiles (in-flight + queued).
   * @param {boolean} [viewChanged] true on frames where the view (camera/zoom) is changing.
   *   On the RISING edge of a navigation the burst is re-baselined to the new view: the stale
   *   progress is dropped so the denominator reflects only what the new view needs (and can
   *   therefore SHRINK on a zoom-out rather than accumulate). Frames where it stays true within
   *   one continuous gesture re-baseline only once, then the burst counts up normally.
   */
  setOutstanding(k, viewChanged = false) {
    k = Math.max(0, Math.trunc(Number(k) || 0));
    if (viewChanged && !this._navigating) this._rebase();  // new gesture → drop the stale burst
    this._navigating = viewChanged;
    if (k > 0) {
      if (!this._active) { this._active = true; this._loaded = 0; this._total = 0; }
      this._total = Math.max(this._total, this._loaded + k);
      this._loaded = this._total - k;
    } else if (this._active) {
      this._loaded = this._total;   // land on x == y for the final frame
      this._active = false;
    }
    return this.state;
  }

  _rebase() { this._loaded = 0; this._total = 0; this._active = false; }

  /** Force back to idle (e.g. on dataset switch, or a 2D navigation gesture). */
  reset() { this._navigating = false; this._rebase(); return this.state; }
}
