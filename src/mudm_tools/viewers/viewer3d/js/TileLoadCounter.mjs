// Pure per-burst tile-load gauge. No DOM, no timers — unit-tested with `node --test`.
// Fed a single "outstanding" number (in-flight + queued tiles) by each viewer's adapter.
// Per-burst semantics: total = max outstanding seen since idle; loaded = total - outstanding.
export class TileLoadCounter {
  constructor() { this._loaded = 0; this._total = 0; this._active = false; }

  get state() { return { loaded: this._loaded, total: this._total, active: this._active }; }

  /** Report the current number of outstanding tiles. Returns the new state. */
  setOutstanding(k) {
    k = Math.max(0, Math.trunc(Number(k) || 0));
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

  /** Force back to idle (e.g. on dataset switch). */
  reset() { this._loaded = 0; this._total = 0; this._active = false; return this.state; }
}
