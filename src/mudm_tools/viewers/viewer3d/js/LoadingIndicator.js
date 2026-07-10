// Renders the bottom-center loading pill from a TileLoadCounter state, and owns the
// hide debounce (so the pill doesn't flicker between back-to-back bursts). DOM only —
// verified via Playwright, not the unit test.
export class LoadingIndicator {
  constructor(el, { settleMs = 250 } = {}) {
    this.el = el;                 // a hidden <div class="loading-pill"> with a .loading-pill__count child
    this.settleMs = settleMs;
    this._hideTimer = null;
    this._countEl = el ? el.querySelector('.loading-pill__count') : null;
  }

  update({ loaded, total, active }) {
    if (!this.el) return;
    if (active && total > 0) {
      this._cancelHide();
      const text = `Loading ${loaded} of ${total} tiles`;
      if (this._countEl) this._countEl.textContent = text; else this.el.textContent = text;
      this.el.hidden = false;
    } else {
      this._scheduleHide();
    }
  }

  _scheduleHide() {
    if (this._hideTimer || !this.el || this.el.hidden) return;
    this._hideTimer = setTimeout(() => { this.el.hidden = true; this._hideTimer = null; }, this.settleMs);
  }
  _cancelHide() { if (this._hideTimer) { clearTimeout(this._hideTimer); this._hideTimer = null; } }
}
