import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TileLoadCounter } from './TileLoadCounter.mjs';

test('starts idle', () => {
  const c = new TileLoadCounter();
  assert.deepEqual(c.state, { loaded: 0, total: 0, active: false });
});

test('a burst activates and reports x of y', () => {
  const c = new TileLoadCounter();
  assert.deepEqual(c.setOutstanding(5), { loaded: 0, total: 5, active: true });
  assert.deepEqual(c.setOutstanding(2), { loaded: 3, total: 5, active: true }); // 2 still outstanding
});

test('within one gesture (no navigation) total grows as more work is discovered', () => {
  // Progressive LOD within a single continuous gesture legitimately adds tiles, so the
  // denominator tracks the cumulative work and the fraction never jumps backwards. The
  // guard against inflation is the per-navigation re-baseline (see the tests below), not
  // capping growth here.
  const c = new TileLoadCounter();
  c.setOutstanding(5);          // total 5
  c.setOutstanding(2);          // loaded 3 / 5
  const s = c.setOutstanding(8); // more requested before idle, no navigation
  assert.equal(s.total, 11);    // 3 loaded + 8 outstanding
  assert.equal(s.loaded, 3);
  assert.equal(s.active, true);
});

test('reaching 0 outstanding ends the burst (x == y, then inactive)', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(4);
  const s = c.setOutstanding(0);
  assert.deepEqual(s, { loaded: 4, total: 4, active: false });
});

test('a new burst after idle resets the counts', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(4);
  c.setOutstanding(0);
  assert.deepEqual(c.setOutstanding(3), { loaded: 0, total: 3, active: true });
});

test('setOutstanding(0) while already idle is a no-op', () => {
  const c = new TileLoadCounter();
  assert.deepEqual(c.setOutstanding(0), { loaded: 0, total: 0, active: false });
});

test('negative / fractional inputs are clamped to a non-negative int', () => {
  const c = new TileLoadCounter();
  assert.deepEqual(c.setOutstanding(-3), { loaded: 0, total: 0, active: false });
  assert.equal(c.setOutstanding(2.9).total, 2);
});

test('reset() forces idle', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(9);
  assert.deepEqual(c.reset(), { loaded: 0, total: 0, active: false });
});

// --- Navigation re-baselining (a zoom-out must not inflate the denominator) ----------

test('a navigation mid-burst re-baselines to the new view (no carry-over inflation)', () => {
  // Reproduces the reported 27 -> 54 bug: a burst is in progress (some tiles loaded), then
  // the user zooms out, adding a fresh set of tiles for the new view. Without re-baselining,
  // the abandoned burst's progress (loaded) inflates the new denominator (loaded + k). The
  // viewChanged flag restarts the burst against the new view instead.
  const c = new TileLoadCounter();
  c.setOutstanding(27);                    // 0 / 27  (fine view)
  c.setOutstanding(7);                     // 20 / 27 (loading)
  const s = c.setOutstanding(27, true);    // zoom-out: 27 tiles for the NEW view
  assert.deepEqual(s, { loaded: 0, total: 27, active: true }); // NOT 20 / 47
});

test('a zoom-out to a smaller view shrinks the denominator', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(27);                    // 0 / 27
  c.setOutstanding(10);                    // 17 / 27
  const s = c.setOutstanding(6, true);     // coarser view needs only 6 tiles
  assert.equal(s.total, 6);                // decreased — did not stay at 27 (or grow)
  assert.equal(s.loaded, 0);
});

test('navigation re-baselines only on the rising edge (once per gesture)', () => {
  // viewChanged stays true for every frame of a continuous gesture; the burst must
  // re-baseline once (at the start) and then count up — not reset to 0 every frame.
  const c = new TileLoadCounter();
  c.setOutstanding(10);                    // 0 / 10
  c.setOutstanding(3);                     // 7 / 10
  assert.deepEqual(c.setOutstanding(20, true), { loaded: 0, total: 20, active: true }); // rebased
  const s = c.setOutstanding(15, true);    // same gesture, tiles now loading
  assert.equal(s.loaded, 5);               // 20 - 15, counting up (no re-rebase)
  assert.equal(s.total, 20);
});

test('the navigation flag clears when it goes false, so the next gesture re-baselines', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(10);                    // 0 / 10
  c.setOutstanding(20, true);              // gesture 1 → rebased → 0 / 20
  c.setOutstanding(5, false);              // motion stopped, 15 / 20
  const s = c.setOutstanding(30, true);    // gesture 2 → rising edge again → rebased
  assert.deepEqual(s, { loaded: 0, total: 30, active: true });
});

test('viewChanged with 0 outstanding re-baselines to idle without a spurious burst', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(8);                     // 0 / 8
  c.setOutstanding(2);                     // 6 / 8
  const s = c.setOutstanding(0, true);     // navigated away with nothing outstanding
  assert.deepEqual(s, { loaded: 0, total: 0, active: false });
});
