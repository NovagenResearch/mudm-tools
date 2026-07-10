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

test('total grows if more work arrives mid-burst (continuous pan)', () => {
  const c = new TileLoadCounter();
  c.setOutstanding(5);          // total 5
  c.setOutstanding(2);          // loaded 3 / 5
  const s = c.setOutstanding(8); // more requested before idle
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
