const test = require('node:test');
const assert = require('node:assert/strict');
const { makeView, pointerToSource } = require('../static/capture_geometry.js');

test('full frame view retains native source dimensions', () => {
  assert.deepEqual(makeView(1920, 1080), { x: 0, y: 0, width: 1920, height: 1080 });
});

test('selected face adds 20 percent padding without changing the input', () => {
  const selected = { x: 100, y: 200, width: 100, height: 50 };
  assert.deepEqual(makeView(800, 600, selected), { x: 80, y: 190, width: 140, height: 70 });
  assert.deepEqual(selected, { x: 100, y: 200, width: 100, height: 50 });
});

test('padding clamps at each frame edge', () => {
  assert.deepEqual(makeView(100, 80, { x: 0, y: 0, width: 50, height: 30 }),
    { x: 0, y: 0, width: 60, height: 36 });
  assert.deepEqual(makeView(100, 80, { x: 80, y: 60, width: 20, height: 20 }),
    { x: 76, y: 56, width: 24, height: 24 });
});

test('fractional padding produces bounded integer dimensions and at least one pixel', () => {
  const view = makeView(15, 12, { x: 1, y: 1, width: 3, height: 2 });
  for (const key of ['x', 'y', 'width', 'height']) assert.ok(Number.isInteger(view[key]), key);
  assert.ok(view.width >= 1 && view.height >= 1);
  assert.ok(view.x >= 0 && view.y >= 0 && view.x + view.width <= 15 && view.y + view.height <= 12);
  assert.ok(view.x <= 1 && view.y <= 1 && view.x + view.width >= 4 && view.y + view.height >= 3);
});

test('pointer inversion accounts for horizontal letterboxing', () => {
  const bounds = { left: 10, top: 20, width: 300, height: 200 };
  const view = { x: 50, y: 25, width: 100, height: 100 };
  assert.deepEqual(pointerToSource(160, 120, bounds, view), { x: 100, y: 75, scale: 2 });
  assert.deepEqual(pointerToSource(10, 20, bounds, view), { x: 50, y: 25, scale: 2 });
  assert.deepEqual(pointerToSource(310, 220, bounds, view), { x: 150, y: 125, scale: 2 });
});

test('pointer inversion accounts for vertical letterboxing and rounds native pixels', () => {
  const bounds = { left: 20, top: 10, width: 400, height: 300 };
  const view = { x: 100, y: 50, width: 200, height: 100 };
  assert.deepEqual(pointerToSource(220, 160, bounds, view), { x: 200, y: 100, scale: 2 });
  assert.deepEqual(pointerToSource(21, 61, bounds, view), { x: 101, y: 51, scale: 2 });
  assert.deepEqual(pointerToSource(420, 310, bounds, view), { x: 300, y: 150, scale: 2 });
});

test('zoom center maps to selected face center and drag delta stays in source pixels', () => {
  const view = makeView(800, 600, { x: 100, y: 200, width: 100, height: 50 });
  const bounds = { left: 50, top: 30, width: 280, height: 140 };
  const start = pointerToSource(190, 100, bounds, view);
  const end = pointerToSource(210, 110, bounds, view);
  assert.deepEqual(start, { x: 150, y: 225, scale: 2 });
  assert.equal(end.x - start.x, 10);
  assert.equal(end.y - start.y, 5);
});

test('invalid frame, rectangle, and hidden preview fail explicitly', () => {
  assert.throws(() => makeView(0, 100), RangeError);
  assert.throws(() => makeView(100, 100, { x: 0, y: 0, width: NaN, height: 10 }), RangeError);
  assert.throws(() => pointerToSource(0, 0, { left: 0, top: 0, width: 0, height: 100 },
    { x: 0, y: 0, width: 100, height: 100 }), RangeError);
});
