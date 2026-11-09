const test = require('node:test');
const assert = require('node:assert/strict');
const { createQueue } = require('../static/restoration_queue.js');

const crop = name => ({ file: { name }, captureId: `capture-${name}`, label: name });
const ids = queue => queue.items.map(item => item.id);
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

test('bounds the queue and keeps each selected crop independently addressable', () => {
  const queue = createQueue(2);
  queue.add(crop('first'));
  queue.add(crop('second'));
  assert.equal(queue.items.length, 2);
  assert.equal(new Set(ids(queue)).size, 2);
  assert.ok(queue.items.every(item => item.id && typeof item.status === 'string'));
  assert.throws(() => queue.add(crop('third')));
  queue.remove(ids(queue)[0]);
  queue.add(crop('third'));
  assert.equal(queue.items.length, 2);
});

test('adds all four detected crops in order as separate pending reviews', async () => {
  const queue = createQueue(4);
  const detected = ['face-1', 'face-2', 'face-3', 'face-4'].map(crop);
  assert.equal(queue.remaining, 4);

  queue.addMany(detected);

  assert.deepEqual(queue.items.map(item => item.captureId), detected.map(item => item.captureId));
  assert.equal(new Set(ids(queue)).size, 4);
  assert.deepEqual(queue.items.map(item => item.status), Array(4).fill('pending'));
  assert.equal(queue.remaining, 0);
  const lastId = ids(queue).at(-1);
  for (const id of ids(queue).slice(0, -1)) queue.approve(id, { id });
  const restored = [];
  await assert.rejects(queue.run(async (_, item) => { restored.push(item.id); }), /review or skip/i);
  assert.deepEqual(restored, []);
  queue.skip(lastId);
  await queue.run(async (_, item) => { restored.push(item.id); });
  assert.deepEqual(restored, ids(queue).slice(0, -1));
});

test('rejects a batch larger than the remaining capacity without adding any crop', () => {
  const queue = createQueue(3);
  const firstId = queue.add(crop('existing'));
  queue.approve(firstId, { mode: 'on' });
  const before = queue.items;
  const detected = ['face-1', 'face-2', 'face-3'].map(crop);

  assert.throws(() => queue.addMany(detected), /holds up to 3 crops/i);
  assert.deepEqual(queue.items, before);
  assert.equal(queue.remaining, 2);
  queue.addMany(detected.slice(0, 2));
  assert.deepEqual(queue.items.map(item => item.captureId), [before[0].captureId, ...detected.slice(0, 2).map(item => item.captureId)]);
  assert.equal(queue.remaining, 0);
});

test('refuses to run until every crop has a review decision', async () => {
  const queue = createQueue();
  queue.add(crop('first'));
  queue.add(crop('second'));
  queue.approve(ids(queue)[0], { mode: 'on' });
  let calls = 0;
  await assert.rejects(queue.run(async () => { calls++; }));
  assert.equal(calls, 0);
  queue.skip(ids(queue)[1]);
  await queue.run(async () => { calls++; return 'done'; });
  assert.equal(calls, 1);
});

test('runs approved crops in selection order with their own requests and results', async () => {
  const queue = createQueue();
  const first = crop('first'), skipped = crop('skipped'), last = crop('last');
  for (const entry of [first, skipped, last]) queue.add(entry);
  const [firstId, skippedId, lastId] = ids(queue);
  const firstRequest = { restoration: 'on' }, lastRequest = { restoration: 'off' };
  queue.approve(lastId, lastRequest);
  queue.skip(skippedId);
  queue.approve(firstId, firstRequest);
  const started = deferred(), release = deferred(), calls = [];
  const running = queue.run(async (request, item) => {
    calls.push({ request, item });
    if (item.id === firstId) { started.resolve(); await release.promise; }
    return `result-${item.id}`;
  });
  await started.promise;
  assert.deepEqual(calls.map(call => call.item.id), [firstId]);
  release.resolve();
  await running;
  assert.deepEqual(calls.map(call => call.item.id), [firstId, lastId]);
  assert.deepEqual(calls.map(call => call.request), [firstRequest, lastRequest]);
  assert.equal(calls[0].item.file, first.file);
  assert.equal(calls[0].item.captureId, first.captureId);
  const items = queue.items;
  assert.equal(items.find(item => item.id === firstId).result, `result-${firstId}`);
  assert.equal(items.find(item => item.id === lastId).result, `result-${lastId}`);
  assert.equal(items.find(item => item.id === skippedId).result, undefined);
});

test('stops on failure and leaves later approved crops unprocessed', async () => {
  const queue = createQueue();
  for (const name of ['first', 'second', 'third']) queue.add(crop(name));
  const [firstId, secondId, thirdId] = ids(queue);
  for (const id of ids(queue)) queue.approve(id, { id });
  const beforeThird = queue.items.find(item => item.id === thirdId).status;
  const failure = new Error('worker failed');
  const calls = [];
  await assert.rejects(queue.run(async (request, item) => {
    calls.push(item.id);
    if (item.id === secondId) throw failure;
    return 'first result';
  }), error => error === failure);
  assert.deepEqual(calls, [firstId, secondId]);
  assert.equal(queue.items.find(item => item.id === firstId).result, 'first result');
  assert.equal(queue.items.find(item => item.id === thirdId).result, undefined);
  assert.equal(queue.items.find(item => item.id === thirdId).status, beforeThird);
});

test('rejects edits and concurrent execution while a crop is processing', async () => {
  const queue = createQueue();
  queue.add(crop('first'));
  const [id] = ids(queue);
  queue.approve(id, { mode: 'on' });
  const started = deferred(), release = deferred();
  const running = queue.run(async () => { started.resolve(); await release.promise; return 'done'; });
  await started.promise;
  assert.throws(() => queue.add(crop('other')));
  assert.throws(() => queue.addMany([crop('other')]), /queue is running/i);
  assert.throws(() => queue.approve(id, { mode: 'off' }));
  assert.throws(() => queue.skip(id));
  assert.throws(() => queue.remove(id));
  await assert.rejects(queue.run(async () => 'duplicate'));
  release.resolve();
  await running;
  assert.equal(queue.items[0].result, 'done');
});
