const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const captureGeometry = require('../static/capture_geometry.js');

const PNG_1X1 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

function makeClassList(element) {
  const set = new Set();
  return {
    add(...names) { for (const n of names) set.add(n); },
    remove(...names) { for (const n of names) set.delete(n); },
    toggle(name, force) {
      if (force === undefined) {
        if (set.has(name)) { set.delete(name); return false; }
        set.add(name); return true;
      }
      if (force) set.add(name); else set.delete(name);
      return Boolean(force);
    },
    contains(name) { return set.has(name); },
  };
}

class MockNode {
  constructor(tagName = 'div', id = '') {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = makeClassList(this);
    this.dataset = {};
    this.style = {};
    this.textContent = '';
    this.value = '';
    this.disabled = false;
    this.checked = false;
    this.open = false;
    this.hidden = false;
    this.width = 0;
    this.height = 0;
    this.files = [];
    this.focused = false;
    this._rect = { left: 0, top: 0, width: 400, height: 300 };
    if (this.tagName === 'SELECT') {
      this.options = this.children;
    }
    if (this.tagName === 'TABLE') {
      this._tHead = null;
      this._bodies = [];
    }
  }
  append(...nodes) {
    for (const n of nodes) {
      if (typeof n === 'string') {
        const txt = new MockNode('#text');
        txt.textContent = n;
        this.children.push(txt);
      } else if (n) {
        n.parentElement = this;
        this.children.push(n);
      }
    }
  }
  appendChild(node) { this.append(node); return node; }
  replaceChildren(...nodes) {
    for (const c of this.children) c.parentElement = null;
    this.children.length = 0;
    this.append(...nodes);
  }
  add(option, before = null) {
    option.parentElement = this;
    const idx = before ? this.children.indexOf(before) : -1;
    if (idx >= 0) this.children.splice(idx, 0, option);
    else this.children.push(option);
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  async dispatchEvent(event) {
    const evt = { preventDefault() {}, target: this, ...event };
    for (const fn of this.listeners.get(evt.type) || []) {
      await fn(evt);
    }
  }
  setAttribute(name, val) { this.attributes.set(name, String(val)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  hasAttribute(name) { return this.attributes.has(name); }
  removeAttribute(name) {
    this.attributes.delete(name);
    if (name === 'href') delete this.href;
    if (name === 'src') delete this.src;
  }
  focus() { this.focused = true; }
  setPointerCapture() {}
  getBoundingClientRect() { return this._rect; }
  closest(selector) {
    if (selector === '.work-screen') return new MockNode('section');
    return null;
  }
  matches() { return false; }
  querySelector(selector) {
    if (selector.startsWith('option[value="') && selector.endsWith('"]')) {
      const val = selector.slice('option[value="'.length, -2);
      return this.children.find(c => c.tagName === 'OPTION' && c.value === val) || null;
    }
    const queue = [...this.children];
    while (queue.length) {
      const item = queue.shift();
      if (selector.startsWith('.') && item.classList.contains(selector.slice(1))) return item;
      if (item.tagName.toLowerCase() === selector.toLowerCase()) return item;
      queue.push(...item.children);
    }
    return null;
  }
  createCaption() { const c = new MockNode('caption'); this.append(c); return c; }
  createTHead() {
    if (!this._tHead) {
      this._tHead = new MockNode('thead');
      this._tHead.insertRow = () => { const r = new MockNode('tr'); this._tHead.append(r); return r; };
      this.append(this._tHead);
    }
    return this._tHead;
  }
  createTBody() {
    const b = new MockNode('tbody');
    b.insertRow = () => {
      const r = new MockNode('tr');
      r.insertCell = () => { const td = new MockNode('td'); r.append(td); return td; };
      b.append(r);
      return r;
    };
    this._bodies.push(b);
    this.append(b);
    return b;
  }
  getContext() {
    const self = this;
    return {
      clearRect() {},
      drawImage() {},
      getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
      putImageData() {},
      save() {},
      restore() {},
      beginPath() {},
      rect() {},
      fill() {},
      strokeRect() {},
      fillRect() {},
      set fillStyle(v) { self._fill = v; },
      set strokeStyle(v) { self._stroke = v; },
      set lineWidth(v) { self._lw = v; },
    };
  }
  toDataURL() { return PNG_1X1; }
}

function createImplementationHarness({ inspectResponse, restoreResponse, referenceScorer } = {}) {
  const elements = new Map();
  const byId = id => {
    if (!elements.has(id)) {
      const tag = id === 'picture-face' || id === 'brush-model' || id === 'brush-restoration' || id === 'brush-preparation'
        ? 'select'
        : id === 'brush-editor' || id === 'picture-crop-canvas' ? 'canvas' : 'div';
      const el = new MockNode(tag, id);
      if (id === 'video-download-smoothed-frame') el.parentElement = new MockNode('div');
      if (id === 'brush-restoration') el.value = 'auto';
      if (id === 'brush-preparation') el.value = 'automatic';
      if (id === 'brush-model') el.value = 'codeformer';
      if (id === 'brush-size') el.value = '20';
      if (id === 'brush-radius') el.value = '3';
      if (id === 'brush-action') el.value = 'paint';
      if (id === 'brush-plane') el.value = 'protect';
      elements.set(id, el);
    }
    return elements.get(id);
  };

  let currentInspect = inspectResponse;
  let currentRestore = restoreResponse;
  const windowListeners = new Map();
  const windowObj = {
    captureGeometry,
    addEventListener(type, fn) {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(fn);
    },
    dispatchEvent(evt) {
      for (const fn of windowListeners.get(evt.type) || []) fn(evt);
    },
    scoreVideoReference: referenceScorer,
  };

  class MockImage {
    constructor() { this.src = ''; this.naturalWidth = 0; this.naturalHeight = 0; }
    async decode() {
      if (this.src === PNG_1X1) {
        this.naturalWidth = currentInspect?.width || 256;
        this.naturalHeight = currentInspect?.height || 256;
      } else {
        this.naturalWidth = 256;
        this.naturalHeight = 256;
      }
    }
  }

  class MockOption extends MockNode {
    constructor(text = '', value = '') {
      super('option');
      this.textContent = text;
      this.value = String(value);
    }
  }

  class MockFormData {
    constructor() { this._entries = new Map(); }
    append(k, v) { this._entries.set(k, v); }
    get(k) { return this._entries.get(k); }
  }

  class MockEvent {
    constructor(type, init = {}) { this.type = type; Object.assign(this, init); }
  }

  const sandbox = {
    window: windowObj,
    document: {
      getElementById: byId,
      querySelectorAll: () => [],
      createElement: tag => new MockNode(tag),
    },
    Image: MockImage,
    ImageData: class { constructor(data, w, h) { this.data = data; this.width = w; this.height = h; } },
    Option: MockOption,
    FormData: MockFormData,
    Event: MockEvent,
    CustomEvent: MockEvent,
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {},
    fetch: async url => {
      if (url === '/face/restoration/status') {
        return {
          ok: true,
          json: async () => ({
            available: true,
            maximum_restore_forwards: 1,
            maximum_off_forwards: 0,
            automatic_preparation: false,
            controller: { selection: 'conditioned_sft_v2_p2_u0224' },
          }),
        };
      }
      if (url === '/face/restoration/inspect') {
        return { ok: true, json: async () => currentInspect };
      }
      if (url === '/face/restoration/restore') {
        return { ok: true, json: async () => currentRestore };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    },
    Uint8ClampedArray,
    Uint8Array,
    Uint32Array,
    Number,
    Array,
    Math,
    JSON,
    Error,
    Promise,
    Set,
    Map,
  };

  const source = fs.readFileSync(path.join(__dirname, '..', 'static', 'implementation.js'), 'utf8');
  vm.runInNewContext(source, sandbox, { filename: 'implementation.js' });

  return {
    byId,
    windowObj,
    setInspect: next => { currentInspect = next; },
    setRestore: next => { currentRestore = next; },
    flushMicrotasks: () => new Promise(resolve => setTimeout(resolve, 0)),
  };
}

test('manual crop validates 16..1600 px bounds, applies, cancels, and switches #picture-face options', async () => {
  const h = createImplementationHarness({
    inspectResponse: {
      width: 240,
      height: 180,
      source: PNG_1X1,
      neural_calls: 0,
      face_crops: [[20, 10, 80, 90]],
    },
  });
  await h.flushMicrotasks();
  const pictureFile = { size: 1024, type: 'image/png', name: 'portrait.png' };
  assert.equal(await h.windowObj.restorationInput.use(pictureFile), true);
  assert.equal(h.byId('brush-editor').width, 80);
  assert.equal(h.byId('brush-editor').height, 90);
  assert.equal(h.byId('picture-manual-crop').disabled, false);

  // Enter manual crop: pre-populates draft from Face 1 crop [20, 10, 80, 90]
  await h.byId('picture-manual-crop').dispatchEvent({ type: 'click' });
  assert.equal(h.byId('picture-preview').classList.contains('is-manual-crop'), true);
  assert.equal(h.byId('picture-crop-canvas').classList.contains('hidden'), false);
  assert.equal(h.byId('picture-manual-apply').disabled, false);
  assert.equal(h.byId('picture-manual-size').textContent, '80 × 90 px at (20, 10)');

  // Cancel exits without creating manual option or changing Face 1 crop
  await h.byId('picture-manual-cancel').dispatchEvent({ type: 'click' });
  assert.equal(h.byId('picture-preview').classList.contains('is-manual-crop'), false);
  assert.equal(h.byId('picture-crop-canvas').width, 0);
  assert.equal(h.byId('picture-face').querySelector('option[value="manual"]'), null);
  assert.equal(h.byId('brush-editor').width, 80);

  // Switch to Full picture (-1), then enter manual crop -> starts with empty draft
  h.byId('picture-face').value = '-1';
  await h.byId('picture-face').dispatchEvent({ type: 'change' });
  assert.equal(h.byId('brush-editor').width, 240);

  await h.byId('picture-manual-crop').dispatchEvent({ type: 'click' });
  assert.equal(h.byId('picture-manual-apply').disabled, true);

  // Dragging too small (<16 px) keeps Apply disabled with guidance
  const cropCanvas = h.byId('picture-crop-canvas');
  cropCanvas._rect = { left: 0, top: 0, width: 240, height: 180 };
  await cropCanvas.dispatchEvent({ type: 'pointerdown', button: 0, clientX: 10, clientY: 10, pointerId: 1 });
  await cropCanvas.dispatchEvent({ type: 'pointermove', clientX: 20, clientY: 20, pointerId: 1 });
  await cropCanvas.dispatchEvent({ type: 'pointerup', pointerId: 1 });
  assert.equal(h.byId('picture-manual-apply').disabled, true);
  assert.match(h.byId('picture-manual-size').textContent, /16–1600 px per side/i);

  // Dragging valid 100x80 rect far from prior draft enables Apply; ArrowRight and Shift+ArrowDown adjust within bounds
  await cropCanvas.dispatchEvent({ type: 'pointerdown', button: 0, clientX: 80, clientY: 60, pointerId: 2 });
  await cropCanvas.dispatchEvent({ type: 'pointermove', clientX: 180, clientY: 140, pointerId: 2 });
  await cropCanvas.dispatchEvent({ type: 'pointerup', pointerId: 2 });
  assert.equal(h.byId('picture-manual-apply').disabled, false);
  await cropCanvas.dispatchEvent({ type: 'keydown', key: 'ArrowRight', shiftKey: false });
  await cropCanvas.dispatchEvent({ type: 'keydown', key: 'ArrowDown', shiftKey: true });
  assert.equal(h.byId('picture-manual-size').textContent, '100 × 81 px at (81, 60)');

  // Apply creates manual option before -1, selects it, and updates canvas dimensions
  await h.byId('picture-manual-apply').dispatchEvent({ type: 'click' });
  const choices = h.byId('picture-face');
  assert.equal(choices.value, 'manual');
  assert.equal(choices.querySelector('option[value="manual"]').textContent, 'Manual crop');
  assert.equal(choices.children.map(o => o.value).join(','), '0,manual,-1');
  assert.equal(h.byId('picture-face-status').textContent, 'Using manual crop (100 × 81 px).');
  assert.equal(h.byId('brush-editor').width, 100);
  assert.equal(h.byId('brush-editor').height, 81);

  // Switching to Face 1 and back to Manual crop preserves manual coordinates
  choices.value = '0';
  await choices.dispatchEvent({ type: 'change' });
  assert.equal(h.byId('brush-editor').width, 80);
  choices.value = 'manual';
  await choices.dispatchEvent({ type: 'change' });
  assert.equal(h.byId('brush-editor').width, 100);
  assert.equal(h.byId('brush-editor').height, 81);

  // Image smaller than 16x16 disables manual crop creation via Enter
  h.setInspect({ width: 12, height: 12, source: PNG_1X1, neural_calls: 0, face_crops: [] });
  assert.equal(await h.windowObj.restorationInput.use(pictureFile), true);
  await h.byId('picture-manual-crop').dispatchEvent({ type: 'click' });
  await cropCanvas.dispatchEvent({ type: 'keydown', key: 'Enter', shiftKey: false });
  assert.equal(h.byId('picture-manual-apply').disabled, true);
  assert.match(h.byId('picture-manual-size').textContent, /at least 16 × 16 px/i);

  // Max-size image (1600x1600): Enter creates 800x800 draft and #brush-continue applies it
  h.setInspect({ width: 1600, height: 1600, source: PNG_1X1, neural_calls: 0, face_crops: [] });
  assert.equal(await h.windowObj.restorationInput.use(pictureFile), true);
  await h.byId('picture-manual-crop').dispatchEvent({ type: 'click' });
  await cropCanvas.dispatchEvent({ type: 'keydown', key: 'Enter', shiftKey: false });
  assert.equal(h.byId('picture-manual-apply').disabled, false);
  await h.byId('brush-continue').dispatchEvent({ type: 'click' });
  assert.equal(h.byId('picture-face').value, 'manual');
  assert.equal(h.byId('brush-editor').width, 800);
});

test('restorationMetrics renders 2-column and 3-column diagnostics, native noise, and reference scores', async () => {
  let nextScores = { before: { mae: '12.34', psnr: '26.30' }, after: { mae: '4.56', psnr: '34.95' } };
  const h = createImplementationHarness({
    inspectResponse: { width: 256, height: 256, source: PNG_1X1, neural_calls: 0, face_crops: [] },
    restoreResponse: {
      original_input: PNG_1X1,
      prepared_input: PNG_1X1,
      result: PNG_1X1,
      metadata: {
        neural_forward_returns: 1,
        neural_forward_attempts: 1,
        selection: 'conditioned_sft_v2_p2_u0224',
        restoration_applied: true,
        preparation_applied: false,
        seconds: 1.2,
        output_diagnostics: {
          before: { blur_variance: 18.5, noise_sigma_255: 3.2, noise_sigma_native: 11.45 },
          after: { blur_variance: 42.1, noise_sigma_255: 1.1 },
        },
      },
    },
    referenceScorer: async () => nextScores,
  });
  await h.flushMicrotasks();
  assert.equal(await h.windowObj.restorationInput.use({ size: 1024, type: 'image/png', name: 'p.png' }), true);

  // Case 1: Without prepared diagnostics -> Metric | Before | After, native noise shown (>= 0.5 delta)
  await h.byId('brush-form').dispatchEvent({ type: 'submit' });
  const container = h.byId('picture-result-metrics');
  const table1 = container.children[0];
  const headers1 = table1.querySelector('thead').children[0].children.map(c => c.textContent);
  assert.deepEqual(headers1, ['Metric', 'Before', 'After']);
  const rows1 = table1._bodies[0].children;
  assert.equal(rows1[1].children[1].textContent, '3.20 (native 11.45)');
  assert.equal(rows1[1].children[2].textContent, '1.10');

  const refInput1 = container.querySelector('input');
  refInput1.files = [{ name: 'ref.png', type: 'image/png', size: 100 }];
  await refInput1.dispatchEvent({ type: 'change' });
  assert.equal(rows1[2].children[1].textContent, '12.34');
  assert.equal(rows1[2].children[2].textContent, '4.56');
  assert.equal(rows1[3].children[1].textContent, '26.30');
  assert.equal(rows1[3].children[2].textContent, '34.95');

  // Case 2: With prepared diagnostics -> Metric | Before | Prepared | After, native omitted (< 0.5 delta)
  h.setRestore({
    original_input: PNG_1X1,
    prepared_input: PNG_1X1,
    result: PNG_1X1,
    metadata: {
      neural_forward_returns: 1,
      neural_forward_attempts: 1,
      selection: 'conditioned_sft_v2_p2_u0224',
      restoration_applied: true,
      preparation_applied: true,
      seconds: 1.4,
      output_diagnostics: {
        before: { blur_variance: 20.0, noise_sigma_255: 4.0, noise_sigma_native: 4.25 },
        prepared: { blur_variance: 15.0, noise_sigma_255: 1.8 },
        after: { blur_variance: 40.0, noise_sigma_255: 1.0 },
      },
    },
  });
  await h.byId('brush-form').dispatchEvent({ type: 'submit' });
  const table2 = container.children[0];
  const headers2 = table2.querySelector('thead').children[0].children.map(c => c.textContent);
  assert.deepEqual(headers2, ['Metric', 'Before', 'Prepared', 'After']);
  const rows2 = table2._bodies[0].children;
  assert.equal(rows2[1].children[1].textContent, '4.00');
  assert.equal(rows2[1].children[2].textContent, '1.80');
  assert.equal(rows2[1].children[3].textContent, '1.00');

  const refInput2 = container.querySelector('input');
  refInput2.files = [{ name: 'ref.png', type: 'image/png', size: 100 }];
  await refInput2.dispatchEvent({ type: 'change' });
  assert.equal(rows2[2].children[1].textContent, '12.34');
  assert.equal(rows2[2].children[2].textContent, '—');
  assert.equal(rows2[2].children[3].textContent, '4.56');
  assert.equal(rows2[3].children[1].textContent, '26.30');
  assert.equal(rows2[3].children[2].textContent, '—');
  assert.equal(rows2[3].children[3].textContent, '34.95');

  // Clearing reference resets Before and After back to em dash
  refInput2.files = [];
  await refInput2.dispatchEvent({ type: 'change' });
  assert.equal(rows2[2].children[1].textContent, '—');
  assert.equal(rows2[2].children[3].textContent, '—');
});
