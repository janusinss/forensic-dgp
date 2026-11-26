'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class MockClassList {
  constructor() { this._set = new Set(); }
  add(...items) { items.forEach(i => this._set.add(i)); }
  remove(...items) { items.forEach(i => this._set.delete(i)); }
  toggle(item, force) {
    if (force === undefined) {
      if (this._set.has(item)) this._set.delete(item);
      else this._set.add(item);
    } else if (force) this._set.add(item);
    else this._set.delete(item);
  }
  contains(item) { return this._set.has(item); }
}

class MockNode {
  constructor(tag, id = '') {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.classList = new MockClassList();
    this.attributes = new Map();
    this.listeners = new Map();
    this.children = [];
    this.textContent = '';
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.style = {};
    this.dataset = {};
    this.src = '';
    this.href = '';
    this.width = 256;
    this.height = 256;
  }
  setAttribute(k, v) { this.attributes.set(k, String(v)); if (k === 'src') this.src = String(v); }
  getAttribute(k) { return k === 'src' ? (this.src || null) : (this.attributes.get(k) ?? null); }
  removeAttribute(k) { this.attributes.delete(k); if (k === 'src') this.src = ''; if (k === 'href') this.href = ''; }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  async dispatchEvent(evt) {
    for (const fn of this.listeners.get(evt.type) || []) await fn(evt);
  }
  closest() { return null; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  replaceChildren() { this.children = []; }
  append() {}
  prepend() {}
  focus() {}
  getContext() {
    return {
      clearRect() {},
      drawImage() {},
      getImageData: () => ({ data: new Uint8ClampedArray(256 * 256 * 4), width: 256, height: 256 }),
      putImageData() {},
    };
  }
  async decode() {}
}

function buildCompletionHarness({ fetchImpl, locationObj = undefined }) {
  const elements = new Map();
  const screens = ['inspect', 'options', 'run', 'result', 'completion-paint', 'completion-tools', 'completion-confirm', 'completion-result'].map(name => {
    const s = new MockNode('section');
    s.dataset = { screen: name };
    s.querySelector = sel => sel === 'h1' ? new MockNode('h1') : sel === '.feedback-slot' ? new MockNode('div') : null;
    s.closest = sel => {
      if (sel === '.hidden') {
        if (['completion-paint', 'completion-tools', 'completion-confirm'].includes(name) && elements.get('completion-editor-section')?.classList.contains('hidden')) {
          return elements.get('completion-editor-section');
        }
        if (name === 'completion-result' && s.classList.contains('hidden')) return s;
        return null;
      }
      return s;
    };
    if (name === 'completion-result') {
      s.id = 'completion-result-section';
      s.classList.add('hidden');
      elements.set('completion-result-section', s);
    }
    return s;
  });

  const confirmScreen = screens.find(s => s.dataset.screen === 'completion-confirm');
  const completionStage = new MockNode('div', 'completion-stage');
  completionStage.classList.add('hidden');
  completionStage.contains = node => screens.some(s => s.dataset.screen.startsWith('completion-') && s === node);
  elements.set('completion-stage', completionStage);

  const observerCallbacks = new Map();
  const byId = id => {
    if (!elements.has(id)) {
      const tag = id === 'brush-form' ? 'form' : id === 'completion-editor' ? 'canvas' : 'div';
      const el = new MockNode(tag, id);
      if (id === 'brush-form') el.contains = () => false;
      if (id === 'completion-loading') el.closest = sel => sel === '.work-screen' ? confirmScreen : null;
      if (id === 'completion-editor-section' || id === 'completion-result-section') el.classList.add('hidden');
      if (id === 'completion-size') el.value = '17';
      if (id === 'completion-action') el.value = 'paint';
      elements.set(id, el);
    }
    return elements.get(id);
  };

  const resultSection = byId('completion-result-section');
  const origRemove = resultSection.classList.remove.bind(resultSection.classList);
  resultSection.classList.remove = (...names) => {
    origRemove(...names);
    if (!resultSection.classList.contains('hidden')) {
      for (const cb of observerCallbacks.get(resultSection) || []) cb();
    }
  };

  const windowListeners = new Map();
  const sandbox = {
    window: {
      location: locationObj,
      restorationInput: { video: false, clearVideoBatch() {}, confirmPreparation: () => true },
      addEventListener(type, fn) {
        if (!windowListeners.has(type)) windowListeners.set(type, []);
        windowListeners.get(type).push(fn);
      },
      dispatchEvent(evt) {
        for (const fn of windowListeners.get(evt.type) || []) fn(evt);
      },
    },
    document: {
      title: '',
      getElementById: byId,
      querySelectorAll: sel => sel === '.work-screen' ? screens : [],
      createElement: tag => new MockNode(tag),
      addEventListener(type, fn) {
        if (!windowListeners.has(`doc:${type}`)) windowListeners.set(`doc:${type}`, []);
        windowListeners.get(`doc:${type}`).push(fn);
      },
    },
    MutationObserver: class {
      constructor(cb) { this.cb = cb; }
      observe(target) {
        if (!observerCallbacks.has(target)) observerCallbacks.set(target, []);
        observerCallbacks.get(target).push(this.cb);
      }
    },
    ResizeObserver: class { observe() {} },
    Image: class {
      constructor() { this.src = ''; this.naturalWidth = 256; this.naturalHeight = 256; }
      async decode() {}
    },
    ImageData: class { constructor(data, w, h) { this.data = data; this.width = w; this.height = h; } },
    Blob: class {
      constructor(parts, opts = {}) { this.parts = parts; this.type = opts.type || ''; }
      stream() { return { pipeThrough: () => ({}) }; }
      async arrayBuffer() { return this.parts[0].buffer || new Uint8Array([1, 2, 3]).buffer; }
    },
    Response: class {
      async arrayBuffer() { return new Uint8Array([1, 2, 3, 4]).buffer; }
    },
    CompressionStream: class {},
    File: class {
      constructor(parts, name, opts = {}) { this.parts = parts; this.name = name; this.type = opts.type || ''; this.size = 10; }
    },
    FormData: class {
      constructor() { this._entries = new Map(); }
      append(k, v) { this._entries.set(k, v); }
      get(k) { return this._entries.get(k); }
    },
    Event: class { constructor(type) { this.type = type; } },
    CustomEvent: class { constructor(type, init = {}) { this.type = type; Object.assign(this, init); } },
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {},
    atob: str => Buffer.from(str, 'base64').toString('binary'),
    fetch: fetchImpl,
    Uint8Array, Uint8ClampedArray, Uint32Array, DataView, TextEncoder, Array, Math, Number, JSON, Promise, Error, Object, Set, Map,
  };

  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', 'static', 'viewport-workbench.js'), 'utf8'), sandbox, { filename: 'viewport-workbench.js' });
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', 'static', 'completion.js'), 'utf8'), sandbox, { filename: 'completion.js' });
  return { sandbox, byId, screens };
}

test('implementation.html exposes Edit removal area and bumps cache-busting query versions', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'implementation.html'), 'utf8');
  assert.match(html, /<button id="completion-edit-mask" class="tool-btn" type="button" data-nav="completion-paint" data-wizard-nav>Edit removal area<\/button>/);
  assert.match(html, /\/static\/implementation\.js\?v=50/);
  assert.match(html, /\/static\/completion\.js\?v=15/);
});

test('completion.js short-circuits 0-pixel proposals to manual editor and forwards proposal_id on manual edits', async () => {
  const PNG_INPUT = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
  const PNG_RESTORED = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
  const PNG_EMPTY = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
  const ZIP_EVIDENCE = 'data:application/zip;base64,UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA==';
  const SESSION_ID = 'a'.repeat(32);
  const PROPOSAL_ID = 'd'.repeat(32);
  const REVIEW_ID = 'e'.repeat(32);

  const calls = [];
  let savedMaskPngDataUrl = PNG_EMPTY;

  const { sandbox, byId, screens } = buildCompletionHarness({
    fetchImpl: async (url, init) => {
      calls.push({ url, form: init?.body });
      if (url === '/face/completion/prepare') {
        return { ok: true, json: async () => ({ session_id: SESSION_ID, bounds: [0, 0, 256, 256], input_png: PNG_RESTORED }) };
      }
      if (url === '/face/completion/propose') {
        return { ok: true, json: async () => ({ proposal_id: PROPOSAL_ID, proposal_png: PNG_EMPTY, metadata: { proposed_mask_pixels: 0 } }) };
      }
      if (url === '/face/completion/review') {
        const blob = init.body.get('mask');
        const buf = Buffer.from(await blob.arrayBuffer());
        savedMaskPngDataUrl = 'data:image/png;base64,' + buf.toString('base64');
        return { ok: true, json: async () => ({ review_id: REVIEW_ID, mask_png: savedMaskPngDataUrl, visibility: { rejected: false } }) };
      }
      if (url === '/face/completion/generate') {
        return {
          ok: true,
          json: async () => ({
            result_png: PNG_RESTORED,
            mask_png: savedMaskPngDataUrl,
            bundle: ZIP_EVIDENCE,
            neural_calls: 0,
            metadata: { seconds: 0.2 },
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    },
  });

  const detail = {
    originalFile: new sandbox.File(['orig'], 'face.png', { type: 'image/png' }),
    result: {
      original_input: PNG_INPUT,
      result: PNG_RESTORED,
      metadata: { selection: 'preconditioned_v34_codeformer', geometry: { observed_bounds_256: [0, 0, 256, 256] } },
    },
  };
  sandbox.window.dispatchEvent(new sandbox.CustomEvent('restoration:completed', { detail }));
  await byId('completion-start').dispatchEvent({ type: 'click' });

  // Short-circuited after /propose: zero /review or /generate calls and switched to completion-paint.
  assert.deepEqual(calls.map(c => c.url), ['/face/completion/prepare', '/face/completion/propose']);
  const paintScreen = screens.find(s => s.dataset.screen === 'completion-paint');
  assert.equal(paintScreen.classList.contains('is-active'), true);
  assert.match(byId('completion-progress').textContent, /No covering was detected automatically/i);

  // Manual review + generate forwards the captured proposal_id.
  byId('completion-mask-confirm').checked = true;
  await byId('completion-mask-confirm').dispatchEvent({ type: 'change' });
  byId('completion-visible-confirm').checked = true;
  await byId('completion-visible-confirm').dispatchEvent({ type: 'change' });
  await byId('completion-generate').dispatchEvent({ type: 'click' });

  assert.deepEqual(calls.map(c => c.url), [
    '/face/completion/prepare',
    '/face/completion/propose',
    '/face/completion/review',
    '/face/completion/generate',
  ]);
  assert.equal(calls[2].form.get('proposal_id'), PROPOSAL_ID);
  assert.equal(calls[3].form.get('proposal_id'), PROPOSAL_ID);
});

test('completion.js disables Remove covering when /face/completion/status reports unavailable in browser', async () => {
  const PNG_INPUT = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
  const PNG_RESTORED = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
  const statusCalls = [];

  const { sandbox, byId } = buildCompletionHarness({
    locationObj: { protocol: 'http:', origin: 'http://127.0.0.1:8000' },
    fetchImpl: async (url) => {
      statusCalls.push(url);
      if (url === '/face/completion/status') {
        return { ok: true, json: async () => ({ available: false, error: 'missing codeformer_completion_r2.pth' }) };
      }
      throw new Error(`Unexpected URL: ${url}`);
    },
  });

  const detail = {
    originalFile: new sandbox.File(['orig'], 'face.png', { type: 'image/png' }),
    result: {
      original_input: PNG_INPUT,
      result: PNG_RESTORED,
      metadata: { selection: 'preconditioned_v34_codeformer', geometry: { observed_bounds_256: [0, 0, 256, 256] } },
    },
  };
  sandbox.window.dispatchEvent(new sandbox.CustomEvent('restoration:completed', { detail }));
  await new Promise(resolve => setTimeout(resolve, 10));

  assert.deepEqual(statusCalls, ['/face/completion/status']);
  assert.equal(byId('completion-start').disabled, true);
  assert.match(byId('completion-progress').textContent, /unavailable/i);
});
