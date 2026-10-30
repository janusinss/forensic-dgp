const test = require('node:test');
const assert = require('node:assert/strict');
const { normalizeRate, stepRate, setup } = require('../static/capture_playback.js');

class Element {
  constructor() { this.value = ''; this.disabled = false; this.textContent = ''; this.listeners = {}; }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  fire(name, detail) { for (const fn of this.listeners[name] || []) fn({ target: this, detail }); }
}

function fixture() {
  const elements = Object.fromEntries(['capture-video', 'capture-speed-select', 'capture-speed-minus',
    'capture-speed-plus', 'capture-speed-error'].map(id => [id, new Element()]));
  elements['capture-video'].playbackRate = 1;
  elements['capture-video'].defaultPlaybackRate = 1;
  const select = elements['capture-speed-select'];
  select.options = [];
  select.appendChild = option => { select.options.push(option); option.remove = () => select.options.splice(select.options.indexOf(option), 1); };
  const document = { getElementById: id => elements[id] || null, createElement: () => new Element() };
  return { elements, document };
}

test('rates accept exact quarter steps between 0.25 and 4', () => {
  for (let i = 1; i <= 16; i++) assert.equal(normalizeRate(String(i / 4)), i / 4);
  for (const invalid of [0, 4.25, 1.1, '', 'NaN', NaN, Infinity, null, true]) {
    assert.throws(() => normalizeRate(invalid), RangeError);
  }
});

test('stepping uses quarter increments and clamps at either end', () => {
  assert.equal(stepRate(1, 1), 1.25);
  assert.equal(stepRate(1, -1), 0.75);
  assert.equal(stepRate(.25, -1), .25);
  assert.equal(stepRate(4, 1), 4);
  assert.throws(() => stepRate(1, 0), RangeError);
  assert.equal(stepRate(1.1, 1), 1.25);
  assert.equal(stepRate(1.1, -1), 1);
});

test('buttons and selector set playback speed without seeking or starting video', () => {
  const { elements: e, document } = fixture();
  e['capture-video'].currentTime = 2;
  e['capture-video'].play = () => { throw Error('unexpected autoplay'); };
  setup(document);
  assert.equal(e['capture-speed-select'].value, '1');
  e['capture-speed-plus'].fire('click');
  assert.equal(e['capture-video'].playbackRate, 1.25);
  assert.equal(e['capture-video'].defaultPlaybackRate, 1.25);
  assert.equal(e['capture-speed-select'].value, '1.25');
  e['capture-speed-select'].value = '0.25';
  e['capture-speed-select'].fire('change');
  assert.equal(e['capture-video'].playbackRate, .25);
  assert.equal(e['capture-speed-minus'].disabled, true);
  assert.equal(e['capture-speed-plus'].disabled, false);
  assert.equal(e['capture-video'].currentTime, 2);
  e['capture-speed-select'].value = '4';
  e['capture-speed-select'].fire('change');
  assert.equal(e['capture-speed-plus'].disabled, true);
});

test('new video metadata resets speed to 1', () => {
  const { elements: e, document } = fixture();
  setup(document);
  e['capture-speed-select'].value = '2';
  e['capture-speed-select'].fire('change');
  e['capture-video'].fire('loadedmetadata');
  assert.equal(e['capture-video'].playbackRate, 1);
  assert.equal(e['capture-video'].defaultPlaybackRate, 1);
  assert.equal(e['capture-speed-select'].value, '1');
  assert.equal(e['capture-speed-minus'].disabled, false);
  assert.equal(e['capture-speed-plus'].disabled, false);
});

test('failed rate change restores selector and displays local error', () => {
  const { elements: e, document } = fixture();
  let rate = 1;
  Object.defineProperty(e['capture-video'], 'playbackRate', {
    get: () => rate,
    set: value => { if (value !== 1) throw new Error('unsupported'); rate = value; },
  });
  setup(document);
  e['capture-speed-select'].value = '4';
  e['capture-speed-select'].fire('change');
  assert.equal(e['capture-video'].playbackRate, 1);
  assert.equal(e['capture-speed-select'].value, '1');
  assert.ok(e['capture-speed-error'].textContent.length > 0);
});

test('setup is safe when this screen is absent', () => {
  assert.doesNotThrow(() => setup({ getElementById: () => null }));
});

test('pending capture locks speed changes and unlock restores boundary state', () => {
  const { elements: e, document } = fixture();
  setup(document);
  e['capture-speed-select'].value = '4';
  e['capture-speed-select'].fire('change');
  e['capture-video'].fire('capture:pending', { pending: true });
  assert.equal(e['capture-speed-select'].disabled, true);
  assert.equal(e['capture-speed-minus'].disabled, true);
  assert.equal(e['capture-speed-plus'].disabled, true);
  e['capture-speed-minus'].fire('click');
  e['capture-speed-select'].value = '1';
  e['capture-speed-select'].fire('change');
  assert.equal(e['capture-video'].playbackRate, 4);
  e['capture-video'].fire('capture:pending', { pending: false });
  assert.equal(e['capture-speed-select'].value, '4');
  assert.equal(e['capture-speed-select'].disabled, false);
  assert.equal(e['capture-speed-minus'].disabled, false);
  assert.equal(e['capture-speed-plus'].disabled, true);
  e['capture-speed-minus'].fire('click');
  assert.equal(e['capture-video'].playbackRate, 3.75);
});

test('new video resets speed while pending without unlocking controls', () => {
  const { elements: e, document } = fixture();
  setup(document);
  e['capture-speed-select'].value = '2';
  e['capture-speed-select'].fire('change');
  e['capture-video'].fire('capture:pending', { pending: true });
  e['capture-video'].fire('loadedmetadata');
  assert.equal(e['capture-video'].playbackRate, 1);
  assert.equal(e['capture-speed-select'].value, '1');
  assert.equal(e['capture-speed-select'].disabled, true);
  assert.equal(e['capture-speed-minus'].disabled, true);
  assert.equal(e['capture-speed-plus'].disabled, true);
});

test('native playback rate change updates selector and subsequent steps', () => {
  const { elements: e, document } = fixture();
  setup(document);
  e['capture-video'].playbackRate = 1.75;
  e['capture-video'].fire('ratechange');
  assert.equal(e['capture-speed-select'].value, '1.75');
  assert.equal(e['capture-video'].defaultPlaybackRate, 1.75);
  e['capture-speed-minus'].fire('click');
  assert.equal(e['capture-video'].playbackRate, 1.5);
  e['capture-video'].fire('ratechange');
  e['capture-speed-plus'].fire('click');
  assert.equal(e['capture-video'].playbackRate, 1.75);
  e['capture-video'].fire('ratechange');
  assert.equal(e['capture-speed-select'].value, '1.75');
});

test('unsupported native speed reverts once to last valid speed and reports error', () => {
  const { elements: e, document } = fixture();
  setup(document);
  e['capture-video'].playbackRate = 0.1;
  e['capture-video'].fire('ratechange');
  assert.equal(e['capture-video'].playbackRate, 1);
  assert.equal(e['capture-speed-select'].value, '1');
  assert.ok(e['capture-speed-error'].textContent);
  e['capture-video'].fire('ratechange');
  assert.equal(e['capture-video'].playbackRate, 1);
});

test('native non-quarter rate remains truthful and steps to neighboring quarters', () => {
  const { elements: e, document } = fixture();
  setup(document);
  const video = e['capture-video'], select = e['capture-speed-select'];
  video.playbackRate = 1.1;
  video.fire('ratechange');
  assert.equal(video.playbackRate, 1.1);
  assert.equal(video.defaultPlaybackRate, 1.1);
  assert.equal(select.value, '1.1');
  assert.equal(select.options.length, 17);
  assert.equal(select.options.filter(o => o.value === '1.1').length, 1);
  video.fire('ratechange');
  assert.equal(select.options.length, 17);
  e['capture-speed-plus'].fire('click');
  assert.equal(video.playbackRate, 1.25);
  assert.equal(select.options.length, 16);
  video.playbackRate = 1.1;
  video.fire('ratechange');
  e['capture-speed-minus'].fire('click');
  assert.equal(video.playbackRate, 1);
  assert.equal(video.defaultPlaybackRate, 1);
});

test('a valid native speed clears a prior rejected-rate error', () => {
  const { elements: e, document } = fixture();
  setup(document);
  const video = e['capture-video'];
  video.playbackRate = .1;
  video.fire('ratechange');
  assert.ok(e['capture-speed-error'].textContent);
  video.fire('ratechange');
  assert.ok(e['capture-speed-error'].textContent);
  video.playbackRate = 1.1;
  video.fire('ratechange');
  assert.equal(e['capture-speed-error'].textContent, '');
  assert.equal(video.playbackRate, 1.1);
  assert.equal(video.defaultPlaybackRate, 1.1);
  assert.equal(e['capture-speed-select'].value, '1.1');
});
