const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = process.env.CAPTURE_SCRIPT || path.join(__dirname, '../static/capture.js');

class Element {
  constructor(tagName = 'DIV') {
    this.tagName = tagName; this.listeners = {}; this.dataset = {}; this.children = [];
    this.attrs = {}; this.value = ''; this.disabled = false; this.textContent = ''; this.files = [];
    this.classList = { values: new Set(), add: (...x) => x.forEach(v => this.classList.values.add(v)),
      remove: (...x) => x.forEach(v => this.classList.values.delete(v)),
      toggle: (v, force) => { const on = force === undefined ? !this.classList.values.has(v) : force; on ? this.classList.values.add(v) : this.classList.values.delete(v); return on; },
      contains: v => this.classList.values.has(v) };
    this.nextElementSibling = { textContent: '' };
  }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  async fire(name, extra = {}) { for (const fn of this.listeners[name] || []) await fn({ target: this, ...extra }); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(event); return true; }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k] || null; }
  removeAttribute(k) { delete this.attrs[k]; if (k === 'src') this.src = ''; }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.append(item); }
  replaceChildren(...items) { this.children = [...items]; }
  querySelectorAll() { return []; }
  querySelector() { return new Element('P'); }
  after() {} focus() {} close() {}
}

function harness({deferFrame = false, deferCrop = false, deferOpen = false} = {}) {
  const elements = new Map();
  const get = id => { if (id === 'capture-draw') return null; if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const ctx = { clearRect() {}, drawImage() {}, save() {}, translate() {}, strokeRect() {}, fillRect() {}, fillText() {}, restore() {} };
  function canvas() { const c = new Element('CANVAS'); c.width = 0; c.height = 0; c.getContext = () => ctx; c.toBlob = fn => fn(new Blob(['frame'], { type: 'image/png' })); c.getBoundingClientRect = () => ({left:0,top:0,width:100,height:80}); c.setPointerCapture = () => {}; return c; }
  elements.set('capture-canvas', canvas());
  const video = get('capture-video');
  Object.assign(video, { videoWidth: 100, videoHeight: 80, duration: 20, currentTime: 0, playbackRate: 1,
    defaultPlaybackRate: 1, paused: true, seeking: false, readyState: 4, src: '', srcObject: null, playCount: 0 });
  video.pause = () => { video.paused = true; };
  video.play = async () => { video.paused = false; video.playCount++; };
  video.load = () => { if (!deferOpen && video.src && video.onloadeddata) queueMicrotask(() => video.onloadeddata?.()); };
  const screen = new Element(); screen.classList.add('is-active');
  const visual = new Element();
  const body = new Element('BODY');
  const docEvents = {}, windowEvents = [], observers = [];
  const doc = { body, activeElement: body, hidden: false, getElementById: get,
    createElement: name => name === 'canvas' ? canvas() : new Element(name.toUpperCase()),
    querySelector: selector => selector === '[data-screen=inspect]' ? screen : selector === '.capture-visual' ? visual : selector === '.capture-transport' ? get('transport') : null,
    addEventListener(name,fn) { (docEvents[name] ||= []).push(fn); } };
  const revoked = [];
  let accepted = true, useCalls = 0;
  const track = { stopCount:0, stop() { this.stopCount++; }, addEventListener() {} };
  const stream = { getTracks:() => [track], getVideoTracks:() => [track] };
  const win = { captureGeometry: { makeView: (width, height) => ({x:0,y:0,width,height}), pointerToSource: (x,y) => ({x,y,scale:1}) },
    restorationInput: { busy:false, use:async () => {useCalls++; return accepted;} }, addEventListener(name,fn) { windowEvents.push([name,fn]); } };
  const reply = data => ({ ok:true, status:200, json:async () => data });
  let requestedCrop, finishFrame, frameStarted, finishCrop, cropStarted;
  const started = new Promise(resolve => {frameStarted=resolve;});
  const cropRequestStarted = new Promise(resolve => {cropStarted=resolve;});
  const fetch = async (url,options) => { if(url === '/capture/status')return reply({token:'token'}); if(url === '/capture/frames') { if(deferFrame) {frameStarted(); return await new Promise(resolve => {finishFrame=() => resolve(reply({frame_id:'frame',faces:[[20,20,30,30]]}));});} return reply({frame_id:'frame', faces:[[20,20,30,30]]}); } requestedCrop=JSON.parse(options.body); if(deferCrop) {cropStarted(); return await new Promise(resolve => {finishCrop=() => resolve(reply({capture_id:'crop',image:'data:image/png;base64,YQ=='}));});} return reply({capture_id:'crop', image:'data:image/png;base64,YQ=='}); };
  class CustomEvent { constructor(type, init) { this.type=type; this.detail=init?.detail; } }
  const sandbox = { document:doc, window:win, navigator:{mediaDevices:{getUserMedia:async () => stream}}, fetch, Blob, File, FormData, Uint8Array, atob,
    CustomEvent, MutationObserver:class { constructor(fn){this.fn=fn;} observe(target){observers.push({target,fn:this.fn});} }, matchMedia:() => ({matches:false,addEventListener(){}}),
    URL:{createObjectURL:() => 'blob:video',revokeObjectURL:url => revoked.push(url)},
    setTimeout, clearTimeout, queueMicrotask };
  vm.runInNewContext(fs.readFileSync(path.resolve(source), 'utf8'), sandbox, {filename:source});
  return { get, video, revoked, track, doc, screen, frameStarted:started, finishFrame:() => finishFrame(), cropStarted:cropRequestStarted, finishCrop:() => finishCrop(),
    docEvent:name => (docEvents[name]||[]).forEach(fn=>fn()), windowEvent:name => windowEvents.filter(([key])=>key===name).forEach(([,fn])=>fn()),
    mutate:target => observers.filter(item=>item.target===target).forEach(item=>item.fn()), requestedCrop:() => requestedCrop, useCalls:() => useCalls, setAccepted:value => { accepted=value; } };
}

async function openFile(h) {
  await h.get('input-mode-video').fire('click');
  h.get('capture-source').value = 'video';
  await h.get('capture-source').fire('change');
  h.get('capture-video-file').files = [{name:'clip.mp4',size:100}];
  await h.get('capture-video-file').fire('change');
}

async function chooseFace(h) {
  await h.get('capture-freeze').fire('click');
  assert.equal(h.get('capture-faces').children.length, 1);
  await h.get('capture-faces').children[0].fire('click');
}

test('return from face review preserves video until explicit Stop', async () => {
  const h = harness(), {get,video,revoked} = h;
  await openFile(h);
  video.currentTime = 8.25;
  video.playbackRate = video.defaultPlaybackRate = 1.5;
  const selectedSource = video.src;
  for (let i=0; i<2; i++) {
    await chooseFace(h);
    const plays = video.playCount;
    await get('capture-use').fire('click');
    await get('capture-back').fire('click');
    assert.equal(video.paused, true);
    assert.equal(video.playCount, plays);
    assert.equal(video.src, selectedSource);
    assert.equal(video.currentTime, 8.25);
    assert.equal(video.playbackRate, 1.5);
    assert.equal(get('capture-resume').disabled, false);
    assert.equal(get('capture-stop').disabled, false);
    assert.deepEqual(revoked, []);
    await get('capture-resume').fire('click');
    assert.equal(video.paused, false);
  }
  await get('capture-stop').fire('click');
  assert.equal(video.src, '');
  assert.deepEqual(revoked, [selectedSource]);
  assert.equal(video.paused, true);
  assert.equal(get('capture-stop').disabled, true);
});

test('rejected picture keeps frozen frame and video resumable', async () => {
  const h = harness(), {get,video,revoked} = h;
  await openFile(h);
  video.currentTime = 4.5;
  await chooseFace(h);
  h.setAccepted(false);
  await get('capture-use').fire('click');
  assert.equal(video.src, 'blob:video');
  assert.equal(video.currentTime, 4.5);
  assert.equal(get('capture-resume').disabled, false);
  assert.deepEqual(revoked, []);
  await get('capture-resume').fire('click');
  assert.equal(video.paused, false);
});

test('switching modes retains frozen source and crop after review, then explicit Stop releases it once', async () => {
  const h = harness(), {get,video,revoked} = h;
  await openFile(h);
  video.currentTime=7.25; video.playbackRate=video.defaultPlaybackRate=1.5;
  await chooseFace(h);
  const fields = () => ['x','y','width','height'].map(key => get('capture-'+key).value);
  const selected = fields();
  await get('capture-use').fire('click');
  await get('input-mode-picture').fire('click');
  await get('input-mode-video').fire('click');
  assert.equal(video.src, 'blob:video');
  assert.equal(video.currentTime, 7.25);
  assert.equal(video.playbackRate, 1.5);
  assert.equal(video.paused, true);
  assert.deepEqual(fields(), selected);
  assert.equal(get('capture-resume').disabled, false);
  assert.deepEqual(revoked, []);
  await get('capture-resume').fire('click');
  assert.equal(video.paused, false);
  await get('capture-stop').fire('click');
  assert.equal(video.src, '');
  assert.deepEqual(revoked, ['blob:video']);
});

test('mode toggles pause playing file without autoplay and source change revokes exactly once', async () => {
  const h = harness(), {get,video,revoked} = h;
  await openFile(h);
  video.currentTime=3.5; video.playbackRate=video.defaultPlaybackRate=.75;
  await video.play();
  const plays = video.playCount;
  for(let i=0;i<2;i++) {
    await get('input-mode-picture').fire('click');
    assert.equal(video.paused, true);
    await get('input-mode-video').fire('click');
    assert.equal(video.src, 'blob:video');
    assert.equal(video.currentTime, 3.5);
    assert.equal(video.playbackRate, .75);
    assert.equal(video.playCount, plays);
    assert.equal(video.paused, true);
    assert.equal(get('capture-freeze').disabled, false);
  }
  get('capture-source').value='webcam';
  await get('capture-source').fire('change');
  assert.deepEqual(revoked,['blob:video']);
  await get('capture-stop').fire('click');
  assert.deepEqual(revoked,['blob:video']);
});

test('late freeze completion after mode switch does not select stale frame', async () => {
  const h = harness({deferFrame:true}), {get,video,revoked} = h;
  await openFile(h);
  const freezing = get('capture-freeze').fire('click');
  await h.frameStarted;
  await get('input-mode-picture').fire('click');
  h.finishFrame();
  await freezing;
  await get('input-mode-video').fire('click');
  assert.equal(video.src, 'blob:video');
  assert.deepEqual(revoked, []);
  assert.equal(get('capture-faces').children.length, 0);
  assert.equal(get('capture-use').disabled, true);
  assert.equal(get('capture-freeze').disabled, false);
});

test('temporary hidden document and leaving inspect pause but retain ready file; pagehide releases it', async () => {
  const h = harness(), {get,video,revoked,doc,screen} = h;
  await openFile(h);
  video.currentTime=6.25; video.playbackRate=video.defaultPlaybackRate=1.75;
  await video.play();
  doc.hidden=true; h.docEvent('visibilitychange');
  assert.equal(video.paused,true);
  assert.equal(video.src,'blob:video');
  assert.deepEqual(revoked,[]);
  doc.hidden=false; h.docEvent('visibilitychange');
  assert.equal(video.paused,true);
  assert.equal(video.currentTime,6.25);
  await video.play();
  screen.classList.remove('is-active'); h.mutate(screen);
  assert.equal(video.paused,true);
  assert.equal(video.src,'blob:video');
  screen.classList.add('is-active'); h.mutate(screen);
  assert.equal(video.playbackRate,1.75);
  assert.equal(get('capture-freeze').disabled,false);
  h.windowEvent('pagehide');
  assert.equal(video.src,'');
  assert.deepEqual(revoked,['blob:video']);
});

test('switching modes during incomplete file opening cancels and releases it', async () => {
  const h = harness({deferOpen:true}), {get,video,revoked} = h;
  await get('input-mode-video').fire('click');
  get('capture-source').value='video';
  await get('capture-source').fire('change');
  get('capture-video-file').files=[{name:'clip.mp4',size:100}];
  const opening=get('capture-video-file').fire('change');
  assert.equal(video.src,'blob:video');
  await get('input-mode-picture').fire('click');
  await opening;
  assert.equal(video.src,'');
  assert.deepEqual(revoked,['blob:video']);
});

test('an opened file survives mode switches while the decoder is seeking', async () => {
  const h = harness(), {get,video,revoked} = h;
  await openFile(h);
  video.currentTime=5.5; video.playbackRate=video.defaultPlaybackRate=1.25;
  video.readyState=1; video.seeking=true;
  await get('input-mode-picture').fire('click');
  await get('input-mode-video').fire('click');
  assert.equal(video.src,'blob:video');
  assert.equal(video.currentTime,5.5);
  assert.equal(video.playbackRate,1.25);
  assert.deepEqual(revoked,[]);
  assert.equal(get('capture-freeze').disabled,true);
  video.seeking=false;
  await video.fire('waiting');
  assert.equal(get('capture-freeze').disabled,true);
  await get('input-mode-picture').fire('click');
  await get('input-mode-video').fire('click');
  assert.equal(video.src,'blob:video');
  assert.deepEqual(revoked,[]);
  video.readyState=4;
  await video.fire('canplay');
  assert.equal(get('capture-freeze').disabled,false);
});

test('late crop completion after mode switch is not adopted', async () => {
  const h = harness({deferCrop:true}), {get,video,revoked} = h;
  await openFile(h);
  await chooseFace(h);
  const using=get('capture-use').fire('click');
  await h.cropStarted;
  await get('input-mode-picture').fire('click');
  h.finishCrop();
  await using;
  assert.equal(h.useCalls(),0);
  await get('input-mode-video').fire('click');
  assert.equal(video.src,'blob:video');
  assert.deepEqual(revoked,[]);
  assert.equal(get('capture-resume').disabled,false);
});

test('accepting a webcam face still releases its track', async () => {
  const h = harness(), {get,track} = h;
  await get('input-mode-video').fire('click');
  get('capture-source').value = 'webcam';
  await get('capture-source').fire('change');
  await get('capture-start').fire('click');
  await chooseFace(h);
  await get('capture-use').fire('click');
  assert.equal(track.stopCount, 1);
});

test('face selection and full-frame view retain the selected crop', async () => {
  const h = harness();
  await openFile(h);
  await chooseFace(h);
  const before = ['x','y','width','height'].map(key => h.get('capture-'+key).value);
  await h.get('capture-full-frame').fire('click');
  assert.deepEqual(['x','y','width','height'].map(key => h.get('capture-'+key).value), before);
  await h.get('capture-use').fire('click');
  assert.deepEqual(Object.values(h.requestedCrop()), before);
});

test('dragging a blank frame area creates a manual crop without a Draw crop button', async () => {
  const h = harness();
  await openFile(h);
  await h.get('capture-freeze').fire('click');
  const canvas = h.get('capture-canvas');
  await canvas.fire('pointerdown', {clientX:60,clientY:5,pointerId:1});
  await canvas.fire('pointermove', {clientX:90,clientY:30,pointerId:1});
  await canvas.fire('pointerup', {pointerId:1});
  assert.equal(h.get('capture-use').disabled, false);
  await h.get('capture-use').fire('click');
  assert.deepEqual(h.requestedCrop(), {x:60,y:5,width:30,height:25});
});

test('keyboard creates and adjusts a centered manual crop when no face is selected', async () => {
  const h = harness();
  await openFile(h);
  await h.get('capture-freeze').fire('click');
  const canvas = h.get('capture-canvas');
  let prevented = false;
  await canvas.fire('keydown', {key:'Enter', preventDefault:() => { prevented=true; }});
  assert.equal(prevented, true);
  assert.equal(h.get('capture-use').disabled, false);
  const fields = () => ['x','y','width','height'].map(key => h.get('capture-'+key).value);
  assert.deepEqual(fields(), [25,20,50,40]);
  await canvas.fire('keydown', {key:'ArrowRight', preventDefault() {}});
  await canvas.fire('keydown', {key:'ArrowDown', shiftKey:true, preventDefault() {}});
  assert.deepEqual(fields(), [26,20,50,41]);
  await canvas.fire('keydown', {key:'Enter', preventDefault() {}});
  assert.deepEqual(fields(), [26,20,50,41]);
  await h.get('capture-use').fire('click');
  assert.deepEqual(h.requestedCrop(), {x:26,y:20,width:50,height:41});
});

test('Enter does not replace a selected face crop', async () => {
  const h = harness();
  await openFile(h);
  await chooseFace(h);
  const fields = () => ['x','y','width','height'].map(key => h.get('capture-'+key).value);
  const selected = fields();
  await h.get('capture-canvas').fire('keydown', {key:'Enter', preventDefault() {}});
  assert.deepEqual(fields(), selected);
});
