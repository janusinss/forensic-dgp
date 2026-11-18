const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function solidRgba(w, h, rgba) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rgba[0];
    data[i + 1] = rgba[1];
    data[i + 2] = rgba[2];
    data[i + 3] = rgba[3];
  }
  return data;
}

function paddedRgba(bounds, innerRgba, padRgba = [128, 128, 128, 255]) {
  const [left, top, right, bottom] = bounds;
  const data = solidRgba(256, 256, padRgba);
  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) {
      const i = (y * 256 + x) * 4;
      data[i] = innerRgba[0];
      data[i + 1] = innerRgba[1];
      data[i + 2] = innerRgba[2];
      data[i + 3] = innerRgba[3];
    }
  }
  return data;
}

function rectInImageRgba(w, h, [rx, ry, rw, rh], insideRgba, outsideRgba) {
  const data = solidRgba(w, h, outsideRgba);
  for (let y = ry; y < ry + rh; y++) {
    for (let x = rx; x < rx + rw; x++) {
      const i = (y * w + x) * 4;
      data[i] = insideRgba[0];
      data[i + 1] = insideRgba[1];
      data[i + 2] = insideRgba[2];
      data[i + 3] = insideRgba[3];
    }
  }
  return data;
}

class MockFormData {
  constructor() {
    this._entries = new Map();
  }
  append(key, value) {
    this._entries.set(key, value);
  }
  get(key) {
    return this._entries.get(key);
  }
}

function loadReferenceHarness(imagesByUrl, options = {}) {
  const revoked = [];
  let nextBlobId = 1;

  class MockImage {
    constructor() {
      this.src = '';
      this.naturalWidth = 0;
      this.naturalHeight = 0;
      this._pixels = null;
    }
    async decode() {
      const entry = imagesByUrl.get(this.src);
      if (!entry || entry.decodeError) {
        throw new Error('decode failed');
      }
      this.naturalWidth = entry.width;
      this.naturalHeight = entry.height;
      this._pixels = entry.pixels;
    }
  }

  class MockContext2D {
    constructor(canvas) {
      this.canvas = canvas;
      this.fillStyle = 'rgb(0, 0, 0)';
    }
    _EnsureBuffer() {
      if (!this.canvas._data) {
        this.canvas._data = new Uint8ClampedArray(this.canvas.width * this.canvas.height * 4);
      }
    }
    fillRect(x, y, w, h) {
      this._EnsureBuffer();
      const match = /rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/.exec(this.fillStyle);
      const r = match ? Number(match[1]) : 0;
      const g = match ? Number(match[2]) : 0;
      const b = match ? Number(match[3]) : 0;
      const x0 = Math.max(0, Math.floor(x));
      const y0 = Math.max(0, Math.floor(y));
      const x1 = Math.min(this.canvas.width, Math.ceil(x + w));
      const y1 = Math.min(this.canvas.height, Math.ceil(y + h));
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const i = (yy * this.canvas.width + xx) * 4;
          this.canvas._data[i] = r;
          this.canvas._data[i + 1] = g;
          this.canvas._data[i + 2] = b;
          this.canvas._data[i + 3] = 255;
        }
      }
    }
    drawImage(image, ...args) {
      this._EnsureBuffer();
      const srcW = image.naturalWidth ?? image.videoWidth;
      const srcH = image.naturalHeight ?? image.videoHeight;
      const srcPixels = typeof image._currentFramePixels === 'function' ? image._currentFramePixels() : image._pixels;
      let sx = 0, sy = 0, sw = srcW, sh = srcH;
      let dx = 0, dy = 0, dw = srcW, dh = srcH;
      if (args.length >= 8) {
        [sx, sy, sw, sh, dx, dy, dw, dh] = args;
      } else if (args.length >= 4) {
        [dx, dy, dw, dh] = args;
      } else if (args.length >= 2) {
        [dx, dy] = args;
      }
      const x0 = Math.max(0, Math.floor(dx));
      const y0 = Math.max(0, Math.floor(dy));
      const x1 = Math.min(this.canvas.width, Math.floor(dx + dw));
      const y1 = Math.min(this.canvas.height, Math.floor(dy + dh));
      for (let yy = y0; yy < y1; yy++) {
        const srcY = Math.max(0, Math.min(srcH - 1, sy + Math.min(sh - 1, Math.floor(((yy - dy) * sh) / dh))));
        for (let xx = x0; xx < x1; xx++) {
          const srcX = Math.max(0, Math.min(srcW - 1, sx + Math.min(sw - 1, Math.floor(((xx - dx) * sw) / dw))));
          const si = (srcY * srcW + srcX) * 4;
          const di = (yy * this.canvas.width + xx) * 4;
          this.canvas._data[di] = srcPixels[si];
          this.canvas._data[di + 1] = srcPixels[si + 1];
          this.canvas._data[di + 2] = srcPixels[si + 2];
          this.canvas._data[di + 3] = srcPixels[si + 3];
        }
      }
    }
    getImageData(sx, sy, sw, sh) {
      this._EnsureBuffer();
      if (sx === 0 && sy === 0 && sw === this.canvas.width && sh === this.canvas.height) {
        return { data: new Uint8ClampedArray(this.canvas._data) };
      }
      throw new Error('Unexpected getImageData slice');
    }
  }

  class MockCanvas {
    constructor() {
      this.width = 0;
      this.height = 0;
      this._data = null;
      this._ctx = new MockContext2D(this);
    }
    getContext(kind) {
      assert.equal(kind, '2d');
      return this._ctx;
    }
    toBlob(callback, type = 'image/png') {
      const pixels = this._data ? new Uint8ClampedArray(this._data) : new Uint8ClampedArray(0);
      callback({
        size: pixels.length || 1024,
        type,
        _image: { width: this.width, height: this.height, pixels },
      });
    }
    toDataURL(type = 'image/png') {
      const header = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
      const meta = Buffer.alloc(8);
      meta.writeUInt32BE(this.width >>> 0, 0);
      meta.writeUInt32BE(this.height >>> 0, 4);
      const digest = crypto.createHash('sha256').update(this._data || Buffer.alloc(0)).digest();
      return `data:${type};base64,${Buffer.concat([header, meta, digest]).toString('base64')}`;
    }
  }

  class MockFile {
    constructor(parts, name, opts = {}) {
      const part = parts[0] || {};
      this.name = name;
      this.type = opts.type || part.type || '';
      this.size = part.size || 0;
      this._image = part._image || null;
      this._parts = parts;
    }
  }

  const videosByUrl = new Map();
  const createdVideos = [];
  class MockVideo {
    constructor() {
      this.muted = false;
      this.playsInline = false;
      this.preload = '';
      this.duration = 0;
      this.videoWidth = 0;
      this.videoHeight = 0;
      this._currentTime = 0;
      this._listeners = new Map();
      this._entry = null;
      this.seekedTimes = [];
      this.src = '';
      createdVideos.push(this);
    }
    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, []);
      this._listeners.get(type).push(fn);
    }
    removeEventListener(type, fn) {
      const list = this._listeners.get(type);
      if (!list) return;
      const idx = list.indexOf(fn);
      if (idx >= 0) list.splice(idx, 1);
    }
    _emit(type) {
      for (const fn of [...(this._listeners.get(type) || [])]) fn();
    }
    load() {
      this._entry = videosByUrl.get(this.src);
      queueMicrotask(() => {
        if (!this._entry || this._entry.decodeError) {
          this._emit('error');
          return;
        }
        this.duration = this._entry.duration ?? 10;
        this.videoWidth = this._entry.width;
        this.videoHeight = this._entry.height;
        this._emit('loadeddata');
      });
    }
    getVideoPlaybackQuality() {
      if (this._entry && typeof this._entry.totalVideoFrames === 'number') {
        return { totalVideoFrames: this._entry.totalVideoFrames };
      }
      return { totalVideoFrames: 1 };
    }
    get currentTime() {
      return this._currentTime;
    }
    set currentTime(val) {
      this._currentTime = val;
      this.seekedTimes.push(val);
      queueMicrotask(() => this._emit('seeked'));
    }
    _currentFramePixels() {
      if (typeof this._entry?.frameAt === 'function') {
        return this._entry.frameAt(this._currentTime);
      }
      return this._entry.pixels;
    }
  }

  const windowObj = options.protocol ? { location: { protocol: options.protocol } } : {};
  const sandbox = {
    window: windowObj,
    Image: MockImage,
    File: MockFile,
    FormData: MockFormData,
    fetch: options.fetch,
    document: {
      createElement(tag) {
        if (tag === 'canvas') {
          const canvas = new MockCanvas();
          if (options.disableCanvasToDataUrl) canvas.toDataURL = undefined;
          return canvas;
        }
        if (tag === 'video') return new MockVideo();
        throw new Error(`Unexpected createElement tag: ${tag}`);
      },
    },
    URL: {
      createObjectURL(file) {
        const url = `blob:ref-${nextBlobId++}`;
        if (file._video) videosByUrl.set(url, file._video);
        else imagesByUrl.set(url, file._image);
        return url;
      },
      revokeObjectURL(url) {
        revoked.push(url);
      },
    },
    setTimeout,
    clearTimeout,
    Uint8ClampedArray,
    Error,
    Number,
    Array,
    Math,
    JSON,
    Promise,
  };
  const source = fs.readFileSync(path.join(__dirname, '..', 'static', 'restoration_reference.js'), 'utf8');
  vm.runInNewContext(source, sandbox, { filename: 'restoration_reference.js' });
  return { scoreVideoReference: windowObj.scoreVideoReference, revoked, createdVideos };
}

function assertReference256(scores, expectedExact = null) {
  const desc = Object.getOwnPropertyDescriptor(scores, 'reference_256');
  assert.ok(desc, 'reference_256 property should exist on scores');
  assert.equal(desc.enumerable, false);
  assert.equal(desc.configurable, true);
  assert.equal(desc.writable, true);
  assert.equal(Object.keys(scores).includes('reference_256'), false);
  if (expectedExact !== null) {
    assert.equal(scores.reference_256, expectedExact);
  } else {
    assert.match(scores.reference_256, /^data:image\/png;base64,iVBORw0KGgo/);
  }
}

test('accepts valid 256x256 padded reference and excludes padding from MAE/PSNR', async () => {
  const bounds = [28, 12, 228, 244];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [90, 90, 90, 255], [0, 0, 0, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [98, 98, 98, 255], [255, 255, 255, 255]) }],
  ]);
  const { scoreVideoReference, revoked } = loadReferenceHarness(images);
  const refFile = {
    type: 'image/png',
    name: 'ref.png',
    size: 2048,
    _image: { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255], [128, 128, 128, 255]) },
  };
  const scores = await scoreVideoReference(refFile, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: { geometry: { observed_bounds_256: bounds, native_size: [200, 232] } },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '2.00', psnr: (10 * Math.log10((255 ** 2) / 4)).toFixed(2), accuracy: '99.22%' },
  });
  assertReference256(scores);
  assert.equal(revoked.length, 1);
});

test('accepts unpadded aspect-matched crop and aligns it into observed_bounds_256', async () => {
  const bounds = [28, 0, 228, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [120, 120, 120, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [125, 125, 125, 255]) }],
  ]);
  const { scoreVideoReference } = loadReferenceHarness(images);
  const refFile = {
    type: 'image/jpeg',
    name: 'crop.jpg',
    size: 4096,
    _image: { width: 150, height: 180, pixels: solidRgba(150, 180, [125, 125, 125, 255]) },
  };
  const scores = await scoreVideoReference(refFile, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: { geometry: { observed_bounds_256: bounds, native_size: [100, 120] } },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '5.00', psnr: (10 * Math.log10((255 ** 2) / 25)).toFixed(2), accuracy: '98.04%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores);
});

test('rejects unpadded 256x256 reference when observed_bounds_256 is padded', async () => {
  const bounds = [28, 12, 228, 244];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
  ]);
  const { scoreVideoReference, revoked } = loadReferenceHarness(images);
  const badPadded = {
    type: 'image/png',
    name: 'unpadded_256.png',
    size: 1024,
    _image: { width: 256, height: 256, pixels: solidRgba(256, 256, [210, 40, 40, 255]) },
  };
  await assert.rejects(
    scoreVideoReference(badPadded, {
      original_input: 'orig.png',
      result: 'out.png',
      metadata: { geometry: { observed_bounds_256: bounds, native_size: [200, 232] } },
    }),
    /not padded to match this crop/i,
  );
  assert.equal(revoked.length, 1);
});

test('rejects reference whose aspect ratio mismatches both native_size and observed bounds', async () => {
  const bounds = [64, 0, 192, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
  ]);
  const { scoreVideoReference } = loadReferenceHarness(images);
  const wrongAspect = {
    type: 'image/png',
    name: 'wide.png',
    size: 1024,
    _image: { width: 200, height: 100, pixels: solidRgba(200, 100, [100, 100, 100, 255]) },
  };
  await assert.rejects(
    scoreVideoReference(wrongAspect, {
      original_input: 'orig.png',
      result: 'out.png',
      metadata: { geometry: { observed_bounds_256: bounds, native_size: [100, 200] } },
    }),
    /aspect ratio does not match/i,
  );
});

test('supports empty-MIME PNG/JPEG extension fallback and rejects invalid types, sizes, bounds, and alpha', async () => {
  const bounds = [0, 0, 256, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: solidRgba(256, 256, [100, 100, 100, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: solidRgba(256, 256, [100, 100, 100, 255]) }],
  ]);
  const { scoreVideoReference } = loadReferenceHarness(images);
  const result = {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: { geometry: { observed_bounds_256: bounds, native_size: [256, 256] } },
  };

  const emptyMimeJpg = {
    type: '',
    name: 'clean.JPEG',
    size: 512,
    _image: { width: 256, height: 256, pixels: solidRgba(256, 256, [100, 100, 100, 255]) },
  };
  const ok = await scoreVideoReference(emptyMimeJpg, result);
  assert.deepEqual(JSON.parse(JSON.stringify(ok)), {
    before: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(ok);

  for (const badFile of [
    { type: 'image/webp', name: 'ref.webp', size: 512 },
    { type: '', name: 'ref.bmp', size: 512 },
    { type: 'image/png', name: 'empty.png', size: 0 },
    { type: 'image/png', name: 'huge.png', size: 10 * 1024 ** 2 + 1 },
  ]) {
    await assert.rejects(scoreVideoReference(badFile, result), /PNG or JPEG reference under 10 MB/i);
  }

  await assert.rejects(
    scoreVideoReference(emptyMimeJpg, {
      ...result,
      metadata: { geometry: { observed_bounds_256: [100, 10, 50, 200] } },
    }),
    /observed crop bounds could not be verified/i,
  );

  const transparentRef = {
    type: 'image/png',
    name: 'alpha.png',
    size: 512,
    _image: { width: 256, height: 256, pixels: solidRgba(256, 256, [100, 100, 100, 128]) },
  };
  await assert.rejects(scoreVideoReference(transparentRef, result), /must be opaque/i);
});

test('slices full uncropped reference using picture_crop and source_size in local fallback', async () => {
  const bounds = [32, 0, 224, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [130, 130, 130, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [140, 140, 140, 255]) }],
  ]);
  const { scoreVideoReference, revoked } = loadReferenceHarness(images);
  const fullScene = {
    type: 'image/png',
    name: 'full.png',
    size: 4096,
    _image: {
      width: 400,
      height: 300,
      pixels: rectInImageRgba(400, 300, [100, 50, 120, 160], [140, 140, 140, 255], [10, 220, 30, 255]),
    },
  };
  const scores = await scoreVideoReference(fullScene, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: {
      picture_crop: [100, 50, 120, 160],
      source_size: [400, 300],
      geometry: { observed_bounds_256: bounds, native_size: [120, 160] },
    },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores);
  assert.equal(revoked.length, 1);

  // Also works when the full uncropped source image happens to be 256x256
  const square256Full = {
    type: 'image/png',
    name: 'full_256.png',
    size: 2048,
    _image: {
      width: 256,
      height: 256,
      pixels: rectInImageRgba(256, 256, [40, 20, 96, 128], [140, 140, 140, 255], [20, 200, 40, 255]),
    },
  };
  const scores256 = await scoreVideoReference(square256Full, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: {
      picture_crop: [40, 20, 96, 128],
      source_size: [256, 256],
      geometry: { observed_bounds_256: bounds, native_size: [96, 128] },
    },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores256)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores256);
});

test('rejects transparent non-256x256 PNG references before gray padding masks alpha', async () => {
  const bounds = [28, 0, 228, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [120, 120, 120, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [120, 120, 120, 255]) }],
  ]);
  const { scoreVideoReference } = loadReferenceHarness(images);
  for (const alpha of [0, 128]) {
    const transparentCrop = {
      type: 'image/png',
      name: `alpha_${alpha}.png`,
      size: 1024,
      _image: { width: 150, height: 180, pixels: solidRgba(150, 180, [120, 120, 120, alpha]) },
    };
    await assert.rejects(
      scoreVideoReference(transparentCrop, {
        original_input: 'orig.png',
        result: 'out.png',
        metadata: { geometry: { observed_bounds_256: bounds, native_size: [100, 120] } },
      }),
      /must be opaque/i,
    );
  }
});

test('uses /face/restoration/reference when fetch succeeds, scores prepared_input, and handles 400 vs 404/network fallback', async () => {
  const bounds = [28, 0, 228, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
    ['prep.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [106, 106, 106, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [110, 110, 110, 255]) }],
    ['server_ref.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [110, 110, 110, 255]) }],
  ]);
  const requests = [];
  const { scoreVideoReference, revoked } = loadReferenceHarness(images, {
    fetch: async (url, init) => {
      requests.push({ url, init });
      return {
        ok: true,
        status: 200,
        json: async () => ({ reference_256: 'server_ref.png' }),
      };
    },
  });
  const refFile = {
    type: 'image/png',
    name: 'full.png',
    size: 4096,
    _image: { width: 400, height: 300, pixels: solidRgba(400, 300, [10, 10, 10, 255]) },
  };
  const result = {
    original_input: 'orig.png',
    prepared_input: 'prep.png',
    result: 'out.png',
    metadata: {
      preparation_applied: true,
      picture_crop: [50, 40, 100, 120],
      source_size: [400, 300],
      geometry: { observed_bounds_256: bounds, native_size: [100, 120] },
    },
  };
  const scores = await scoreVideoReference(refFile, result);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/face/restoration/reference');
  assert.equal(requests[0].init.method, 'POST');
  assert.equal(requests[0].init.body.get('file'), refFile);
  assert.deepEqual(JSON.parse(requests[0].init.body.get('geometry')), {
    observed_bounds_256: bounds,
    native_size: [100, 120],
    picture_crop: [50, 40, 100, 120],
    source_size: [400, 300],
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    prepared: { mae: '4.00', psnr: (10 * Math.log10((255 ** 2) / 16)).toFixed(2), accuracy: '98.43%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores, 'server_ref.png');
  assert.equal(revoked.length, 0);

  // HTTP 400 propagates server detail without falling back
  const err400 = loadReferenceHarness(images, {
    fetch: async () => ({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'The reference must be opaque.' }),
    }),
  });
  await assert.rejects(err400.scoreVideoReference(refFile, result), /The reference must be opaque\./);

  // HTTP 404 and network errors fall back to local Canvas2D
  const localRef = {
    type: 'image/png',
    name: 'crop.png',
    size: 2048,
    _image: { width: 100, height: 120, pixels: solidRgba(100, 120, [110, 110, 110, 255]) },
  };
  for (const fetchImpl of [
    async () => ({ ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) }),
    async () => { throw new Error('network down'); },
  ]) {
    const fallback = loadReferenceHarness(images, { fetch: fetchImpl });
    const fbScores = await fallback.scoreVideoReference(localRef, {
      ...result,
      metadata: { ...result.metadata, picture_crop: null, source_size: null },
    });
    assert.deepEqual(JSON.parse(JSON.stringify(fbScores)), {
      before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
      prepared: { mae: '4.00', psnr: (10 * Math.log10((255 ** 2) / 16)).toFixed(2), accuracy: '98.43%' },
      after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
    });
    assertReference256(fbScores);
    assert.equal(fallback.revoked.length, 1);
  }

  // localOnly skips fetch even when fetch is available
  let fetchCalled = false;
  const localOnlyHarness = loadReferenceHarness(images, {
    fetch: async () => { fetchCalled = true; return { ok: true, status: 200, json: async () => ({}) }; },
  });
  await localOnlyHarness.scoreVideoReference(localRef, {
    ...result,
    metadata: { ...result.metadata, picture_crop: null, source_size: null },
  }, { localOnly: true });
  assert.equal(fetchCalled, false);
});

test('slices >1600px video frame reference using capture_crop and frame_size and forwards them to server', async () => {
  const bounds = [32, 0, 224, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [130, 130, 130, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [145, 145, 145, 255]) }],
    ['server_ref.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [145, 145, 145, 255]) }],
  ]);
  const videoResult = {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: {
      capture_crop: [840, 400, 180, 240],
      frame_size: [1920, 1080],
      source_size: [180, 240],
      geometry: { observed_bounds_256: bounds, native_size: [180, 240] },
    },
  };
  const fullHdFrame = {
    type: 'image/png',
    name: 'frame_1080p.png',
    size: 8192,
    _image: {
      width: 1920,
      height: 1080,
      pixels: rectInImageRgba(1920, 1080, [840, 400, 180, 240], [145, 145, 145, 255], [20, 200, 30, 255]),
    },
  };

  // Local fallback slices 1920x1080 frame via capture_crop + frame_size even when source_size is the crop size
  const localHarness = loadReferenceHarness(images);
  const localScores = await localHarness.scoreVideoReference(fullHdFrame, videoResult, { localOnly: true });
  assert.deepEqual(JSON.parse(JSON.stringify(localScores)), {
    before: { mae: '15.00', psnr: (10 * Math.log10((255 ** 2) / 225)).toFixed(2), accuracy: '94.12%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(localScores);
  assert.equal(localHarness.revoked.length, 1);

  // Server fetch maps capture_crop -> picture_crop and frame_size -> source_size
  const requests = [];
  const serverHarness = loadReferenceHarness(images, {
    fetch: async (url, init) => {
      requests.push({ url, init });
      return { ok: true, status: 200, json: async () => ({ reference_256: 'server_ref.png' }) };
    },
  });
  const serverScores = await serverHarness.scoreVideoReference(fullHdFrame, videoResult);
  assert.equal(requests.length, 1);
  assert.deepEqual(JSON.parse(requests[0].init.body.get('geometry')), {
    observed_bounds_256: bounds,
    native_size: [180, 240],
    picture_crop: [840, 400, 180, 240],
    source_size: [1920, 1080],
  });
  assert.deepEqual(JSON.parse(JSON.stringify(serverScores)), {
    before: { mae: '15.00', psnr: (10 * Math.log10((255 ** 2) / 225)).toFixed(2), accuracy: '94.12%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(serverScores, 'server_ref.png');
});

test('fetchServerReference appends data PNG original_input and scoreVideoReference scores over eval_bounds_256', async () => {
  const observedBounds = [20, 20, 236, 236];
  const evalBounds = [64, 64, 192, 192];
  // Outer ring [20..236] outside evalBounds [64..192] differs drastically (255 vs 0),
  // while inside evalBounds [64..192] original=100, output=110, reference=110.
  const origPixels = rectInImageRgba(256, 256, [64, 64, 128, 128], [100, 100, 100, 255], [0, 0, 0, 255]);
  const outPixels = rectInImageRgba(256, 256, [64, 64, 128, 128], [110, 110, 110, 255], [255, 255, 255, 255]);
  const refPixels = rectInImageRgba(256, 256, [64, 64, 128, 128], [110, 110, 110, 255], [50, 50, 50, 255]);

  const origDataUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB';
  const images = new Map([
    [origDataUrl, { width: 256, height: 256, pixels: origPixels }],
    ['out.png', { width: 256, height: 256, pixels: outPixels }],
    ['aligned_ref.png', { width: 256, height: 256, pixels: refPixels }],
  ]);

  const requests = [];
  const harness = loadReferenceHarness(images, {
    fetch: async (url, init) => {
      requests.push({ url, init });
      return {
        ok: true,
        status: 200,
        json: async () => ({
          reference_256: 'aligned_ref.png',
          eval_bounds_256: evalBounds,
        }),
      };
    },
  });

  const refFile = {
    type: 'image/png',
    name: 'full.png',
    size: 4096,
    _image: { width: 400, height: 300, pixels: solidRgba(400, 300, [110, 110, 110, 255]) },
  };
  const scores = await harness.scoreVideoReference(refFile, {
    original_input: origDataUrl,
    result: 'out.png',
    metadata: {
      picture_crop: [50, 40, 100, 120],
      source_size: [400, 300],
      geometry: { observed_bounds_256: observedBounds, native_size: [100, 120] },
    },
  });

  assert.equal(requests.length, 1);
  assert.equal(requests[0].init.body.get('original_input'), origDataUrl);
  // Scoring over evalBounds [64, 64, 192, 192] yields exact 10.00 MAE for before and 0.00 MAE for after
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores, 'aligned_ref.png');

  // Non-data-PNG original_input is NOT appended, and invalid eval_bounds_256 falls back to observed_bounds_256
  const requestsNonPng = [];
  const fallbackImages = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(observedBounds, [100, 100, 100, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(observedBounds, [110, 110, 110, 255]) }],
    ['aligned_ref2.png', { width: 256, height: 256, pixels: paddedRgba(observedBounds, [110, 110, 110, 255]) }],
  ]);
  const nonPngHarness = loadReferenceHarness(fallbackImages, {
    fetch: async (url, init) => {
      requestsNonPng.push({ url, init });
      return {
        ok: true,
        status: 200,
        json: async () => ({
          reference_256: 'aligned_ref2.png',
          eval_bounds_256: [10, 10, 5, 200],
        }),
      };
    },
  });
  const nonPngScores = await nonPngHarness.scoreVideoReference(refFile, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: { geometry: { observed_bounds_256: observedBounds, native_size: [216, 216] } },
  });
  assert.equal(requestsNonPng[0].init.body.get('original_input'), undefined);
  assert.deepEqual(JSON.parse(JSON.stringify(nonPngScores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(nonPngScores, 'aligned_ref2.png');
});

test('extracts video frame at result.metadata.source_time as PNG File and sends it to /face/restoration/reference', async () => {
  const bounds = [32, 0, 224, 256];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [120, 120, 120, 255]) }],
    ['server_video_ref.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [120, 120, 120, 255]) }],
  ]);
  const requests = [];
  const harness = loadReferenceHarness(images, {
    fetch: async (url, init) => {
      requests.push({ url, init });
      return {
        ok: true,
        status: 200,
        json: async () => ({ reference_256: 'server_video_ref.png', eval_bounds_256: bounds }),
      };
    },
  });

  const videoFile = {
    type: 'video/mp4',
    name: 'clean-source.mp4',
    size: 5 * 1024 * 1024,
    _video: {
      width: 640,
      height: 360,
      duration: 8.0,
      frameAt(time) {
        const val = Math.abs(time - 3.25) < 0.01 ? 120 : 10;
        return rectInImageRgba(640, 360, [200, 60, 180, 240], [val, val, val, 255], [15, 25, 35, 255]);
      },
    },
  };
  const result = {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: {
      source_time: 3.25,
      capture_crop: [200, 60, 180, 240],
      frame_size: [640, 360],
      geometry: { observed_bounds_256: bounds, native_size: [180, 240] },
    },
  };

  const scores = await harness.scoreVideoReference(videoFile, result);
  assert.equal(harness.createdVideos.length, 1);
  assert.equal(harness.createdVideos[0].muted, true);
  assert.equal(harness.createdVideos[0].playsInline, true);
  assert.deepEqual(harness.createdVideos[0].seekedTimes, [3.25]);
  assert.equal(harness.revoked.length, 1);

  assert.equal(requests.length, 1);
  const uploaded = requests[0].init.body.get('file');
  assert.equal(uploaded.name, 'reference-frame.png');
  assert.equal(uploaded.type, 'image/png');
  assert.equal(uploaded._image.width, 640);
  assert.equal(uploaded._image.height, 360);
  // Check pixel inside capture_crop reflects frame at source_time=3.25 (120)
  const sampleIdx = (100 * 640 + 250) * 4;
  assert.equal(uploaded._image.pixels[sampleIdx], 120);

  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '20.00', psnr: (10 * Math.log10((255 ** 2) / 400)).toFixed(2), accuracy: '92.16%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(scores, 'server_video_ref.png');

  // Also test localOnly fallback with .m4v extension (empty MIME) and source_time exceeding duration
  const localScores = await harness.scoreVideoReference({
    type: '',
    name: 'clean-clip.m4v',
    size: 2048,
    _video: {
      width: 640,
      height: 360,
      duration: 2.0,
      frameAt(time) {
        const val = Math.abs(time - 1.999) < 0.002 ? 120 : 10;
        return rectInImageRgba(640, 360, [200, 60, 180, 240], [val, val, val, 255], [15, 25, 35, 255]);
      },
    },
  }, {
    ...result,
    metadata: { ...result.metadata, source_time: 5.0 },
  }, { localOnly: true });
  assert.deepEqual(harness.createdVideos[1].seekedTimes, [1.999]);
  assert.deepEqual(JSON.parse(JSON.stringify(localScores)), {
    before: { mae: '20.00', psnr: (10 * Math.log10((255 ** 2) / 400)).toFixed(2), accuracy: '92.16%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assertReference256(localScores);
  // 1 revoke for extractVideoFrameAsPngFile + 1 revoke for local fallback referencePixels
  assert.equal(harness.revoked.length, 3);

  // Oversized (> 500 MB) and decode errors reject cleanly and still revoke object URL
  await assert.rejects(
    harness.scoreVideoReference({ type: 'video/mp4', name: 'huge.mp4', size: 500 * 1024 ** 2 + 1 }, result),
    /video reference under 500 MB/i,
  );
  await assert.rejects(
    harness.scoreVideoReference({
      type: 'video/mp4',
      name: 'corrupt.mp4',
      size: 1024,
      _video: { decodeError: true },
    }, result),
    /cannot decode this video reference/i,
  );
  assert.equal(harness.revoked.length, 4);
});

test('omits reference_256 gracefully when canvas.toDataURL is unavailable in local fallback', async () => {
  const bounds = [28, 12, 228, 244];
  const images = new Map([
    ['orig.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [90, 90, 90, 255]) }],
    ['out.png', { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) }],
  ]);
  const { scoreVideoReference, revoked } = loadReferenceHarness(images, { disableCanvasToDataUrl: true });
  const refFile = {
    type: 'image/png',
    name: 'ref.png',
    size: 2048,
    _image: { width: 256, height: 256, pixels: paddedRgba(bounds, [100, 100, 100, 255]) },
  };
  const scores = await scoreVideoReference(refFile, {
    original_input: 'orig.png',
    result: 'out.png',
    metadata: { geometry: { observed_bounds_256: bounds, native_size: [200, 232] } },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(scores)), {
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2), accuracy: '96.08%' },
    after: { mae: '0.00', psnr: '∞', accuracy: '100.00%' },
  });
  assert.equal('reference_256' in scores, false);
  assert.equal(revoked.length, 1);
});
