const test = require('node:test');
const assert = require('node:assert/strict');
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
      const srcW = image.naturalWidth;
      const srcH = image.naturalHeight;
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
          this.canvas._data[di] = image._pixels[si];
          this.canvas._data[di + 1] = image._pixels[si + 1];
          this.canvas._data[di + 2] = image._pixels[si + 2];
          this.canvas._data[di + 3] = image._pixels[si + 3];
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
  }

  const windowObj = options.protocol ? { location: { protocol: options.protocol } } : {};
  const sandbox = {
    window: windowObj,
    Image: MockImage,
    FormData: MockFormData,
    fetch: options.fetch,
    document: {
      createElement(tag) {
        assert.equal(tag, 'canvas');
        return new MockCanvas();
      },
    },
    URL: {
      createObjectURL(file) {
        const url = `blob:ref-${nextBlobId++}`;
        imagesByUrl.set(url, file._image);
        return url;
      },
      revokeObjectURL(url) {
        revoked.push(url);
      },
    },
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
  return { scoreVideoReference: windowObj.scoreVideoReference, revoked };
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
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2) },
    after: { mae: '2.00', psnr: (10 * Math.log10((255 ** 2) / 4)).toFixed(2) },
  });
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
    before: { mae: '5.00', psnr: (10 * Math.log10((255 ** 2) / 25)).toFixed(2) },
    after: { mae: '0.00', psnr: '∞' },
  });
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
    before: { mae: '0.00', psnr: '∞' },
    after: { mae: '0.00', psnr: '∞' },
  });

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
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2) },
    after: { mae: '0.00', psnr: '∞' },
  });
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
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2) },
    after: { mae: '0.00', psnr: '∞' },
  });
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
    before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2) },
    prepared: { mae: '4.00', psnr: (10 * Math.log10((255 ** 2) / 16)).toFixed(2) },
    after: { mae: '0.00', psnr: '∞' },
  });
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
      before: { mae: '10.00', psnr: (10 * Math.log10((255 ** 2) / 100)).toFixed(2) },
      prepared: { mae: '4.00', psnr: (10 * Math.log10((255 ** 2) / 16)).toFixed(2) },
      after: { mae: '0.00', psnr: '∞' },
    });
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
