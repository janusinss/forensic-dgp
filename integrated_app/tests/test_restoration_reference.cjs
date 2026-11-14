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

function loadReferenceHarness(imagesByUrl) {
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
    drawImage(image, dx = 0, dy = 0, dw = image.naturalWidth, dh = image.naturalHeight) {
      this._EnsureBuffer();
      const srcW = image.naturalWidth;
      const srcH = image.naturalHeight;
      const x0 = Math.max(0, Math.floor(dx));
      const y0 = Math.max(0, Math.floor(dy));
      const x1 = Math.min(this.canvas.width, Math.floor(dx + dw));
      const y1 = Math.min(this.canvas.height, Math.floor(dy + dh));
      for (let yy = y0; yy < y1; yy++) {
        const sy = Math.min(srcH - 1, Math.floor(((yy - dy) * srcH) / dh));
        for (let xx = x0; xx < x1; xx++) {
          const sx = Math.min(srcW - 1, Math.floor(((xx - dx) * srcW) / dw));
          const si = (sy * srcW + sx) * 4;
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

  const windowObj = {};
  const sandbox = {
    window: windowObj,
    Image: MockImage,
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
