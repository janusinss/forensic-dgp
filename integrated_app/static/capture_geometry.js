/* Preview-only view windows. Crops always remain in original source pixels. */
((root) => {
  'use strict';
  function makeView(frameWidth, frameHeight, rect = null, padding = .2) {
    if (!Number.isInteger(frameWidth) || !Number.isInteger(frameHeight) || frameWidth < 1 || frameHeight < 1) throw new RangeError('Invalid frame dimensions.');
    if (!rect) return {x:0, y:0, width:frameWidth, height:frameHeight};
    if (![rect.x, rect.y, rect.width, rect.height, padding].every(Number.isFinite) || rect.width <= 0 || rect.height <= 0 || padding < 0) throw new RangeError('Invalid preview rectangle.');
    const x = Math.max(0, Math.min(frameWidth - 1, Math.floor(rect.x - rect.width * padding)));
    const y = Math.max(0, Math.min(frameHeight - 1, Math.floor(rect.y - rect.height * padding)));
    const right = Math.max(x + 1, Math.min(frameWidth, Math.ceil(rect.x + rect.width * (1 + padding))));
    const bottom = Math.max(y + 1, Math.min(frameHeight, Math.ceil(rect.y + rect.height * (1 + padding))));
    return {x, y, width:right-x, height:bottom-y};
  }
  function pointerToSource(clientX, clientY, bounds, view) {
    if (![clientX,clientY,bounds.left,bounds.top,bounds.width,bounds.height,view.x,view.y,view.width,view.height].every(Number.isFinite) || view.width<=0 || view.height<=0) throw new RangeError('Invalid preview coordinates.');
    const scale = Math.min(bounds.width / view.width, bounds.height / view.height);
    if (!Number.isFinite(scale) || scale <= 0) throw new RangeError('The preview is not visible.');
    const offsetX = (bounds.width - view.width * scale) / 2;
    const offsetY = (bounds.height - view.height * scale) / 2;
    return {
      x:Math.max(view.x, Math.min(view.x + view.width, Math.round(view.x + (clientX-bounds.left-offsetX)/scale))),
      y:Math.max(view.y, Math.min(view.y + view.height, Math.round(view.y + (clientY-bounds.top-offsetY)/scale))),
      scale
    };
  }
  function isValidCropRect(rect, frameWidth, frameHeight, minSide = 16, maxSide = 1600) {
    if (!rect || typeof rect !== 'object') return false;
    if (!Number.isInteger(frameWidth) || !Number.isInteger(frameHeight) || frameWidth < 1 || frameHeight < 1) return false;
    if (!Number.isInteger(minSide) || !Number.isInteger(maxSide) || minSide < 1 || maxSide < minSide) return false;
    const { x, y, width, height } = rect;
    if (![x, y, width, height].every(Number.isInteger)) return false;
    return (
      x >= 0 &&
      y >= 0 &&
      width >= minSide &&
      height >= minSide &&
      width <= maxSide &&
      height <= maxSide &&
      x + width <= frameWidth &&
      y + height <= frameHeight
    );
  }
  function clampCropRect(rect, frameWidth, frameHeight, minSide = 16, maxSide = 1600) {
    if (!Number.isInteger(frameWidth) || !Number.isInteger(frameHeight) || frameWidth < 1 || frameHeight < 1) throw new RangeError('Invalid frame dimensions.');
    if (!Number.isInteger(minSide) || !Number.isInteger(maxSide) || minSide < 1 || maxSide < minSide) throw new RangeError('Invalid crop bounds.');
    if (frameWidth < minSide || frameHeight < minSide) throw new RangeError('Frame dimensions are smaller than minimum crop side.');
    if (!rect || typeof rect !== 'object' || ![rect.x, rect.y, rect.width, rect.height].every(Number.isFinite)) throw new RangeError('Invalid crop rectangle.');
    const roundInt = value => { const n = Math.round(value); return n === 0 ? 0 : n; };
    const maxWidth = Math.min(maxSide, frameWidth);
    const maxHeight = Math.min(maxSide, frameHeight);
    const width = Math.max(minSide, Math.min(maxWidth, roundInt(rect.width)));
    const height = Math.max(minSide, Math.min(maxHeight, roundInt(rect.height)));
    const x = Math.max(0, Math.min(frameWidth - width, roundInt(rect.x)));
    const y = Math.max(0, Math.min(frameHeight - height, roundInt(rect.y)));
    return { x, y, width, height };
  }
  const api = Object.freeze({makeView, pointerToSource, clampCropRect, isValidCropRect});
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.captureGeometry = api;
})(globalThis);
