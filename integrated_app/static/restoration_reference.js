(() => {
    'use strict';
    const side = 256;

    async function decodeImage(source) {
        const image = new Image();
        image.src = source;
        try {
            await image.decode();
        } catch {
            throw new Error('Choose a valid PNG or JPEG reference image.');
        }
        return image;
    }

    async function pixels(source) {
        const image = await decodeImage(source);
        if (image.naturalWidth !== side || image.naturalHeight !== side) {
            throw new Error('Choose a clean, aligned 256 × 256 PNG or JPEG.');
        }
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = side;
        const context = canvas.getContext('2d');
        context.drawImage(image, 0, 0);
        return context.getImageData(0, 0, side, side).data;
    }

    function hasUniformGrayPadding(data, bounds) {
        const [left, top, right, bottom] = bounds;
        const margin = 2;
        const checkPixels = useMargin => {
            let checked = 0, mismatched = 0;
            for (let y = 0; y < side; y++) {
                for (let x = 0; x < side; x++) {
                    if (x >= left && x < right && y >= top && y < bottom) continue;
                    if (useMargin && x >= left - margin && x < right + margin && y >= top - margin && y < bottom + margin) continue;
                    checked++;
                    const i = (y * side + x) * 4;
                    if (data[i + 3] !== 255
                        || Math.abs(data[i] - 128) > 16
                        || Math.abs(data[i + 1] - 128) > 16
                        || Math.abs(data[i + 2] - 128) > 16) {
                        mismatched++;
                    }
                }
            }
            return { checked, mismatched };
        };
        let stats = checkPixels(true);
        if (stats.checked === 0) stats = checkPixels(false);
        if (stats.checked === 0) return true;
        return stats.mismatched / stats.checked <= 0.02;
    }

    function validPositivePair(pair) {
        return Array.isArray(pair) && pair.length === 2
            && pair.every(n => Number.isInteger(n) && n > 0);
    }

    function validCropWithinSource(crop, sourceSize) {
        if (!Array.isArray(crop) || crop.length !== 4 || !crop.every(n => Number.isInteger(n))) return false;
        const [cx, cy, cw, ch] = crop;
        return cx >= 0 && cy >= 0 && cw > 0 && ch > 0
            && cx + cw <= sourceSize[0] && cy + ch <= sourceSize[1];
    }

    function drawCroppedToBounds(image, bounds, cropRect) {
        const [left, top, right, bottom] = bounds;
        const obsW = right - left, obsH = bottom - top;
        const draw = ctx => {
            if (cropRect) {
                const [cx, cy, cw, ch] = cropRect;
                ctx.drawImage(image, cx, cy, cw, ch, left, top, obsW, obsH);
            } else {
                ctx.drawImage(image, left, top, obsW, obsH);
            }
        };
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = side;
        const context = canvas.getContext('2d');
        // Check alpha on an unfilled canvas first so rgb(128, 128, 128) padding cannot mask transparency.
        draw(context);
        const alphaData = context.getImageData(0, 0, side, side).data;
        for (let y = top; y < bottom; y++) {
            for (let x = left; x < right; x++) {
                if (alphaData[(y * side + x) * 4 + 3] !== 255) {
                    throw new Error('The reference must be opaque.');
                }
            }
        }
        context.fillStyle = 'rgb(128, 128, 128)';
        context.fillRect(0, 0, side, side);
        draw(context);
        return context.getImageData(0, 0, side, side).data;
    }

    async function referencePixels(source, result) {
        const image = await decodeImage(source);
        const natW = image.naturalWidth, natH = image.naturalHeight;
        if (!natW || !natH) throw new Error('Choose a valid PNG or JPEG reference image.');
        const geometry = result?.metadata?.geometry ?? result?.geometry ?? result;
        const bounds = geometry.observed_bounds_256;
        const [left, top, right, bottom] = bounds;
        const obsW = right - left, obsH = bottom - top;
        const isPadded = left > 0 || top > 0 || right < side || bottom < side;
        const pictureCrop = result?.metadata?.picture_crop ?? geometry?.picture_crop;
        const sourceSize = result?.metadata?.source_size ?? geometry?.source_size;
        const canSliceSource = validPositivePair(sourceSize)
            && natW === sourceSize[0] && natH === sourceSize[1]
            && validCropWithinSource(pictureCrop, sourceSize);

        if (natW === side && natH === side) {
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = side;
            const context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);
            const data = context.getImageData(0, 0, side, side).data;
            if (isPadded) {
                if (hasUniformGrayPadding(data, bounds)) {
                    return data;
                }
                if (canSliceSource) {
                    return drawCroppedToBounds(image, bounds, pictureCrop);
                }
                throw new Error('The 256 × 256 reference is not padded to match this crop. Upload the clean unpadded crop or a 256 × 256 padded frame.');
            }
            if (canSliceSource) {
                return drawCroppedToBounds(image, bounds, pictureCrop);
            }
            return data;
        }

        if (canSliceSource) {
            return drawCroppedToBounds(image, bounds, pictureCrop);
        }

        const nativeSize = Array.isArray(geometry.native_size) && geometry.native_size.length === 2 ? geometry.native_size : null;
        const expectedW = nativeSize && nativeSize[0] > 0 ? nativeSize[0] : obsW;
        const expectedH = nativeSize && nativeSize[1] > 0 ? nativeSize[1] : obsH;
        const refAspect = natW / natH;
        const expectedAspect = expectedW / expectedH;
        const obsAspect = obsW / obsH;
        const diffExpected = Math.abs(refAspect - expectedAspect) / Math.max(refAspect, expectedAspect);
        const diffObs = Math.abs(refAspect - obsAspect) / Math.max(refAspect, obsAspect);
        if (diffExpected > 0.05 && diffObs > 0.05) {
            throw new Error('Reference aspect ratio does not match the input crop. Use the full original photo, a clean crop matching the input, or a 256 × 256 padded frame.');
        }
        return drawCroppedToBounds(image, bounds, null);
    }

    function score(target, candidate, bounds) {
        const [left, top, right, bottom] = bounds;
        let absolute = 0, squared = 0;
        for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) {
            const i = (y * side + x) * 4;
            if (target[i + 3] !== 255) throw new Error('The reference must be opaque.');
            for (let channel = 0; channel < 3; channel++) {
                const delta = target[i + channel] - candidate[i + channel];
                absolute += Math.abs(delta);
                squared += delta ** 2;
            }
        }
        const count = (right - left) * (bottom - top) * 3;
        const psnr = squared ? (10 * Math.log10(255 ** 2 / (squared / count))).toFixed(2) : '∞';
        return { mae: (absolute / count).toFixed(2), psnr };
    }

    async function fetchServerReference(file, result, geometry, bounds) {
        const form = new FormData();
        form.append('file', file);
        const payload = { observed_bounds_256: bounds };
        if (geometry?.native_size != null) payload.native_size = geometry.native_size;
        const pictureCrop = result.metadata?.picture_crop ?? geometry?.picture_crop;
        if (pictureCrop != null) payload.picture_crop = pictureCrop;
        const sourceSize = result.metadata?.source_size ?? geometry?.source_size;
        if (sourceSize != null) payload.source_size = sourceSize;
        form.append('geometry', JSON.stringify(payload));
        let response;
        try {
            response = await fetch('/face/restoration/reference', { method: 'POST', body: form });
        } catch {
            return null;
        }
        if (response.status === 404 || response.status === 405) {
            return null;
        }
        let body = null;
        try {
            body = await response.json();
        } catch {
            body = null;
        }
        if (!response.ok) {
            const detail = typeof body?.detail === 'string'
                ? body.detail
                : Array.isArray(body?.detail)
                    ? body.detail.map(item => item?.msg || JSON.stringify(item)).join('; ')
                    : (body?.error || body?.message || `Reference preparation failed (${response.status}).`);
            throw new Error(detail);
        }
        if (!body || typeof body.reference_256 !== 'string' || !body.reference_256) {
            throw new Error('The prepared reference image is unavailable.');
        }
        return body.reference_256;
    }

    window.scoreVideoReference = async (file, result, options = {}) => {
        const allowedTypes = ['image/png', 'image/jpeg'];
        const hasAllowedType = allowedTypes.includes(file?.type);
        const hasAllowedExt = !file?.type && /\.(png|jpe?g)$/i.test(file?.name || '');
        if (!file || !file.size || file.size > 10 * 1024 ** 2 || (!hasAllowedType && !hasAllowedExt)) {
            throw new Error('Choose a PNG or JPEG reference under 10 MB.');
        }
        const geometry = result.metadata?.geometry;
        const bounds = geometry?.observed_bounds_256;
        if (bounds?.length !== 4 || !bounds.every(n => Number.isInteger(n) && n >= 0 && n <= side)
            || bounds[0] >= bounds[2] || bounds[1] >= bounds[3]) {
            throw new Error('The observed crop bounds could not be verified.');
        }

        const canFetch = typeof fetch === 'function'
            && typeof FormData === 'function'
            && !(options && options.localOnly)
            && typeof window !== 'undefined'
            && window.location?.protocol !== 'file:';

        let serverRefUrl = null;
        if (canFetch) {
            serverRefUrl = await fetchServerReference(file, result, geometry, bounds);
        }

        const hasPrepared = Boolean(result.metadata?.preparation_applied && result.prepared_input);
        let url = null;
        try {
            const targetPromise = serverRefUrl
                ? pixels(serverRefUrl)
                : (() => {
                    url = URL.createObjectURL(file);
                    return referencePixels(url, result);
                })();
            const [target, original, output, prepared] = await Promise.all([
                targetPromise,
                pixels(result.original_input),
                pixels(result.result),
                hasPrepared ? pixels(result.prepared_input) : Promise.resolve(null),
            ]);
            const scores = {
                before: score(target, original, bounds),
                after: score(target, output, bounds),
            };
            if (prepared) {
                scores.prepared = score(target, prepared, bounds);
            }
            return scores;
        } finally {
            if (url) URL.revokeObjectURL(url);
        }
    };
})();
