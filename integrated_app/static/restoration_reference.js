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

    function canvasResult(canvas, data) {
        return {
            data,
            dataUrl: typeof canvas.toDataURL === 'function' ? canvas.toDataURL('image/png') : null,
        };
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
        return canvasResult(canvas, context.getImageData(0, 0, side, side).data);
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
        const pictureCrop = result?.metadata?.picture_crop ?? result?.metadata?.capture_crop ?? geometry?.picture_crop;
        const sourceSize = result?.metadata?.picture_crop != null
            ? (result?.metadata?.source_size ?? geometry?.source_size)
            : (result?.metadata?.frame_size ?? result?.metadata?.source_size ?? geometry?.source_size);
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
                    return canvasResult(canvas, data);
                }
                if (canSliceSource) {
                    return drawCroppedToBounds(image, bounds, pictureCrop);
                }
                throw new Error('The 256 × 256 reference is not padded to match this crop. Upload the clean unpadded crop or a 256 × 256 padded frame.');
            }
            if (canSliceSource) {
                return drawCroppedToBounds(image, bounds, pictureCrop);
            }
            return canvasResult(canvas, data);
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
        const maeNum = absolute / count;
        const accuracy = `${((1 - maeNum / 255) * 100).toFixed(2)}%`;
        const psnr = squared ? (10 * Math.log10(255 ** 2 / (squared / count))).toFixed(2) : '∞';
        return { mae: maeNum.toFixed(2), psnr, accuracy };
    }

    function waitForVideoEvent(video, eventName, timeoutMessage, trigger) {
        return new Promise((resolve, reject) => {
            let settled = false;
            let timer = null;
            const cleanup = () => {
                video.removeEventListener?.(eventName, onSuccess);
                video.removeEventListener?.('error', onError);
                if (video.onloadeddata === onSuccess) video.onloadeddata = null;
                if (video.onseeked === onSuccess) video.onseeked = null;
                if (video.onerror === onError) video.onerror = null;
                if (timer != null && typeof clearTimeout === 'function') clearTimeout(timer);
            };
            const finish = problem => {
                if (settled) return;
                settled = true;
                cleanup();
                problem ? reject(problem) : resolve();
            };
            const onSuccess = () => finish();
            const onError = () => finish(new Error('This browser cannot decode this video reference. Try MP4 or WebM.'));
            if (typeof setTimeout === 'function') {
                timer = setTimeout(() => finish(new Error(timeoutMessage)), 15000);
            }
            if (typeof video.addEventListener === 'function') {
                video.addEventListener(eventName, onSuccess, { once: true });
                video.addEventListener('error', onError, { once: true });
            } else if (eventName === 'loadeddata') {
                video.onloadeddata = onSuccess;
                video.onerror = onError;
            } else {
                video.onseeked = onSuccess;
                video.onerror = onError;
            }
            if (trigger) trigger();
        });
    }

    async function extractVideoFrameAsPngFile(file, result) {
        if (!file || !file.size || file.size > 500 * 1024 ** 2) {
            throw new Error('Choose a video reference under 500 MB.');
        }
        const targetTime = Number.isFinite(result?.metadata?.source_time) && result.metadata.source_time >= 0
            ? result.metadata.source_time
            : 0;
        const objectUrl = URL.createObjectURL(file);
        try {
            const video = document.createElement('video');
            video.muted = true;
            video.playsInline = true;
            video.preload = 'auto';
            await waitForVideoEvent(
                video,
                'loadeddata',
                'The video reference could not be decoded within 15 seconds. Try another file.',
                () => {
                    video.src = objectUrl;
                    if (typeof video.load === 'function') video.load();
                }
            );
            const hasFiniteDuration = Number.isFinite(video.duration) && video.duration > 0;
            const maxSeek = hasFiniteDuration
                ? Math.max(0, (video.duration || 0) - 0.001)
                : targetTime;
            const zeroFrames = video.getVideoPlaybackQuality?.().totalVideoFrames === 0;
            if (targetTime > 0 || zeroFrames) {
                const seekTo = targetTime > 0
                    ? Math.min(targetTime, maxSeek)
                    : (hasFiniteDuration ? Math.min(0.001, maxSeek) : 0);
                if (seekTo > 0 || zeroFrames) {
                    await waitForVideoEvent(
                        video,
                        'seeked',
                        'Timed out while seeking to the matching video reference frame.',
                        () => {
                            video.currentTime = seekTo;
                        }
                    );
                }
            }
            const width = video.videoWidth;
            const height = video.videoHeight;
            if (!width || !height) {
                throw new Error('The browser could not decode a frame from this video reference.');
            }
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, width, height);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            if (!blob || !blob.size) {
                throw new Error('Could not encode the extracted video reference frame.');
            }
            return new File([blob], 'reference-frame.png', { type: 'image/png' });
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    async function fetchServerReference(file, result, geometry, bounds) {
        const form = new FormData();
        form.append('file', file);
        const payload = { observed_bounds_256: bounds };
        if (geometry?.native_size != null) payload.native_size = geometry.native_size;
        const pictureCrop = result?.metadata?.picture_crop ?? result?.metadata?.capture_crop ?? geometry?.picture_crop;
        if (pictureCrop != null) payload.picture_crop = pictureCrop;
        const sourceSize = result?.metadata?.picture_crop != null
            ? (result?.metadata?.source_size ?? geometry?.source_size)
            : (result?.metadata?.frame_size ?? result?.metadata?.source_size ?? geometry?.source_size);
        if (sourceSize != null) payload.source_size = sourceSize;
        form.append('geometry', JSON.stringify(payload));
        const alignmentOriginal = typeof result?._alignmentOriginalInput === 'string' && result._alignmentOriginalInput.startsWith('data:image/png;base64,')
            ? result._alignmentOriginalInput
            : result?.original_input;
        if (typeof alignmentOriginal === 'string' && alignmentOriginal.startsWith('data:image/png;base64,')) {
            form.append('original_input', alignmentOriginal);
        }
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
        return {
            reference_256: body.reference_256,
            eval_bounds_256: body.eval_bounds_256,
        };
    }

    function validBounds256(bounds) {
        return Array.isArray(bounds)
            && bounds.length === 4
            && bounds.every(n => Number.isInteger(n) && n >= 0 && n <= side)
            && bounds[0] < bounds[2]
            && bounds[1] < bounds[3];
    }

    window.scoreVideoReference = async (file, result, options = {}) => {
        const isVideo = /^video\//i.test(file?.type || '') || /\.(mp4|webm|mov|m4v)$/i.test(file?.name || '');
        if (isVideo) {
            if (!file || !file.size || file.size > 500 * 1024 ** 2) {
                throw new Error('Choose a video reference under 500 MB.');
            }
        } else {
            const allowedTypes = ['image/png', 'image/jpeg'];
            const hasAllowedType = allowedTypes.includes(file?.type);
            const hasAllowedExt = !file?.type && /\.(png|jpe?g)$/i.test(file?.name || '');
            if (!file || !file.size || file.size > 10 * 1024 ** 2 || (!hasAllowedType && !hasAllowedExt)) {
                throw new Error('Choose a PNG or JPEG reference under 10 MB.');
            }
        }
        const geometry = result.metadata?.geometry;
        const bounds = geometry?.observed_bounds_256;
        if (!validBounds256(bounds)) {
            throw new Error('The observed crop bounds could not be verified.');
        }

        const resolvedFile = isVideo ? await extractVideoFrameAsPngFile(file, result) : file;

        const canFetch = typeof fetch === 'function'
            && typeof FormData === 'function'
            && !(options && options.localOnly)
            && typeof window !== 'undefined'
            && window.location?.protocol !== 'file:';

        let serverRef = null;
        if (canFetch) {
            serverRef = await fetchServerReference(resolvedFile, result, geometry, bounds);
        }

        const scoreBounds = validBounds256(serverRef?.eval_bounds_256) ? serverRef.eval_bounds_256 : bounds;
        const hasPrepared = Boolean(result.metadata?.preparation_applied && result.prepared_input);
        let url = null;
        try {
            const targetPromise = serverRef?.reference_256
                ? pixels(serverRef.reference_256).then(data => ({ data, dataUrl: serverRef.reference_256 }))
                : (() => {
                    url = URL.createObjectURL(resolvedFile);
                    return referencePixels(url, result);
                })();
            const [targetResult, original, output, prepared] = await Promise.all([
                targetPromise,
                pixels(result.original_input),
                pixels(result.result),
                hasPrepared ? pixels(result.prepared_input) : Promise.resolve(null),
            ]);
            const target = targetResult.data;
            const refDataUrl = targetResult.dataUrl;
            const scores = {
                before: score(target, original, scoreBounds),
                after: score(target, output, scoreBounds),
            };
            if (prepared) {
                scores.prepared = score(target, prepared, scoreBounds);
            }
            if (typeof refDataUrl === 'string' && refDataUrl) {
                Object.defineProperty(scores, 'reference_256', {
                    value: refDataUrl,
                    enumerable: false,
                    configurable: true,
                    writable: true,
                });
            }
            return scores;
        } finally {
            if (url) URL.revokeObjectURL(url);
        }
    };
})();
