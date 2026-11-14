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

    async function referencePixels(source, geometry) {
        const image = await decodeImage(source);
        const natW = image.naturalWidth, natH = image.naturalHeight;
        if (!natW || !natH) throw new Error('Choose a valid PNG or JPEG reference image.');
        const bounds = geometry.observed_bounds_256;
        const [left, top, right, bottom] = bounds;
        const obsW = right - left, obsH = bottom - top;
        const isPadded = left > 0 || top > 0 || right < side || bottom < side;
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = side;
        const context = canvas.getContext('2d');
        if (natW === side && natH === side) {
            context.drawImage(image, 0, 0);
            const data = context.getImageData(0, 0, side, side).data;
            if (isPadded && !hasUniformGrayPadding(data, bounds)) {
                throw new Error('The 256 × 256 reference is not padded to match this crop. Upload the clean unpadded crop or a 256 × 256 padded frame.');
            }
            return data;
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
            throw new Error('Reference aspect ratio does not match the input crop. Use a 256 × 256 padded frame or clean face crop matching the input.');
        }
        context.fillStyle = 'rgb(128, 128, 128)';
        context.fillRect(0, 0, side, side);
        context.drawImage(image, left, top, obsW, obsH);
        return context.getImageData(0, 0, side, side).data;
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

    window.scoreVideoReference = async (file, result) => {
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
        const url = URL.createObjectURL(file);
        try {
            const [target, original, output] = await Promise.all([
                referencePixels(url, geometry),
                pixels(result.original_input),
                pixels(result.result),
            ]);
            return { before: score(target, original, bounds), after: score(target, output, bounds) };
        } finally {
            URL.revokeObjectURL(url);
        }
    };
})();
