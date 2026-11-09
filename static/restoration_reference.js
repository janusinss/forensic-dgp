(() => {
    'use strict';
    const side = 256;

    async function pixels(source) {
        const image = new Image();
        image.src = source;
        await image.decode();
        if (image.naturalWidth !== side || image.naturalHeight !== side) {
            throw new Error('Choose a clean, aligned 256 × 256 PNG or JPEG.');
        }
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = side;
        const context = canvas.getContext('2d');
        context.drawImage(image, 0, 0);
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
        if (!['image/png', 'image/jpeg'].includes(file.type) || !file.size || file.size > 10 * 1024 ** 2) {
            throw new Error('Choose a PNG or JPEG reference under 10 MB.');
        }
        const bounds = result.metadata.geometry?.observed_bounds_256;
        if (bounds?.length !== 4 || !bounds.every(n => Number.isInteger(n) && n >= 0 && n <= side)
            || bounds[0] >= bounds[2] || bounds[1] >= bounds[3]) {
            throw new Error('The observed crop bounds could not be verified.');
        }
        const url = URL.createObjectURL(file);
        try {
            const [target, original, output] = await Promise.all([pixels(url), pixels(result.original_input), pixels(result.result)]);
            return { before: score(target, original, bounds), after: score(target, output, bounds) };
        } finally {
            URL.revokeObjectURL(url);
        }
    };
})();
