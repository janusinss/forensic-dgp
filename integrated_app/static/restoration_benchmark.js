(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const dialog = byId('benchmark-dialog');
    const field = byId('benchmark-file');
    const confirm = byId('benchmark-confirm');
    const run = byId('benchmark-run');
    const demo = byId('benchmark-demo');
    const images = byId('benchmark-images');
    const status = byId('benchmark-status');
    let reference, degraded, busy = false, selection = 0, illustrative = false;
    function update() {
        field.disabled = demo.disabled = byId('benchmark-close').disabled = busy;
        confirm.disabled = busy || !reference;
        run.disabled = busy || !degraded || !confirm.checked;
    }
    function canvas() {
        const result = document.createElement('canvas');
        result.width = result.height = 256;
        return result;
    }
    async function load(source) {
        const image = new Image();
        image.src = source;
        await image.decode();
        if (image.naturalWidth !== image.naturalHeight || image.naturalWidth < 64 || image.naturalWidth > 1600) {
            throw new Error('Choose a square face crop from 64 to 1600 pixels per side.');
        }
        const output = canvas();
        output.getContext('2d').drawImage(image, 0, 0, 256, 256);
        return output;
    }
    function pixels(image) { return image.getContext('2d').getImageData(0, 0, 256, 256).data; }
    function boxBlur(input) {
        const output = canvas();
        const context = output.getContext('2d');
        const blurred = context.createImageData(256, 256);
        const clean = pixels(input);
        for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
            const pos = (y * 256 + x) * 4;
            const sums = [0, 0, 0];
            for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
                const sample = (Math.max(0, Math.min(255, y + dy)) * 256 + Math.max(0, Math.min(255, x + dx))) * 4;
                for (let channel = 0; channel < 3; channel++) sums[channel] += clean[sample + channel];
            }
            for (let channel = 0; channel < 3; channel++) blurred.data[pos + channel] = Math.round(sums[channel] / 25);
            blurred.data[pos + 3] = 255;
        }
        context.putImageData(blurred, 0, 0);
        return output;
    }
    function scores(target, candidate) {
        const a = pixels(target), b = pixels(candidate);
        let absolute = 0, squared = 0;
        for (let i = 0; i < a.length; i += 4) for (let channel = 0; channel < 3; channel++) {
            const error = a[i + channel] - b[i + channel];
            absolute += Math.abs(error);
            squared += error ** 2;
        }
        const count = 256 * 256 * 3;
        const psnr = squared ? `${(10 * Math.log10(255 ** 2 / (squared / count))).toFixed(2)} dB` : '∞ dB (exact match)';
        return `MAE ${(absolute / count).toFixed(2)} RGB levels, PSNR ${psnr}`;
    }
    async function select(source, isDemo) {
        const version = ++selection;
        reference = degraded = undefined;
        confirm.checked = false;
        images.classList.add('hidden');
        byId('benchmark-restored').removeAttribute('src');
        byId('benchmark-restored').closest('figure').classList.add('hidden');
        update();
        if (!source) return;
        try {
            const clean = await load(source);
            if (version !== selection) return;
            if (pixels(clean).some((value, index) => index % 4 === 3 && value !== 255)) throw new Error('Choose an opaque reference image.');
            illustrative = isDemo;
            reference = clean;
            degraded = boxBlur(clean);
            byId('benchmark-clean').src = reference.toDataURL();
            byId('benchmark-degraded').src = degraded.toDataURL();
            images.classList.remove('hidden');
            status.textContent = isDemo ? 'Illustrated example for demonstrating the benchmark. Use a clean photograph for a photographic test.' : 'Check that the reference is clear and contains one face.';
        } catch (error) {
            if (version === selection) status.textContent = error.message;
        } finally { update(); }
    }
    field.addEventListener('change', async () => {
        const file = field.files[0];
        if (!file) { select(undefined, false); return; }
        if (file.size === 0 || file.size > 10 * 1024 ** 2 || !['image/png', 'image/jpeg'].includes(file.type)) {
            select(undefined, false);
            status.textContent = 'Choose a PNG or JPEG smaller than 10 MB.';
            return;
        }
        const url = URL.createObjectURL(file);
        try { await select(url, false); } finally { URL.revokeObjectURL(url); }
    });
    demo.addEventListener('click', () => { field.value = ''; select('/static/benchmark-face.svg', true); });
    confirm.addEventListener('change', update);
    dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
    run.addEventListener('click', async () => {
        if (!degraded || !confirm.checked || busy) return;
        busy = true;
        update();
        status.textContent = 'Restoring the blurred reference.';
        byId('benchmark-restored').closest('figure').classList.add('hidden');
        try {
            const blob = await new Promise(resolve => degraded.toBlob(resolve, 'image/png'));
            if (!blob) throw new Error('The blurred image could not be encoded.');
            const result = await window.runRestorationBenchmark(new File([blob], 'controlled-blur.png', { type: 'image/png' }));
            if (result.metadata.restoration_applied !== true) throw new Error('Restoration did not run for this test.');
            const restored = await load(result.result);
            byId('benchmark-restored').src = result.result;
            byId('benchmark-restored').closest('figure').classList.remove('hidden');
            const label = illustrative ? 'Illustrated test. ' : '';
            status.textContent = `${label}Against the clean reference. Blurred input: ${scores(reference, degraded)}. Restored output: ${scores(reference, restored)}.`;
        } catch (error) { status.textContent = error.message; }
        finally { busy = false; update(); }
    });
    update();
})();
