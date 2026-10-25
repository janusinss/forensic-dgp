(() => {
    'use strict';
    const api = '/restoration-brush-test', byId = id => document.getElementById(id);
    const canvas = byId('brush-editor'), context = canvas.getContext('2d');
    let sourceFile, sourceImage, originalPixels, width = 0, height = 0, planes, undo = [];
    let busy = false, available = false, revision = 0, previewId, masks, drawing = false, previous;
    const error = message => { byId('brush-error').textContent = message; byId('brush-error').classList.toggle('hidden', !message); };
    const clearResults = () => {
        byId('brush-results').classList.add('hidden');
        for (const id of ['brush-download-original', 'brush-download-derived', 'brush-result-bundle']) byId(id).removeAttribute('href');
        for (const id of ['brush-result-input', 'brush-result-prepared', 'brush-result-original', 'brush-result-derived']) byId(id).removeAttribute('src');
    };
    const update = () => {
        document.querySelectorAll('#brush-form input, #brush-form select, #brush-form button').forEach(item => { item.disabled = busy; });
        byId('brush-preview').disabled = busy || !sourceFile;
        byId('brush-run').disabled = busy || !available || !previewId || !byId('brush-confirm').checked || byId('brush-review').value !== 'usable';
        byId('brush-bypass').disabled = busy || !available || !sourceFile || byId('brush-review').value !== 'usable';
        byId('brush-undo').disabled = busy || !undo.length;
        byId('brush-confirm').disabled = busy || !previewId;
        canvas.setAttribute('aria-disabled', String(busy));
    };
    const invalidate = () => {
        revision++; previewId = undefined; masks = undefined;
        byId('brush-confirm').checked = false; byId('brush-preview-section').classList.add('hidden');
        for (const id of ['brush-prepared-download', 'brush-preview-download']) byId(id).removeAttribute('href');
        clearResults(); update();
    };
    const setBusy = value => { busy = value; update(); };
    const checkpoint = () => {
        undo.push({ protect: planes.protect.slice(), apply: planes.apply.slice() });
        if (undo.length > 10) undo.shift();
    };
    const point = event => {
        const rect = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(width - 1, Math.floor((event.clientX - rect.left) * width / rect.width))),
            Math.max(0, Math.min(height - 1, Math.floor((event.clientY - rect.top) * height / rect.height)))];
    };
    const cursor = (x, y) => {
        byId('brush-x').value = x; byId('brush-y').value = y;
        byId('brush-cursor').textContent = `${byId('brush-plane').value}, ${byId('brush-action').value}; native cursor ${x}, ${y}.`;
    };
    const render = () => {
        if (!planes) return;
        const pixels = new Uint8ClampedArray(originalPixels.data);
        for (let i = 0; i < width * height; i++) {
            const color = planes.protect[i] ? [45, 185, 90] : (planes.apply[i] ? [55, 125, 220] : null);
            if (color) for (let c = 0; c < 3; c++) pixels[i * 4 + c] = Math.round(pixels[i * 4 + c] * .6 + color[c] * .4);
        }
        context.putImageData(new ImageData(pixels, width, height), 0, 0);
        byId('brush-cursor').textContent = `${byId('brush-plane').value}, ${byId('brush-action').value}; native cursor ${byId('brush-x').value}, ${byId('brush-y').value}. ${planes.protect.reduce((n, v) => n + (v > 0), 0)} Protect pixels; ${planes.apply.reduce((n, v) => n + (v > 0), 0)} Apply pixels. Protect wins overlap.`;
    };
    const disk = (x, y) => {
        const size = Number(byId('brush-size').value);
        if (!Number.isInteger(size) || size < 1 || size > 160) throw new Error('Use a brush diameter from 1 to 160 native pixels.');
        const radius = (size - 1) / 2, plane = planes[byId('brush-plane').value], value = byId('brush-action').value === 'erase' ? 0 : 255;
        for (let yy = Math.max(0, Math.ceil(y - radius)); yy <= Math.min(height - 1, Math.floor(y + radius)); yy++) {
            for (let xx = Math.max(0, Math.ceil(x - radius)); xx <= Math.min(width - 1, Math.floor(x + radius)); xx++) {
                if ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) plane[yy * width + xx] = value;
            }
        }
        cursor(x, y);
    };
    const edit = callback => {
        if (busy || !planes) return;
        try { checkpoint(); callback(); invalidate(); render(); } catch (problem) { error(problem.message); }
    };
    canvas.addEventListener('pointerdown', event => {
        if (busy || !planes || event.button !== 0) return;
        event.preventDefault(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
        checkpoint(); invalidate(); drawing = true; previous = point(event);
        try { disk(...previous); render(); } catch (problem) { drawing = false; error(problem.message); }
    });
    canvas.addEventListener('pointermove', event => {
        if (!drawing || busy) return;
        const next = point(event), count = Math.max(Math.abs(next[0] - previous[0]), Math.abs(next[1] - previous[1]), 1);
        try {
            for (let i = 1; i <= count; i++) disk(Math.round(previous[0] + (next[0] - previous[0]) * i / count), Math.round(previous[1] + (next[1] - previous[1]) * i / count));
            previous = next; render();
        } catch (problem) { drawing = false; error(problem.message); }
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drawing = false; });
    canvas.addEventListener('keydown', event => {
        if (busy || !planes) return;
        let x, y;
        try { x = coordinate('brush-x', width - 1); y = coordinate('brush-y', height - 1); }
        catch (problem) { error(problem.message); return; }
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (moves[event.key]) { event.preventDefault(); cursor(Math.max(0, Math.min(width - 1, x + moves[event.key][0])), Math.max(0, Math.min(height - 1, y + moves[event.key][1]))); }
        if (event.key === ' ') { event.preventDefault(); edit(() => disk(x, y)); }
    });
    const coordinate = (id, max) => {
        const value = Number(byId(id).value);
        if (!Number.isInteger(value) || value < 0 || value > max) throw new Error('Choose whole-pixel coordinates inside the uploaded crop.');
        return value;
    };
    byId('brush-stamp').addEventListener('click', () => edit(() => disk(coordinate('brush-x', width - 1), coordinate('brush-y', height - 1))));
    byId('brush-fill').addEventListener('click', () => edit(() => { for (let i = 0; i < planes.apply.length; i++) planes.apply[i] = planes.protect[i] ? 0 : 255; }));
    byId('brush-clear').addEventListener('click', () => edit(() => planes[byId('brush-plane').value].fill(0)));
    byId('brush-reset').addEventListener('click', () => edit(() => { planes.protect.fill(0); planes.apply.fill(0); }));
    byId('brush-undo').addEventListener('click', () => { if (busy || !undo.length) return; planes = undo.pop(); invalidate(); render(); });
    byId('brush-rectangle').addEventListener('click', () => edit(() => {
        const left = coordinate('brush-left', width - 1), top = coordinate('brush-top', height - 1);
        const right = coordinate('brush-right', width - 1), bottom = coordinate('brush-bottom', height - 1);
        if (right < left || bottom < top) throw new Error('Right/bottom must follow left/top.');
        for (let y = top; y <= bottom; y++) planes.protect.fill(255, y * width + left, y * width + right + 1);
    }));
    byId('brush-radius').addEventListener('input', invalidate);
    byId('brush-confirm').addEventListener('change', update);
    byId('brush-review').addEventListener('change', () => { clearResults(); update(); });
    byId('brush-model').addEventListener('change', () => { clearResults(); update(); });

    // PNG color type0 / depth8: preserve native binary mask pixels, no Canvas RGBA conversion.
    // W3C PNG specification and WHATWG Compression Streams; decode is independently checked server-side.
    const crcTable = Uint32Array.from({ length: 256 }, (_, n) => { for (let i = 0; i < 8; i++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0; });
    const chunk = (type, bytes) => {
        const result = new Uint8Array(bytes.length + 12), view = new DataView(result.buffer);
        view.setUint32(0, bytes.length); result.set(new TextEncoder().encode(type), 4); result.set(bytes, 8);
        let crc = 0xffffffff; for (let i = 4; i < bytes.length + 8; i++) crc = crcTable[(crc ^ result[i]) & 255] ^ (crc >>> 8);
        view.setUint32(bytes.length + 8, (crc ^ 0xffffffff) >>> 0); return result;
    };
    const grayPNG = async values => {
        if (typeof CompressionStream !== 'function') throw new Error('This browser cannot export exact masks. Use the current Chrome or Edge browser.');
        const rows = new Uint8Array((width + 1) * height);
        for (let y = 0; y < height; y++) rows.set(values.subarray(y * width, (y + 1) * width), y * (width + 1) + 1);
        const compressed = new Uint8Array(await new Response(new Blob([rows]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
        const header = new Uint8Array(13), view = new DataView(header.buffer);
        view.setUint32(0, width); view.setUint32(4, height); header[8] = 8;
        return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', compressed), chunk('IEND', new Uint8Array())], { type: 'image/png' });
    };
    const post = async (route, form) => {
        const response = await fetch(api + route, { method: 'POST', body: form });
        const answer = await response.json();
        if (!response.ok) throw new Error(typeof answer.detail === 'string' ? answer.detail : 'The request could not be completed.');
        return answer;
    };
    const form = () => { const data = new FormData(); data.append('file', sourceFile); return data; };
    const maskedForm = () => {
        const data = form(); data.append('protect_mask', masks.protect, 'Protect.png'); data.append('apply_mask', masks.apply, 'Apply.png');
        data.append('radius', byId('brush-radius').value); return data;
    };
    byId('brush-file').addEventListener('change', async () => {
        sourceFile = byId('brush-file').files[0]; sourceImage = undefined; planes = undefined; undo = [];
        invalidate(); error(''); byId('brush-edit-section').classList.add('hidden'); if (!sourceFile) return;
        setBusy(true); byId('brush-review').value = ''; const current = revision;
        try {
            const result = await post('/inspect', form()); if (current !== revision) return;
            sourceImage = new Image(); sourceImage.src = result.source; await sourceImage.decode();
            width = result.width; height = result.height; canvas.width = width; canvas.height = height;
            context.drawImage(sourceImage, 0, 0); originalPixels = context.getImageData(0, 0, width, height);
            planes = { protect: new Uint8Array(width * height), apply: new Uint8Array(width * height) };
            byId('brush-dimensions').textContent = `${width} × ${height} oriented native pixels. Both areas start empty.`;
            for (const id of ['brush-x', 'brush-left', 'brush-right']) byId(id).max = width - 1;
            for (const id of ['brush-y', 'brush-top', 'brush-bottom']) byId(id).max = height - 1;
            for (const id of ['brush-x', 'brush-y', 'brush-left', 'brush-top', 'brush-right', 'brush-bottom']) byId(id).value = 0;
            byId('brush-edit-section').classList.remove('hidden'); render();
            byId('brush-progress').textContent = 'Review the face. Protect it before permitting surroundings blur, or bypass preparation.';
        } catch (problem) { sourceFile = undefined; error(problem.message); }
        finally { setBusy(false); }
    });
    byId('brush-preview').addEventListener('click', async () => {
        if (busy || !planes) return;
        invalidate(); error(''); setBusy(true); const current = revision;
        try {
            masks = { protect: await grayPNG(planes.protect), apply: await grayPNG(planes.apply) };
            const result = await post('/preview', maskedForm()); if (current !== revision) return;
            previewId = result.preview_id;
            byId('brush-original').src = result.original; byId('brush-overlay').src = result.protection_overlay; byId('brush-derived').src = result.prepared_input;
            byId('brush-prepared-download').href = result.prepared_input; byId('brush-preview-download').href = result.bundle;
            byId('brush-preservation').textContent = `Every protected native pixel and every pixel outside Apply stayed exact. ${result.metadata.changed_native_pixels} native pixels changed. ${result.metadata.no_effect ? 'This preparation has no effect; bypass is available.' : 'Review the visible transition and prepared input before confirming.'}`;
            byId('brush-preview-section').classList.remove('hidden');
            byId('brush-progress').textContent = 'Preview rebuilt from the original. No model ran.';
        } catch (problem) { previewId = undefined; masks = undefined; error(problem.message); }
        finally { setBusy(false); }
    });
    const restore = async bypass => {
        if (busy || !sourceFile) return;
        error(''); clearResults(); setBusy(true);
        byId('brush-progress').textContent = bypass ? 'Restoring the original with one frozen model call.' : 'Comparing original and reviewed preparation with two frozen model calls.';
        try {
            const data = bypass ? form() : maskedForm();
            data.append('input_review', byId('brush-review').value); data.append('selection', byId('brush-model').value);
            if (!bypass) { data.append('preview_id', previewId); data.append('confirmed', byId('brush-confirm').checked ? 'true' : 'false'); }
            const result = await post(bypass ? '/bypass' : '/compare', data);
            byId('brush-prepared-input-heading').textContent = bypass ? 'Original bypass / resize control' : 'Reviewed preparation / resize control';
            byId('brush-prepared-result-heading').textContent = bypass ? 'Original bypass / same result' : 'Restoration from reviewed input';
            byId('brush-result-input').src = result.original_input; byId('brush-result-prepared').src = result.prepared_input;
            byId('brush-result-original').src = result.original_result; byId('brush-result-derived').src = result.prepared_result;
            byId('brush-download-original').href = result.original_result; byId('brush-download-derived').href = result.prepared_result;
            byId('brush-result-bundle').href = result.bundle; byId('brush-results').classList.remove('hidden');
            byId('brush-result-bundle').textContent = bypass ? 'Download original + result + RAW records' : 'Download original + masks + results + RAW records';
            byId('brush-progress').textContent = `${bypass ? 'Original bypass' : 'Manual assisted comparison'} complete in ${result.metadata.seconds.toFixed(1)} seconds. ${result.metadata.neural_forward_returns} model calls. Experimental results.`;
        } catch (problem) { error(problem.message); byId('brush-progress').textContent = 'The request stopped. Review the message before retrying.'; }
        finally { setBusy(false); }
    };
    byId('brush-form').addEventListener('submit', event => { event.preventDefault(); restore(false); });
    byId('brush-bypass').addEventListener('click', () => restore(true));
    fetch(api + '/status').then(async response => {
        if (!response.ok) throw new Error('The reviewed local model binding is unavailable.');
        const result = await response.json(); available = result.available;
        byId('brush-progress').textContent = 'Ready. Preparation is optional and manual; U96 and u64 remain frozen.'; update();
    }).catch(problem => { error(problem.message); update(); });
    update();
})();
