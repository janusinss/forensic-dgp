/* Unmounted app-style V2 workflow. Existing brush geometry and exact PNG
   encoding are retained; this page never runs comparison or covering APIs. */
(() => {
    'use strict';
    const api = '/face/restoration', byId = id => document.getElementById(id);
    const canvas = byId('brush-editor'), context = canvas.getContext('2d');
    let sourceFile, originalPixels, width = 0, height = 0, planes, undo = [];
    let available = false, busy = false, revision = 0, operation = 0;
    let previewId, masks, drawing = false, previous;
    const error = message => {
        byId('brush-error').textContent = message;
        byId('brush-error').classList.toggle('hidden', !message);
    };
    const progress = message => { byId('brush-progress').textContent = message; };
    const manual = () => byId('brush-preparation').value === 'manual';
    const off = () => byId('brush-restoration').value === 'off';
    const clearResults = () => {
        byId('brush-results').classList.add('hidden');
        byId('brush-result-summary').classList.add('hidden');
        byId('brush-result-downloads').classList.add('hidden');
        byId('brush-standby').classList.remove('hidden');
        for (const id of ['brush-download-input', 'brush-download-prepared', 'brush-download-output', 'brush-result-bundle']) byId(id).removeAttribute('href');
        for (const id of ['brush-result-input', 'brush-result-prepared', 'brush-result-output']) byId(id).removeAttribute('src');
    };
    const update = () => {
        document.querySelectorAll('#brush-form input, #brush-form select, #brush-form button').forEach(item => { item.disabled = busy; });
        const readable = byId('brush-review').value === 'usable';
        const confirmed = Boolean(previewId && masks && byId('brush-confirm').checked);
        byId('brush-preview').disabled = busy || !available || !planes || !readable || !manual();
        byId('brush-run').disabled = busy || !available || !sourceFile || !planes || !readable || (manual() && !off() && !confirmed);
        byId('brush-undo').disabled = busy || !undo.length;
        byId('brush-confirm').disabled = busy || !previewId || !manual();
        byId('brush-edit-section').classList.toggle('hidden', !planes || !manual());
        canvas.setAttribute('aria-disabled', String(busy));
    };
    const invalidate = () => {
        revision++; previewId = undefined; masks = undefined;
        byId('brush-confirm').checked = false;
        byId('brush-preview-section').classList.add('hidden');
        for (const id of ['brush-prepared-download', 'brush-preview-download']) byId(id).removeAttribute('href');
        for (const id of ['brush-original', 'brush-overlay', 'brush-derived']) byId(id).removeAttribute('src');
        clearResults(); update();
    };
    const begin = () => { busy = true; const token = ++operation; update(); return token; };
    const finish = token => { if (token === operation) { busy = false; update(); } };
    const current = (token, version) => token === operation && version === revision;
    const restorationLoading = active => {
        byId('restoration-idle').classList.toggle('hidden', active);
        byId('restoration-loading').classList.toggle('hidden', !active);
        byId('restoration-loading').closest('.work-screen').classList.toggle('is-loading', active);
    };
    const validPNG = value => {
        if (typeof value !== 'string' || !value.startsWith('data:image/png;base64,')) throw new Error('The service returned an invalid image.');
        return value;
    };
    const validZIP = value => {
        if (typeof value !== 'string' || !value.startsWith('data:application/zip;base64,')) throw new Error('The download could not be prepared. Please try again.');
        return value;
    };
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
        byId('brush-cursor').textContent = `${byId('brush-plane').value}, ${byId('brush-action').value}; position ${x}, ${y}.`;
    };
    const render = () => {
        if (!planes) return;
        const pixels = new Uint8ClampedArray(originalPixels.data);
        for (let i = 0; i < width * height; i++) {
            const color = planes.protect[i] ? [45, 185, 90] : (planes.apply[i] ? [55, 125, 220] : null);
            if (color) for (let c = 0; c < 3; c++) pixels[i * 4 + c] = Math.round(pixels[i * 4 + c] * .6 + color[c] * .4);
        }
        context.putImageData(new ImageData(pixels, width, height), 0, 0);
        byId('brush-cursor').textContent = `${byId('brush-plane').value}, ${byId('brush-action').value}; position ${byId('brush-x').value}, ${byId('brush-y').value}. ${planes.protect.reduce((n, v) => n + (v > 0), 0)} protected pixels; ${planes.apply.reduce((n, v) => n + (v > 0), 0)} selected pixels. Protected areas take priority where they overlap.`;
    };
    const coordinate = (id, max) => {
        const value = Number(byId(id).value);
        if (byId(id).value.trim() === '' || !Number.isInteger(value) || value < 0 || value > max) throw new Error('Choose whole numbers within the picture.');
        return value;
    };
    const disk = (x, y) => {
        const size = Number(byId('brush-size').value);
        if (!Number.isInteger(size) || size < 1 || size > 160) throw new Error('Choose a brush size from 1 to 160 pixels.');
        const radius = (size - 1) / 2, plane = planes[byId('brush-plane').value], value = byId('brush-action').value === 'erase' ? 0 : 255;
        for (let yy = Math.max(0, Math.ceil(y - radius)); yy <= Math.min(height - 1, Math.floor(y + radius)); yy++) {
            for (let xx = Math.max(0, Math.ceil(x - radius)); xx <= Math.min(width - 1, Math.floor(x + radius)); xx++) {
                if ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) plane[yy * width + xx] = value;
            }
        }
        cursor(x, y);
    };
    const edit = callback => {
        if (busy || !planes || !manual()) return;
        try { checkpoint(); callback(); invalidate(); render(); error(''); }
        catch (problem) { error(problem.message); }
    };
    canvas.addEventListener('pointerdown', event => {
        if (busy || !planes || !manual() || event.button !== 0) return;
        event.preventDefault(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
        checkpoint(); invalidate(); drawing = true; previous = point(event);
        try { disk(...previous); render(); error(''); }
        catch (problem) { drawing = false; error(problem.message); }
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
        if (busy || !planes || !manual()) return;
        let x, y;
        try { x = coordinate('brush-x', width - 1); y = coordinate('brush-y', height - 1); }
        catch (problem) { error(problem.message); return; }
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (moves[event.key]) { event.preventDefault(); cursor(Math.max(0, Math.min(width - 1, x + moves[event.key][0])), Math.max(0, Math.min(height - 1, y + moves[event.key][1]))); }
        if (event.key === ' ') { event.preventDefault(); edit(() => disk(x, y)); }
    });
    byId('brush-stamp').addEventListener('click', () => edit(() => disk(coordinate('brush-x', width - 1), coordinate('brush-y', height - 1))));
    byId('brush-fill').addEventListener('click', () => edit(() => { for (let i = 0; i < planes.apply.length; i++) planes.apply[i] = planes.protect[i] ? 0 : 255; }));
    byId('brush-clear').addEventListener('click', () => edit(() => { planes.protect.fill(0); planes.apply.fill(0); }));
    byId('brush-reset').addEventListener('click', () => edit(() => { planes.protect.fill(0); planes.apply.fill(0); }));
    byId('brush-undo').addEventListener('click', () => { if (busy || !undo.length) return; planes = undo.pop(); invalidate(); render(); });
    byId('brush-rectangle').addEventListener('click', () => edit(() => {
        const left = coordinate('brush-left', width - 1), top = coordinate('brush-top', height - 1);
        const right = coordinate('brush-right', width - 1), bottom = coordinate('brush-bottom', height - 1);
        if (right < left || bottom < top) throw new Error('The right and bottom edges must come after the left and top edges.');
        for (let y = top; y <= bottom; y++) planes.protect.fill(255, y * width + left, y * width + right + 1);
    }));
    for (const id of ['brush-radius', 'brush-size']) byId(id).addEventListener('input', invalidate);
    for (const id of ['brush-plane', 'brush-action', 'brush-review', 'brush-restoration', 'brush-preparation']) byId(id).addEventListener('change', () => { drawing = false; invalidate(); });
    byId('brush-confirm').addEventListener('change', () => { revision++; clearResults(); update(); });
    byId('brush-include-bundle').addEventListener('change', () => { revision++; clearResults(); update(); });

    // Exact grayscale8 binary mask PNG, retained from the reviewed brush UI.
    const crcTable = Uint32Array.from({ length: 256 }, (_, n) => { for (let i = 0; i < 8; i++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0; });
    const chunk = (type, bytes) => {
        const result = new Uint8Array(bytes.length + 12), view = new DataView(result.buffer);
        view.setUint32(0, bytes.length); result.set(new TextEncoder().encode(type), 4); result.set(bytes, 8);
        let crc = 0xffffffff; for (let i = 4; i < bytes.length + 8; i++) crc = crcTable[(crc ^ result[i]) & 255] ^ (crc >>> 8);
        view.setUint32(bytes.length + 8, (crc ^ 0xffffffff) >>> 0); return result;
    };
    const grayPNG = async (values, nativeWidth, nativeHeight) => {
        if (typeof CompressionStream !== 'function') throw new Error('This browser cannot save the selected areas. Use an up-to-date Chrome or Edge browser.');
        const rows = new Uint8Array((nativeWidth + 1) * nativeHeight);
        for (let y = 0; y < nativeHeight; y++) rows.set(values.subarray(y * nativeWidth, (y + 1) * nativeWidth), y * (nativeWidth + 1) + 1);
        const compressed = new Uint8Array(await new Response(new Blob([rows]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
        const header = new Uint8Array(13), view = new DataView(header.buffer);
        view.setUint32(0, nativeWidth); view.setUint32(4, nativeHeight); header[8] = 8;
        return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', compressed), chunk('IEND', new Uint8Array())], { type: 'image/png' });
    };
    const post = async (route, data) => {
        const response = await fetch(api + route, { method: 'POST', body: data });
        let answer;
        try { answer = await response.json(); }
        catch { throw new Error('The app could not get a valid response. Select Check service before trying again.'); }
        if (!response.ok) {
            const detail = typeof answer.detail === 'string' ? answer.detail : '';
            if (/could not be confirmed stopped|do not retry/i.test(detail)) throw new Error('The previous task may still be running. Do not retry. Check the saved request and stop the task before restarting the app.');
            if (/previous worker stopped unexpectedly/i.test(detail)) throw new Error('The previous task stopped unexpectedly. Review the saved request and restart the app before trying again.');
            if (/model|\bv2\b|conditioned_sft/i.test(detail)) throw new Error('The restoration service is not ready. Check service or restart the app.');
            if (/mask/i.test(detail)) throw new Error('The app could not use the marked areas. Review your selections or choose Bypass.');
            throw new Error(detail || 'The request stopped. Check the picture and service status.');
        }
        return answer;
    };
    const form = file => { const data = new FormData(); data.append('file', file); return data; };
    const maskedForm = (file, savedMasks, radius) => {
        const data = form(file);
        data.append('protect_mask', savedMasks.protect, 'Protect.png'); data.append('apply_mask', savedMasks.apply, 'Apply.png');
        data.append('radius', radius); return data;
    };
    const resetInput = () => {
        sourceFile = undefined; originalPixels = undefined; planes = undefined; undo = []; width = 0; height = 0; drawing = false;
        canvas.width = 0; canvas.height = 0;
        byId('upload-preview').removeAttribute('src'); byId('upload-preview').classList.add('hidden');
        byId('brush-review').value = ''; byId('brush-review').dispatchEvent(new Event('change', { bubbles: true }));
    };
    byId('brush-file').addEventListener('change', async () => {
        const file = byId('brush-file').files[0];
        resetInput(); error('');
        if (busy) { error('Wait for the current task to finish, then choose the picture again.'); return; }
        if (!file) return;
        if (file.size === 0 || file.size > 10 * 1024 ** 2 || !['image/png', 'image/jpeg'].includes(file.type)) { error('Choose a PNG or JPEG face picture smaller than 10 MB.'); return; }
        const token = begin(), version = revision;
        progress('Checking your picture.');
        try {
            const result = await post('/inspect', form(file));
            if (!current(token, version)) return;
            if (!Number.isInteger(result.width) || !Number.isInteger(result.height) || result.width < 1 || result.height < 1 || result.width > 1600 || result.height > 1600 || result.width * result.height > 2560000 || result.neural_calls !== 0) throw new Error('The app could not read the picture size. Please choose another image.');
            const image = new Image(); image.src = validPNG(result.source); await image.decode();
            if (!current(token, version)) return;
            if (image.naturalWidth !== result.width || image.naturalHeight !== result.height) throw new Error('The picture could not be prepared correctly. Please choose it again.');
            sourceFile = file; width = result.width; height = result.height; canvas.width = width; canvas.height = height;
            context.drawImage(image, 0, 0); originalPixels = context.getImageData(0, 0, width, height);
            planes = { protect: new Uint8Array(width * height), apply: new Uint8Array(width * height) };
            byId('upload-preview').src = image.src; byId('upload-preview').classList.remove('hidden');
            byId('brush-dimensions').textContent = `Picture size: ${width} × ${height} pixels. No areas are marked yet.`;
            for (const id of ['brush-x', 'brush-left', 'brush-right']) byId(id).max = width - 1;
            for (const id of ['brush-y', 'brush-top', 'brush-bottom']) byId(id).max = height - 1;
            for (const id of ['brush-x', 'brush-y', 'brush-left', 'brush-top', 'brush-right', 'brush-bottom']) byId(id).value = 0;
            render(); progress('Check the face before continuing. Marking areas is optional.');
        } catch (problem) { if (current(token, version)) { resetInput(); error(problem.message); progress('The picture could not be checked. Choose a clear face picture.'); } }
        finally { finish(token); }
    });
    byId('brush-preview').addEventListener('click', async () => {
        if (busy || !planes || !sourceFile || !manual() || byId('brush-review').value !== 'usable') return;
        const radius = byId('brush-radius').value;
        if (radius.trim() === '' || !Number.isFinite(Number(radius)) || Number(radius) < 0 || Number(radius) > 12 || Number(radius) * 2 % 1 !== 0) { error('Choose a blur radius from 0 to 12 in steps of 0.5.'); return; }
        invalidate(); error(''); const token = begin(), version = revision;
        const file = sourceFile, nativeWidth = width, nativeHeight = height;
        const savedPlanes = { protect: planes.protect.slice(), apply: planes.apply.slice() };
        progress('Preparing a preview from your original picture.');
        try {
            const savedMasks = { protect: await grayPNG(savedPlanes.protect, nativeWidth, nativeHeight), apply: await grayPNG(savedPlanes.apply, nativeWidth, nativeHeight) };
            if (!current(token, version)) return;
            const data = maskedForm(file, savedMasks, radius); data.append('input_review', 'usable');
            const result = await post('/preview', data);
            if (!current(token, version)) return;
            if (!/^[a-f0-9]{64}$/.test(result.preview_id) || result.neural_calls !== 0 || result.metadata?.every_protected_native_RGB_exact !== true || result.metadata?.every_outside_Apply_native_RGB_exact !== true) throw new Error('The app could not verify that protected areas stayed unchanged. Use Bypass instead.');
            const original = validPNG(result.original), overlay = validPNG(result.protection_overlay), prepared = validPNG(result.prepared_input), bundle = validZIP(result.bundle);
            previewId = result.preview_id; masks = savedMasks;
            byId('brush-original').src = original; byId('brush-overlay').src = overlay; byId('brush-derived').src = prepared;
            byId('brush-prepared-download').href = prepared; byId('brush-preview-download').href = bundle;
            byId('brush-preservation').textContent = `Protected areas stayed unchanged during preparation. ${result.metadata.changed_native_pixels} pixels changed. ${result.metadata.no_effect ? 'These selections made no changes; you can choose Bypass.' : 'Review the prepared picture and its edges before confirming.'} Restoration can still change facial appearance.`;
            byId('brush-preview-section').classList.remove('hidden');
            progress('Preview ready. Review it and confirm your selections before continuing.');
        } catch (problem) { if (current(token, version)) { invalidate(); error(problem.message); progress('Could not prepare the picture. Review the selected areas or choose Bypass.'); } }
        finally { finish(token); }
    });
    byId('brush-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (busy || !sourceFile || !planes || !available || byId('brush-review').value !== 'usable') return;
        const restoration = byId('brush-restoration').value;
        const mode = restoration === 'off' ? 'bypass' : byId('brush-preparation').value;
        if (mode === 'manual' && (!previewId || !masks || !byId('brush-confirm').checked)) { error('First preview and confirm the areas you marked.'); return; }
        const data = mode === 'manual' ? maskedForm(sourceFile, masks, byId('brush-radius').value) : form(sourceFile);
        data.append('input_review', 'usable'); data.append('restoration', restoration); data.append('preparation_mode', mode);
        data.append('model', byId('brush-model').value);
        const includeBundle = byId('brush-include-bundle').checked;
        data.append('include_bundle', includeBundle ? 'true' : 'false');
        if (mode === 'manual') { data.append('preview_id', previewId); data.append('confirmed', 'true'); }
        error(''); clearResults(); const token = begin(), version = revision;
        restorationLoading(true);
        progress(restoration === 'off' ? 'Keeping your original picture and resizing it.' : 'Restoring your picture. Auto may keep the original.');
        try {
            const result = await post('/restore', data);
            if (!current(token, version)) return;
            const metadata = result.metadata;
            if (!metadata || ![0, 1].includes(metadata.neural_forward_returns) || metadata.neural_forward_attempts !== metadata.neural_forward_returns || metadata.selection !== 'conditioned_sft_v2_p2_u0224') throw new Error('The app could not verify the result. Check the service before trying again.');
            const original = validPNG(result.original_input), prepared = validPNG(result.prepared_input), output = validPNG(result.result);
            const bundle = includeBundle ? validZIP(result.bundle) : undefined;
            byId('brush-result-input').src = original; byId('brush-result-prepared').src = prepared; byId('brush-result-output').src = output;
            await Promise.all(['brush-result-input', 'brush-result-prepared', 'brush-result-output'].map(id => byId(id).decode()));
            if (!current(token, version)) return;
            byId('brush-download-input').href = original; byId('brush-download-prepared').href = prepared; byId('brush-download-output').href = output;
            byId('brush-download-output').download = 'restoration-result.png';
            byId('brush-result-bundle').download = 'restoration-evidence.zip';
            byId('brush-result-output').alt = 'Restored picture, or resized original when restoration is off';
            byId('brush-selected-heading').textContent = metadata.preparation_applied ? 'PREPARED PICTURE' : 'ORIGINAL PICTURE';
            byId('brush-output-heading').textContent = metadata.restoration_applied ? 'RESTORED PICTURE' : 'ORIGINAL / RESTORATION OFF';
            if (bundle) { byId('brush-result-bundle').href = bundle; byId('brush-result-downloads').classList.remove('hidden'); }
            byId('brush-results').classList.remove('hidden'); byId('brush-standby').classList.add('hidden');
            const seconds = Number.isFinite(metadata.seconds) ? `${metadata.seconds.toFixed(1)} seconds. ` : '';
            const bypass = metadata.preparation_applied ? 'Your reviewed preparation was used.' : metadata.preparation_bypass_reason === 'Auto_did_not_suggest_restoration' ? 'Auto kept the resized original.' : 'The original picture was used without preparation.';
            const summary = `${seconds}${bypass} This result is experimental. Changes can affect appearance and do not confirm identity.`;
            byId('brush-result-summary').textContent = summary; byId('brush-result-summary').classList.remove('hidden'); progress(summary);
            window.dispatchEvent(new CustomEvent('restoration:completed', { detail: { result, originalFile: sourceFile } }));
        } catch (problem) { if (current(token, version)) { restorationLoading(false); clearResults(); error(problem.message); progress('The request stopped. Read the message before trying again.'); } }
        finally { if (token === operation) restorationLoading(false); finish(token); }
    });
    const checkService = async () => {
        if (busy) return;
        const token = begin(), version = revision;
        try {
            const response = await fetch(api + '/status');
            if (!response.ok) throw new Error('The restoration service is unavailable. Start the app and try again.');
            const result = await response.json();
            if (!current(token, version)) return;
            available = result.available === true && result.maximum_restore_forwards === 1 && result.maximum_off_forwards === 0 && result.automatic_preparation === false && result.controller?.selection === 'conditioned_sft_v2_p2_u0224';
            if (!available) throw new Error('The restoration service is not ready. Restart the app or run installation again.');
            error(''); byId('brush-service-status').textContent = 'SERVICE READY'; byId('brush-status-indicator').classList.add('online');
            progress('Ready. Check your picture before continuing.');
        } catch (problem) {
            if (current(token, version)) { available = false; byId('brush-service-status').textContent = 'SERVICE UNAVAILABLE'; byId('brush-status-indicator').classList.remove('online'); error(problem.message); progress('The app is not ready yet. Select Check service after restarting it.'); }
        } finally { finish(token); }
    };
    byId('brush-refresh').addEventListener('click', checkService);
    byId('brush-model').addEventListener('change', () => { clearResults(); update(); });
    update(); checkService();
})();
