/* Unmounted app-style V2 workflow. Existing brush geometry and exact PNG
   encoding are retained; this page never runs comparison or covering APIs. */
(() => {
    'use strict';
    const api = '/face/restoration', byId = id => document.getElementById(id);
    const queue = window.createRestorationQueue();
    const queueDialog = byId('restoration-queue-dialog');
    let reviewItem;
    const canvas = byId('brush-editor'), context = canvas.getContext('2d');
    const protectColor = [45, 185, 90], applyColor = [55, 125, 220];
    let sourceFile, captureId, originalPixels, width = 0, height = 0, planes, undo = [];
    let available = false, busy = false, revision = 0, operation = 0;
    let previewId, masks, drawing = false, previous, cursorX = 0, cursorY = 0;
    let overlayPixels, dirty, renderFrame = 0, protectCount = 0, applyCount = 0;
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
        document.querySelectorAll('#brush-form input, #brush-form select, #brush-form button').forEach(item => { if (!item.closest('[data-capture-controls], .capture-visual')) item.disabled = busy; });
        const readable = byId('brush-review').value === 'usable';
        const confirmed = Boolean(previewId && masks && byId('brush-confirm').checked);
        byId('brush-preview').disabled = busy || !available || !planes || !readable || !manual();
        byId('brush-run').disabled = busy || !available || !sourceFile || !planes || !readable || (manual() && !off() && !confirmed);
        byId('brush-run').textContent = reviewItem ? 'Approve crop for queue' : 'Restore selected input';
        byId('queue-add-picture').disabled = busy || !sourceFile || Boolean(reviewItem);
        byId('brush-undo').disabled = busy || !undo.length;
        byId('brush-confirm').disabled = busy || !previewId || !manual();
        byId('brush-edit-section').classList.toggle('hidden', !planes || !manual());
        canvas.setAttribute('aria-disabled', String(busy));
        if (queueDialog.open) renderQueue();
        window.dispatchEvent(new CustomEvent('restoration:busy', {detail:{busy}}));
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
    const brushMode = () => byId('brush-action').value === 'erase' ? 'Erasing green and blue marks' : byId('brush-plane').value === 'protect' ? 'Painting green protection' : 'Painting blue blur';
    const cursorStatus = () => {
        const caption = byId('brush-cursor');
        const label = `${brushMode()} · ${cursorX}, ${cursorY}`;
        const title = `${label}. ${protectCount.toLocaleString()} protected pixels; ${applyCount.toLocaleString()} selected for blur. Green protection takes priority.`;
        if (caption.textContent !== label) caption.textContent = label;
        if (caption.title !== title) caption.title = title;
    };
    const cursor = (x, y) => {
        cursorX = x; cursorY = y;
        render();
    };
    const markDirty = (left, top, right, bottom) => {
        if (!dirty) dirty = [left, top, right, bottom];
        else { dirty[0] = Math.min(dirty[0], left); dirty[1] = Math.min(dirty[1], top); dirty[2] = Math.max(dirty[2], right); dirty[3] = Math.max(dirty[3], bottom); }
    };
    // Update masks immediately, but paint only touched pixels once per frame.
    const render = () => {
        if (!planes || !overlayPixels || renderFrame) return;
        renderFrame = requestAnimationFrame(() => {
            renderFrame = 0;
            if (!planes || !overlayPixels || !originalPixels) return;
            if (dirty) {
                const [left, top, right, bottom] = dirty;
                dirty = undefined;
                const original = originalPixels.data, pixels = overlayPixels.data;
                for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
                    const i = y * width + x, offset = i * 4;
                    const color = planes.protect[i] ? protectColor : (planes.apply[i] ? applyColor : null);
                    for (let channel = 0; channel < 3; channel++) pixels[offset + channel] = color ? Math.round(original[offset + channel] * .6 + color[channel] * .4) : original[offset + channel];
                }
                context.putImageData(overlayPixels, 0, 0, left, top, right - left + 1, bottom - top + 1);
            }
            cursorStatus();
        });
    };
    const repaintAll = () => {
        protectCount = 0; applyCount = 0;
        for (let i = 0; i < planes.protect.length; i++) {
            if (planes.protect[i]) protectCount++;
            if (planes.apply[i]) applyCount++;
        }
        markDirty(0, 0, width - 1, height - 1);
        render();
    };
    const brushSettings = () => {
        const size = Number(byId('brush-size').value);
        if (!Number.isInteger(size) || size < 1 || size > 100) throw new Error('Choose a brush size from 1 to 100 pixels.');
        return { radius: (size - 1) / 2, erase: byId('brush-action').value === 'erase', planeName: byId('brush-plane').value };
    };
    const disk = (x, y, { radius, erase, planeName } = brushSettings()) => {
        const plane = planes[planeName];
        const left = Math.max(0, Math.ceil(x - radius)), right = Math.min(width - 1, Math.floor(x + radius));
        const top = Math.max(0, Math.ceil(y - radius)), bottom = Math.min(height - 1, Math.floor(y + radius));
        let changed = false;
        for (let yy = top; yy <= bottom; yy++) {
            for (let xx = left; xx <= right; xx++) {
                if ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) {
                    const index = yy * width + xx;
                    if (erase) {
                        if (planes.protect[index]) { protectCount--; changed = true; }
                        if (planes.apply[index]) { applyCount--; changed = true; }
                        planes.protect[index] = 0; planes.apply[index] = 0;
                    } else if (!plane[index]) {
                        plane[index] = 255; changed = true;
                        if (planeName === 'protect') protectCount++; else applyCount++;
                    }
                }
            }
        }
        if (changed) markDirty(left, top, right, bottom);
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
            const brush = brushSettings();
            for (let i = 1; i <= count; i++) disk(Math.round(previous[0] + (next[0] - previous[0]) * i / count), Math.round(previous[1] + (next[1] - previous[1]) * i / count), brush);
            previous = next; render();
        } catch (problem) { drawing = false; error(problem.message); }
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drawing = false; });
    canvas.addEventListener('keydown', event => {
        if (busy || !planes || !manual()) return;
        const x = cursorX, y = cursorY;
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (moves[event.key]) { event.preventDefault(); cursor(Math.max(0, Math.min(width - 1, x + moves[event.key][0])), Math.max(0, Math.min(height - 1, y + moves[event.key][1]))); }
        if (event.key === ' ') { event.preventDefault(); edit(() => disk(x, y)); }
    });
    byId('brush-clear').addEventListener('click', () => edit(() => { planes.protect.fill(0); planes.apply.fill(0); protectCount = 0; applyCount = 0; markDirty(0, 0, width - 1, height - 1); }));
    byId('brush-undo').addEventListener('click', () => { if (busy || !undo.length) return; planes = undo.pop(); invalidate(); repaintAll(); });
    byId('brush-size').addEventListener('input', event => { byId('brush-size-value').textContent = `${event.target.value} px`; });
    byId('brush-radius').addEventListener('input', event => { byId('brush-radius-value').textContent = `${event.target.value} px`; });
    const updateBrushMode = () => {
        byId('brush-mode-summary').textContent = brushMode();
        if (planes) render();
    };
    for (const id of ['brush-plane', 'brush-action']) byId(id).addEventListener('change', updateBrushMode);
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
        if (renderFrame) cancelAnimationFrame(renderFrame);
        renderFrame = 0; overlayPixels = undefined; dirty = undefined; protectCount = 0; applyCount = 0;
        sourceFile = undefined; captureId = undefined; originalPixels = undefined; planes = undefined; undo = []; width = 0; height = 0; drawing = false;
        canvas.width = 0; canvas.height = 0;
        byId('input-picture').classList.add('hidden');
        byId('upload-preview').removeAttribute('src'); byId('upload-preview').classList.add('hidden');
        byId('brush-review').value = ''; byId('brush-review').dispatchEvent(new Event('change', { bubbles: true }));
    };
    const usePicture = async (file, options = {}) => {
        if (busy) { error('Wait for the current task to finish, then choose the picture again.'); return false; }
        if (!options.queueReview) reviewItem = undefined;
        resetInput(); error('');
        if (!file) return;
        if (file.size === 0 || file.size > 10 * 1024 ** 2 || !['image/png', 'image/jpeg'].includes(file.type)) { error('Choose a PNG or JPEG face picture smaller than 10 MB.'); return; }
        const token = begin(), version = revision;
        progress('Checking your picture.');
        try {
            const result = await post('/inspect', form(file));
            if (!current(token, version) || (options.isCurrent && !options.isCurrent())) return false;
            if (!Number.isInteger(result.width) || !Number.isInteger(result.height) || result.width < 1 || result.height < 1 || result.width > 1600 || result.height > 1600 || result.width * result.height > 2560000 || result.neural_calls !== 0) throw new Error('The app could not read the picture size. Please choose another image.');
            const image = new Image(); image.src = validPNG(result.source); await image.decode();
            if (!current(token, version) || (options.isCurrent && !options.isCurrent())) return false;
            if (image.naturalWidth !== result.width || image.naturalHeight !== result.height) throw new Error('The picture could not be prepared correctly. Please choose it again.');
            sourceFile = file; captureId = options.captureId; width = result.width; height = result.height; canvas.width = width; canvas.height = height;
            context.drawImage(image, 0, 0); originalPixels = context.getImageData(0, 0, width, height);
            overlayPixels = new ImageData(new Uint8ClampedArray(originalPixels.data), width, height);
            planes = { protect: new Uint8Array(width * height), apply: new Uint8Array(width * height) };
            byId('input-picture').classList.remove('hidden');
            byId('input-picture-label').textContent = options.label || 'Uploaded picture';
            byId('upload-preview').alt = options.label || 'Original uploaded picture';
            byId('upload-preview').src = image.src; byId('upload-preview').classList.remove('hidden');
            byId('brush-dimensions').textContent = `Picture size: ${width} × ${height} pixels. No areas are marked yet.`;
            cursorX = 0; cursorY = 0;
            render(); progress('Check the face before continuing. Marking areas is optional.');
            window.dispatchEvent(new CustomEvent('picture:changed', {detail:{captured:!!captureId,file}}));
            return true;
        } catch (problem) { if (current(token, version)) { resetInput(); error(problem.message); progress('The picture could not be checked. Choose a clear face picture.'); } }
        finally { finish(token); }
    };
    window.restorationInput = {use:usePicture, get busy() {return busy;}};
    byId('brush-file').addEventListener('change', () => { reviewItem = undefined; usePicture(byId('brush-file').files[0]); });
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
            let explanation;
            if (result.metadata.no_effect) {
                if (Number(radius) === 0) explanation = 'Blur strength is 0, so the picture did not change. Increase the strength to add blur.';
                else if (!savedPlanes.protect.some(Boolean) && !savedPlanes.apply.some(Boolean)) explanation = 'No area was selected. Paint blue where you want blur, or paint green to protect an area and blur its surroundings.';
                else if (result.metadata.effective_Apply_native_pixels === 0) explanation = 'No area is available to blur. Green protection or its border covers the selection. Erase some green or paint blue outside it.';
                else explanation = 'The picture did not change. Try a larger blue area or a higher blur strength.';
            } else {
                explanation = `Blur strength: ${radius} px. ${result.metadata.changed_native_pixels.toLocaleString()} pixels changed. Review the prepared picture and its edges.`;
            }
            byId('brush-preservation').textContent = `${explanation} Protected areas stay unchanged during preparation; restoration can still change facial appearance.`;
            byId('brush-preview-section').classList.remove('hidden');
            progress('Preview ready. Review it and confirm your selections before continuing.');
        } catch (problem) { if (current(token, version)) { invalidate(); error(problem.message); progress('Could not prepare the picture. Review the selected areas or choose Bypass.'); } }
        finally { finish(token); }
    });
    async function restoreRequest(data) {
        const result = await post('/restore', data);
        const metadata = result.metadata;
        if (!metadata || ![0, 1].includes(metadata.neural_forward_returns) || metadata.neural_forward_attempts !== metadata.neural_forward_returns || metadata.selection !== 'conditioned_sft_v2_p2_u0224') throw new Error('The app could not verify the result. Check the service before trying again.');
        await Promise.all([result.original_input, result.prepared_input, result.result].map(async url => {
            const image = new Image(); image.src = validPNG(url); await image.decode();
            if (image.naturalWidth !== 256 || image.naturalHeight !== 256) throw new Error('The service returned an invalid result size.');
        }));
        if (data.get('include_bundle') === 'true') validZIP(result.bundle);
        return result;
    }
    function displayResult(result, originalFile, fromQueue = false) {
        const metadata = result.metadata;
        byId('brush-result-input').src = result.original_input;
        byId('brush-result-prepared').src = result.prepared_input;
        byId('brush-result-output').src = result.result;
        byId('brush-download-input').href = result.original_input;
        byId('brush-download-prepared').href = result.prepared_input;
        byId('brush-download-output').href = result.result;
        byId('brush-selected-heading').textContent = metadata.preparation_applied ? 'PREPARED PICTURE' : 'ORIGINAL PICTURE';
        byId('brush-output-heading').textContent = metadata.restoration_applied ? 'RESTORED PICTURE' : 'ORIGINAL / RESTORATION OFF';
        byId('brush-result-downloads').classList.toggle('hidden', !result.bundle);
        if (result.bundle) byId('brush-result-bundle').href = result.bundle;
        byId('brush-results').classList.remove('hidden');
        byId('brush-standby').classList.add('hidden');
        const seconds = Number.isFinite(metadata.seconds) ? `${metadata.seconds.toFixed(1)} seconds. ` : '';
        const summary = `${seconds}${metadata.preparation_applied ? 'Your reviewed preparation was used.' : 'No preparation was applied.'}`;
        byId('brush-result-summary').textContent = summary;
        byId('brush-result-summary').classList.remove('hidden');
        progress('Result ready to review.');
        byId('restoration-results-heading').closest('.work-controls').querySelector('[data-nav="options"]').classList.toggle('hidden', fromQueue);
        const diagnostics = metadata.output_diagnostics;
        const value = number => Number.isFinite(number) ? number.toFixed(2) : 'Unavailable';
        byId('brush-diagnostics').textContent = diagnostics ? `Original → result. Edge variation: ${value(diagnostics.before.blur_variance)} → ${value(diagnostics.after.blur_variance)}. Noise estimate: ${value(diagnostics.before.noise_sigma_255)} → ${value(diagnostics.after.noise_sigma_255)}.` : '';
        window.dispatchEvent(new CustomEvent('restoration:completed', { detail: { result, originalFile } }));
    }
    byId('brush-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (busy || !sourceFile || !planes || !available || byId('brush-review').value !== 'usable') return;
        const restoration = byId('brush-restoration').value;
        const mode = restoration === 'off' ? 'bypass' : byId('brush-preparation').value;
        if (mode === 'manual' && (!previewId || !masks || !byId('brush-confirm').checked)) { error('First preview and confirm the areas you marked.'); return; }
        const data = mode === 'manual' ? maskedForm(sourceFile, masks, byId('brush-radius').value) : form(sourceFile);
        data.append('input_review', 'usable'); data.append('restoration', restoration); data.append('preparation_mode', mode);
        data.append('model', byId('brush-model').value);
        if (captureId) data.append('capture_id', captureId);
        const includeBundle = byId('brush-include-bundle').checked;
        data.append('include_bundle', includeBundle ? 'true' : 'false');
        if (mode === 'manual') { data.append('preview_id', previewId); data.append('confirmed', 'true'); }
        if (reviewItem) {
            queue.approve(reviewItem, data);
            reviewItem = undefined;
            resetInput();
            window.dispatchEvent(new Event('queue:review'));
            openQueue();
            return;
        }
        error(''); clearResults(); const token = begin(), version = revision;
        restorationLoading(true);
        progress(restoration === 'off' ? 'Keeping your original picture and resizing it.' : 'Restoring your picture. Auto may keep the original.');
        try {
            const result = await restoreRequest(data);
            if (!current(token, version)) return;
            displayResult(result, sourceFile);
        } catch (problem) { if (current(token, version)) { restorationLoading(false); clearResults(); error(problem.message); progress('The request stopped. Read the message before trying again.'); } }
        finally { if (token === operation) restorationLoading(false); finish(token); }
    });
    function queueButton(label, action) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.disabled = busy || queue.running;
        button.addEventListener('click', action);
        return button;
    }
    function renderQueue() {
        const items = queue.items;
        byId('queue-items').replaceChildren(...items.map(item => {
            const row = document.createElement('li');
            row.textContent = `${item.id}. ${item.label} · ${item.status} `;
            if (item.status === 'completed') {
                row.append(queueButton('View result', () => {
                    queueDialog.close();
                    displayResult(item.result, item.file, true);
                }));
            } else {
                row.append(queueButton('Review', async () => {
                    queue.review(item.id);
                    reviewItem = item.id;
                    queueDialog.close();
                    window.dispatchEvent(new Event('queue:review'));
                    if (!await usePicture(item.file, { captureId: item.captureId, label: item.label, queueReview: true })) reviewItem = undefined;
                    update();
                }), queueButton('Skip', () => { queue.skip(item.id); clearQueueReview(item.id); renderQueue(); }));
            }
            row.append(queueButton('Remove', () => {
                queue.remove(item.id);
                clearQueueReview(item.id);
                renderQueue();
                window.dispatchEvent(new Event('queue:changed'));
            }));
            return row;
        }));
        byId('queue-empty').classList.toggle('hidden', Boolean(items.length));
        byId('queue-run').disabled = busy || queue.running || !available || !items.some(item => item.status === 'approved') || items.some(item => ['pending', 'failed'].includes(item.status));
        byId('queue-close').disabled = queue.running;
    }
    function clearQueueReview(id) {
        if (reviewItem !== id) return;
        reviewItem = undefined;
        resetInput();
        window.dispatchEvent(new Event('queue:review'));
    }
    function openQueue() {
        renderQueue();
        if (!queueDialog.open) queueDialog.showModal();
    }
    window.restorationQueue = {
        get full() { return queue.full; },
        get remaining() { return queue.remaining; },
        get captureIds() { return new Set(queue.items.map(item => item.captureId)); },
        add(file, captureId) {
            const id = queue.add({ file, captureId, label: captureId ? 'Captured face' : file.name });
            byId('queue-status').textContent = `Crop ${id} added.`;
            window.dispatchEvent(new Event('queue:changed'));
            return id;
        },
        addDetected(crops) {
            const ids = queue.addMany(crops.map(({ file, captureId, faceNumber }) => ({ file, captureId, label: `Detected face ${faceNumber}` })));
            byId('queue-status').textContent = `${ids.length} ${ids.length === 1 ? 'face added' : 'faces added'}. Review or skip each crop.`;
            window.dispatchEvent(new Event('queue:changed'));
            openQueue();
        }
    };
    window.runRestorationBenchmark = async file => {
        if (busy || !available) throw new Error('The restoration service is busy or unavailable.');
        const data = form(file);
        data.append('input_review', 'usable');
        data.append('restoration', 'on');
        data.append('preparation_mode', 'bypass');
        data.append('model', byId('brush-model').value);
        data.append('include_bundle', 'false');
        const token = begin();
        try { return await restoreRequest(data); }
        finally { finish(token); }
    };
    document.querySelectorAll('[data-open-queue]').forEach(button => button.addEventListener('click', openQueue));
    byId('queue-add-picture').addEventListener('click', () => {
        try { window.restorationQueue.add(sourceFile, captureId); openQueue(); }
        catch (problem) { error(problem.message); }
    });
    queueDialog.addEventListener('cancel', event => { if (queue.running) event.preventDefault(); });
    byId('queue-run').addEventListener('click', async () => {
        if (busy || queue.running) return;
        const token = begin();
        try {
            await queue.run(async (request, item) => {
                byId('queue-status').textContent = `Processing crop ${item.id}.`;
                renderQueue();
                return restoreRequest(request);
            });
            byId('queue-status').textContent = 'Approved crops processed. Choose a result to review or download.';
        } catch (problem) {
            byId('queue-status').textContent = `Queue paused. ${problem.message}`;
        } finally { finish(token); renderQueue(); }
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
