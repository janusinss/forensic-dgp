/* Optional R1 stage. V2 restoration remains owned by implementation.js. */
(() => {
    'use strict';
    const byId = id => document.getElementById(id), api = '/face/completion';
    const canvas = byId('completion-editor'), context = canvas.getContext('2d');
    let source, session, imagePixels, bounds, mask = new Uint8Array(256 * 256), undo = [];
    let proposalId, reviewId, busy = false, revision = 0, operation = 0, drawing = false, previous;
    const error = message => { byId('completion-error').textContent = message; byId('completion-error').classList.toggle('hidden', !message); };
    const progress = message => { byId('completion-progress').textContent = message; };
    const pngURL = value => {
        if (typeof value !== 'string' || !value.startsWith('data:image/png;base64,')) throw new Error('The image could not be read. Please try again.');
        return value;
    };
    const zipURL = value => {
        if (typeof value !== 'string' || !value.startsWith('data:application/zip;base64,')) throw new Error('The download could not be created. Please try again.');
        return value;
    };
    const pngBlob = value => {
        const encoded = pngURL(value).split(',')[1], binary = atob(encoded);
        return new Blob([Uint8Array.from(binary, character => character.charCodeAt(0))], { type: 'image/png' });
    };
    const friendlyServiceError = (detail, route) => {
        if (typeof detail !== 'string') return 'The request stopped. Check the image and selected area, then try again.';
        if (/could not be confirmed stopped|do not retry/i.test(detail)) return 'The previous task may still be running. Do not retry. Check the saved request and stop the task before restarting the app.';
        if (/worker needs review|previous restoration worker error/i.test(detail)) return 'The previous task needs review. Check the saved request and restart the app before trying again.';
        const messages = [
            [/Use an exact RGB256 image or opaque black-and-white L256 mask|Removal masks must contain only black and white|mask must be opaque and contain only black and white pixels/i, 'The selected-area image must contain only solid black and white pixels.'],
            [/Use a PNG smaller than 1 MiB/i, 'Choose a PNG smaller than 1 MiB.'],
            [/Cannot read this PNG/i, 'The PNG could not be read. Choose another file.'],
            [/Removal must not include neutral padding/i, 'Keep the selected area inside the photo.'],
            [/Too little visible face remains/i, 'Too much of the face is hidden. Choose a clearer photo with more of the face visible.'],
            [/proposal is stale/i, 'The suggested area has changed. Check and confirm the area again.'],
            [/Confirm the removal area and sufficient visible facial structure/i, 'Confirm the selected area and that enough of the face is visible.'],
            [/not produced in the current app session/i, 'This result is from an earlier session. Restore the photo again or choose Off.'],
            [/Completion review is full/i, 'Too many images are open for review. Restart the app to continue.'],
            [/at least 2\.5 GiB available RAM/i, 'There is not enough memory. Close unused apps and try again.'],
            [/exceeded 120 seconds/i, 'The request took too long and stopped. Your saved files are still available.'],
            [/6 GiB owned-worker memory limit/i, 'The request used too much memory and stopped. Close unused apps and try again.'],
        ];
        for (const [pattern, message] of messages) if (pattern.test(detail)) return message;
        if (/\b(?:R1|V2|model|detector|worker|provenance|fingerprint|sha256|metadata|artifact|mask|binary|padding|bounds|receipt|session|review_id|proposal_id|neural)\b/i.test(detail)) {
            return route === '/propose' ? 'Could not suggest an area automatically. You can mark it yourself.' : 'The request could not be completed. Please review your image and try again.';
        }
        return detail;
    };
    const post = async (route, data) => {
        const response = await fetch(api + route, { method: 'POST', body: data });
        let answer;
        try { answer = await response.json(); }
        catch { throw new Error('The request returned an invalid response. Your saved files are still available. Please try again.'); }
        if (!response.ok) throw new Error(friendlyServiceError(answer.detail, route));
        return answer;
    };
    const clearResult = () => {
        byId('completion-result-section').classList.add('hidden');
        for (const id of ['completion-original', 'completion-input', 'completion-result-mask', 'completion-result']) byId(id).removeAttribute('src');
        for (const id of ['completion-original-download', 'completion-input-download', 'completion-mask-download', 'completion-result-download', 'completion-bundle']) byId(id).removeAttribute('href');
        byId('completion-summary').textContent = '';
    };
    const update = () => {
        const ready = Boolean(session && imagePixels);
        document.querySelectorAll('#completion-stage input, #completion-stage select, #completion-stage button:not([data-wizard-nav])').forEach(item => { item.disabled = busy || !ready; });
        byId('completion-start').disabled = busy || !source;
        byId('completion-undo').disabled = busy || !ready || !undo.length;
        byId('completion-generate').disabled = busy || !ready || !reviewId || !byId('completion-mask-confirm').checked || !byId('completion-visible-confirm').checked;
        canvas.setAttribute('aria-disabled', String(busy || !ready));
    };
    const invalidateReview = () => {
        revision++; reviewId = undefined;
        byId('completion-mask-confirm').checked = false;
        byId('completion-visible-confirm').checked = false;
        clearResult(); update();
    };
    const reset = () => {
        operation++; revision++; source = undefined; session = undefined; imagePixels = undefined; bounds = undefined;
        proposalId = undefined; reviewId = undefined; busy = false; drawing = false; undo = []; mask.fill(0);
        byId('completion-mask-confirm').checked = false; byId('completion-visible-confirm').checked = false;
        byId('completion-editor-section').classList.add('hidden'); byId('completion-stage').classList.add('hidden');
        byId('completion-import').value = ''; clearResult(); error(''); context.clearRect(0, 0, 256, 256); update();
    };
    const begin = () => { busy = true; const token = ++operation; update(); return token; };
    const current = (token, version) => token === operation && version === revision && Boolean(source);
    const finish = token => { if (token === operation) { busy = false; update(); } };
    const remember = () => { undo.push(mask.slice()); if (undo.length > 10) undo.shift(); };
    const inside = (x, y) => Boolean(bounds && x >= bounds[0] && y >= bounds[1] && x < bounds[2] && y < bounds[3]);
    const setCursor = (x, y) => { byId('completion-x').value = x; byId('completion-y').value = y; };
    const render = () => {
        if (!imagePixels) return;
        const pixels = new Uint8ClampedArray(imagePixels.data);
        let count = 0;
        for (let i = 0; i < mask.length; i++) if (mask[i]) {
            count++;
            for (let channel = 0; channel < 3; channel++) pixels[i * 4 + channel] = Math.round(pixels[i * 4 + channel] * .55 + [238, 151, 55][channel] * .45);
        }
        context.putImageData(new ImageData(pixels, 256, 256), 0, 0);
        byId('completion-cursor').textContent = `${count.toLocaleString()} pixels selected. ${byId('completion-action').value === 'erase' ? 'Erase' : 'Paint'}; position ${byId('completion-x').value}, ${byId('completion-y').value}. ${count ? 'Check all orange areas before continuing.' : 'Nothing is selected. Check for any coverings that need to be marked.'}`;
    };
    const coordinate = id => {
        const value = Number(byId(id).value);
        if (byId(id).value.trim() === '' || !Number.isInteger(value) || value < 0 || value > 255) throw new Error('Enter a whole number from 0 to 255.');
        return value;
    };
    const disk = (x, y) => {
        const size = Number(byId('completion-size').value);
        if (byId('completion-size').value.trim() === '' || !Number.isInteger(size) || size < 1 || size > 80) throw new Error('Choose a brush size from 1 to 80.');
        const radius = (size - 1) / 2, value = byId('completion-action').value === 'erase' ? 0 : 255;
        for (let yy = Math.max(0, Math.ceil(y - radius)); yy <= Math.min(255, Math.floor(y + radius)); yy++) {
            for (let xx = Math.max(0, Math.ceil(x - radius)); xx <= Math.min(255, Math.floor(x + radius)); xx++) {
                if (inside(xx, yy) && (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) mask[yy * 256 + xx] = value;
            }
        }
        setCursor(x, y);
    };
    const point = event => {
        const rect = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(255, Math.floor((event.clientX - rect.left) * 256 / rect.width))), Math.max(0, Math.min(255, Math.floor((event.clientY - rect.top) * 256 / rect.height)))];
    };
    const edit = callback => {
        if (busy || !session || !imagePixels) return;
        try { remember(); callback(); invalidateReview(); render(); error(''); progress('The selected area changed. Check it and confirm your choices again.'); }
        catch (problem) { error(problem.message); }
    };
    canvas.addEventListener('pointerdown', event => {
        if (busy || !session || !imagePixels || event.button !== 0) return;
        event.preventDefault(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
        remember(); invalidateReview(); drawing = true; previous = point(event);
        try { disk(...previous); render(); error(''); }
        catch (problem) { drawing = false; error(problem.message); }
    });
    canvas.addEventListener('pointermove', event => {
        if (!drawing || busy) return;
        const next = point(event), count = Math.max(Math.abs(next[0] - previous[0]), Math.abs(next[1] - previous[1]), 1);
        try {
            if (reviewId || byId('completion-mask-confirm').checked || byId('completion-visible-confirm').checked) invalidateReview();
            for (let i = 1; i <= count; i++) disk(Math.round(previous[0] + (next[0] - previous[0]) * i / count), Math.round(previous[1] + (next[1] - previous[1]) * i / count));
            previous = next; render();
        } catch (problem) { drawing = false; error(problem.message); }
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drawing = false; });
    canvas.addEventListener('keydown', event => {
        if (busy || !session || !imagePixels) return;
        let x, y;
        try { x = coordinate('completion-x'); y = coordinate('completion-y'); }
        catch (problem) { error(problem.message); return; }
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (moves[event.key]) { event.preventDefault(); setCursor(Math.max(0, Math.min(255, x + moves[event.key][0])), Math.max(0, Math.min(255, y + moves[event.key][1]))); render(); }
        if (event.key === ' ') { event.preventDefault(); edit(() => disk(x, y)); }
    });
    byId('completion-stamp').addEventListener('click', () => edit(() => disk(coordinate('completion-x'), coordinate('completion-y'))));
    byId('completion-clear').addEventListener('click', () => edit(() => mask.fill(0)));
    byId('completion-undo').addEventListener('click', () => {
        if (busy || !session || !undo.length) return;
        mask = undo.pop(); invalidateReview(); render(); error(''); progress('Your last change was undone. Check the area again.');
    });
    byId('completion-rectangle').addEventListener('click', () => edit(() => {
        const left = coordinate('completion-left'), top = coordinate('completion-top'), right = coordinate('completion-right'), bottom = coordinate('completion-bottom');
        if (right < left || bottom < top) throw new Error('The right and bottom values must be greater than the left and top values.');
        if (!inside(left, top) || !inside(right, bottom)) throw new Error('Keep the rectangle inside the photo.');
        const value = byId('completion-action').value === 'erase' ? 0 : 255;
        for (let y = top; y <= bottom; y++) mask.fill(value, y * 256 + left, y * 256 + right + 1);
    }));
    byId('completion-action').addEventListener('change', render);
    // Only pixel changes invalidate a reviewed mask; changing brush size does not.
    const crcTable = Uint32Array.from({ length: 256 }, (_, n) => { for (let i = 0; i < 8; i++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0; });
    const chunk = (type, bytes) => {
        const result = new Uint8Array(bytes.length + 12), view = new DataView(result.buffer);
        view.setUint32(0, bytes.length); result.set(new TextEncoder().encode(type), 4); result.set(bytes, 8);
        let crc = 0xffffffff;
        for (let i = 4; i < bytes.length + 8; i++) crc = crcTable[(crc ^ result[i]) & 255] ^ (crc >>> 8);
        view.setUint32(bytes.length + 8, (crc ^ 0xffffffff) >>> 0); return result;
    };
    const maskPNG = async values => {
        if (typeof CompressionStream !== 'function') throw new Error('Use an up-to-date version of Chrome or Edge to download the selected area.');
        const rows = new Uint8Array(257 * 256);
        for (let y = 0; y < 256; y++) rows.set(values.subarray(y * 256, (y + 1) * 256), y * 257 + 1);
        const compressed = new Uint8Array(await new Response(new Blob([rows]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
        const header = new Uint8Array(13), view = new DataView(header.buffer);
        view.setUint32(0, 256); view.setUint32(4, 256); header[8] = 8;
        return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', compressed), chunk('IEND', new Uint8Array())], { type: 'image/png' });
    };
    const readMask = async imageSource => {
        const image = new Image(); image.src = imageSource; await image.decode();
        if (image.naturalWidth !== 256 || image.naturalHeight !== 256) throw new Error('Choose a black-and-white PNG that is exactly 256 × 256 pixels.');
        const buffer = document.createElement('canvas'); buffer.width = buffer.height = 256;
        const ctx = buffer.getContext('2d'); ctx.drawImage(image, 0, 0);
        const pixels = ctx.getImageData(0, 0, 256, 256).data, values = new Uint8Array(256 * 256);
        for (let i = 0; i < values.length; i++) {
            const r = pixels[i * 4];
            if ((r !== 0 && r !== 255) || pixels[i * 4 + 1] !== r || pixels[i * 4 + 2] !== r || pixels[i * 4 + 3] !== 255) throw new Error('The selected-area image must contain only solid black and white pixels.');
            if (r && !inside(i % 256, Math.floor(i / 256))) throw new Error('The selected area must stay inside the photo.');
            values[i] = r;
        }
        return values;
    };
    byId('completion-import').addEventListener('change', async () => {
        const file = byId('completion-import').files[0];
        if (busy || !session || !file) return;
        if (!file.size || file.size > 1024 * 1024 || file.type !== 'image/png') { error('Choose a black-and-white PNG smaller than 1 MiB.'); return; }
        invalidateReview(); const token = begin(), version = revision, objectURL = URL.createObjectURL(file);
        try {
            const values = await readMask(objectURL);
            if (!current(token, version)) return;
            remember(); mask = values; render(); error(''); progress('Area imported. Check all selected parts before continuing.');
        } catch (problem) { if (current(token, version)) error(problem.message); }
        finally { URL.revokeObjectURL(objectURL); byId('completion-import').value = ''; finish(token); }
    });
    byId('completion-download-mask').addEventListener('click', async () => {
        if (busy || !session) return;
        const version = revision;
        try {
            const bytes = await maskPNG(mask.slice());
            if (version !== revision || !session) return;
            const objectURL = URL.createObjectURL(bytes), link = document.createElement('a');
            link.href = objectURL; link.download = 'reviewed-removal-mask.png'; document.body.appendChild(link); link.click(); link.remove();
            setTimeout(() => URL.revokeObjectURL(objectURL), 1000);
        } catch (problem) { error(problem.message); }
    });
    const propose = async (token, version) => {
        const data = new FormData(); data.append('session_id', session);
        progress('Finding an area to remove. Please check and correct it before continuing.');
        const result = await post('/propose', data);
        if (!current(token, version)) return;
        const values = await readMask(pngURL(result.proposal_png));
        if (!current(token, version)) return;
        if (typeof result.proposal_id !== 'string' || !/^[a-f0-9]{24,64}$/.test(result.proposal_id)) throw new Error('The suggested area could not be loaded. Please select it yourself.');
        remember(); mask = values; proposalId = result.proposal_id; render();
        const empty = !mask.some(Boolean);
        progress(empty ? 'No area was selected automatically. Check the photo and mark any covering yourself.' : 'A suggested area is ready. Add any missed covering and remove marks from visible features.');
    };
    byId('completion-start').addEventListener('click', async () => {
        if (busy || !source) return;
        invalidateReview(); error(''); const token = begin(), version = revision, fixed = source;
        progress('Preparing your result for covering removal.');
        try {
            const data = new FormData(); data.append('file', pngBlob(fixed.result.result), 'restoration-result.png');
            data.append('original', fixed.originalFile, fixed.originalFile.name); data.append('restoration_metadata', JSON.stringify(fixed.result.metadata));
            const result = await post('/prepare', data);
            if (!current(token, version)) return;
            if (typeof result.session_id !== 'string' || !/^[a-f0-9]{24,64}$/.test(result.session_id) || !Array.isArray(result.bounds) || result.bounds.length !== 4 || result.bounds.some(value => !Number.isInteger(value))) throw new Error('The image could not be prepared. Please try again.');
            const b = result.bounds;
            if (!(0 <= b[0] && b[0] < b[2] && b[2] <= 256 && 0 <= b[1] && b[1] < b[3] && b[3] <= 256)) throw new Error('The image could not be prepared. Please try again.');
            const image = new Image(); image.src = pngURL(result.input_png); await image.decode();
            if (!current(token, version)) return;
            if (image.naturalWidth !== 256 || image.naturalHeight !== 256) throw new Error('The restored image has an unexpected size. Please restore it again.');
            session = result.session_id; bounds = b.slice(); mask = new Uint8Array(256 * 256); undo = []; proposalId = undefined;
            context.drawImage(image, 0, 0); imagePixels = context.getImageData(0, 0, 256, 256); render();
            byId('completion-editor-section').classList.remove('hidden'); update();
            try { await propose(token, version); }
            catch (problem) { if (current(token, version)) { error(problem.message); progress('We could not suggest an area automatically. You can still mark or import the area yourself.'); } }
        } catch (problem) { if (current(token, version)) { error(problem.message); progress('Could not prepare covering removal. You can keep your result or check the error and try again.'); } }
        finally { finish(token); }
    });
    byId('completion-propose').addEventListener('click', async () => {
        if (busy || !session) return;
        invalidateReview(); error(''); const token = begin(), version = revision;
        try { await propose(token, version); }
        catch (problem) { if (current(token, version)) { error(problem.message); progress('We could not suggest an area automatically. You can still adjust the selected area yourself.'); } }
        finally { finish(token); }
    });
    byId('completion-mask-confirm').addEventListener('change', async () => {
        reviewId = undefined; clearResult(); update();
        if (!byId('completion-mask-confirm').checked || busy || !session) return;
        error(''); const token = begin(), version = revision;
        progress('Saving your selected area for this image.');
        try {
            const data = new FormData(); data.append('session_id', session); data.append('mask', await maskPNG(mask.slice()), 'reviewed-mask.png');
            if (proposalId) data.append('proposal_id', proposalId);
            if (!current(token, version)) return;
            const result = await post('/review', data);
            if (!current(token, version)) return;
            if (typeof result.review_id !== 'string' || !/^[a-f0-9]{24,64}$/.test(result.review_id)) throw new Error('Could not save the selected area. Please check and confirm it again.');
            if (result.visibility?.rejected === true) throw new Error('Too much of the face is hidden. Choose a clearer photo with more of the face visible.');
            const checked = await readMask(pngURL(result.mask_png));
            if (!current(token, version)) return;
            if (checked.some((value, index) => value !== mask[index])) throw new Error('The saved area does not match your selection. Please check and confirm it again.');
            reviewId = result.review_id;
            progress('Area saved. Confirm that enough of the face is visible, then continue.');
        } catch (problem) { if (current(token, version)) { reviewId = undefined; byId('completion-mask-confirm').checked = false; error(problem.message); progress('Could not save the area. Adjust it or choose a clearer photo.'); } }
        finally { finish(token); }
    });
    byId('completion-visible-confirm').addEventListener('change', () => { clearResult(); update(); });
    byId('completion-generate').addEventListener('click', async () => {
        if (busy || !session || !reviewId || !byId('completion-mask-confirm').checked || !byId('completion-visible-confirm').checked) return;
        error(''); clearResult(); const token = begin(), version = revision, fixed = source, fixedMask = mask.slice(), fixedReview = reviewId;
        progress(mask.some(Boolean) ? 'Creating an estimate of the covered area. Please keep the app open.' : 'No area is selected, so your restored image will stay the same.');
        try {
            const data = new FormData(); data.append('session_id', session); data.append('mask', await maskPNG(fixedMask), 'reviewed-mask.png');
            data.append('review_id', fixedReview); data.append('mask_confirmed', 'true'); data.append('visible_confirmed', 'true');
            if (proposalId) data.append('proposal_id', proposalId);
            if (!current(token, version)) return;
            const result = await post('/generate', data);
            if (!current(token, version)) return;
            const output = pngURL(result.result_png), removal = pngURL(result.mask_png), evidence = zipURL(result.bundle);
            const original = pngURL(fixed.result.original_input), input = pngURL(fixed.result.result);
            byId('completion-original').src = original; byId('completion-original-download').href = original;
            byId('completion-input').src = input; byId('completion-input-download').href = input;
            byId('completion-result-mask').src = removal; byId('completion-mask-download').href = removal;
            byId('completion-result').src = output; byId('completion-result-download').href = output; byId('completion-bundle').href = evidence;
            const calls = result.neural_calls ?? result.metadata?.neural_calls;
            if (![0, 1].includes(calls) || calls !== Number(fixedMask.some(Boolean))) throw new Error('The result did not match the selected area. Please review the area and try again.');
            await Promise.all(['completion-original', 'completion-input', 'completion-result-mask', 'completion-result'].map(id => byId(id).decode()));
            if (!current(token, version)) return;
            const seconds = result.metadata?.seconds;
            const timing = Number.isFinite(seconds) ? `${seconds.toFixed(1)} seconds. ` : '';
            const summary = `${timing}${calls ? 'The image outside the selected area is unchanged. Hidden features are estimated and may not match the real person.' : 'Nothing was selected, so the restored image is unchanged.'}`;
            byId('completion-summary').textContent = summary; byId('completion-result-section').classList.remove('hidden'); progress(summary);
            // The server consumes a review after one successful generation.
            // Keep this result visible, but require a fresh review for another.
            reviewId = undefined;
            byId('completion-mask-confirm').checked = false;
            byId('completion-visible-confirm').checked = false;
        } catch (problem) { if (current(token, version)) { clearResult(); error(problem.message); progress('Covering removal stopped. Your saved files are still available. Check the message before trying again.'); } }
        finally { finish(token); }
    });
    window.addEventListener('restoration:completed', event => {
        reset();
        const detail = event.detail;
        try {
            if (!(detail?.originalFile instanceof File) || detail.result?.metadata?.selection !== 'conditioned_sft_v2_p2_u0224') throw new Error('The restored image is not available. Please restore the photo first.');
            pngURL(detail.result.result); pngURL(detail.result.original_input);
            source = { originalFile: detail.originalFile, result: detail.result };
            byId('completion-stage').classList.remove('hidden');
            progress('Your restored image is ready. You can review a covering for removal or keep the image as it is.'); update();
        } catch (problem) { error(problem.message); }
    });
    // Existing restoration invalidations remove/hide its output. Observe only;
    // do not replace its network calls, controls, state or result pixels.
    new MutationObserver(() => {
        if (source && (byId('brush-result-output').getAttribute('src') !== source.result.result || byId('brush-results').classList.contains('hidden'))) reset();
    }).observe(byId('brush-results'), { attributes: true, attributeFilter: ['src', 'class'], subtree: true });
    update();
})();
