/* Unmounted app-style V2 workflow. Existing brush geometry and exact PNG
   encoding are retained; this page never runs comparison or covering APIs. */
(() => {
    'use strict';
    const api = '/face/restoration', byId = id => document.getElementById(id);
    let videoBatch, videoRunMode, videoToken, pictureMode;
    let videoRetryBlocked = false;
    const canvas = byId('brush-editor'), context = canvas.getContext('2d');
    const protectColor = [45, 185, 90], applyColor = [55, 125, 220];
    let sourceFile, captureId, originalPixels, width = 0, height = 0, planes, undo = [];
    let pictureImage, pictureCrops = [], pictureCrop, pictureLabel;
    let available = false, busy = false, revision = 0, operation = 0;
    let previewId, confirmedPreviewId, masks, drawing = false, previous, cursorX = 0, cursorY = 0;
    let overlayPixels, dirty, renderFrame = 0, protectCount = 0, applyCount = 0;
    const error = message => {
        byId('brush-error').textContent = message;
        byId('brush-error').classList.toggle('hidden', !message);
    };
    const progress = message => { byId('brush-progress').textContent = message; };
    const preparationSummary = metadata => {
        const decision = metadata.automatic_preparation;
        if (decision?.route === 'jpeg_mosaic_coordinated') return 'Auto lightly smoothed the detected mosaic and reduced JPEG seams in the surrounding image before face restoration.';
        if (decision?.route === 'mosaic_regions_coordinated') return 'Auto applied separate preparation to distinct mosaic regions before face restoration.';
        if (decision?.route === 'scanline_mosaic_coordinated') return 'Auto reduced horizontal scanlines and smoothed a separately detected mosaic before face restoration.';
        if (decision?.route === 'perspective_mosaic_gaussian') return 'Auto smoothed a suspected perspective-warped mosaic before face restoration.';
        if (decision?.route === 'irregular_mosaic_gaussian') return 'Auto smoothed a suspected irregular mosaic before face restoration.';
        if (decision?.route === 'jpeg_deblocking') return 'Auto reduced strong JPEG block seams with FBCNN before face restoration.';
        if (decision?.route === 'rotated_mosaic_gaussian') return 'Auto found a tilted mosaic and smoothed the detected region before restoration.';
        if (decision?.route === 'tiny_mosaic_gaussian') return 'Auto smoothed a strong block grid in this small crop before face restoration.';
        if (decision?.route === 'direct' && metadata.softened_mosaic_detection?.crop_intersects) {
            return 'Auto flagged a possible softened mosaic and used direct restoration.';
        }
        if (decision?.route === 'scanline_cleanup') return 'Auto reduced repeating horizontal scanlines and lightly smoothed the image before restoration.';
        const stageLabels = {
            scanline_cleanup: 'scanline cleanup',
            jpeg_deblocking: 'JPEG deblocking',
            macroblock_fbcnn: 'FBCNN macroblock deblocking',
            macroblock_hevc: 'adaptive macroblock deblocking',
            edge_deringing: 'edge deringing',
            gradient_debanding: 'gradient debanding',
            tangent_antialiasing: 'tangent anti-aliasing',
            chroma_debleeding: 'chroma debleeding',
        };
        if (decision?.route === 'compound_artifact_preparation') {
            const stages = Array.isArray(decision.stages) && decision.stages.length
                ? decision.stages
                : (Array.isArray(decision.applied_stages) ? decision.applied_stages : []);
            const names = stages.map(stage => stageLabels[stage] || String(stage).replaceAll('_', ' '));
            return names.length
                ? `Auto applied ${names.join(', ')} before face restoration.`
                : 'Auto applied multi-stage artifact preparation before face restoration.';
        }
        if (decision?.route === 'macroblock_fbcnn') return 'Auto reduced compression macroblocking with FBCNN before face restoration.';
        if (decision?.route === 'macroblock_hevc') return 'Auto smoothed DCT macroblock boundaries with adaptive boundary deblocking before face restoration.';
        if (decision?.route === 'edge_deringing') return 'Auto suppressed ringing halos near sharp edges before face restoration.';
        if (decision?.route === 'gradient_debanding') return 'Auto smoothed quantized gradient banding with subtle dither before face restoration.';
        if (decision?.route === 'tangent_antialiasing') return 'Auto smoothed jagged diagonal contours along the edge tangent before face restoration.';
        if (decision?.route === 'chroma_debleeding') return 'Auto reduced chromatic bleeding across luma boundaries before face restoration.';
        if (['picture_region_gaussian', 'frame_region_gaussian', 'face_gaussian'].includes(decision?.route)) {
            const area = decision.route === 'face_gaussian' ? 'the detected face' : 'flagged regions';
            const radius = decision.Gaussian_radius_at256.toFixed(1);
            if (decision.strength_basis === 'measured_artifact_detail_tradeoff') {
                return `Auto chose ${radius} px smoothing for ${area} based on remaining block edges and retained detail.`;
            }
            if (decision.strength_basis === 'default_weak_evidence') {
                return `Auto used the ${radius} px default for ${area} because the signal was too weak to estimate a different strength.`;
            }
            return `Auto applied ${radius} px smoothing to ${area} before restoration.`;
        }
        if (decision) return decision.route === 'protect_auto'
            ? 'Auto used Protect + Auto-match before restoration.'
            : 'Auto used direct restoration.';
        return metadata.preparation_applied ? 'Your reviewed preparation was used.' : 'No preparation was applied.';
    };
    const manual = () => byId('brush-preparation').value === 'manual';
    const off = () => byId('brush-restoration').value === 'off';
    const automatic = () => byId('brush-restoration').value === 'auto';
    const matchBlur = () => byId('brush-match-blur').checked;
    const updateBlurStrength = () => {
        byId('brush-radius').disabled = busy || (matchBlur() && protectCount > 0 && applyCount === 0);
    };
    const preparationConfirmed = () => Boolean(previewId && masks && confirmedPreviewId === previewId);
    const clearResults = () => {
        byId('brush-results').classList.add('hidden');
        byId('picture-result-metrics').replaceChildren();
        byId('brush-result-summary').classList.add('hidden');
        byId('brush-result-downloads').classList.add('hidden');
        byId('brush-full-picture-result').classList.add('hidden');
        byId('brush-standby').classList.remove('hidden');
        for (const id of ['brush-download-input', 'brush-download-prepared', 'brush-download-output', 'brush-result-bundle', 'brush-download-full-picture']) byId(id).removeAttribute('href');
        for (const id of ['brush-result-input', 'brush-result-prepared', 'brush-result-output']) byId(id).removeAttribute('src');
    };
    const update = () => {
        document.querySelectorAll('#brush-form input, #brush-form select, #brush-form button').forEach(item => {
            if (!item.closest('[data-capture-controls], .capture-visual, #video-face-selector') && !item.matches('.video-comparison [data-zoom]')) item.disabled = busy;
        });
        byId('brush-continue').disabled = busy || !sourceFile || !planes;
        byId('brush-preview').disabled = busy || !available || !planes || !manual();
        byId('brush-preview-continue').disabled = busy || !previewId || !masks;
        updateBlurStrength();
        byId('brush-run').disabled = busy || !available || (videoBatch ? videoRetryBlocked || !videoBatch.items.some(item => item.status !== 'completed') && videoRunMode === byId('brush-restoration').value
            : !sourceFile || !planes || (manual() && !off() && !automatic() && !preparationConfirmed()));
        byId('picture-preparation').classList.toggle('hidden', Boolean(videoBatch) || automatic());
        byId('picture-preparation-details').classList.toggle('hidden', Boolean(videoBatch) || automatic());
        byId('brush-undo').disabled = busy || !undo.length;
        byId('brush-edit-section').classList.toggle('hidden', !planes || !manual());
        canvas.setAttribute('aria-disabled', String(busy));
        window.dispatchEvent(new CustomEvent('restoration:busy', {detail:{busy}}));
    };
    const invalidate = () => {
        revision++; previewId = confirmedPreviewId = masks = undefined;
        byId('brush-preview-section').classList.add('hidden');
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
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => {
        drawing = false;
        updateBlurStrength();
    });
    canvas.addEventListener('keydown', event => {
        if (busy || !planes || !manual()) return;
        const x = cursorX, y = cursorY;
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (moves[event.key]) { event.preventDefault(); cursor(Math.max(0, Math.min(width - 1, x + moves[event.key][0])), Math.max(0, Math.min(height - 1, y + moves[event.key][1]))); }
        if (event.key === ' ') { event.preventDefault(); edit(() => disk(x, y)); }
    });
    byId('brush-clear').addEventListener('click', () => edit(() => { planes.protect.fill(0); planes.apply.fill(0); protectCount = 0; applyCount = 0; markDirty(0, 0, width - 1, height - 1); }));
    byId('brush-undo').addEventListener('click', () => { if (busy || !undo.length) return; planes = undo.pop(); repaintAll(); invalidate(); });
    byId('brush-size').addEventListener('input', event => { byId('brush-size-value').textContent = `${event.target.value} px`; });
    byId('brush-radius').addEventListener('input', event => { byId('brush-radius-value').textContent = `${event.target.value} px`; });
    const updateBrushMode = () => {
        byId('brush-mode-summary').textContent = brushMode();
        if (planes) render();
    };
    for (const id of ['brush-plane', 'brush-action']) byId(id).addEventListener('change', updateBrushMode);
    for (const id of ['brush-radius', 'brush-size']) byId(id).addEventListener('input', invalidate);
    for (const id of ['brush-plane', 'brush-action', 'brush-restoration', 'brush-preparation', 'brush-match-blur']) byId(id).addEventListener('change', () => { drawing = false; invalidate(); });
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
    const post = async (route, data, headers) => {
        const response = await fetch(api + route, { method: 'POST', body: data, headers });
        let answer;
        try { answer = await response.json(); }
        catch { throw new Error('The app could not get a valid response. Select Check service before trying again.'); }
        if (!response.ok) {
            const detail = typeof answer.detail === 'string' ? answer.detail : '';
            if (/could not be confirmed stopped|do not retry|previous worker stopped unexpectedly/i.test(detail)) {
                const problem = new Error('Check the saved request and restart the app before restoring another face.');
                problem.restartRequired = true;
                throw problem;
            }
            if (/model|\bv2\b|conditioned_sft/i.test(detail)) throw new Error('The restoration service is not ready. Check service or restart the app.');
            if (/mask/i.test(detail)) throw new Error('The app could not use the marked areas. Review your selections or choose Bypass.');
            throw new Error(detail || 'The request stopped. Check the picture and service status.');
        }
        return answer;
    };
    const form = file => { const data = new FormData(); data.append('file', file); return data; };
    const pictureForm = data => {
        data.append('input_review', 'picture_upload');
        if (pictureCrop) data.append('picture_crop', JSON.stringify(pictureCrop));
        return data;
    };
    const maskedForm = (file, savedMasks, radius) => {
        const data = form(file);
        data.append('protect_mask', savedMasks.protect, 'Protect.png'); data.append('apply_mask', savedMasks.apply, 'Apply.png');
        data.append('radius', radius);
        data.append('match_blur', String(matchBlur()));
        return data;
    };
    const resetInput = () => {
        if (renderFrame) cancelAnimationFrame(renderFrame);
        renderFrame = 0; overlayPixels = undefined; dirty = undefined; protectCount = 0; applyCount = 0;
        sourceFile = undefined; captureId = undefined; originalPixels = undefined; planes = undefined; undo = []; width = 0; height = 0; drawing = false;
        pictureImage = pictureCrop = undefined; pictureCrops = [];
        byId('picture-face').replaceChildren();
        byId('picture-face-field').classList.add('hidden');
        byId('picture-face-status').classList.add('hidden');
        canvas.width = 0; canvas.height = 0;
        byId('input-picture').classList.add('hidden');
        byId('upload-preview').removeAttribute('src'); byId('upload-preview').classList.add('hidden');
        const mainPreview = byId('picture-main-preview');
        if (mainPreview) {
            mainPreview.removeAttribute('src');
            mainPreview.classList.add('hidden');
        }
        byId('picture-empty')?.classList.remove('hidden');
        invalidate();
    };
    function selectPictureCrop(rectangle, label) {
        pictureCrop = rectangle;
        const [x, y, w, h] = rectangle || [0, 0, pictureImage.naturalWidth, pictureImage.naturalHeight];
        width = canvas.width = w;
        height = canvas.height = h;
        context.drawImage(pictureImage, x, y, w, h, 0, 0, w, h);
        originalPixels = context.getImageData(0, 0, w, h);
        const preview = rectangle ? canvas.toDataURL('image/png') : pictureImage.src;
        overlayPixels = new ImageData(new Uint8ClampedArray(originalPixels.data), w, h);
        planes = { protect: new Uint8Array(w * h), apply: new Uint8Array(w * h) };
        undo = []; dirty = undefined; protectCount = applyCount = 0;
        cursorX = cursorY = 0;
        byId('input-picture').classList.remove('hidden');
        byId('input-picture-label').textContent = byId('upload-preview').alt = label;
        byId('upload-preview').src = preview;
        byId('upload-preview').classList.remove('hidden');
        const mainPreview = byId('picture-main-preview');
        if (mainPreview) {
            mainPreview.alt = label;
            mainPreview.width = w;
            mainPreview.height = h;
            mainPreview.src = preview;
            mainPreview.classList.remove('hidden');
        }
        byId('picture-empty')?.classList.add('hidden');
        byId('brush-dimensions').textContent = `Picture size: ${w} × ${h} pixels. No areas are marked yet.`;
        invalidate(); render();
    }
    byId('picture-face').addEventListener('change', event => {
        if (busy || !pictureImage || captureId) return;
        const index = Number(event.target.value);
        const selected = Number.isInteger(index) && index >= 0 ? pictureCrops[index] : undefined;
        if (!selected) {
            selectPictureCrop(undefined, pictureLabel);
            byId('picture-face-status').textContent = 'Using the full picture.';
            return;
        }
        selectPictureCrop(selected, `Face ${index + 1} crop`);
        byId('picture-face-status').textContent = `Using the crop for Face ${index + 1}.`;
    });
    const usePicture = async (file, options = {}) => {
        if (busy) { error('Wait for the current task to finish, then choose the picture again.'); return false; }
        clearVideoBatch();
        resetInput(); error('');
        if (!file) return;
        if (file.size === 0 || file.size > 10 * 1024 ** 2 || !['image/png', 'image/jpeg'].includes(file.type)) { error('Choose a PNG or JPEG face picture smaller than 10 MB.'); return; }
        const token = begin(), version = revision;
        progress('Checking your picture.');
        try {
            const data = form(file);
            if (!options.captureId) data.append('scan_faces', 'true');
            const result = await post('/inspect', data);
            if (!current(token, version) || (options.isCurrent && !options.isCurrent())) return false;
            if (!Number.isInteger(result.width) || !Number.isInteger(result.height) || result.width < 1 || result.height < 1 || result.width > 1600 || result.height > 1600 || result.width * result.height > 2560000 || result.neural_calls !== 0) throw new Error('The app could not read the picture size. Please choose another image.');
            const image = new Image(); image.src = validPNG(result.source); await image.decode();
            if (!current(token, version) || (options.isCurrent && !options.isCurrent())) return false;
            if (image.naturalWidth !== result.width || image.naturalHeight !== result.height) throw new Error('The picture could not be prepared correctly. Please choose it again.');
            sourceFile = file; captureId = options.captureId; pictureImage = image;
            pictureLabel = options.label || 'Full uploaded picture';
            pictureCrops = Array.isArray(result.face_crops) ? result.face_crops : [];
            const first = pictureCrops[0];
            selectPictureCrop(first, first ? 'Face 1 crop' : pictureLabel);
            if (!captureId) {
                const choices = byId('picture-face');
                choices.replaceChildren();
                pictureCrops.forEach((_, index) => choices.add(new Option(`Face ${index + 1}`, String(index))));
                choices.add(new Option('Full picture', '-1'));
                choices.value = first ? '0' : '-1';
                byId('picture-face-field').classList.toggle('hidden', !pictureCrops.length);
                const status = byId('picture-face-status');
                status.textContent = result.face_scan_unavailable ? 'Face scanning is unavailable. Use the full picture or upload a close-up.'
                    : !first ? 'No face detected. Use the full picture or upload a close-up.'
                    : pictureCrops.length > 1 ? 'Face 1 is selected. Faces are numbered from left to right.'
                    : 'One face detected. Its crop is selected.';
                status.classList.remove('hidden');
            }
            progress('Picture ready. Select Continue.');
            window.dispatchEvent(new CustomEvent('picture:changed', {detail:{captured:!!captureId,file}}));
            return true;
        } catch (problem) { if (current(token, version)) { resetInput(); error(problem.message); progress('The picture could not be checked. Choose a clear face picture.'); } }
        finally { finish(token); }
    };
    function confirmPreparation() {
        if (busy || !previewId || !masks || !manual() || off()) return false;
        confirmedPreviewId = previewId;
        update();
        return true;
    }
    window.restorationInput = {use:usePicture, useVideo, clearVideoBatch, confirmPreparation, get video() {return Boolean(videoBatch);}, get busy() {return busy;}};
    byId('brush-file').addEventListener('change', () => usePicture(byId('brush-file').files[0]));
    byId('brush-preview').addEventListener('click', async () => {
        if (busy || !planes || !sourceFile || !manual()) return;
        const radius = byId('brush-radius').value;
        if (radius.trim() === '' || !Number.isFinite(Number(radius)) || Number(radius) < 0 || Number(radius) > 12 || Number(radius) * 2 % 1 !== 0) { error('Choose a blur radius from 0 to 12 in steps of 0.5.'); return; }
        invalidate(); error(''); const token = begin(), version = revision;
        const file = sourceFile, nativeWidth = width, nativeHeight = height;
        const savedPlanes = { protect: planes.protect.slice(), apply: planes.apply.slice() };
        progress('Preparing a preview from your original picture.');
        try {
            const savedMasks = { protect: await grayPNG(savedPlanes.protect, nativeWidth, nativeHeight), apply: await grayPNG(savedPlanes.apply, nativeWidth, nativeHeight) };
            if (!current(token, version)) return;
            const data = pictureForm(maskedForm(file, savedMasks, radius));
            const result = await post('/preview', data);
            if (!current(token, version)) return;
            if (!/^[a-f0-9]{64}$/.test(result.preview_id) || result.neural_calls !== 0 || result.metadata?.every_protected_native_RGB_exact !== true || result.metadata?.every_outside_Apply_native_RGB_exact !== true) throw new Error('The app could not verify that protected areas stayed unchanged. Use Bypass instead.');
            const original = validPNG(result.original), overlay = validPNG(result.protection_overlay), prepared = validPNG(result.prepared_input);
            previewId = result.preview_id; masks = savedMasks;
            byId('brush-original').src = original; byId('brush-overlay').src = overlay; byId('brush-derived').src = prepared;
            let explanation;
            const matching = result.metadata.algorithm.blur_matching;
            if (matching?.status === 'insufficient_edges') {
                explanation = 'Auto found too few reliable edges to estimate blur. No blur was added. Turn off Auto to set a strength manually.';
            } else if (matching) {
                const target = matching.target_radius_at256;
                const added = matching.maximum_added_radius_at256;
                explanation = result.metadata.no_effect ? `Auto estimated a ${target} px target. No visible blur was added.`
                    : `Auto estimated a ${target} px target and added up to ${added} px to sharper areas.`;
            } else if (result.metadata.no_effect) {
                if (Number(radius) === 0) explanation = 'Blur strength is 0, so the picture did not change. Increase the strength to add blur.';
                else if (result.metadata.resolved_Apply_native_pixels === 0 && !savedPlanes.protect.some(Boolean)) explanation = 'Paint green to blur the surroundings, or blue to blur a specific area.';
                else if (result.metadata.effective_Apply_native_pixels === 0) explanation = 'Green protects the entire blur area. Erase some green to allow blur.';
                else explanation = 'The selected pixels did not visibly change. Try a higher blur strength.';
            } else {
                const area = result.metadata.Apply_selection_mode === 'automatic_surroundings' ? 'Surrounding blur' : 'Painted blur';
                explanation = `${area} at ${radius} px. ${result.metadata.changed_native_pixels.toLocaleString()} pixels changed.`;
            }
            byId('brush-preservation').textContent = `${explanation} Protected areas stay unchanged during preparation; restoration can still change facial appearance.`;
            byId('brush-preview-section').classList.remove('hidden');
            progress('Review the preview. Select Continue to use it.');
        } catch (problem) { if (current(token, version)) { invalidate(); error(problem.message); progress('Could not prepare the picture. Review the selected areas or choose Bypass.'); } }
        finally { finish(token); }
    });
    async function restoreRequest(data, headers) {
        const result = await post('/restore', data, headers);
        const metadata = result.metadata;
        if (!metadata || ![0, 1].includes(metadata.neural_forward_returns) || metadata.neural_forward_attempts !== metadata.neural_forward_returns || metadata.selection !== 'conditioned_sft_v2_p2_u0224') throw new Error('The app could not verify the result. Check the service before trying again.');
        await Promise.all([result.original_input, result.prepared_input, result.result].map(async url => {
            const image = new Image(); image.src = validPNG(url); await image.decode();
            if (image.naturalWidth !== 256 || image.naturalHeight !== 256) throw new Error('The service returned an invalid result size.');
        }));
        const fullImage = result.scanline_cleaned_full_image || result.smoothed_full_picture || result.smoothed_full_frame;
        if (fullImage) {
            const image = new Image();
            image.src = validPNG(fullImage);
            await image.decode();
            const [width, height] = (metadata.scanline_cleanup || metadata.picture_pixelation || metadata.frame_pixelation).native_size;
            if (image.naturalWidth !== width || image.naturalHeight !== height) throw new Error('The full image has an invalid size.');
        }
        if (data.get('include_bundle') === 'true') validZIP(result.bundle);
        return result;
    }
    function displayResult(result, originalFile) {
        const metadata = result.metadata;
        byId('brush-result-input').src = result.original_input;
        byId('brush-result-prepared').src = result.prepared_input;
        byId('brush-result-output').src = result.result;
        byId('brush-download-input').href = result.original_input;
        byId('brush-download-prepared').href = result.prepared_input;
        byId('brush-download-output').href = result.result;
        const prepared = byId('brush-prepared-details');
        prepared.open = false;
        prepared.classList.toggle('hidden', !metadata.preparation_applied);
        byId('brush-output-heading').textContent = metadata.restoration_applied ? 'After' : 'After (resized original)';
        byId('brush-result-downloads').classList.toggle('hidden', !result.bundle);
        if (result.bundle) byId('brush-result-bundle').href = result.bundle;
        const fullPictureDownload = byId('brush-download-full-picture');
        const fullPicture = result.scanline_cleaned_full_image || result.smoothed_full_picture;
        byId('brush-full-picture-result').classList.toggle('hidden', !fullPicture);
        if (fullPicture) {
            const scanlines = Boolean(result.scanline_cleaned_full_image);
            fullPictureDownload.href = fullPicture;
            fullPictureDownload.download = scanlines ? 'scanline-cleaned-picture.png' : 'full-image-smoothed.png';
            fullPictureDownload.textContent = scanlines ? 'Download scanline-cleaned picture (PNG)' : 'Download smoothed full picture (PNG)';
            byId('brush-full-picture-note').textContent = scanlines
                ? 'Original framing with repeating horizontal lines reduced.'
                : 'Original framing with Gaussian smoothing in the regions Auto flagged.';
        }
        else fullPictureDownload.removeAttribute('href');
        byId('brush-results').classList.remove('hidden');
        byId('brush-standby').classList.add('hidden');
        const seconds = Number.isFinite(metadata.seconds) ? `${metadata.seconds.toFixed(1)} seconds. ` : '';
        const summary = `${seconds}${preparationSummary(metadata)}`;
        byId('brush-result-summary').textContent = metadata.picture_crop ? `Before and After show the selected face crop. ${summary}` : summary;
        byId('brush-result-summary').classList.remove('hidden');
        progress('Result ready to review.');
        const pictureMetrics = byId('picture-result-metrics');
        pictureMetrics.replaceChildren();
        appendResultMetrics(pictureMetrics, 'Picture metrics', result);
        window.dispatchEvent(new CustomEvent('restoration:completed', { detail: { result, originalFile } }));
    }
    byId('brush-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (videoBatch) { await restoreVideo(); return; }
        if (busy || !sourceFile || !planes || !available) return;
        const restoration = byId('brush-restoration').value;
        const mode = restoration === 'auto' ? 'automatic' : restoration === 'off' ? 'bypass' : byId('brush-preparation').value;
        if (mode === 'manual' && !preparationConfirmed()) { error('Review the marked areas and select Continue.'); return; }
        const data = pictureForm(mode === 'manual' ? maskedForm(sourceFile, masks, byId('brush-radius').value) : form(sourceFile));
        data.append('restoration', restoration); data.append('preparation_mode', mode);
        data.append('model', byId('brush-model').value);
        if (captureId) data.append('capture_id', captureId);
        const includeBundle = byId('brush-include-bundle').checked;
        data.append('include_bundle', includeBundle ? 'true' : 'false');
        if (mode === 'manual') { data.append('preview_id', previewId); data.append('confirmed', 'true'); }
        error(''); clearResults(); const token = begin(), version = revision;
        restorationLoading(true);
        progress(restoration === 'off' ? 'Keeping your original picture and resizing it.' : 'Restoring your picture.');
        try {
            const result = await restoreRequest(data);
            if (!current(token, version)) return;
            displayResult(result, sourceFile);
        } catch (problem) { if (current(token, version)) { restorationLoading(false); clearResults(); error(problem.message); progress('The request stopped. Read the message before trying again.'); } }
        finally { if (token === operation) restorationLoading(false); finish(token); }
    });
    function videoVisibility(active) {
        for (const id of ['video-settings-gallery', 'video-restore-gallery']) byId(id).classList.toggle('hidden', !active);
        for (const id of ['options-preview', 'options-preview-caption', 'restoration-idle', 'picture-preparation', 'picture-preparation-details', 'picture-restore-note']) {
            byId(id).classList.toggle('hidden', active);
        }
        byId('brush-form').classList.toggle('video-batch', active);
    }
    function clearVideoBatch() {
        if (!videoBatch || busy) return;
        for (const item of videoBatch.items) URL.revokeObjectURL(item.url);
        videoBatch = videoRunMode = videoToken = undefined;
        byId('video-settings-gallery').replaceChildren();
        byId('video-face-selector').replaceChildren();
        byId('video-face-panels').replaceChildren();
        setFrameDownload();
        byId('brush-run').textContent = 'Restore selected input';
        byId('brush-restoration').value = pictureMode;
        byId('brush-restoration').dispatchEvent(new Event('change', { bubbles: true }));
        videoVisibility(false);
        update();
    }
    function setFrameDownload(source, scanlines = false) {
        const link = byId('video-download-smoothed-frame');
        link.parentElement.classList.toggle('hidden', !source);
        if (source) {
            link.href = source;
            link.download = scanlines ? 'scanline-cleaned-frame.png' : 'frozen-frame-smoothed.png';
            link.textContent = scanlines ? 'Download scanline-cleaned frame (PNG)' : 'Download smoothed full frame (PNG)';
        }
        else link.removeAttribute('href');
    }
    function useVideo(crops, options) {
        if (busy || !options.isCurrent() || !options.token || !crops.length || crops.length > window.restorationQueueLimit) return false;
        clearVideoBatch();
        pictureMode = byId('brush-restoration').value;
        error('');
        clearResults();
        videoBatch = window.createRestorationQueue();
        videoToken = options.token;
        videoBatch.addMany(crops.map(crop => ({ ...crop, url: URL.createObjectURL(crop.file) })));
        const items = videoBatch.items;
        for (const item of items) {
            byId('video-settings-gallery').append(videoCard(item, false));
            byId('video-face-panels').append(videoCard(item, true));
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'tool-btn';
            button.textContent = `Face ${item.faceNumber}`;
            button.setAttribute('aria-controls', `video-face-${item.id}`);
            button.addEventListener('click', () => selectVideoFace(item.id));
            byId('video-face-selector').append(button);
        }
        selectVideoFace(items[0].id);
        byId('brush-restoration').value = 'auto';
        byId('brush-restoration').dispatchEvent(new Event('change', { bubbles: true }));
        videoVisibility(true);
        byId('brush-run').textContent = crops.length === 1 ? 'Restore face' : 'Restore faces';
        progress('Choose one restoration setting for the selected faces.');
        update();
        return true;
    }
    function selectVideoFace(id) {
        const panel = `video-face-${id}`;
        for (const button of byId('video-face-selector').children) {
            button.setAttribute('aria-pressed', String(button.getAttribute('aria-controls') === panel));
        }
        for (const card of byId('video-face-panels').children) card.hidden = card.id !== panel;
    }
    function videoImage(src, alt) {
        const image = new Image();
        image.src = src;
        image.alt = alt;
        return image;
    }
    function videoCard(item, showResult) {
        const card = document.createElement('article');
        card.className = 'video-face-card';
        const name = `Face ${item.faceNumber}`;
        const title = document.createElement('h2');
        title.textContent = name;
        const images = document.createElement('div');
        images.className = 'video-face-images';
        card.append(title, images);
        if (!showResult) {
            images.append(videoImage(item.url, `${name} original crop`));
            return card;
        }
        card.id = `video-face-${item.id}`;
        card.setAttribute('aria-label', `${name} comparison`);
        images.classList.add('video-comparison');
        const details = document.createElement('div');
        details.className = 'video-face-details';
        card.append(details);
        showVideoResult(item, card);
        return card;
    }
    function videoComparisonImage(item, stage, source, bundle) {
        const id = `video-face-${item.id}-${stage.toLowerCase()}`;
        const name = `Face ${item.faceNumber}`;
        const figure = document.createElement('figure');
        const image = videoImage(source, `${name} ${stage.toLowerCase()} restoration`);
        image.id = id;
        const caption = document.createElement('figcaption');
        caption.textContent = stage;
        figure.append(image, caption);
        const download = document.createElement('a');
        download.className = 'tool-btn';
        download.href = source;
        download.download = `face-${item.faceNumber}-${stage.toLowerCase()}.png`;
        download.textContent = 'Download PNG';
        download.setAttribute('aria-label', `Download PNG for ${name} ${stage.toLowerCase()}`);
        const enlarge = document.createElement('button');
        enlarge.type = 'button';
        enlarge.className = 'tool-btn';
        enlarge.dataset.zoom = id;
        enlarge.textContent = 'Enlarge';
        enlarge.setAttribute('aria-label', `Enlarge ${name} ${stage.toLowerCase()}`);
        const actions = document.createElement('div');
        actions.className = 'video-image-actions';
        actions.append(download, enlarge);
        if (bundle) {
            const zip = document.createElement('a');
            zip.className = 'tool-btn';
            zip.href = bundle;
            zip.download = `face-${item.faceNumber}-review-files.zip`;
            zip.textContent = 'Download ZIP';
            actions.append(zip);
        }
        figure.append(actions);
        return figure;
    }
    function restorationMetrics(caption, diagnostics) {
        const table = document.createElement('table');
        table.className = 'video-metrics';
        table.createCaption().textContent = caption;
        const header = table.createTHead().insertRow();
        for (const title of ['Metric', 'Before', 'After']) {
            const cell = document.createElement('th');
            cell.scope = 'col';
            cell.textContent = title;
            header.append(cell);
        }
        const body = table.createTBody();
        function metric(label, before, after) {
            const row = body.insertRow();
            const name = document.createElement('th');
            name.scope = 'row';
            name.textContent = label;
            row.append(name);
            return [before, after].map(value => {
                const cell = row.insertCell();
                cell.textContent = value;
                return cell;
            });
        }
        const value = n => Number.isFinite(n) ? n.toFixed(2) : 'Unavailable';
        metric('Sharpness (edge variation)', value(diagnostics?.before?.blur_variance), value(diagnostics?.after?.blur_variance));
        metric('Noise estimate', value(diagnostics?.before?.noise_sigma_255), value(diagnostics?.after?.noise_sigma_255));
        const mae = metric('MAE (RGB levels)', '—', '—');
        const psnr = metric('PSNR (dB)', '—', '—');
        return {
            table,
            setReference(scores) {
                [scores?.before, scores?.after].forEach((score, i) => {
                    mae[i].textContent = score?.mae ?? '—';
                    psnr[i].textContent = score?.psnr ?? '—';
                });
            }
        };
    }
    function showVideoResult(item, card = byId(`video-face-${item.id}`), failure = '') {
        const images = card.querySelector('.video-face-images');
        const details = card.querySelector('.video-face-details');
        images.replaceChildren(videoComparisonImage(item, 'Before', item.result?.original_input || item.url));
        details.replaceChildren();
        const status = document.createElement('p');
        status.textContent = failure || (item.status === 'running' ? 'Processing…' : 'Waiting to restore.');
        status.setAttribute('role', 'status');
        details.append(status);
        if (!item.result) return;
        const { result } = item;
        images.append(videoComparisonImage(item, 'After', result.result, result.bundle));
        status.textContent = result.metadata.automatic_preparation ? preparationSummary(result.metadata)
            : result.metadata.restoration_applied ? 'Original and restored result' : 'Original and resized result';
        appendResultMetrics(details, `Face ${item.faceNumber} metrics`, result);
    }
    function appendResultMetrics(container, caption, result) {
        const metrics = restorationMetrics(caption, result.metadata.output_diagnostics);
        const score = document.createElement('p');
        score.textContent = 'Ground truth unavailable.';
        score.setAttribute('role', 'status');
        const reference = document.createElement('input');
        reference.type = 'file';
        reference.accept = 'image/png,image/jpeg';
        const label = document.createElement('label');
        label.append('Add aligned clean reference (256 × 256)', reference);
        const hint = document.createElement('p');
        hint.textContent = 'Use the same face, pose, and crop. Scores measure pixel agreement.';
        let referenceRevision = 0;
        reference.addEventListener('change', async () => {
            const version = ++referenceRevision;
            const file = reference.files[0];
            metrics.setReference();
            score.textContent = file ? 'Comparing reference…' : 'Ground truth unavailable.';
            if (!file) return;
            try {
                const scores = await window.scoreVideoReference(file, result);
                if (version === referenceRevision) {
                    metrics.setReference(scores);
                    score.textContent = 'MAE and PSNR compare both images with your supplied reference. Padding is excluded.';
                }
            } catch (problem) {
                if (version === referenceRevision) score.textContent = `Ground truth unavailable. ${problem.message}`;
            }
        });
        container.append(metrics.table, score, label, hint);
    }
    async function restoreVideo() {
        if (busy || !available || !videoBatch || videoRetryBlocked) return;
        const mode = byId('brush-restoration').value;
        const includeBundle = byId('brush-include-bundle').checked;
        if (videoRunMode && videoRunMode !== mode) {
            setFrameDownload();
            for (const item of videoBatch.items) {
                videoBatch.review(item.id);
                showVideoResult({ ...item, status: 'pending', result: undefined });
            }
        }
        videoRunMode = mode;
        for (const item of videoBatch.items) {
            if (item.status === 'completed') continue;
            const data = form(item.file);
            data.append('capture_id', item.captureId);
            data.append('input_review', 'video_capture');
            data.append('restoration', mode);
            data.append('preparation_mode', mode === 'auto' ? 'automatic' : 'bypass');
            data.append('model', byId('brush-model').value);
            data.append('include_bundle', String(includeBundle));
            videoBatch.approve(item.id, data);
            showVideoResult({ ...item, status: 'pending' });
        }
        const token = begin();
        let activeId;
        error('');
        progress('Processing the selected faces one at a time.');
        try {
            await videoBatch.run(async (request, item) => {
                activeId = item.id;
                showVideoResult(item);
                request.set('include_smoothed_frame', String(!byId('video-download-smoothed-frame').hasAttribute('href')));
                const result = await restoreRequest(request, { 'X-Capture-Token': videoToken });
                if (result.scanline_cleaned_full_image || result.smoothed_full_frame) {
                    setFrameDownload(result.scanline_cleaned_full_image || result.smoothed_full_frame,
                                     Boolean(result.scanline_cleaned_full_image));
                    delete result.scanline_cleaned_full_image;
                    delete result.smoothed_full_frame;
                }
                showVideoResult({ ...item, result });
                return result;
            });
            progress('All selected faces are ready to inspect.');
        } catch (problem) {
            videoRetryBlocked = Boolean(problem.restartRequired);
            const failed = videoBatch.items.find(item => item.id === activeId);
            if (failed) showVideoResult(failed, undefined, `Processing stopped. ${problem.message}`);
            error(videoRetryBlocked ? problem.message : 'The queue paused. Select Retry remaining faces to continue.');
        } finally {
            byId('brush-run').textContent = videoRetryBlocked ? 'Restart required' : videoBatch.items.some(item => item.status === 'failed') ? 'Retry remaining faces'
                : videoBatch.items.length === 1 ? 'Restore face' : 'Restore faces';
            finish(token);
        }
    }
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
