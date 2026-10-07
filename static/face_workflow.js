/* Review first, one estimate, and explicit restoration override. */
document.addEventListener('DOMContentLoaded', () => {
    const byId = id => document.getElementById(id);
    const editor = byId('mask-editor');
    const ctx = editor.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    const overlay = document.createElement('canvas');
    const overlayCtx = overlay.getContext('2d');
    const results = byId('results-grid');
    const standby = byId('standby').cloneNode(true);
    let file = null, original = null, originalWidth = 0, originalHeight = 0;
    let detectedMask = null, tool = 'paint', history = [], version = 0;
    let controller = null, busy = false, maskReady = false, drawing = false;
    let brushPoint = null, lastPoint = null, brushVisible = false;

    const imageFrom = source => new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('Cannot read this image. Use a PNG or JPEG.'));
        image.src = source;
    });
    const showError = message => {
        byId('workflow-error').textContent = message;
        byId('workflow-error').classList.toggle('hidden', !message);
    };
    const syncControls = () => {
        byId('analyze-btn').disabled = !file || !maskReady || busy || !byId('review-confirm').checked || byId('input-review-decision').value !== 'usable';
        byId('loading-state').classList.toggle('hidden', !busy);
        ['paint-btn', 'erase-btn', 'clear-btn', 'detect-btn', 'brush-size', 'mask-input', 'mask-download-btn', 'review-confirm', 'restoration-mode', 'input-review-decision'].forEach(id => {
            byId(id).disabled = busy || !file;
        });
        byId('undo-btn').disabled = busy || !history.length;
        byId('reset-btn').disabled = busy || !detectedMask;
        editor.setAttribute('aria-disabled', String(busy));
    };
    const clearResults = () => {
        results.replaceChildren(standby.cloneNode(true));
        results.classList.remove('has-estimate');
        byId('result-message').classList.add('hidden');
        byId('result-downloads').classList.add('hidden');
        byId('image-download').removeAttribute('href');
        byId('bundle-download').removeAttribute('href');
    };
    const changed = () => {
        byId('review-confirm').checked = false;
        clearResults();
        showError('');
        syncControls();
    };
    const saveUndo = () => {
        history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
        if (history.length > 12) history.shift();
        syncControls();
    };
    const renderMask = () => {
        if (!original) return;
        const pixels = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const green = overlayCtx.createImageData(maskCanvas.width, maskCanvas.height);
        for (let i = 0; i < pixels.data.length; i += 4) {
            const selected = pixels.data[i] >= 128;
            pixels.data[i] = pixels.data[i+1] = pixels.data[i+2] = selected ? 255 : 0;
            pixels.data[i+3] = 255;
            green.data[i] = 16; green.data[i+1] = 185; green.data[i+2] = 129;
            green.data[i+3] = selected ? 135 : 0;
        }
        maskCtx.putImageData(pixels, 0, 0);
        overlayCtx.putImageData(green, 0, 0);
        ctx.drawImage(original, 0, 0, editor.width, editor.height);
        ctx.drawImage(overlay, 0, 0);
        if (brushVisible && brushPoint) {
            ctx.beginPath();
            ctx.arc(brushPoint.x, brushPoint.y, Number(byId('brush-size').value)*editor.width/512, 0, Math.PI*2);
            ctx.strokeStyle = '#ffffff'; ctx.lineWidth = Math.max(1, editor.width/256); ctx.stroke();
        }
    };
    const installOriginal = image => {
        original = image; originalWidth = image.naturalWidth; originalHeight = image.naturalHeight;
        const scale = Math.min(1, 512/Math.max(originalWidth, originalHeight));
        editor.width = Math.max(1, Math.round(originalWidth*scale));
        editor.height = Math.max(1, Math.round(originalHeight*scale));
        maskCanvas.width = overlay.width = editor.width;
        maskCanvas.height = overlay.height = editor.height;
        maskCtx.fillStyle = '#000000'; maskCtx.fillRect(0, 0, editor.width, editor.height);
        brushPoint = { x: editor.width/2, y: editor.height/2 };
        detectedMask = null; history = []; maskReady = false;
        byId('preview-image').src = image.src;
        byId('preview-image').classList.remove('hidden');
        byId('drop-zone').classList.add('has-image');
        byId('drop-zone').querySelector('.drop-zone-ui').classList.add('hidden');
        byId('mask-review').classList.remove('hidden');
        byId('input-review').classList.remove('hidden');
        renderMask();
    };
    const loadMask = image => {
        maskCtx.imageSmoothingEnabled = false;
        maskCtx.drawImage(image, 0, 0, editor.width, editor.height);
        maskReady = true;
        renderMask();
    };
    const maskBlob = () => new Promise(resolve => {
        const exported = document.createElement('canvas');
        exported.width = originalWidth; exported.height = originalHeight;
        const exportCtx = exported.getContext('2d');
        exportCtx.imageSmoothingEnabled = false;
        exportCtx.drawImage(maskCanvas, 0, 0, originalWidth, originalHeight);
        exported.toBlob(resolve, 'image/png');
    });
    const responseJson = async response => {
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The image or removal mask is invalid.');
        return data;
    };
    const detect = async () => {
        if (!file || busy) return;
        const requestVersion = version;
        controller = new AbortController();
        busy = true;
        byId('loading-text').textContent = 'ESTIMATING REMOVAL AREA…';
        byId('mask-message').textContent = 'Estimating the covering. Review the marked area before generation.';
        changed();
        try {
            const form = new FormData(); form.append('file', file);
            const data = await responseJson(await fetch('/face/mask', { method: 'POST', body: form, signal: controller.signal }));
            const [source, mask] = await Promise.all([imageFrom(data.original), imageFrom(data.mask)]);
            if (version !== requestVersion) return;
            installOriginal(source);
            loadMask(mask);
            detectedMask = maskCtx.getImageData(0, 0, editor.width, editor.height);
            byId('mask-message').textContent = data.message;
        } catch (error) {
            if (version !== requestVersion || error.name === 'AbortError') return;
            maskReady = true;
            byId('mask-message').textContent = 'Automatic detection unavailable. Paint or import the covering area, or leave it empty for an uncovered face.';
            showError(error.message);
        } finally {
            if (version === requestVersion) { busy = false; syncControls(); }
        }
    };
    const acceptFile = async candidate => {
        if (!candidate) return;
        if (!['image/png', 'image/jpeg'].includes(candidate.type) || candidate.size > 10*1024*1024) {
            showError('Upload a PNG or JPEG smaller than 10 MB.'); return;
        }
        const requestVersion = ++version;
        if (controller) controller.abort();
        busy = false; drawing = false; brushVisible = false;
        clearResults(); showError(''); file = candidate; maskReady = false;
        const sourceUrl = URL.createObjectURL(candidate);
        try {
            const image = await imageFrom(sourceUrl);
            if (version !== requestVersion) return;
            if (Math.max(image.naturalWidth, image.naturalHeight) > 4096 || image.naturalWidth*image.naturalHeight > 16000000) {
                throw new Error('Use a face crop no larger than 4096 pixels per side.');
            }
            installOriginal(image);
            byId('input-status-label').textContent = 'REVIEW THE REMOVAL AREA';
            byId('review-confirm').checked = false;
            byId('input-review-decision').value = '';
            await detect();
        } catch (error) {
            if (version !== requestVersion) return;
            file = null; maskReady = false; byId('mask-review').classList.add('hidden'); byId('input-review').classList.add('hidden');
            byId('preview-image').classList.add('hidden');
            byId('drop-zone').querySelector('.drop-zone-ui').classList.remove('hidden');
            showError(error.message); syncControls();
        } finally { URL.revokeObjectURL(sourceUrl); }
    };
    byId('file-input').addEventListener('change', event => { acceptFile(event.target.files[0]); event.target.value = ''; });
    byId('drop-zone').addEventListener('click', () => byId('file-input').click());
    byId('drop-zone').addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); byId('file-input').click(); } });
    byId('drop-zone').addEventListener('dragover', event => { event.preventDefault(); byId('drop-zone').classList.add('drag-over'); });
    byId('drop-zone').addEventListener('dragleave', () => byId('drop-zone').classList.remove('drag-over'));
    byId('drop-zone').addEventListener('drop', event => { event.preventDefault(); byId('drop-zone').classList.remove('drag-over'); acceptFile(event.dataTransfer.files[0]); });
    byId('detect-btn').addEventListener('click', detect);
    ['paint', 'erase'].forEach(name => byId(name+'-btn').addEventListener('click', () => {
        tool = name;
        ['paint', 'erase'].forEach(value => { byId(value+'-btn').classList.toggle('active', value === name); byId(value+'-btn').setAttribute('aria-pressed', String(value === name)); });
    }));
    byId('brush-size').addEventListener('input', () => { byId('brush-value').textContent = byId('brush-size').value; renderMask(); });
    byId('clear-btn').addEventListener('click', () => { saveUndo(); maskCtx.fillStyle = '#000000'; maskCtx.fillRect(0, 0, editor.width, editor.height); maskReady = true; renderMask(); changed(); });
    byId('undo-btn').addEventListener('click', () => { const previous = history.pop(); if (previous) { maskCtx.putImageData(previous, 0, 0); renderMask(); changed(); } });
    byId('reset-btn').addEventListener('click', () => { if (detectedMask) { saveUndo(); maskCtx.putImageData(detectedMask, 0, 0); renderMask(); changed(); } });
    byId('review-confirm').addEventListener('change', syncControls);
    byId('input-review-decision').addEventListener('change', () => {
        clearResults(); syncControls();
        const choice = byId('input-review-decision').value;
        showError(choice === 'needs_clearer' ? 'Facial structure is insufficient. Upload a clearer face crop.' : choice === 'out_of_scope' ? 'Upload one already cropped frontal or mildly turned face.' : '');
    });
    byId('restoration-mode').addEventListener('change', clearResults);
    byId('mask-input').addEventListener('change', async event => {
        const candidate = event.target.files[0]; event.target.value = '';
        if (!candidate || !file || busy) return;
        const requestVersion = version;
        busy = true;
        byId('loading-text').textContent = 'LOADING REMOVAL AREA…';
        changed();
        const sourceUrl = URL.createObjectURL(candidate);
        try {
            if (candidate.size > 10*1024*1024) throw new Error('Use a removal mask smaller than 10 MB.');
            const image = await imageFrom(sourceUrl);
            if (version !== requestVersion) return;
            if (image.naturalWidth !== originalWidth || image.naturalHeight !== originalHeight) throw new Error('The mask must match the original image dimensions. Use black for visible and white for removed regions.');
            saveUndo(); loadMask(image); changed();
            byId('mask-message').textContent = 'Imported removal area. Review the green region before generation.';
        } catch (error) { if (version === requestVersion) showError(error.message); }
        finally {
            URL.revokeObjectURL(sourceUrl);
            if (version === requestVersion) { busy = false; syncControls(); }
        }
    });
    const pointFrom = event => {
        const rect = editor.getBoundingClientRect();
        return { x: Math.max(0, Math.min(editor.width, (event.clientX-rect.left)*editor.width/rect.width)), y: Math.max(0, Math.min(editor.height, (event.clientY-rect.top)*editor.height/rect.height)) };
    };
    const stroke = (from, to) => {
        maskCtx.strokeStyle = maskCtx.fillStyle = tool === 'paint' ? '#ffffff' : '#000000';
        maskCtx.lineWidth = Number(byId('brush-size').value)*editor.width/256;
        maskCtx.lineCap = 'round'; maskCtx.lineJoin = 'round';
        maskCtx.beginPath(); maskCtx.moveTo(from.x, from.y); maskCtx.lineTo(to.x, to.y); maskCtx.stroke();
        maskCtx.beginPath(); maskCtx.arc(to.x, to.y, maskCtx.lineWidth/2, 0, Math.PI*2); maskCtx.fill();
        maskReady = true; renderMask();
    };
    editor.addEventListener('pointerdown', event => {
        if (!original || busy || event.button !== 0) return;
        event.preventDefault(); editor.focus(); editor.setPointerCapture(event.pointerId);
        saveUndo(); changed(); drawing = true; lastPoint = brushPoint = pointFrom(event); brushVisible = true; stroke(lastPoint, lastPoint);
    });
    editor.addEventListener('pointermove', event => {
        if (!original || busy) return;
        brushPoint = pointFrom(event); brushVisible = true;
        if (drawing) { stroke(lastPoint, brushPoint); lastPoint = brushPoint; } else renderMask();
    });
    const finishStroke = () => { drawing = false; lastPoint = null; syncControls(); };
    editor.addEventListener('pointerup', finishStroke); editor.addEventListener('pointercancel', finishStroke);
    editor.addEventListener('lostpointercapture', finishStroke);
    editor.addEventListener('pointerleave', () => { if (!drawing) { brushVisible = false; renderMask(); } });
    editor.addEventListener('blur', () => { brushVisible = false; renderMask(); });
    editor.addEventListener('keydown', event => {
        if (!original || busy) return;
        const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (directions[event.key]) {
            event.preventDefault(); brushVisible = true;
            const step = (event.shiftKey ? 10 : 2)*editor.width/256;
            brushPoint.x = Math.max(0, Math.min(editor.width, brushPoint.x+directions[event.key][0]*step));
            brushPoint.y = Math.max(0, Math.min(editor.height, brushPoint.y+directions[event.key][1]*step)); renderMask();
        } else if (event.key === ' ' || event.key === 'Enter') {
            event.preventDefault(); saveUndo(); changed(); brushVisible = true; stroke(brushPoint, brushPoint); syncControls();
        }
    });
    byId('mask-download-btn').addEventListener('click', async () => {
        const blob = await maskBlob(); const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'removal-mask.png'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    const renderResult = data => {
        const cards = [['Original', data.original, 'Input resized to 256×256; native original in bundle'], ['Removal area', data.mask, 'White marks generated regions'], ['Face estimate', data.output, '256×256 plausible estimate']];
        results.replaceChildren(); results.classList.add('has-estimate');
        for (const [label, source, caption] of cards) {
            const card = document.createElement('article'); card.className = 'result-card';
            const header = document.createElement('div'); header.className = 'result-card-header';
            const title = document.createElement('h3'); title.className = 'rank-pill'; title.textContent = label; header.append(title);
            const holder = document.createElement('div'); holder.className = 'result-image-container';
            const image = document.createElement('img'); image.className = 'result-image workflow-image'; image.src = source; image.alt = label; holder.append(image);
            const detail = document.createElement('p'); detail.className = 'workflow-card-caption'; detail.textContent = caption;
            card.append(header, holder, detail); results.append(card);
        }
        const applied = data.metadata.restoration_applied ? 'DGP visible-region restoration applied.' : 'Resized visible input preserved; restoration off.';
        const maskSource = data.metadata.mask_source === 'automatic_reviewed' ? 'Reviewed automatic area.' : 'Assisted removal area.';
        byId('result-message').textContent = data.message+' '+applied+' '+maskSource+' Processing: '+data.metadata.elapsed_seconds.toFixed(1)+' seconds.';
        byId('result-message').classList.remove('hidden');
        byId('image-download').href = data.output;
        byId('bundle-download').classList.toggle('hidden', !data.bundle);
        if (data.bundle) byId('bundle-download').href = data.bundle;
        byId('result-downloads').classList.remove('hidden');
    };
    byId('analyze-btn').addEventListener('click', async () => {
        if (!file || !maskReady || busy || !byId('review-confirm').checked || byId('input-review-decision').value !== 'usable') return;
        const requestVersion = version;
        controller = new AbortController(); busy = true;
        byId('loading-text').textContent = 'LOADING MODELS / GENERATING ESTIMATE…';
        showError(''); clearResults(); syncControls();
        try {
            const form = new FormData(); form.append('file', file); form.append('mask', await maskBlob(), 'removal-mask.png');
            form.append('restoration', byId('restoration-mode').value); form.append('include_bundle', 'true');
            form.append('input_review', byId('input-review-decision').value); form.append('mask_reviewed', String(byId('review-confirm').checked));
            const data = await responseJson(await fetch('/face/generate', { method: 'POST', body: form, signal: controller.signal }));
            if (version === requestVersion) { renderResult(data); byId('input-status-label').textContent = 'ESTIMATE READY FOR REVIEW'; }
        } catch (error) { if (version === requestVersion && error.name !== 'AbortError') showError(error.message); }
        finally { if (version === requestVersion) { busy = false; syncControls(); } }
    });
    fetch('/face/status').then(responseJson).then(data => {
        byId('engine-status').textContent = 'DGP 256 // '+data.device.toUpperCase();
    }).catch(() => { byId('engine-status').textContent = 'CHECK LOCAL SERVER'; });
    syncControls();
});
