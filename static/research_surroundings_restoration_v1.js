(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const api = '/restoration-context-test';
    const fields = ['left', 'top', 'right', 'bottom'].map(name => byId('context-' + name));
    const canvas = byId('context-editor');
    const context = canvas.getContext('2d');
    let sourceFile, sourceImage, size, protection, prepared, busy = false, dragStart;
    const clearResults = () => {
        byId('context-results').classList.add('hidden');
        for (const id of ['context-bundle', 'download-original-context', 'download-prepared-context']) byId(id).removeAttribute('href');
        for (const id of ['result-original-input', 'result-prepared-input', 'result-original-context', 'result-prepared-context']) byId(id).removeAttribute('src');
    };
    const update = () => {
        byId('context-preview-run').disabled = busy || !sourceFile || !protection;
        byId('context-run').disabled = busy || !prepared || !byId('context-confirm').checked || byId('context-review').value !== 'usable';
    };
    const invalidate = () => {
        prepared = undefined;
        byId('context-confirm').checked = false;
        byId('context-preview-section').classList.add('hidden');
        for (const id of ['context-input-download', 'context-preparation-download']) byId(id).removeAttribute('href');
        clearResults(); update();
    };
    const error = message => {
        byId('context-error').textContent = message;
        byId('context-error').classList.toggle('hidden', !message);
    };
    const setBusy = value => {
        busy = value;
        byId('context-form').setAttribute('aria-busy', String(value));
        byId('context-form').querySelectorAll('input, select, button').forEach(control => { control.disabled = value; });
        update();
    };
    const post = async (path, extra = {}) => {
        const body = new FormData();
        body.append('file', sourceFile);
        for (const [name, value] of Object.entries(extra)) body.append(name, value);
        const response = await fetch(api + path, { method: 'POST', body });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be completed.');
        return result;
    };
    const rectangle = () => protection && protection.rectangle;
    const paint = () => {
        if (!sourceImage) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(sourceImage, 0, 0);
        const bounds = rectangle();
        if (!bounds) return;
        const [left, top, right, bottom] = bounds;
        context.fillStyle = 'rgba(16,185,129,.28)'; context.fillRect(left, top, right - left + 1, bottom - top + 1);
        context.strokeStyle = '#10b981'; context.lineWidth = Math.max(2, canvas.width / 180);
        context.strokeRect(left, top, right - left + 1, bottom - top + 1);
    };
    const setRectangle = bounds => {
        protection = { rectangle: bounds };
        fields.forEach((field, index) => { field.value = bounds[index]; });
        byId('context-protection-note').textContent = 'Manually selected rectangle. Preview it before confirming protection.';
        invalidate(); paint();
    };
    const inspect = async (file, preset = false) => {
        invalidate(); error(''); sourceFile = file; sourceImage = undefined; size = undefined; protection = undefined;
        fields.forEach(field => { field.value = ''; }); byId('context-review').value = '';
        byId('context-editor-section').classList.add('hidden');
        if (!file) { update(); return; }
        setBusy(true); byId('context-progress').textContent = 'Reading original pixels. No restoration runs during preparation.';
        try {
            const result = await post('/inspect');
            sourceImage = new Image(); sourceImage.src = result.source; await sourceImage.decode();
            size = [result.width, result.height]; canvas.width = size[0]; canvas.height = size[1];
            fields.forEach((field, index) => { field.max = size[index % 2] - 1; });
            byId('context-dimensions').textContent = size[0] + ' × ' + size[1] + ' original pixels. Bounds refer to this displayed orientation.';
            byId('context-editor-section').classList.remove('hidden');
            if (preset) {
                protection = { preset: 'reviewed-wild5' };
                byId('context-protection-note').textContent = 'Saved reviewed wild5 protection. Preview to see its exact shape. Drawing or editing fields replaces it with your rectangle.';
            } else byId('context-protection-note').textContent = 'No protection selected. Draw a rectangle or use the pixel fields.';
            paint(); byId('context-progress').textContent = 'Inspect the original, protect its face and existing blur, and preview.';
        } catch (problem) { sourceFile = undefined; error(problem.message); }
        finally { setBusy(false); }
    };
    byId('context-file').addEventListener('change', () => inspect(byId('context-file').files[0]));
    byId('context-example').addEventListener('click', async () => {
        if (busy) return; error(''); setBusy(true);
        try {
            const response = await fetch(api + '/example');
            if (!response.ok) throw new Error('The retained wild5 example is unavailable.');
            const file = new File([await response.blob()], 'wild5.png', { type: 'image/png' });
            byId('context-file').value = ''; await inspect(file, true);
        } catch (problem) { error(problem.message); }
        finally { setBusy(false); }
    });
    fields.forEach(field => field.addEventListener('input', () => {
        if (busy || !size) return;
        const bounds = fields.map(input => input.value === '' ? NaN : Number(input.value));
        if (bounds.every(Number.isInteger)) setRectangle(bounds);
        else { protection = undefined; invalidate(); paint(); }
    }));
    const position = event => {
        const bounds = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(size[0] - 1, Math.floor((event.clientX - bounds.left) * size[0] / bounds.width))),
            Math.max(0, Math.min(size[1] - 1, Math.floor((event.clientY - bounds.top) * size[1] / bounds.height)))];
    };
    canvas.addEventListener('pointerdown', event => {
        if (busy || !size) return;
        event.preventDefault(); dragStart = position(event); canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
        if (!dragStart || busy) return;
        const end = position(event);
        setRectangle([Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]), Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1])]);
    });
    const endDrag = event => {
        if (dragStart && !busy) {
            const end = position(event);
            setRectangle([Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]), Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1])]);
        }
        dragStart = undefined;
    };
    canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', () => { dragStart = undefined; });
    byId('context-center').addEventListener('click', () => {
        if (size) setRectangle([Math.floor(size[0] * .15), Math.floor(size[1] * .08), Math.floor(size[0] * .85), Math.floor(size[1] * .92)]);
    });
    byId('context-clear').addEventListener('click', () => {
        protection = undefined; fields.forEach(field => { field.value = ''; }); invalidate(); paint();
        byId('context-protection-note').textContent = 'No protection selected. Draw a rectangle or use the pixel fields.';
    });
    byId('context-preview-run').addEventListener('click', async () => {
        if (busy || !protection) return;
        invalidate(); setBusy(true); error('');
        byId('context-progress').textContent = 'Preparing outside-only blur and checking protected pixels. No model call.';
        try {
            prepared = await post('/preview', { protection: JSON.stringify(protection) });
            byId('context-protection').src = prepared.protection_overlay;
            byId('context-prepared').src = prepared.prepared_input;
            byId('context-input-download').href = prepared.prepared_input;
            byId('context-preparation-download').href = prepared.bundle;
            byId('context-preservation').textContent = 'Protected pixels verified unchanged. Added blur starts outside the 8-pixel guard and blends over 16 pixels. Its strength is approximate.';
            byId('context-preview-section').classList.remove('hidden');
            byId('context-progress').textContent = 'Review the full protected area before confirming. Appearance accuracy remains unknown.';
        } catch (problem) { prepared = undefined; error(problem.message); }
        finally { setBusy(false); }
    });
    byId('context-confirm').addEventListener('change', update);
    byId('context-review').addEventListener('change', () => {
        clearResults(); update();
        const choice = byId('context-review').value;
        error(choice === 'needs_clearer' ? 'Upload a clearer or less-covered face crop.' : choice === 'out_of_scope' ? 'Upload one already cropped frontal or mildly turned face.' : '');
    });
    byId('context-selection').addEventListener('change', clearResults);
    byId('context-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (busy || !prepared || !byId('context-confirm').checked || byId('context-review').value !== 'usable') return;
        clearResults(); error(''); setBusy(true); byId('context-progress').textContent = 'Restoring the two inputs locally. Two model forwards; keep this page open.';
        try {
            const result = await post('/compare', { protection: JSON.stringify(protection), preview_id: prepared.preview_id,
                confirmed: 'true', input_review: byId('context-review').value, selection: byId('context-selection').value });
            byId('result-original-input').src = result.original_input;
            byId('result-prepared-input').src = result.prepared_input;
            byId('result-original-context').src = result.original_result;
            byId('result-prepared-context').src = result.prepared_result;
            byId('download-original-context').href = result.original_result;
            byId('download-prepared-context').href = result.prepared_result;
            byId('context-bundle').href = result.bundle; byId('context-results').classList.remove('hidden');
            byId('context-progress').textContent = 'Comparison ready. Review visible facial changes before choosing an output.';
        } catch (problem) { error(problem.message); byId('context-progress').textContent = 'Comparison stopped. No automatic retry.'; }
        finally { setBusy(false); }
    });
    update();
})();
