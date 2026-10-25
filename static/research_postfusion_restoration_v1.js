(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const arms = ['incumbent_U96', 'postfusion_LR_u64'];
    let previewUrl;
    let available = false;
    let busy = false;
    const clearResults = () => {
        byId('research-results').classList.add('hidden');
        byId('research-bundle').removeAttribute('href');
        byId('result-input').removeAttribute('src');
        for (const arm of arms) {
            byId('download-' + arm).removeAttribute('href');
            byId('result-' + arm).removeAttribute('src');
        }
    };
    const updateAvailability = () => { byId('research-run').disabled = busy || !available; };
    byId('research-file').addEventListener('change', () => {
        clearResults(); byId('research-review').value = ''; byId('research-error').classList.add('hidden');
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = undefined;
        const file = byId('research-file').files[0];
        byId('upload-preview').classList.toggle('hidden', !file);
        byId('upload-preview').removeAttribute('src');
        if (file) { previewUrl = URL.createObjectURL(file); byId('upload-preview').src = previewUrl; }
    });
    byId('research-review').addEventListener('change', clearResults);
    window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
    byId('research-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (busy) return;
        clearResults();
        const payload = new FormData(event.target);
        const controls = ['research-file', 'research-review'].map(byId);
        busy = true; updateAvailability(); controls.forEach(control => { control.disabled = true; });
        byId('research-error').classList.add('hidden'); byId('research-form').setAttribute('aria-busy', 'true');
        byId('research-progress').textContent = 'Restoring both models locally on CPU. Keep this page open.';
        try {
            const response = await fetch('/restoration-postfusion-test/compare', { method: 'POST', body: payload });
            const result = await response.json();
            if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Comparison failed.');
            byId('result-input').src = result.original;
            for (const arm of arms) {
                byId('result-' + arm).src = result.outputs[arm];
                byId('download-' + arm).href = result.outputs[arm];
                byId('label-' + arm).textContent = result.metadata.outputs[arm].label;
            }
            byId('research-bundle').href = result.bundle; byId('research-results').classList.remove('hidden');
            byId('research-progress').textContent = 'Comparison ready. ' + result.metadata.seconds.toFixed(1) + ' seconds. ' + result.metadata.quality_status;
        } catch (error) {
            byId('research-error').textContent = error.message; byId('research-error').classList.remove('hidden');
            byId('research-progress').textContent = '';
        } finally {
            busy = false; controls.forEach(control => { control.disabled = false; });
            byId('research-form').removeAttribute('aria-busy'); updateAvailability();
        }
    });
    fetch('/restoration-postfusion-test/status').then(async response => {
        if (!response.ok) throw new Error('Cannot verify the experimental comparison.');
        const result = await response.json();
        byId('quality-status').textContent = result.quality_status;
        available = arms.every(arm => result.models[arm] && result.models[arm].available);
        byId('research-progress').textContent = result.binding_status;
        updateAvailability();
    }).catch(error => { byId('research-progress').textContent = error.message; });
})();
