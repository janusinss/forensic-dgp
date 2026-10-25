(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const arms = ['original', 'r2', 'spatial_u', 'spatial_s'];
    let previewUrl;
    let available = {};
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
    const updateAvailability = () => {
        const selected = byId('research-selection').value;
        const enabled = selected === 'all' ? arms.every(arm => available[arm]) : available[selected];
        byId('research-run').disabled = busy || !enabled;
    };
    byId('research-file').addEventListener('change', () => {
        clearResults(); byId('research-review').value = ''; byId('research-error').classList.add('hidden');
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        const file = byId('research-file').files[0];
        byId('upload-preview').classList.toggle('hidden', !file);
        if (file) { previewUrl = URL.createObjectURL(file); byId('upload-preview').src = previewUrl; }
    });
    for (const id of ['research-review', 'research-selection']) byId(id).addEventListener('change', () => { clearResults(); updateAvailability(); });
    byId('research-form').addEventListener('submit', async event => {
        event.preventDefault(); clearResults();
        const payload = new FormData(event.target);
        const controls = ['research-file', 'research-review', 'research-selection'].map(byId);
        busy = true; updateAvailability(); controls.forEach(control => { control.disabled = true; });
        byId('research-error').classList.add('hidden'); byId('research-form').setAttribute('aria-busy', 'true');
        byId('research-progress').textContent = 'Restoring locally on CPU. Keep this page open.';
        try {
            const response = await fetch('/restoration-spatial-test/compare', { method: 'POST', body: payload });
            const result = await response.json();
            if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Comparison failed.');
            byId('result-input').src = result.original;
            for (const arm of arms) {
                const exists = Boolean(result.outputs[arm]);
                byId(arm + '-result').classList.toggle('hidden', !exists);
                if (exists) {
                    byId('result-' + arm).src = result.outputs[arm]; byId('download-' + arm).href = result.outputs[arm];
                    byId('label-' + arm).textContent = result.metadata.outputs[arm].metadata.label;
                    if (arm.startsWith('spatial_')) byId('status-' + arm).textContent = result.metadata.outputs[arm].metadata.quality_status;
                }
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
    fetch('/restoration-spatial-test/status').then(async response => {
        if (!response.ok) throw new Error('Cannot verify model availability.');
        const result = await response.json();
        byId('quality-status').textContent = result.quality_status;
        for (const arm of arms) {
            const row = result.models[arm]; available[arm] = Boolean(row && row.available);
            const option = Array.from(byId('research-selection').options).find(item => item.value === arm);
            option.disabled = !available[arm]; if (row) option.textContent = row.label;
        }
        byId('research-progress').textContent = result.binding_status || 'Ready for a cropped face with readable visible structure.';
        updateAvailability();
    }).catch(error => { byId('research-progress').textContent = error.message; });
})();
