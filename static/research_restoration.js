(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    let previewUrl;
    const clearResults = () => {
        byId('research-results').classList.add('hidden');
        for (const id of ['download-original', 'download-r2', 'research-bundle']) byId(id).removeAttribute('href');
        for (const id of ['result-input', 'result-original', 'result-r2']) byId(id).removeAttribute('src');
    };
    byId('research-file').addEventListener('change', () => {
        clearResults(); byId('research-review').value = ''; byId('research-error').classList.add('hidden');
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        const file = byId('research-file').files[0];
        byId('upload-preview').classList.toggle('hidden', !file);
        if (file) { previewUrl = URL.createObjectURL(file); byId('upload-preview').src = previewUrl; }
    });
    byId('research-form').addEventListener('submit', async event => {
        event.preventDefault(); clearResults();
        byId('research-error').classList.add('hidden');
        byId('research-run').disabled = true;
        byId('research-progress').textContent = 'Restoring locally on CPU. Keep this page open.';
        try {
            const response = await fetch('/restoration-test/compare', { method: 'POST', body: new FormData(event.target) });
            const result = await response.json();
            if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Comparison failed.');
            byId('result-input').src = result.original;
            for (const arm of ['original', 'r2']) {
                const exists = Boolean(result.outputs[arm]);
                byId(arm + '-result').classList.toggle('hidden', !exists);
                if (exists) { byId('result-' + arm).src = result.outputs[arm]; byId('download-' + arm).href = result.outputs[arm]; }
            }
            byId('research-bundle').href = result.bundle;
            byId('research-results').classList.remove('hidden');
            byId('research-progress').textContent = 'Comparison ready. ' + result.metadata.seconds.toFixed(1) + ' seconds. R2 remains experimental.';
        } catch (error) {
            byId('research-error').textContent = error.message;
            byId('research-error').classList.remove('hidden');
            byId('research-progress').textContent = '';
        } finally { byId('research-run').disabled = false; }
    });
})();
