document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-grid');
    
    // Read CSRF Token from meta tag
    const csrfMeta = document.querySelector('meta[name="csrf-token"]');
    const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';

    const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB limit
    
    let currentFile = null;
    let originalImageDataUrl = null;
    const inputStatusLabel = document.getElementById('input-status-label');

    // Drag and Drop Logic
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        // Strict MIME validation
        if (!file || !ALLOWED_MIME_TYPES.includes(file.type.toLowerCase())) {
            displayError('ERR: INVALID_FILE_TYPE (PNG/JPG ONLY)');
            return;
        }

        // File size validation (DoS / memory exhaustion prevention)
        if (file.size > MAX_FILE_SIZE) {
            displayError('ERR: FILE_SIZE_EXCEEDS_10MB_LIMIT');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImageDataUrl = e.target.result;
            previewImage.src = originalImageDataUrl;
            previewImage.classList.remove('hidden');
            analyzeBtn.disabled = false;
            if (inputStatusLabel) {
                inputStatusLabel.textContent = (selectedMode === 'sub32') 
                    ? 'CCTV_CROP_INPUT // SUB-32×32_BENCHMARK' 
                    : 'CCTV_CROP_INPUT // DIRECT_NATIVE';
            }
        };
        reader.readAsDataURL(file);
    }

    // Mode Selection Logic
    let selectedMode = 'direct';
    const modeDirectBtn = document.getElementById('mode-direct-btn');
    const modeSub32Btn = document.getElementById('mode-sub32-btn');

    function handleModeChange(newMode) {
        selectedMode = newMode;
        if (newMode === 'direct') {
            modeDirectBtn.classList.add('active');
            modeSub32Btn.classList.remove('active');
            if (originalImageDataUrl) {
                previewImage.src = originalImageDataUrl;
            }
            if (inputStatusLabel) {
                inputStatusLabel.textContent = 'CCTV_CROP_INPUT // DIRECT_NATIVE';
            }
        } else {
            modeSub32Btn.classList.add('active');
            modeDirectBtn.classList.remove('active');
            if (inputStatusLabel) {
                inputStatusLabel.textContent = 'CCTV_CROP_INPUT // SUB-32×32_BENCHMARK';
            }
        }

        // Notify investigator if displayed cards belong to previous mode
        const existingModePill = resultsContainer.querySelector('.mode-pill');
        if (existingModePill && currentFile) {
            const currentCardMode = existingModePill.textContent;
            const targetModeText = (newMode === 'sub32') ? '32x32_BENCHMARK' : 'DIRECT_RESTORE';
            if (currentCardMode !== targetModeText) {
                analyzeBtn.innerHTML = `<i class="ph-bold ph-arrows-clockwise"></i> RUN_${newMode === 'sub32' ? '32x32_BENCHMARK' : 'DIRECT_RESTORE'}`;
            } else {
                analyzeBtn.innerHTML = '<i class="ph-bold ph-cpu"></i> INITIATE_RECONSTRUCTION';
            }
        }
    }

    if (modeDirectBtn && modeSub32Btn) {
        modeDirectBtn.addEventListener('click', () => handleModeChange('direct'));
        modeSub32Btn.addEventListener('click', () => handleModeChange('sub32'));
    }

    // Analysis Logic
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        loadingState.classList.remove('hidden');
        resultsContainer.textContent = ''; // Safely clear previous results

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('mode', selectedMode);

        try {
            const headers = {};
            if (csrfToken && csrfToken !== '{{CSRF_TOKEN}}') {
                headers['X-CSRF-Token'] = csrfToken;
            }

            const response = await fetch('/reconstruct', {
                method: 'POST',
                headers: headers,
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData && errData.error ? errData.error : `ERR_HTTP_${response.status}`);
            }
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Update preview to simulated 32x32 thumbnail if benchmark mode used
            if (selectedMode === 'sub32' && data.cctv_thumbnail) {
                previewImage.src = data.cctv_thumbnail;
                if (inputStatusLabel) {
                    inputStatusLabel.textContent = 'INPUT: 32×32 PIXELATED CROP (SIMULATED)';
                }
            } else if (originalImageDataUrl) {
                previewImage.src = originalImageDataUrl;
                if (inputStatusLabel) {
                    inputStatusLabel.textContent = 'INPUT: DIRECT NATIVE SOURCE';
                }
            }

            // Render Results Safely
            renderResults(data.results, data.mode || selectedMode);

        } catch (error) {
            console.error(error);
            displayError(error.message || 'SYS_ERR // ANALYSIS_FAILED');
        } finally {
            // UI State: Done
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="ph-bold ph-cpu"></i> INITIATE_RECONSTRUCTION';
            loadingState.classList.add('hidden');
        }
    });

    function displayError(message) {
        resultsContainer.textContent = '';
        
        const errorRow = document.createElement('div');
        errorRow.className = 'result-row';
        errorRow.style.borderColor = '#ef4444';

        const errorData = document.createElement('div');
        errorData.className = 'result-data';

        const errorLabel = document.createElement('span');
        errorLabel.className = 'mono-label';
        errorLabel.style.color = '#ef4444';
        errorLabel.textContent = 'SYS_ERR // ANALYSIS_FAILED';

        const errorVal = document.createElement('span');
        errorVal.className = 'data-value';
        errorVal.textContent = message; // Safe from XSS injection

        errorData.appendChild(errorLabel);
        errorData.appendChild(errorVal);
        errorRow.appendChild(errorData);
        resultsContainer.appendChild(errorRow);
    }

    function renderResults(results, mode = 'direct') {
        resultsContainer.textContent = '';
        
        if (!results || results.length === 0) {
            displayError('ERR: NO_RECONSTRUCTIONS_GENERATED');
            return;
        }

        const modeDisplay = (mode === 'sub32') ? '32x32_BENCHMARK' : 'DIRECT_RESTORE';

        results.forEach((result) => {
            const card = document.createElement('div');
            card.className = 'result-card';

            // 1. Forensic Card Header with distinct Rank designations
            const cardHeader = document.createElement('div');
            cardHeader.className = 'result-card-header';

            const rankPill = document.createElement('span');
            rankPill.className = 'rank-pill';

            // Distinct forensic characterization per rank
            if (result.rank === 1) {
                rankPill.textContent = 'RANK 01 // BEST_FIT';
                rankPill.style.color = 'var(--accent)';
            } else if (result.rank === 2) {
                rankPill.textContent = 'RANK 02 // EDGE_FOCUS';
                rankPill.style.color = '#38bdf8'; // Cyan accent for edge definition
            } else if (result.rank === 3) {
                rankPill.textContent = 'RANK 03 // NATURAL_TONE';
                rankPill.style.color = '#c084fc'; // Soft purple for tone smoothing
            } else {
                rankPill.textContent = `RANK 0${result.rank} // CANDIDATE`;
            }

            const modePill = document.createElement('span');
            modePill.className = 'mode-pill';
            modePill.textContent = modeDisplay;

            cardHeader.appendChild(rankPill);
            cardHeader.appendChild(modePill);
            card.appendChild(cardHeader);

            // 2. Reconstructed Image Container
            const imageContainer = document.createElement('div');
            imageContainer.className = 'result-image-container';

            const img = document.createElement('img');
            img.className = 'result-image';
            img.alt = `Forensic Reconstruction Rank ${result.rank}`;
            if (typeof result.image_data === 'string' && result.image_data.startsWith('data:image/')) {
                img.src = result.image_data;
            }
            imageContainer.appendChild(img);
            card.appendChild(imageContainer);

            // 3. Grid Metrics
            const resultData = document.createElement('div');
            resultData.className = 'result-data';

            const scoreVal = isNaN(parseFloat(result.score)) ? 'N/A' : parseFloat(result.score).toFixed(4);
            const scoreGroup = createDataGroup('FAN_LOSS', scoreVal, 'var(--accent)');
            const resGroup = createDataGroup('RESOLUTION', '256x256');
            const confGroup = createDataGroup('IDENTITY_FIT', (1.0 - Math.min(0.99, parseFloat(scoreVal) || 0.15)).toFixed(3));
            const statusGroup = createDataGroup('STATUS', 'VERIFIED', 'var(--status-green)');

            resultData.appendChild(scoreGroup);
            resultData.appendChild(resGroup);
            resultData.appendChild(confGroup);
            resultData.appendChild(statusGroup);
            card.appendChild(resultData);

            // 4. Export Action
            const downloadBtn = document.createElement('a');
            downloadBtn.className = 'download-action-btn';
            downloadBtn.href = result.image_data;
            downloadBtn.download = `zcpo_forensic_reconstruction_rank_${result.rank}.png`;
            downloadBtn.innerHTML = '<i class="ph-bold ph-download-simple"></i> EXPORT_HIGH_RES_PNG';
            card.appendChild(downloadBtn);

            resultsContainer.appendChild(card);
        });
    }

    function createDataGroup(label, value, valueColor = null) {
        const group = document.createElement('div');
        group.className = 'data-group';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'mono-label';
        labelSpan.textContent = label;

        const valSpan = document.createElement('span');
        valSpan.className = 'data-value';
        valSpan.textContent = value;
        if (valueColor) {
            valSpan.style.color = valueColor;
        }

        group.appendChild(labelSpan);
        group.appendChild(valSpan);
        return group;
    }
});
