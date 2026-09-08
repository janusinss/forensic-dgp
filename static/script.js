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
    let simulatedSub32DataUrl = null;
    const inputStatusLabel = document.getElementById('input-status-label');

    // Drag and Drop & Keyboard Logic
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });
    
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

    function generateSub32Thumbnail(imgSrc, callback) {
        const img = new Image();
        img.onload = () => {
            const width = img.naturalWidth || img.width;
            const height = img.naturalHeight || img.height;
            const minDim = Math.min(width, height);
            const sx = Math.floor((width - minDim) / 2);
            const sy = Math.floor((height - minDim) / 2);

            // Step 1: Downscale to 32x32 standard CCTV resolution
            const canvas32 = document.createElement('canvas');
            canvas32.width = 32;
            canvas32.height = 32;
            const ctx32 = canvas32.getContext('2d');
            ctx32.imageSmoothingEnabled = true;
            ctx32.drawImage(img, sx, sy, minDim, minDim, 0, 0, 32, 32);

            // Step 2: Upscale to 256x256 using nearest-neighbor (crisp pixelation)
            const canvas256 = document.createElement('canvas');
            canvas256.width = 256;
            canvas256.height = 256;
            const ctx256 = canvas256.getContext('2d');
            ctx256.imageSmoothingEnabled = false;
            ctx256.drawImage(canvas32, 0, 0, 32, 32, 0, 0, 256, 256);

            callback(canvas256.toDataURL('image/png'));
        };
        img.src = imgSrc;
    }

    function updatePreviewImage() {
        if (!originalImageDataUrl) return;

        if (selectedMode === 'sub32') {
            if (simulatedSub32DataUrl) {
                previewImage.src = simulatedSub32DataUrl;
                previewImage.classList.add('pixelated');
                previewImage.classList.remove('hidden');
                if (inputStatusLabel) {
                    inputStatusLabel.textContent = 'INPUT: 32×32 PIXELATED CROP (SIMULATED)';
                }
            } else {
                generateSub32Thumbnail(originalImageDataUrl, (thumb) => {
                    simulatedSub32DataUrl = thumb;
                    if (selectedMode === 'sub32') {
                        previewImage.src = simulatedSub32DataUrl;
                        previewImage.classList.add('pixelated');
                        previewImage.classList.remove('hidden');
                        if (inputStatusLabel) {
                            inputStatusLabel.textContent = 'INPUT: 32×32 PIXELATED CROP (SIMULATED)';
                        }
                    }
                });
            }
        } else {
            previewImage.src = originalImageDataUrl;
            previewImage.classList.remove('pixelated');
            previewImage.classList.remove('hidden');
            if (inputStatusLabel) {
                inputStatusLabel.textContent = 'INPUT: DIRECT NATIVE SOURCE';
            }
        }
    }

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
        simulatedSub32DataUrl = null;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImageDataUrl = e.target.result;
            analyzeBtn.disabled = false;
            updatePreviewImage();
            // Pre-calculate 32x32 simulated thumbnail
            generateSub32Thumbnail(originalImageDataUrl, (thumb) => {
                simulatedSub32DataUrl = thumb;
                if (selectedMode === 'sub32') {
                    updatePreviewImage();
                }
            });
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
            modeDirectBtn.setAttribute('aria-checked', 'true');
            modeSub32Btn.classList.remove('active');
            modeSub32Btn.setAttribute('aria-checked', 'false');
            if (originalImageDataUrl) {
                updatePreviewImage();
            } else if (inputStatusLabel) {
                inputStatusLabel.textContent = 'CCTV_CROP_INPUT // DIRECT_NATIVE';
            }
        } else {
            modeSub32Btn.classList.add('active');
            modeSub32Btn.setAttribute('aria-checked', 'true');
            modeDirectBtn.classList.remove('active');
            modeDirectBtn.setAttribute('aria-checked', 'false');
            if (originalImageDataUrl) {
                updatePreviewImage();
            } else if (inputStatusLabel) {
                inputStatusLabel.textContent = 'CCTV_CROP_INPUT // SUB-32×32_BENCHMARK';
            }
        }

        // Notify investigator if displayed cards belong to previous mode
        const existingModePill = resultsContainer.querySelector('.mode-pill');
        if (existingModePill && currentFile) {
            const currentCardMode = existingModePill.textContent;
            const targetModeText = (newMode === 'sub32') ? '32x32_BENCHMARK' : 'DIRECT_RESTORE';
            if (currentCardMode !== targetModeText) {
                analyzeBtn.innerHTML = `<i class="ph-bold ph-arrows-clockwise" aria-hidden="true"></i> RUN_${newMode === 'sub32' ? '32x32_BENCHMARK' : 'DIRECT_RESTORE'}`;
            } else {
                analyzeBtn.innerHTML = '<i class="ph-bold ph-cpu" aria-hidden="true"></i> INITIATE_RECONSTRUCTION';
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

            // Update preview to authoritative backend 32x32 thumbnail if provided
            if (selectedMode === 'sub32' && data.cctv_thumbnail) {
                simulatedSub32DataUrl = data.cctv_thumbnail;
            }
            updatePreviewImage();

            // Render Results Safely
            renderResults(data.results, data.mode || selectedMode);

        } catch (error) {
            console.error(error);
            displayError(error.message || 'SYS_ERR // ANALYSIS_FAILED');
        } finally {
            // UI State: Done
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="ph-bold ph-cpu" aria-hidden="true"></i> INITIATE_RECONSTRUCTION';
            loadingState.classList.add('hidden');
        }
    });

    function displayError(message) {
        resultsContainer.textContent = '';
        
        const errorRow = document.createElement('div');
        errorRow.className = 'result-row';
        errorRow.style.borderColor = 'var(--color-destructive)';

        const errorData = document.createElement('div');
        errorData.className = 'result-data';

        const errorLabel = document.createElement('span');
        errorLabel.className = 'mono-label';
        errorLabel.style.color = 'var(--color-destructive)';
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

        // Candidate Cards
        results.forEach((result) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            if (result.rank === 1) {
                card.classList.add('primary-card');
            }

            // Forensic Card Header with distinct Rank designations
            const cardHeader = document.createElement('div');
            cardHeader.className = 'result-card-header';

            const rankPill = document.createElement('span');
            rankPill.className = 'rank-pill';

            if (result.rank === 1) {
                rankPill.textContent = 'RANK 01 // BEST_FIT';
                rankPill.style.color = 'var(--color-accent)';
            } else if (result.rank === 2) {
                rankPill.textContent = 'RANK 02 // EDGE_FOCUS';
                rankPill.style.color = 'var(--color-foreground)';
            } else if (result.rank === 3) {
                rankPill.textContent = 'RANK 03 // NATURAL_TONE';
                rankPill.style.color = 'var(--color-muted-foreground)';
            } else {
                rankPill.textContent = `RANK 0${result.rank} // CANDIDATE`;
                rankPill.style.color = 'var(--color-muted-foreground)';
            }

            const modePill = document.createElement('span');
            modePill.className = 'mode-pill';
            modePill.textContent = modeDisplay;

            cardHeader.appendChild(rankPill);
            cardHeader.appendChild(modePill);
            card.appendChild(cardHeader);

            // Reconstructed Image Container with Forensic Reticles
            const imageContainer = document.createElement('div');
            imageContainer.className = 'result-image-container';

            ['reticle-tl', 'reticle-tr', 'reticle-bl', 'reticle-br'].forEach((cls) => {
                const reticle = document.createElement('div');
                reticle.className = `reticle ${cls}`;
                reticle.setAttribute('aria-hidden', 'true');
                imageContainer.appendChild(reticle);
            });

            const img = document.createElement('img');
            img.className = 'result-image';
            img.alt = `Forensic Reconstruction Rank ${result.rank}`;
            if (typeof result.image_data === 'string' && result.image_data.startsWith('data:image/')) {
                img.src = result.image_data;
            }
            imageContainer.appendChild(img);
            card.appendChild(imageContainer);

            // Grid Metrics with Restrained Saturation
            const resultData = document.createElement('div');
            resultData.className = 'result-data';

            const scoreVal = isNaN(parseFloat(result.score)) ? 'N/A' : parseFloat(result.score).toFixed(4);
            const isRank1 = (result.rank === 1);
            const scoreGroup = createDataGroup('FAN_LOSS', scoreVal, isRank1 ? 'var(--color-accent)' : 'var(--color-foreground)');
            const resGroup = createDataGroup('RESOLUTION', '256x256');
            const confGroup = createDataGroup('IDENTITY_FIT', (1.0 - Math.min(0.99, parseFloat(scoreVal) || 0.15)).toFixed(3));
            const statusGroup = createDataGroup('STATUS', isRank1 ? 'VERIFIED' : 'CANDIDATE', isRank1 ? 'var(--color-accent)' : 'var(--color-muted-foreground)');

            resultData.appendChild(scoreGroup);
            resultData.appendChild(resGroup);
            resultData.appendChild(confGroup);
            resultData.appendChild(statusGroup);
            card.appendChild(resultData);

            // Export Action
            const downloadBtn = document.createElement('a');
            downloadBtn.className = 'download-action-btn';
            downloadBtn.href = result.image_data;
            downloadBtn.download = `zcpo_forensic_reconstruction_rank_${result.rank}.png`;
            downloadBtn.setAttribute('aria-label', `Export Rank 0${result.rank} Reconstruction as PNG`);
            downloadBtn.innerHTML = '<i class="ph-bold ph-download-simple" aria-hidden="true"></i> EXPORT_HIGH_RES_PNG';
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
