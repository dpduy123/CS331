// Coffee Bean Pseudo-Labeling Tool - Frontend JavaScript

// State management
const state = {
    selectedFiles: [],
    results: [],
    sessionId: null,
    currentModel: null
};

// Model Selection
async function selectModel(modelKey) {
    const statusEl = document.getElementById('currentModelStatus');
    statusEl.innerHTML = 'Loading model...';

    try {
        const response = await fetch('/models/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelKey })
        });

        const data = await response.json();

        if (data.success) {
            state.currentModel = data.current;
            statusEl.innerHTML = `Current: <strong>${getModelName(data.current)}</strong>`;

            // Update UI
            document.querySelectorAll('.model-option').forEach(el => {
                el.classList.remove('active');
                if (el.dataset.model === modelKey) {
                    el.classList.add('active');
                }
            });

            alert(data.message);
        } else {
            statusEl.innerHTML = `<span class="warning">Error: ${data.error}</span>`;
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error selecting model:', error);
        statusEl.innerHTML = `<span class="warning">Error loading model</span>`;
        alert('Error selecting model: ' + error.message);
    }
}

function getModelName(key) {
    const names = {
        'yolov8n': 'YOLOv8n (Nano)',
        'yolov8s': 'YOLOv8s (Small)',
        'ssvit': 'SSViT-YOLOv11n'
    };
    return names[key] || key;
}

// Class colors (RGB) - Must match CSS settings in style.css
const CLASS_COLORS = {
    'barely-riped': 'rgb(255, 0, 0)',      // Red (#FF0000)
    'over-riped': 'rgb(117, 16, 16)',      // Dark Red (#751010)
    'riped': 'rgb(255, 165, 0)',           // Orange (#FFA500)
    'semi-riped': 'rgb(255, 255, 0)',      // Yellow (#FFFF00)
    'unriped': 'rgb(0, 255, 0)'            // Green (#00FF00)
};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const fileCount = document.getElementById('fileCount');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const downloadAllBtn = document.getElementById('downloadAllBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingProgress = document.getElementById('loadingProgress');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
processBtn.addEventListener('click', processImages);
downloadAllBtn.addEventListener('click', downloadAllLabels);
clearBtn.addEventListener('click', clearResults);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files).filter(isValidFile);
    addFiles(files);
}

// File Selection Handler
function handleFileSelect(e) {
    const files = Array.from(e.target.files).filter(isValidFile);
    addFiles(files);
}

// Validate file type
function isValidFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
    return validTypes.includes(file.type);
}

// Add files to selection
function addFiles(files) {
    state.selectedFiles = [...state.selectedFiles, ...files];
    updatePreview();
}

// Update preview grid
function updatePreview() {
    if (state.selectedFiles.length === 0) {
        previewSection.hidden = true;
        processBtn.disabled = true;
        return;
    }

    previewSection.hidden = false;
    processBtn.disabled = false;
    fileCount.textContent = state.selectedFiles.length;

    previewGrid.innerHTML = '';

    state.selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'preview-item';

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.onload = () => URL.revokeObjectURL(img.src);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeFile(index);
        };

        item.appendChild(img);
        item.appendChild(removeBtn);
        previewGrid.appendChild(item);
    });
}

// Remove file from selection
function removeFile(index) {
    state.selectedFiles.splice(index, 1);
    updatePreview();
}

// Process images
async function processImages() {
    if (state.selectedFiles.length === 0) return;

    showLoading();
    loadingProgress.textContent = `Processing ${state.selectedFiles.length} image(s)...`;

    const formData = new FormData();
    state.selectedFiles.forEach(file => {
        formData.append('files[]', file);
    });

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        state.sessionId = data.session_id;
        state.results = data.results;

        displayResults();

        // Clear selection
        state.selectedFiles = [];
        updatePreview();
        fileInput.value = '';

    } catch (error) {
        console.error('Error processing images:', error);
        alert('Error processing images: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults() {
    if (state.results.length === 0) {
        resultsSection.hidden = true;
        return;
    }

    resultsSection.hidden = false;
    resultsGrid.innerHTML = '';

    state.results.forEach((result, index) => {
        const card = createResultCard(result, index);
        resultsGrid.appendChild(card);
    });
}

// Create result card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.dataset.index = index;

    if (result.error) {
        card.classList.add('error');
        card.innerHTML = `
            <div class="result-image-container">
                <span>Error: ${result.error}</span>
            </div>
            <div class="result-info">
                <div class="result-filename">${result.filename}</div>
            </div>
        `;
        return card;
    }

    // Image container (clickable for fullscreen)
    const imageContainer = document.createElement('div');
    imageContainer.className = 'result-image-container clickable';
    imageContainer.title = 'Click to view fullscreen';
    imageContainer.onclick = () => openModal(index);

    const img = document.createElement('img');
    img.src = `/static/results/${result.result_image}`;
    img.alt = result.filename;
    imageContainer.appendChild(img);

    // Info section
    const info = document.createElement('div');
    info.className = 'result-info';

    // Filename
    const filename = document.createElement('div');
    filename.className = 'result-filename';
    filename.textContent = result.filename;

    // Stats
    const stats = document.createElement('div');
    stats.className = 'result-stats';
    stats.innerHTML = `
        <span>Detections: ${result.num_detections}</span>
    `;

    // Detection list
    const detectionList = document.createElement('div');
    detectionList.className = 'detection-list';

    if (result.detections.length === 0) {
        detectionList.innerHTML = '<div class="no-detections">No detections found</div>';
    } else {
        result.detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';

            const color = CLASS_COLORS[det.class_name] || 'rgb(255, 255, 255)';

            item.innerHTML = `
                <span class="detection-class" style="--color: ${color}">${det.class_name}</span>
                <span class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</span>
            `;
            detectionList.appendChild(item);
        });
    }

    // Actions
    const actions = document.createElement('div');
    actions.className = 'result-actions';

    // Accept checkbox
    const acceptLabel = document.createElement('label');
    acceptLabel.className = 'accept-checkbox';

    const acceptCheckbox = document.createElement('input');
    acceptCheckbox.type = 'checkbox';
    acceptCheckbox.checked = result.num_detections > 0;
    acceptCheckbox.dataset.index = index;
    acceptCheckbox.addEventListener('change', (e) => {
        card.classList.toggle('accepted', e.target.checked);
    });

    acceptLabel.appendChild(acceptCheckbox);
    acceptLabel.appendChild(document.createTextNode('Accept Label'));

    // Download label button
    const downloadLabelBtn = document.createElement('button');
    downloadLabelBtn.className = 'btn btn-small btn-secondary';
    downloadLabelBtn.textContent = 'Download Label';
    downloadLabelBtn.onclick = () => downloadSingleLabel(result);

    // Download image button
    const downloadImgBtn = document.createElement('button');
    downloadImgBtn.className = 'btn btn-small btn-secondary';
    downloadImgBtn.textContent = 'Download Image';
    downloadImgBtn.onclick = () => downloadImage(result.result_image);

    // View Detail button (opens fullscreen modal)
    const viewDetailBtn = document.createElement('button');
    viewDetailBtn.className = 'btn btn-small btn-primary';
    viewDetailBtn.textContent = 'View Detail';
    viewDetailBtn.onclick = (e) => {
        e.stopPropagation();
        openModal(index);
    };

    actions.appendChild(acceptLabel);
    actions.appendChild(viewDetailBtn);
    actions.appendChild(downloadLabelBtn);
    actions.appendChild(downloadImgBtn);

    // Assemble card
    info.appendChild(filename);
    info.appendChild(stats);
    info.appendChild(detectionList);
    info.appendChild(actions);

    card.appendChild(imageContainer);
    card.appendChild(info);

    // Auto-accept if detections found
    if (result.num_detections > 0) {
        card.classList.add('accepted');
    }

    return card;
}

// Download single label
function downloadSingleLabel(result) {
    const blob = new Blob([result.label_content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = result.filename.replace(/\.[^/.]+$/, '') + '.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Download image
function downloadImage(filename) {
    const a = document.createElement('a');
    a.href = `/download_image/${filename}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Download all accepted labels
async function downloadAllLabels() {
    const acceptedLabels = [];

    document.querySelectorAll('.result-card').forEach((card, index) => {
        const checkbox = card.querySelector('input[type="checkbox"]');
        if (checkbox && checkbox.checked && state.results[index] && !state.results[index].error) {
            acceptedLabels.push({
                filename: state.results[index].filename,
                content: state.results[index].label_content
            });
        }
    });

    if (acceptedLabels.length === 0) {
        alert('No labels accepted. Please check at least one image.');
        return;
    }

    try {
        const response = await fetch('/download_all', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                labels: acceptedLabels
            })
        });

        if (!response.ok) {
            throw new Error('Download failed');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `labels_${state.sessionId}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error downloading labels:', error);
        alert('Error downloading labels: ' + error.message);
    }
}

// Clear results
function clearResults() {
    state.results = [];
    state.sessionId = null;
    resultsSection.hidden = true;
    resultsGrid.innerHTML = '';
}

// Loading overlay
function showLoading() {
    loadingOverlay.hidden = false;
}

function hideLoading() {
    loadingOverlay.hidden = true;
}

// Modal Elements
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalFilename = document.getElementById('modalFilename');
const modalDetections = document.getElementById('modalDetections');

// Open fullscreen modal with box-only image
function openModal(resultIndex) {
    console.log('openModal called with index:', resultIndex);
    const result = state.results[resultIndex];
    console.log('Result:', result);
    if (!result || result.error) {
        console.log('No result or error, returning');
        return;
    }

    // Request box-only image from backend
    const boxOnlyUrl = `/boxonly/${result.result_image}`;
    console.log('Box only URL:', boxOnlyUrl);

    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    const fname = document.getElementById('modalFilename');
    const det = document.getElementById('modalDetections');

    console.log('Modal element:', modal);

    if (modal && img && fname && det) {
        img.src = boxOnlyUrl;
        fname.textContent = result.filename;
        det.textContent = `${result.num_detections} detections`;
        modal.hidden = false;
        document.body.style.overflow = 'hidden';
        console.log('Modal opened');
    } else {
        console.error('Modal elements not found');
    }
}

// Close modal
function closeModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.hidden = true;
        document.body.style.overflow = ''; // Restore scrolling
    }
}

// Close modal on Escape key
document.addEventListener('keydown', function(e) {
    const modal = document.getElementById('imageModal');
    if (e.key === 'Escape' && modal && !modal.hidden) {
        closeModal();
    }
});

// Initialize on page load - ensure loading overlay is hidden
document.addEventListener('DOMContentLoaded', function() {
    hideLoading();
});
