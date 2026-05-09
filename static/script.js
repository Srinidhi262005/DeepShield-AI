document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const filePreview = document.getElementById('filePreview');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const dropZoneContent = document.querySelector('.drop-zone-content');

    // UI Elements for updates
    const predictionBadge = document.getElementById('predictionBadge');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const heatmapImg = document.getElementById('heatmapImg');
    const fftImg = document.getElementById('fftImg');

    let selectedFile = null;

    // Trigger file input on click
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and Drop handlers
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
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
            alert('Please upload an image or video file.');
            return;
        }

        selectedFile = file;
        submitBtn.disabled = false;
        
        // Clear previous preview
        filePreview.innerHTML = '';
        filePreview.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');

        const reader = new FileReader();
        reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.src = e.target.result;
                filePreview.appendChild(img);
            } else {
                const video = document.createElement('video');
                video.src = e.target.result;
                video.controls = true;
                filePreview.appendChild(video);
            }
        };
        reader.readAsDataURL(file);

        // Hide results if a new file is picked
        resultsSection.classList.add('hidden');
    }

    // Submit handler
    submitBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // UI Loading State
        setLoading(true);
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please ensure the backend is running at /predict.');
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        const btnText = submitBtn.querySelector('.btn-text');
        const loader = submitBtn.querySelector('.loader');
        
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = 'Analyzing...';
            loader.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            btnText.textContent = 'Analyze Media';
            loader.classList.add('hidden');
        }
    }

    // System Status Check
    async function checkStatus() {
        const simulationBanner = document.getElementById('simulationBanner');
        try {
            const response = await fetch('/status');
            const data = await response.json();
            if (data.mode === 'Simulation') {
                console.warn('DeepShield is running in Simulation Mode.');
                simulationBanner.classList.remove('hidden');
            } else {
                simulationBanner.classList.add('hidden');
            }
        } catch (e) {
            console.error('Could not connect to backend status endpoint.');
        }
    }
    checkStatus();

    function displayResults(data) {
        setLoading(false);
        resultsSection.classList.remove('hidden');

        // Update Prediction
        const isFake = data.prediction.toLowerCase() === 'fake';
        predictionBadge.textContent = data.prediction + (data.simulation_mode ? ' (Simulated)' : '');
        predictionBadge.className = 'badge ' + (isFake ? 'badge-fake' : 'badge-real');
        if (data.simulation_mode) {
            predictionBadge.style.opacity = '0.8';
            predictionBadge.title = "TensorFlow is not available on this environment. Result is based on FFT heuristic or mock logic.";
        }

        // Update Confidence
        const confidence = parseFloat(data.confidence).toFixed(2);
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence}%`;
        
        // Set bar color based on result
        confidenceBar.style.background = isFake ? 'var(--fake)' : 'var(--real)';
        confidenceBar.style.boxShadow = `0 0 10px ${isFake ? 'var(--fake)' : 'var(--real)'}`;

        // Update Images
        heatmapImg.src = data.heatmap_url + '?t=' + new Date().getTime();
        fftImg.src = data.fft_url + '?t=' + new Date().getTime();

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
