# DeepShield – AI Deepfake Detector 🛡️

DeepShield is a high-performance, full-stack deepfake detection system. It leverages state-of-the-art Convolutional Neural Networks (XceptionNet) combined with frequency analysis and explainable AI (Grad-CAM) to identify synthetic face manipulations in images and videos.

## ✨ Features

-   **Multi-Frame Video Robustness**: Analyzes multiple frames per second to ensure detection consistency across the entire video.
-   **Neural Integrity Check**: Powered by a pre-trained Xception model optimized for face-forensics.
-   **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to visualize where the AI detects anomalies.
-   **Advanced Frequency Domain Analysis**: Uses Fast Fourier Transform (FFT) with Magma-spectrum visualization to identify GAN-generated checkerboard artifacts.
-   **Simulation & Resilience Mode**: Robust architecture that remains functional even in resource-constrained environments (Simulation Mode).
-   **Modern Dashboard**: A sleek, responsive glassmorphic UI built for speed and clarity.

## 🏗️ Architecture & Pipeline

DeepShield employs a multi-stage pipeline for digital integrity verification:

1.  **Face Acquisition**: OpenCV-based Haar Cascades or MTCNN are used to isolate facial regions.
2.  **Spatial Analysis (Deep Learning)**: An XceptionNet-based CNN analyzes the spatial pixels for manipulation signatures.
3.  **Frequency Analysis (Signal Processing)**: FFT is used to detect high-frequency noise that is characteristic of AI generation but invisible to the human eye.
4.  **Ensemble Scoring**: Results from multiple frames and analysis modes are aggregated for a final confidence score.
5.  **Interpretability**: Grad-CAM is applied to provide a visual 'why' behind the AI's decision.

## 🚀 Getting Started

### 1. Prerequisites
-   Python 3.9 or higher
-   Pip package manager

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Training (Optional)
If you have a dataset (e.g., FaceForensics++), place it in `data/` and run:
```bash
python train.py
```
This will generate `model.h5`. If this file is missing, the system runs in **Simulation Mode**.

### 4. Running the App
Start the Flask development server:
```bash
python app.py
```
Access the application at `http://127.0.0.1:5050`.

## ☁️ Deployment (Render)

1.  **Repository**: Push your code to GitHub.
2.  **Deploy**:
    -   **Fastest**: Use Render "Blueprint" deploy (this repo includes `render.yaml`).
    -   Or create a new Web Service manually on [Render](https://render.com).
3.  **Environment**: 
    -   **Runtime**: Python 3
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
4.  **Environment Variables**: Set `PYTHON_VERSION` to `3.10.x` (recommended).

## 🤝 Contributing
DeepShield is an open-source project aimed at digital integrity. Feel free to submit PRs for improved detection models or better UI features.

## 👤 Author
**Kaiytha Srinidhi Reddy**  
Connect on [LinkedIn](https://www.linkedin.com/in/kaiytha-srinidhi-reddy-27a655282/)

## 📜 License
MIT License. Created by the DeepShield AI Team.
