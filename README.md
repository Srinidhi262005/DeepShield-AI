# DeepShield – AI Deepfake Detector 🛡️

DeepShield is a high-performance, full-stack deepfake detection system. It leverages state-of-the-art Convolutional Neural Networks (XceptionNet) combined with frequency analysis and explainable AI (Grad-CAM) to identify synthetic face manipulations in images and videos.

## ✨ Features

-   **Multi-Media Analysis**: Supports both image and video uploads.
-   **Neural Integrity Check**: Powered by a pre-trained Xception model optimized for face-forensics.
-   **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to visualize where the AI detects anomalies.
-   **Frequency Domain Analysis**: Uses Fast Fourier Transform (FFT) to identify GAN-generated artifacts.
-   **Modern Dashboard**: A sleek, responsive glassmorphic UI built for speed and clarity.

## 🏗️ Architecture

-   **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (Fetch API).
-   **Backend**: Flask (Python) with a modular utility structure.
-   **ML Pipeline**: 
    -   Face Extraction (OpenCV)
    -   Classification (TensorFlow/Keras)
    -   Explainability (Grad-CAM)
    -   Frequency Analysis (FFT)

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
Access the application at `http://127.0.0.1:5000`.

## ☁️ Deployment (Render)

1.  **Repository**: Push your code to GitHub.
2.  **New Web Service**: Create a new Web Service on [Render](https://render.com).
3.  **Environment**: 
    -   **Runtime**: Python 3
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `gunicorn app:app`
4.  **Environment Variables**: Ensure `PYTHON_VERSION` is set to `3.9.0` or higher.

## 🤝 Contributing
DeepShield is an open-source project aimed at digital integrity. Feel free to submit PRs for improved detection models or better UI features.

## 📜 License
MIT License. Created by the DeepShield AI Team.
