import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from face_extractor import FaceExtractor
from frame_extractor import FrameExtractor
from fft_analysis import FFTAnalyzer
from gradcam import GradCAM

app = Flask(__name__)

# --- Load Model & Tools ---
MODEL_PATH = 'model.h5'
# If model doesn't exist, we'll use a placeholder or handle the error
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print("Warning: model.h5 not found. Please train the model first.")

face_extractor = FaceExtractor()
gradcam = GradCAM(model) if model else None

# Ensure output directories exist
os.makedirs('static/results', exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join('static/results', filename)
    file.save(filepath)

    # 1. Handle Image or Video
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(filepath)
    else:
        # For video, extract the middle frame for simplicity in this demo
        frames = FrameExtractor.extract_frames(filepath, n_frames=1)
        if not frames:
            return jsonify({'error': 'Could not extract frames'}), 400
        image = frames[0]

    # 2. Extract Face
    face = face_extractor.extract_face(image)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400

    # 3. Preprocess for Model
    face_input = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_input = cv2.resize(face_input, (224, 224))
    face_input = face_input / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    # 4. Model Prediction
    if model:
        prediction_score = model.predict(face_input)[0][0]
        prediction_label = "Fake" if prediction_score > 0.5 else "Real"
        confidence = float(prediction_score if prediction_score > 0.5 else 1 - prediction_score) * 100
    else:
        # Mock response if model is not trained
        prediction_label = "Fake"
        confidence = 85.0

    # 5. Explainability & Analysis
    # A. FFT Spectrum
    fft_filename = f'fft_{filename}.png'
    fft_path = os.path.join('static/results', fft_filename)
    FFTAnalyzer.save_spectrum_plot(face, fft_path)

    # B. Grad-CAM
    heatmap_filename = f'heatmap_{filename}.png'
    heatmap_path = os.path.join('static/results', heatmap_filename)
    if gradcam:
        heatmap = gradcam.compute_heatmap(face_input)
        overlay = gradcam.overlay_heatmap(heatmap, face)
        cv2.imwrite(heatmap_path, overlay)
    else:
        # Fallback placeholder
        cv2.imwrite(heatmap_path, face)

    return jsonify({
        'prediction': prediction_label,
        'confidence': confidence,
        'heatmap_url': f'/static/results/{heatmap_filename}',
        'fft_url': f'/static/results/{fft_filename}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
