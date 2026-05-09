import os
import cv2
import numpy as np
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not found. Inference features will be disabled.")

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

# Modular imports from utils/
from utils.face_extractor import FaceExtractor
from utils.frame_extractor import FrameExtractor
from utils.fft_analysis import FFTAnalyzer
from utils.gradcam import GradCAM

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
MODEL_PATH = 'model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Global Components ---
# Initialize ML tools once to save memory/time
face_extractor = FaceExtractor()
model = None
gradcam = None

def load_ml_components():
    global model, gradcam
    if not HAS_TENSORFLOW:
        print("Skipping ML component loading: TensorFlow is not available.")
        return

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            gradcam = GradCAM(model)
            print("ML Components loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: model.h5 not found. System will run in 'Simulated Mode'.")

load_ml_components()

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_face(face_img):
    """Prepares extracted face for model prediction."""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)

# --- Endpoints ---

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Check the health and mode of the system."""
    return jsonify({
        'status': 'online',
        'tensorflow_available': HAS_TENSORFLOW,
        'model_loaded': model is not None,
        'mode': 'Full Integrity' if (HAS_TENSORFLOW and model is not None) else 'Simulation'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction pipeline:
    1. Validate & Save Upload
    2. Extract Frames (if video)
    3. Extract & Preprocess Face
    4. Model Inference
    5. Generate Analytics (Heatmap, FFT)
    6. Return JSON response
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Use PNG, JPG, or MP4.'}), 400

    # Securely save the file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 1. Processing Pipeline
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # Extract multiple frames for a more robust analysis
            frames = FrameExtractor.extract_frames(filepath, n_frames=5)
            if not frames:
                return jsonify({'error': 'Failed to process video frames.'}), 400
            
            # We'll use the middle frame for the visual heatmaps/FFT
            image = frames[len(frames)//2]
            
            # Multi-frame inference
            scores = []
            for frame in frames:
                face_tmp = face_extractor.extract_face(frame)
                if face_tmp is not None:
                    face_input_tmp = preprocess_face(face_tmp)
                    if model:
                        scores.append(float(model.predict(face_input_tmp)[0][0]))
                    else:
                        # Simulation: random slight variance around a "Real" base
                        scores.append(0.01 + np.random.uniform(0, 0.02))
            
            if not scores:
                return jsonify({'error': 'No human face detected in any of the video frames.'}), 400
            
            prediction_score = sum(scores) / len(scores)
        else:
            image = cv2.imread(filepath)
            # 2. Face Extraction (for image)
            face = face_extractor.extract_face(image)
            if face is None:
                return jsonify({'error': 'No human face detected in the image.'}), 400
            
            face_input = preprocess_face(face)
            if model:
                prediction_score = float(model.predict(face_input)[0][0])
            else:
                prediction_score = 0.008

        # 3. Final Prediction Logic
        prediction_label = "Fake" if prediction_score > 0.5 else "Real"
        confidence = float(prediction_score if prediction_score > 0.5 else 1 - prediction_score) * 100
        
        # Prepare the face for visuals (if it hasn't been extracted yet for image)
        if not is_video:
            visual_face = face
            visual_input = face_input
        else:
            # For video, extract face from the selected middle frame
            visual_face = face_extractor.extract_face(image)
            visual_input = preprocess_face(visual_face)

        # 4. Generate Analysis Assets
        # A. FFT Analysis
        fft_filename = f"fft_{filename}.png"
        fft_path = os.path.join(app.config['RESULTS_FOLDER'], fft_filename)
        FFTAnalyzer.save_spectrum_plot(visual_face, fft_path)

        # B. Grad-CAM (Spatial Analysis)
        heatmap_filename = f"heatmap_{filename}.png"
        heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], heatmap_filename)
        if gradcam:
            heatmap = gradcam.compute_heatmap(visual_input)
            overlay = gradcam.overlay_heatmap(heatmap, visual_face)
            cv2.imwrite(heatmap_path, overlay)
        else:
            # Fallback to saving the cropped face
            cv2.imwrite(heatmap_path, visual_face)

        # 5. Build Response
        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'processed_image': f'/static/uploads/{filename}',
            'heatmap_url': f'/static/results/{heatmap_filename}',
            'fft_url': f'/static/results/{fft_filename}',
            'simulation_mode': not (HAS_TENSORFLOW and model is not None)
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': 'Internal server error during processing.'}), 500

# Route to serve static files if needed (though Flask does this by default)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Run server
    print("DeepShield Backend starting on http://127.0.0.1:5050")
    app.run(host='0.0.0.0', port=5050, debug=True)
