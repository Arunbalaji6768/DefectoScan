import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import onnxruntime as ort
from PIL import Image
from datetime import datetime
import numpy as np
from backend.routes.otp import otp_bp
from backend.routes.oauth import oauth_bp

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_super_secret_key_123')
CORS(app)
app.register_blueprint(otp_bp)
app.register_blueprint(oauth_bp)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_dir = os.path.dirname(os.path.abspath(__file__))
# ONNX export of the trained MobileNetV2 pneumonia classifier (~10 MB, runs on onnxruntime
# instead of TensorFlow so it fits in Render's free 512 MB instances).
MODEL_PATH = os.path.join(current_dir, 'model', 'model.onnx')
print(f"Looking for model at: {MODEL_PATH}")

model = None
MODEL_INPUT = None
try:
    _opts = ort.SessionOptions()
    _opts.intra_op_num_threads = 1
    _opts.inter_op_num_threads = 1
    model = ort.InferenceSession(MODEL_PATH, sess_options=_opts, providers=['CPUExecutionProvider'])
    MODEL_INPUT = model.get_inputs()[0].name
    print("ONNX model loaded successfully!")
except Exception as e_onnx:
    print(f"Loading ONNX model failed: {e_onnx}")


def run_model(arr):
    """arr: float32 array of shape (1, 224, 224, 3) scaled to [0, 1]. Returns sigmoid score."""
    out = model.run(None, {MODEL_INPUT: arr.astype(np.float32)})[0]
    return float(np.asarray(out).reshape(-1)[0])


MONGO_URI = os.environ.get('MONGO_URI')  # set in the host's environment; never commit credentials
client = None
db = None
predictions_col = None
try:
    if not MONGO_URI:
        raise RuntimeError('MONGO_URI not set - predictions will not be saved')
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['defectoscan']
    predictions_col = db['predictions']
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"Warning: Could not connect to MongoDB: {e}")

CLASS_INDICES = {'NORMAL': 0, 'PNEUMONIA': 1}
try:
    if hasattr(model, 'class_indices'):
        CLASS_INDICES = model.class_indices
except Exception:
    pass

def preprocess(img_path):
    # Same as the training pipeline: RGB, 224x224 nearest-neighbour resize, scaled to [0, 1]
    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
def index():
    return jsonify({
        'message': 'DefectoScan API is running',
        'model_loaded': model is not None,
        'endpoints': ['/predict', '/health']
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mongodb_connected': predictions_col is not None
    })

@app.route('/test')
def test_model():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = run_model(test_image)
        return jsonify({
            'message': 'Model is working correctly',
            'prediction': f'{prediction:.4f}',
            'model_type': 'MobileNetV2 (trained weights, ONNX Runtime)',
            'test_image_shape': list(test_image.shape)
        })
    except Exception as e:
        return jsonify({'error': f'Model test failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    label = None
    confidence = None
    record_id = None
    try:
        img_tensor = preprocess(filepath)
        print(f"Filepath: {filepath}")
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Image tensor min/max: {img_tensor.min()}/{img_tensor.max()}")
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
        score = run_model(img_tensor)
        threshold = 0.5
        if CLASS_INDICES.get('NORMAL', 0) == 0:
            label = 'Pneumonia' if score >= threshold else 'Normal'
            confidence = score if score >= threshold else (1 - score)
        else:
            label = 'Normal' if score >= threshold else 'Pneumonia'
            confidence = score if score >= threshold else (1 - score)
        print(f"DEBUG: score={score}, label={label}, confidence={confidence}")
        if predictions_col is not None:
            try:
                record = {
                    'filename': filename,
                    'label': label,
                    'confidence': round(confidence, 4),
                    'timestamp': datetime.utcnow()
                }
                result = predictions_col.insert_one(record)
                record_id = str(result.inserted_id)
            except Exception as db_error:
                print(f"Warning: Could not save to database: {db_error}")
        response_data = {
            'label': label,
            'confidence': round(confidence, 4)
        }
        if record_id:
            response_data['id'] = record_id
        return jsonify(response_data), 201
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return predict()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)