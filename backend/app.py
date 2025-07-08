from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from datetime import datetime
import os
import numpy as np
from routes.otp import otp_bp
from routes.oauth import oauth_bp

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_super_secret_key_123')
CORS(app)
app.register_blueprint(otp_bp)
app.register_blueprint(oauth_bp)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path_keras = os.path.join(current_dir, 'model', 'model_mobilenetv2.keras')
model_path_h5 = os.path.join(current_dir, 'model', 'model_mobilenetv2.h5')
print(f"Looking for model at: {model_path_keras} or {model_path_h5}")

try:
    model = tf.keras.models.load_model(model_path_keras)
    print("Model loaded successfully from .keras file!")
except Exception as e_keras:
    print(f"Loading .keras model failed: {e_keras}")
    try:
        model = tf.keras.models.load_model(model_path_h5, compile=False)
        print("Model loaded successfully from .h5 file!")
    except Exception as e_h5:
        print(f"Loading .h5 model failed: {e_h5}")
    try:
        IMG_SIZE = (224, 224)
        base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        if os.path.exists(model_path_h5):
            try:
                model.load_weights(model_path_h5)
                print("Model created and weights loaded successfully!")
            except Exception as e2:
                print(f"Loading weights failed: {e2}")
                print("Using model with ImageNet weights only.")
        else:
            print("Model file not found. Using model with ImageNet weights only.")
    except Exception as e3:
        print(f"Creating model failed: {e3}")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            print("Created fallback model for testing!")
        except Exception as e4:
            print(f"Fallback model creation failed: {e4}")

MONGO_URI = os.environ.get('MONGO_URI', "mongodb+srv://prarunbalaji853:Arun6768@cluster0.k8zw842.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = None
db = None
predictions_col = None
try:
    client = MongoClient(MONGO_URI)
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
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
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
        test_image = np.random.random((1, 224, 224, 3))
        prediction = float(model.predict(test_image)[0][0])
        model_type = "MobileNetV2 with ImageNet weights"
        if "Sequential" in str(type(model)):
            if "MobileNetV2" in str(model.layers[0]):
                model_type = "MobileNetV2 (trained weights)" if model.layers[0].trainable else "MobileNetV2 (ImageNet weights)"
            else:
                model_type = "Simple CNN (fallback)"
        return jsonify({
            'message': 'Model is working correctly',
            'prediction': f'{prediction:.4f}',
            'model_type': model_type,
            'test_image_shape': test_image.shape
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
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.imsave('debug_uploaded_image.png', np.squeeze(img_tensor) if img_tensor.shape[-1]==3 else img_tensor[0])
            print("Saved debug_uploaded_image.png for inspection.")
        except Exception as debug_img_exc:
            print(f"Could not save debug image: {debug_img_exc}")
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
        score = float(model.predict(img_tensor)[0][0])
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