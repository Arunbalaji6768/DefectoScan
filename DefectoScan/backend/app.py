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

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Try to load the model using the same architecture as train_mobilenetv2.py
model = None
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'model_mobilenetv2.h5')
print(f"Looking for model at: {model_path}")

# Method 1: Try to load the saved model
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully from saved file!")
except Exception as e:
    print(f"Loading saved model failed: {e}")
    
    # Method 2: Create the model using the same architecture as train_mobilenetv2.py
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
        
        # Try to load weights if the model file exists
        if os.path.exists(model_path):
            try:
                model.load_weights(model_path)
                print("Model created and weights loaded successfully!")
            except Exception as e2:
                print(f"Loading weights failed: {e2}")
                print("Using model with ImageNet weights only.")
        else:
            print("Model file not found. Using model with ImageNet weights only.")
            
    except Exception as e3:
        print(f"Creating model failed: {e3}")
        
        # Method 3: Create a simple fallback model
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

# MongoDB setup
MONGO_URI = "mongodb+srv://prarunbalaji853:Arun6768@cluster0.k8zw842.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
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

    try:
        img_tensor = preprocess(filepath)
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
        
        score = float(model.predict(img_tensor)[0][0])
        label = 'Pneumonia' if score >= 0.5 else 'Normal'
        confidence = score if score >= 0.5 else (1 - score)

        # Save to MongoDB if available
        record_id = None
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

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        os.remove(filepath)

    response_data = {
        'label': label,
        'confidence': round(confidence, 4)
    }
    
    if record_id:
        response_data['id'] = record_id

    return jsonify(response_data), 201

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
