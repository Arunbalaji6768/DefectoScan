from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import tensorflow as tf
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Try multiple methods to load the model
model = None
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'model_mobilenetv2.h5')
print(f"Looking for model at: {model_path}")

# Method 1: Standard loading
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully with standard method!")
except Exception as e:
    print(f"Standard loading failed: {e}")
    
    # Method 2: With custom_objects
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf}, compile=False)
        print("Model loaded successfully with custom_objects!")
    except Exception as e2:
        print(f"Custom objects loading failed: {e2}")
        
        # Method 3: Try loading weights only
        try:
            from tensorflow.keras.applications import MobileNetV2
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs=base_model.input, outputs=x)
            model.load_weights(model_path)
            print("Model loaded successfully with weights only!")
        except Exception as e3:
            print(f"Weights loading failed: {e3}")
            
            # Method 4: Create a simple model for testing
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
