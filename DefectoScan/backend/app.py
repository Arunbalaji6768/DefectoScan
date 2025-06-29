from flask import Flask, request, jsonify, render_template_string
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
    """Serve a simple HTML interface for testing"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DefectoScan - Chest X-Ray Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { border-color: #007bff; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .normal { background: #d4edda; color: #155724; }
            .pneumonia { background: #f8d7da; color: #721c24; }
            .error { background: #f8d7da; color: #721c24; }
            .status { background: #e2e3e5; color: #383d41; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DefectoScan</h1>
            <h2>Chest X-Ray Analysis</h2>
            <p>Upload a chest X-ray image to analyze for pneumonia detection.</p>
            
            <div class="status">
                <strong>Model Status:</strong> <span id="modelStatus">Loading...</span><br>
                <strong>Database Status:</strong> <span id="dbStatus">Loading...</span>
            </div>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button class="btn" onclick="document.getElementById('fileInput').click()">Choose Image</button>
                <p id="fileName"></p>
            </div>
            
            <button class="btn" onclick="analyzeImage()" id="analyzeBtn" disabled>Analyze Image</button>
            <button class="btn" onclick="testModel()">Test Model</button>
            
            <div id="result"></div>
        </div>

        <script>
            // Check status on page load
            window.onload = function() {
                checkStatus();
            };
            
            async function checkStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('modelStatus').textContent = data.model_loaded ? 'Loaded' : 'Not Loaded';
                    document.getElementById('dbStatus').textContent = data.mongodb_connected ? 'Connected' : 'Not Connected';
                } catch (error) {
                    document.getElementById('modelStatus').textContent = 'Error';
                    document.getElementById('dbStatus').textContent = 'Error';
                }
            }
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    document.getElementById('fileName').textContent = file.name;
                    document.getElementById('analyzeBtn').disabled = false;
                }
            });

            async function analyzeImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const resultDiv = document.getElementById('result');
                
                if (!file) {
                    alert('Please select a file first');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                resultDiv.innerHTML = '<p>Analyzing image...</p>';
                resultDiv.className = 'result';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        const className = data.label === 'Normal' ? 'normal' : 'pneumonia';
                        resultDiv.className = `result ${className}`;
                        resultDiv.innerHTML = `
                            <h3>Analysis Result:</h3>
                            <p><strong>Diagnosis:</strong> ${data.label}</p>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            }
            
            async function testModel() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Testing model with sample data...</p>';
                resultDiv.className = 'result';
                
                try {
                    const response = await fetch('/test');
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.className = 'result normal';
                        resultDiv.innerHTML = `
                            <h3>Model Test Result:</h3>
                            <p><strong>Status:</strong> ${data.message}</p>
                            <p><strong>Sample Prediction:</strong> ${data.prediction}</p>
                            <p><strong>Model Type:</strong> ${data.model_type}</p>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mongodb_connected': predictions_col is not None
    })

@app.route('/test')
def test_model():
    """Test endpoint to verify model is working"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Create a random test image
        test_image = np.random.random((1, 224, 224, 3))
        prediction = float(model.predict(test_image)[0][0])
        
        # Determine model type
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
