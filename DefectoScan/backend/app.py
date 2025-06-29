from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import tensorflow as tf
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Try to load the model, but don't fail if it doesn't work
model = None
# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'model_mobilenetv2.h5')
print(f"Looking for model at: {model_path}")
try:
    # Try to load with custom_objects to handle compatibility issues
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("App will start but prediction functionality will be limited.")
    # Try alternative loading method
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        print("Model loaded with custom_objects!")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")

# Replace <db_password> with your actual MongoDB Atlas password
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
    print("App will work but predictions won't be saved to database.")

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
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .normal { background: #d4edda; color: #155724; }
            .pneumonia { background: #f8d7da; color: #721c24; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DefectoScan</h1>
            <h2>Chest X-Ray Analysis</h2>
            <p>Upload a chest X-ray image to analyze for pneumonia detection.</p>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button class="btn" onclick="document.getElementById('fileInput').click()">Choose Image</button>
                <p id="fileName"></p>
            </div>
            
            <button class="btn" onclick="analyzeImage()" id="analyzeBtn" disabled>Analyze Image</button>
            
            <div id="result"></div>
        </div>

        <script>
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
        </script>
    </body>
    </html>
    """
    return html

@app.route('/health')
def health():
    """Health check endpoint"""
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

        # Try to save to MongoDB if available
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
