from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import tensorflow as tf
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)

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
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("App will start but prediction functionality will be limited.")

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
    app.run(debug=True)
