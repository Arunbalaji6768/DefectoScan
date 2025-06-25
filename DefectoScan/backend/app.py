from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = os.path.join(os.getcwd(), 'model', 'model_mobilenetv2.h5')
model = load_model(model_path)

MONGO_URI = "mongodb+srv://prarunbalaji853:<db_password>@cluster0.k8zw842.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['defectoscan']
predictions_col = db['predictions']

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
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
        score = float(model.predict(img_tensor)[0][0])
        label = 'Pneumonia' if score >= 0.5 else 'Normal'
        confidence = score if score >= 0.5 else (1 - score)

        record = {
            'filename': filename,
            'label': label,
            'confidence': round(confidence, 4),
            'timestamp': datetime.utcnow()
        }
        result = predictions_col.insert_one(record)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        os.remove(filepath)

    return jsonify({
        'label': label,
        'confidence': round(confidence, 4),
        'id': str(result.inserted_id)
    }), 201

if __name__ == '__main__':
    app.run(debug=True)
