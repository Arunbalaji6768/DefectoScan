it import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

IMG_SIZE = (224, 224)
model = tf.keras.models.load_model('backend/model/model_mobilenetv2.keras')
test_dir = 'data/Chest Xray Dataset/chest_xray/chest_xray/test'

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=1, class_mode='binary', shuffle=False
)

count = 0
for i in range(len(test_data.filepaths)):
    img_path = test_data.filepaths[i]
    true_label_idx = test_data.classes[i]
    true_label = list(test_data.class_indices.keys())[list(test_data.class_indices.values()).index(true_label_idx)]
    if true_label == 'PNEUMONIA' and count < 10:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        pred_label = 'Pneumonia' if prediction >= 0.3 else 'Normal'
        print(f"{os.path.basename(img_path)} | True: {true_label} | Pred: {pred_label} | Confidence: {prediction:.4f}")
        count += 1 