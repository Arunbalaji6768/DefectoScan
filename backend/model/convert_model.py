import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

IMG_SIZE = (224, 224)

# Recreate the model architecture
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Load weights from the .h5 file
model.load_weights('backend/model/model_mobilenetv2.h5')

# Save as .keras format
keras_path = os.path.abspath('backend/model/model_mobilenetv2.keras')
model.save(keras_path)
print(f"Model architecture recreated, weights loaded, and saved as: {keras_path}") 