import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing import image

np.random.seed(42)
tf.random.set_seed(42)

# Update paths to be relative to the root directory
train_dir = 'data/Chest Xray Dataset/chest_xray/chest_xray/train'
test_dir  = 'data/Chest Xray Dataset/chest_xray/chest_xray/test'

if not os.path.exists(train_dir): raise FileNotFoundError(train_dir)
if not os.path.exists(test_dir): raise FileNotFoundError(test_dir)

IMG_SIZE = (224, 224) 
BATCH_SIZE = 16
EPOCHS = 20

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

print("Class Indices:", train_data.class_indices)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
]

history1 = model.fit(train_data, validation_data=test_data, epochs=EPOCHS, class_weight=class_weights_dict, callbacks=callbacks)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_data, validation_data=test_data, epochs=EPOCHS, class_weight=class_weights_dict, callbacks=callbacks)

loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc:.4f}")

model_dir = os.getcwd()
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'model_mobilenetv2.h5'))
print("Model saved!")

history_data = {
    'phase1': history1.history,
    'phase2': history2.history
}
with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
    json.dump(history_data, f, indent=2)
print("Training history saved.")

for i in range(5):
    img_path = test_data.filepaths[i]
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = 'Pneumonia' if prediction >= 0.5 else 'Normal'
    print(f"{os.path.basename(img_path)} → Pred: {label}, Confidence: {prediction:.4f}")
