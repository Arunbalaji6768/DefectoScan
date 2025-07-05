from tensorflow.keras.models import load_model, save_model

# Load the original model (trained/saved in newer TF)
model = load_model("DefectoScan/backend/model/model_mobilenetv2.h5", compile=False)

# Re-save in a format compatible with TF 2.13
save_model(model, "DefectoScan/backend/model/model_compatible.h5", include_optimizer=False)

print("Model re-saved as model_compatible.h5 for TensorFlow 2.13 compatibility.") 