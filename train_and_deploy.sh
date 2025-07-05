#!/bin/bash
set -e

echo "Step 1: Ensure correct TensorFlow version..."
pip install tensorflow==2.13.0

echo "Step 2: Train the model..."
python DefectoScan/backend/model/train_mobilenetv2.py

echo "Step 3: Add model file to git..."
git add DefectoScan/backend/model/model_mobilenetv2.h5

echo "Step 4: Commit changes..."
git commit -m 'Add trained MobileNetV2 model'

echo "Step 5: Push to GitHub..."
git push

echo "All done! Now redeploy on Render." 