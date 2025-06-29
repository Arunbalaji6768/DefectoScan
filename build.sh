#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 