#!/usr/bin/env bash
# exit on error
set -o errexit

# Install numpy first with explicit version
pip install numpy==1.26.2

# Install other dependencies
pip install Flask==3.0.0
pip install gunicorn==21.2.0
pip install tensorflow-cpu==2.15.0
pip install pillow==10.1.0
pip install pymongo==4.6.0
pip install requests==2.31.0

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 