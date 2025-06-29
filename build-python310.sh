#!/usr/bin/env bash
# exit on error
set -o errexit

# Force Python 3.10.11 environment
echo "Setting up Python 3.10.11 environment..."

# Install Python 3.10.11 compatible dependencies
pip install --upgrade pip
pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install keras==2.13.1
pip install Flask==2.3.3
pip install gunicorn==21.2.0
pip install pymongo==4.5.0
pip install requests==2.31.0
pip install pillow==10.0.0
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install scipy==1.11.1

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Python 3.10.11 environment setup complete!" 