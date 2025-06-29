#!/usr/bin/env bash
# Aggressive Python 3.10.11 build script
set -o errexit

echo "=== AGGRESSIVELY FORCING PYTHON 3.10.11 ENVIRONMENT ==="

# Force Python 3.10.11 environment variables
export PYTHON_VERSION=3.10.11
export PYTHONPATH="/opt/render/project/src"

# Upgrade pip and install setuptools first
pip install --upgrade pip
pip install setuptools==65.0.0 wheel==0.40.0

# Install Python 3.10.11 compatible packages one by one
echo "Installing Python 3.10.11 compatible packages..."

# Core packages
pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install keras==2.13.1

# Web framework
pip install Flask==2.3.3
pip install Werkzeug==2.3.7
pip install gunicorn==21.2.0

# Database and utilities
pip install pymongo==4.5.0
pip install requests==2.31.0
pip install python-dotenv==1.0.0

# Image and data processing
pip install pillow==10.0.0
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install scipy==1.11.1

# Create necessary directories
mkdir -p uploads
mkdir -p DefectoScan/uploads

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== PYTHON 3.10.11 AGGRESSIVE BUILD COMPLETE ==="
echo "Python version: $(python --version)" 