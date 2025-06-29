#!/usr/bin/env bash
# Force Python 3.10.11 build script
set -o errexit

echo "=== FORCING PYTHON 3.10.11 ENVIRONMENT ==="

# Check Python version
python --version

# Upgrade pip
pip install --upgrade pip

# Install setuptools first to avoid build issues
pip install setuptools wheel

# Install Python 3.10.11 compatible packages one by one
echo "Installing Python 3.10.11 compatible packages..."

pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install keras==2.13.1
pip install Flask==2.3.3
pip install Werkzeug==2.3.7
pip install gunicorn==21.2.0
pip install pymongo==4.5.0
pip install requests==2.31.0
pip install pillow==10.0.0
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install scipy==1.11.1
pip install python-dotenv==1.0.0

# Create necessary directories
mkdir -p uploads

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== PYTHON 3.10.11 BUILD COMPLETE ===" 