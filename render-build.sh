#!/bin/bash
# Force Python 3.10.11 for Render
set -e

echo "Forcing Python 3.10.11 environment..."

# Set Python version environment variable
export PYTHON_VERSION=3.10.11

# Upgrade pip
pip install --upgrade pip

# Install setuptools and wheel first
pip install setuptools==65.0.0 wheel==0.40.0

# Install requirements
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads

echo "Python 3.10.11 build complete!" 