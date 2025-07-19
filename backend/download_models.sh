#!/bin/bash
set -e
mkdir -p backend/model

# Download .h5 file
pip install gdown
# Download model_mobilenetv2.h5
# File ID: 1KvdAYE4ioKIAwn5R4zHR-kTcxb_Em0iW
gdown --id 1KvdAYE4ioKIAwn5R4zHR-kTcxb_Em0iW -O backend/model/model_mobilenetv2.h5

# Download model_mobilenetv2.keras
# File ID: 19XHzNIiIihW_vDjWK0WX2KyXsqbX0cou
gdown --id 19XHzNIiIihW_vDjWK0WX2KyXsqbX0cou -O backend/model/model_mobilenetv2.keras 