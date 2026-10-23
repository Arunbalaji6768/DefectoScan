#!/bin/bash
# The trained model is now committed to the repo as backend/model/model.onnx (~10 MB),
# so nothing needs to be downloaded at build time.
set -e
test -f backend/model/model.onnx && echo "model.onnx present" || { echo "backend/model/model.onnx is missing"; exit 1; }
