# Render Deployment Guide for DefectoScan

## Deployment Steps

1. **Connect your GitHub repository to Render**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure the Web Service**
   - **Name**: `defectoscan` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT backend.app:app`
   - **Root Directory**: Leave empty (or set to `DefectoScan` if needed)

3. **Environment Variables** (Optional)
   - `TF_CPP_MIN_LOG_LEVEL=2` (reduces TensorFlow logging)
   - `PYTHONPATH=/opt/render/project/src`

## Troubleshooting

### If you get TensorFlow compatibility errors:

1. **Use the CPU-only version**:
   - Rename `requirements-render.txt` to `requirements.txt`
   - This uses `tensorflow-cpu==2.11.0` instead of the full TensorFlow

2. **Alternative: Use even older versions**:
   ```txt
   tensorflow==2.10.0
   keras==2.10.0
   ```

3. **Check Python version compatibility**:
   - Ensure `runtime.txt` contains `python-3.10.10`
   - TensorFlow 2.11.0 works well with Python 3.10

### Common Issues and Solutions:

1. **"No matching distribution found for tensorflow"**
   - Solution: Use `tensorflow-cpu` instead of `tensorflow`
   - Or downgrade to TensorFlow 2.10.0

2. **Model loading issues**
   - Ensure your model file is in the correct path: `backend/model/model_mobilenetv2.h5`
   - Check that the model file is committed to your repository

3. **Memory issues**
   - Use `tensorflow-cpu` to reduce memory usage
   - Consider using a larger instance type on Render

## File Structure for Render

Your repository should have this structure:
```
DefectoScan/
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── model_mobilenetv2.h5
│   └── uploads/
├── requirements.txt
├── runtime.txt
└── Procfile
```

## Build Script (Optional)

If you want to use a custom build script:
1. Rename `build.sh` to `render-build.sh`
2. In Render settings, set Build Command to: `./render-build.sh`

## Monitoring

- Check Render logs for any errors
- Monitor memory usage (TensorFlow can be memory-intensive)
- Ensure your MongoDB connection string is correct

## Performance Tips

1. **Use CPU-only TensorFlow** for better compatibility
2. **Enable caching** if possible
3. **Monitor memory usage** and upgrade instance if needed
4. **Use environment variables** for configuration 