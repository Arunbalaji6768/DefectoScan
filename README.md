# DefectoScan


> **Demo Video:**
> [![Watch the demo](demo-video-placeholder.png)](https://your-demo-video-link-here)

## Overview
DefectoScan is a full-stack application for automated chest X-ray analysis using deep learning. It provides a user-friendly interface for uploading X-ray images and delivers instant predictions for pneumonia detection. The project features secure authentication, a modern React frontend, and a Flask backend with TensorFlow-powered inference.

## Features
- Upload and analyze chest X-ray images
- Deep learning model (MobileNetV2) for pneumonia detection
- Google OAuth login
- RESTful API backend
- MongoDB integration for results storage
- Responsive, modern UI (React + Tailwind CSS)

## Tech Stack
- **Frontend:** React, Vite, Tailwind CSS, React Router, Google OAuth
- **Backend:** Flask, TensorFlow, MongoDB, scikit-learn

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm 8+

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python app.py
```

### Frontend Setup
```bash
# Install frontend dependencies
cd DefectoScan/frontend
npm install

# Start the development server
npm run dev
```

The frontend will be available at [http://localhost:5173](http://localhost:5173) and the backend at [http://localhost:5000](http://localhost:5000).

## Usage
1. Register or log in with Google OAuth.
2. Upload a chest X-ray image.
3. View the prediction and confidence score.

## Project Structure
```
DefectoScan/
  backend/         # Flask API, ML model, routes
  DefectoScan/frontend/  # React frontend
  data/            # Datasets (not tracked in git)
  uploads/         # Uploaded images
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **Demo Video:**
>  [![Watch the demo](demo-video-placeholder.png)](https://your-demo-video-link-here)
