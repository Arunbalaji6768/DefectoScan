import requests
import os

# Test the Flask API
url = "http://127.0.0.1:5000/predict"
test_image_path = "data/Chest Xray Dataset/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

if os.path.exists(test_image_path):
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
else:
    print(f"Test image not found at: {test_image_path}") 