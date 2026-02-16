import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

# Sample image path for testing
test_image = Path("../data/processed/test/cats/31.jpg")  # Update path as needed

with open(test_image, "rb") as f:
    files = {"file": f}
    r = requests.post(f"{BASE_URL}/predict", files=files, timeout=20)
    
    print("RAW RESPONSE:", r.text)   # <-- ADD THIS
    result = r.json()
    print("JSON:", result)           # <-- ADD THIS
