import requests
from PIL import Image
import io
import numpy as np

BASE_URL = "http://localhost:8000"

def create_test_image():
    """
    Create a synthetic RGB image for CI smoke testing.
    Avoids dependency on dataset files.
    """
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("[OK] Health endpoint working")


def test_prediction():
    img_bytes = create_test_image()

    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files)

    assert r.status_code == 200

    data = r.json()
    print(f"[DEBUG] Response data: {data}")

    # Validate structure (NOT accuracy — just service sanity)
    assert "label" in data
    assert "cat_probability" in data
    assert "dog_probability" in data

    print("[OK] Prediction endpoint working")


if __name__ == "__main__":
    test_health()
    test_prediction()
