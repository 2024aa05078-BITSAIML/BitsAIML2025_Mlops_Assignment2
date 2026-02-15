import sys
import os
import pytest
import torch
import numpy as np
from PIL import Image

# --------------------------------------------------
# Make 'src' importable in CI environment
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from src.models.model import SimpleCNN


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    This fixture prepares CI-safe environment:
    - creates dummy trained model
    - creates minimal dataset folder structure with fake images
    """

    # -----------------------------
    # Create dummy model artifact
    # -----------------------------
    os.makedirs("artifacts", exist_ok=True)

    model = SimpleCNN()
    torch.save(model.state_dict(), "artifacts/model.pt")

    # -----------------------------
    # Create fake dataset structure with dummy images
    # -----------------------------
    base = "data/processed"
    splits = ["train", "val", "test"]
    classes = ["cats", "dogs"]

    for split in splits:
        for cls in classes:
            path = os.path.join(base, split, cls)
            os.makedirs(path, exist_ok=True)
            
            # Create 2 dummy images per class/split
            for i in range(2):
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(path, f"{i}.jpg"))

    yield
