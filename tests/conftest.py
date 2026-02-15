import os
import torch
import pytest

from src.model.model import SimpleCNN


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    This fixture prepares CI-safe environment:
    - creates dummy trained model
    - creates minimal dataset folder structure
    """

    # -----------------------------
    # Create dummy model artifact
    # -----------------------------
    os.makedirs("artifacts", exist_ok=True)

    model = SimpleCNN()
    torch.save(model.state_dict(), "artifacts/model.pt")

    # -----------------------------
    # Create fake dataset structure
    # -----------------------------
    base = "data/processed"
    splits = ["train", "val", "test"]
    classes = ["cats", "dogs"]

    for split in splits:
        for cls in classes:
            path = os.path.join(base, split, cls)
            os.makedirs(path, exist_ok=True)

    yield
