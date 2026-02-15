import os
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.models.model import SimpleCNN   # your model class


# --------------------------------------------------
# Configuration (NO hardcoding — use env or defaults)
# --------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")

DEFAULT_IMAGE_PATH = os.getenv(
    "DEFAULT_IMAGE_PATH",
    "data/processed/test/cats/31.jpg"   # your provided image
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Lazy Model Loader (Important for CI/CD)
# --------------------------------------------------

_model = None  # global cache


def load_model() -> nn.Module:
    """
    Loads model only when needed.
    Prevents CI failures and speeds up repeated inference.
    """
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Ensure training ran or CI created a dummy model."
        )

    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


# --------------------------------------------------
# Image Preprocessing (Same as Training)
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(image_path) -> torch.Tensor:
    """
    Preprocess image from file path or PIL Image object.
    """
    # Handle PIL Image objects
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    elif isinstance(image_path, str):
        # Handle file paths
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
    else:
        raise TypeError(f"image_path must be a string (file path) or PIL Image, got {type(image_path)}")
    
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension

    return image.to(DEVICE)


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

CLASS_NAMES = ["cat", "dog"]


def predict_image(image_path = None) -> Dict:
    """
    Run inference on a single image (file path or PIL Image object).
    
    Args:
        image_path: File path (str) or PIL Image object. Cannot be None.
    
    Raises:
        ValueError: If image_path is None.
    """

    if image_path is None:
        raise ValueError("image_path cannot be None. Provide a file path or PIL Image object.")

    model = load_model()
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # Get individual probabilities for each class
    cat_prob = float(probs[0, 0].item())
    dog_prob = float(probs[0, 1].item())

    result = {
        "label": CLASS_NAMES[predicted.item()],
        "cat_probability": cat_prob,
        "dog_probability": dog_prob
    }

    return result


# --------------------------------------------------
# CLI Entry (for manual testing)
# --------------------------------------------------

if __name__ == "__main__":
    result = predict_image()
    print(result)
