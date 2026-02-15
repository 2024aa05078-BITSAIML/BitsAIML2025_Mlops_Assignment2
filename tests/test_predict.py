import pytest
import numpy as np
from PIL import Image

from src.inference.predict import predict_image


def create_dummy_image(size=(224, 224), color=0):
    """Utility to create synthetic images"""
    arr = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def test_prediction_returns_valid_structure():
    """Prediction should return expected keys."""
    img = create_dummy_image()

    result = predict_image(img)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"label", "cat_probability", "dog_probability"}


def test_probabilities_are_valid():
    """Probabilities should be between 0 and 1 and sum ~1."""
    img = create_dummy_image(color=128)

    result = predict_image(img)

    cat_p = result["cat_probability"]
    dog_p = result["dog_probability"]

    assert 0.0 <= cat_p <= 1.0
    assert 0.0 <= dog_p <= 1.0
    assert abs((cat_p + dog_p) - 1.0) < 1e-3


def test_model_handles_different_image_sizes():
    """Model should auto-resize non-224 images."""
    img = create_dummy_image(size=(500, 300))

    result = predict_image(img)

    assert "label" in result


def test_model_handles_rgb_conversion():
    """Grayscale images should be converted to RGB safely."""
    gray = np.zeros((224, 224), dtype=np.uint8)
    img = Image.fromarray(gray)  # single channel

    result = predict_image(img)

    assert "label" in result


def test_extreme_input_image():
    """All-white image shouldn't crash model."""
    img = create_dummy_image(color=255)

    result = predict_image(img)

    assert result["label"] in ["cat", "dog"]


def test_invalid_input_raises_error():
    """Passing invalid input should raise an exception."""
    with pytest.raises(Exception):
        predict_image(None)
