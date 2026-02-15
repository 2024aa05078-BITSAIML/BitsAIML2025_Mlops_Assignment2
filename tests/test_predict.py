from PIL import Image
import numpy as np

from src.inference.predict import predict_image


def test_prediction_output_format():
    """
    Ensure prediction returns expected keys.
    """

    # Create dummy image (black image)
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    result = predict_image(img)

    assert "label" in result
    assert "cat_probability" in result
    assert "dog_probability" in result
