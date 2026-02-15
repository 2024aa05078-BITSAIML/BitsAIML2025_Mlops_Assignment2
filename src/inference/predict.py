import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.models.model import SimpleCNN

MODEL_PATH = "artifacts/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup (important for performance)
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(image: Image.Image):
    """
    Takes a PIL image and returns prediction + probabilities.
    """

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    cat_prob = probs[0][0].item()
    dog_prob = probs[0][1].item()

    label = "dog" if dog_prob > cat_prob else "cat"

    return {
        "label": label,
        "cat_probability": round(cat_prob, 4),
        "dog_probability": round(dog_prob, 4)
    }

