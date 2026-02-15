import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow

from src.models.model import SimpleCNN

DATA_PATH = "data/processed"
MODEL_PATH = "artifacts/model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():
    mlflow.set_experiment("cats_vs_dogs_baseline")

    with mlflow.start_run(run_name="evaluation"):

        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "test"), transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        mlflow.log_metric("test_accuracy", acc)

        print(f"Test Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Cat", "Dog"],
                    yticklabels=["Cat", "Dog"])
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)


if __name__ == "__main__":
    evaluate()
