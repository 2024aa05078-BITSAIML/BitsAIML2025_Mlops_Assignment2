import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import SimpleCNN

# Paths
DATA_PATH = "data/processed"
MODEL_PATH = "artifacts/model.pt"

# Hyperparameters (will be logged in MLflow)
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_loaders():
    """
    Creates dataloaders with augmentation only for training data.
    """

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def train():
    mlflow.set_experiment("cats_vs_dogs_baseline")

    with mlflow.start_run():

        # Log hyperparameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)

        train_loader, val_loader, _ = get_data_loaders()

        model = SimpleCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

            # Log metric
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

        # Log model artifact
        mlflow.log_artifact(MODEL_PATH)

        # Also log full model
        mlflow.pytorch.log_model(model, "model")

        print("Training completed and logged to MLflow.")


if __name__ == "__main__":
    train()
