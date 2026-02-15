import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

IMG_SIZE = (224, 224)
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test


def create_folder_structure():
    """
    Creates train/val/test directory structure.
    """
    for split in ["train", "val", "test"]:
        for label in ["cats", "dogs"]:
            os.makedirs(os.path.join(PROCESSED_DATA_PATH, split, label), exist_ok=True)


def load_image_paths():
    """
    Collects all image file paths and labels.
    """
    image_paths = []
    labels = []

    for label in ["cats", "dogs"]:
        folder = os.path.join(RAW_DATA_PATH, label)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(folder, file))
                labels.append(label)

    return image_paths, labels


def split_dataset(image_paths, labels):
    """
    Splits dataset into train/val/test.
    """
    train_x, temp_x, train_y, temp_y = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, stratify=temp_y, random_state=42
    )

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def process_and_save(images, labels, split):
    """
    Resize and save images into processed folder.
    """
    for img_path, label in tqdm(zip(images, labels), total=len(images), desc=f"Processing {split}"):

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            filename = os.path.basename(img_path)
            save_path = os.path.join(PROCESSED_DATA_PATH, split, label, filename)

            img.save(save_path)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")


def main():
    print("Starting preprocessing...")

    create_folder_structure()

    image_paths, labels = load_image_paths()

    train, val, test = split_dataset(image_paths, labels)

    process_and_save(*train, split="train")
    process_and_save(*val, split="val")
    process_and_save(*test, split="test")

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
