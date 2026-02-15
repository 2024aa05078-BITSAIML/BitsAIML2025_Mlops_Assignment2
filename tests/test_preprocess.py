import os


def test_dataset_structure():
    """Ensure dataset folders follow expected structure."""
    base = "data/processed"

    for split in ["train", "val", "test"]:
        split_path = os.path.join(base, split)
        assert os.path.exists(split_path)

        # Must contain class folders
        classes = os.listdir(split_path)
        assert "cat" in classes or "cats" in classes
        assert "dog" in classes or "dogs" in classes


def test_dataset_not_empty():
    """Ensure preprocessing actually produced images."""
    base = "data/processed/train"

    total_files = 0
    for root, _, files in os.walk(base):
        total_files += len(files)

    assert total_files > 0
