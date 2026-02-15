# BitsAIML2025_Mlops_Assignment2
MLOps ((S1-25_AIMLCZG523) Assignment 2

# 🐶🐱 MLOps Pipeline — Cats vs Dogs Classification

## 📌 Project Overview

This project implements an **end-to-end MLOps pipeline** for a Binary Image Classification use case (Cats vs Dogs) designed for a Pet Adoption Platform.

The pipeline covers:

* Data Versioning
* Model Development
* Experiment Tracking
* Reproducible Training Workflow

This README documents **Module M1: Model Development & Experiment Tracking**.

---

# ✅ M1 — Model Development & Experiment Tracking

## 🎯 Objective

Build a baseline deep learning model and ensure:

* Code is versioned using Git
* Data is versioned using DVC
* Experiments are tracked using MLflow
* Artifacts are reproducible

---

## 📂 Project Structure

```
mlops-cats-dogs/
│
├── src/
│   ├── data/            # Data preprocessing pipeline
│   ├── models/          # CNN model definition
│   ├── training/        # Training & evaluation scripts
│   └── inference/       # (Used in M2)
│
├── data/
│   ├── raw/             # Original dataset (DVC tracked)
│   └── processed/       # Resized + split dataset (DVC tracked)
│
├── artifacts/           # Saved model & evaluation outputs
├── logs/
├── tests/
│
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Cats vs Dogs image dataset
* Images resized to **224 × 224 RGB**
* Dataset split:

  * **80% Training**
  * **10% Validation**
  * **10% Test**

---

## 🔄 Data Versioning (DVC)

We used **DVC** to track:

* Raw dataset
* Processed dataset after preprocessing

This ensures:

* No large files stored in Git
* Full reproducibility of training pipeline
* Dataset changes are version controlled

Commands used:

```
dvc init
dvc add data/raw
dvc add data/processed
```

---

## 🧹 Data Preprocessing Pipeline

Implemented in:

```
src/data/preprocess.py
```

Steps performed:

* Image loading from raw dataset
* Conversion to RGB
* Resize to 224×224
* Stratified train/val/test split
* Saved into structured directory format

---

## 🧠 Baseline Model

Implemented in:

```
src/models/model.py
```

Architecture:

* 3 Convolutional Layers
* MaxPooling
* Fully Connected Layers
* Dropout Regularization

Purpose:

* Provide a lightweight **baseline CNN**
* Validate full MLOps lifecycle before optimization

---

## 🏋️ Training Pipeline

Implemented in:

```
src/training/train.py
```

Features:

* Data augmentation (train only)
* Adam optimizer
* Cross-entropy loss
* GPU support (if available)
* Model saved to `artifacts/model.pt`

---

## 📈 Experiment Tracking (MLflow)

MLflow used to log:

* Hyperparameters (batch size, learning rate, epochs)
* Training loss per epoch
* Trained model artifact
* Reproducible runs

Run UI using:

```
mlflow ui
```

---

## 📊 Evaluation & Metrics Logging

Implemented in:

```
src/training/evaluate.py
```

Evaluation includes:

* Test Accuracy calculation
* Confusion Matrix generation
* Artifact logging to MLflow

Artifacts saved:

```
artifacts/confusion_matrix.png
```

---

## ✅ M1 Deliverables Achieved

✔ Git-based code versioning
✔ DVC-based dataset tracking
✔ Automated preprocessing pipeline
✔ Baseline CNN model training
✔ MLflow experiment tracking
✔ Evaluation metrics & artifacts logged
✔ Reproducible ML workflow

---

## ▶️ Next Phase

Module **M2** will package this trained model into a containerized inference service using FastAPI and Docker.

