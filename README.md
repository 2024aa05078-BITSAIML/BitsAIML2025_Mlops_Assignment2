# BitsAIML2025 MLOps Assignment 2: Cats vs Dogs Classification

## Project Overview

This project implements a comprehensive end-to-end Machine Learning Operations (MLOps) pipeline for binary image classification (Cats vs Dogs) designed for a Pet Adoption Platform. The pipeline demonstrates industry best practices for:

- Version control of code and datasets
- Reproducible machine learning workflows
- Experiment tracking and management
- Model containerization and deployment
- Continuous Integration/Continuous Deployment (CI/CD)
- Automated testing
- Production monitoring and logging

The implementation covers the entire lifecycle from data preprocessing through model inference, containerization, and deployment readiness.

---

## Project Architecture and Structure

```
BitsAIML2025_Mlops_Assignment2/
│
├── src/
│   ├── data/
│   │   └── preprocess.py           # Data preprocessing and preparation
│   ├── models/
│   │   └── model.py                # CNN model definition (SimpleCNN)
│   ├── training/
│   │   ├── train.py                # Training pipeline with MLflow integration
│   │   └── evaluate.py             # Model evaluation and metrics logging
│   └── inference/
│       ├── predict.py              # Prediction module
│       └── service.py              # FastAPI inference service
│
├── data/
│   ├── raw/                        # Original dataset (DVC tracked)
│   └── processed/                  # Preprocessed and split dataset (DVC tracked)
│       ├── train/
│       │   ├── cats/
│       │   └── dogs/
│       ├── val/
│       │   ├── cats/
│       │   └── dogs/
│       └── test/
│           ├── cats/
│           └── dogs/
│
├── artifacts/                      # Model artifacts and outputs
│   ├── model.pt                    # Trained model weights
│   └── confusion_matrix.png        # Evaluation metrics visualization
│
├── logs/                           # Application and training logs
├── tests/                          # Unit and integration tests
│   ├── conftest.py                # Pytest configuration and fixtures
│   ├── test_predict.py            # Prediction module tests
│   └── test_preprocess.py         # Data preprocessing tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
│
├── .gitignore                     # Git ignore configuration
├── .dockerignore                  # Docker ignore configuration
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies
├── requirements-docker.txt        # Docker-specific dependencies
├── dvc.yaml                       # DVC pipeline configuration
├── setup.py                       # Package setup configuration
└── README.md                      # Project documentation
```

---

## Dataset Description and Management

### Dataset Overview

The Cats vs Dogs dataset is a binary classification dataset containing:

- High-resolution images of cats and dogs
- Images standardized to 224 x 224 pixels in RGB format
- Total samples distributed across training, validation, and test sets

### Data Distribution

- Training Set: 80% of total samples
- Validation Set: 10% of total samples
- Test Set: 10% of total samples

### Data Versioning with DVC

Data versioning is implemented using DVC (Data Version Control) to:

- Track raw and processed datasets without storing large files in Git
- Ensure reproducibility of data pipelines
- Enable version control of data transformations
- Maintain data lineage and dependencies

DVC Configuration:

```bash
dvc init
dvc add data/raw
dvc add data/processed
dvc commit
```

The `.dvc` files are committed to Git while actual data files are stored in DVC remote storage or local cache.

---

## Data Preprocessing Pipeline

### Module Location

Implementation: `src/data/preprocess.py`

### Preprocessing Steps

1. **Data Loading**
   - Load images from raw dataset directory
   - Validate image format and integrity
   - Handle corrupted or missing files

2. **Image Normalization**
   - Convert all images to RGB color space (handles grayscale conversion)
   - Standardize pixel values to [0, 1] range

3. **Resizing**
   - Resize all images to 224 x 224 pixels
   - Maintain aspect ratio using padding when necessary
   - Consistency ensures uniform input to CNN model

4. **Dataset Splitting**
   - Stratified split to maintain class distribution
   - 80% training, 10% validation, 10% test
   - Reproducible splitting with fixed random seeds

5. **Organizational Structure**
   - Organize processed data in hierarchical directory structure
   - Separate training, validation, and test sets
   - Separate by class labels (cats, dogs)

6. **Data Augmentation**
   - Training set augmentation: random rotations, flips, brightness adjustments
   - Validation and test sets processed without augmentation
   - Augmentation improves model generalization

---

## Model Architecture and Design

### Model Implementation

Location: `src/models/model.py`

Class: `SimpleCNN`

### Architecture Specification

The SimpleCNN baseline model comprises:

**Convolutional Feature Extraction Layers:**
- Layer 1: 32 filters, 3x3 kernel, ReLU activation
  - MaxPooling 2x2
- Layer 2: 64 filters, 3x3 kernel, ReLU activation
  - MaxPooling 2x2
- Layer 3: 128 filters, 3x3 kernel, ReLU activation
  - MaxPooling 2x2

**Classification Head:**
- Flattening layer
- Fully connected layer: 256 units, ReLU activation
- Dropout layer: 0.5 dropout rate (reduces overfitting)
- Output layer: 2 units (binary classification), softmax activation

### Model Characteristics

- Total Parameters: Approximately 1.2M
- Input Shape: (224, 224, 3)
- Output Shape: (batch_size, 2) - logits for [cat_probability, dog_probability]
- Design Rationale: Lightweight baseline for validation before optimization

---

## Training Pipeline

### Module Location

Implementation: `src/training/train.py`

### Training Configuration

**Hyperparameters:**
- Batch Size: 32 samples per batch
- Learning Rate: 0.001 (Adam optimizer)
- Epochs: 20 maximum (with early stopping if applicable)
- Loss Function: Cross-Entropy Loss (standard for multi-class classification)
- Optimizer: Adam with default momentum parameters

**Device Management:**
- Automatic GPU detection and utilization if available
- Falls back to CPU if GPU not present
- Enables `torch.cuda` for acceleration

**Data Augmentation (Training Only):**
- Random horizontal flips
- Random rotations (-15 to +15 degrees)
- Random brightness and contrast adjustments
- Helps model generalize to variations in real-world data

### Training Process

1. Model initialization and weight setup
2. Batch-wise gradient computation using backpropagation
3. Parameter updates using Adam optimizer
4. Loss tracking per epoch
5. Validation evaluation after each epoch
6. Model checkpoint saving based on validation performance
7. Final trained model serialization to `artifacts/model.pt`

### MLflow Integration

Training metrics are automatically logged to MLflow:

- Hyperparameters (batch_size, learning_rate, epochs)
- Training loss per epoch
- Validation metrics per epoch
- Model artifact (trained weights)
- Training duration
- Environment information

**Accessing Training History:**

```bash
mlflow ui
# Navigate to http://localhost:5000 in browser
```

---

## Model Evaluation and Metrics

### Module Location

Implementation: `src/training/evaluate.py`

### Evaluation Methodology

**Metrics Computed:**

1. **Accuracy**: (TP + TN) / Total
   - Proportion of correct predictions
   - Overall model performance

2. **Precision**: TP / (TP + FP)
   - Reliability of positive predictions
   - Important for reducing false positives

3. **Recall (Sensitivity)**: TP / (TP + FN)
   - Ability to find all positive instances
   - Important for reducing false negatives

4. **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   - Balanced performance metric

5. **Confusion Matrix**:
   ```
   [[True Negatives,   False Positives],
    [False Negatives,  True Positives]]
   ```
   - Visual breakdown of prediction outcomes

### Evaluation Process

1. Load trained model from `artifacts/model.pt`
2. Process test dataset in batches
3. Generate predictions for all test samples
4. Compute metrics against ground truth labels
5. Generate confusion matrix visualization
6. Log metrics and artifacts to MLflow
7. Save confusion matrix PNG to `artifacts/confusion_matrix.png`

### MLflow Artifact Logging

- Confusion matrix PNG visualization
- Evaluation metrics as parameters
- Summary statistics for model performance documentation

---

## Experiment Tracking with MLflow

### Purpose and Benefits

MLflow provides centralized experiment management:

- Track all training runs with full metadata
- Compare hyperparameters across experiments
- Version and manage model artifacts
- Enable reproducible research
- Maintain audit trail of model development

### Logged Information

**Per Run:**
- Start and end timestamps
- Duration of training
- Hyperparameters (batch size, learning rate, epochs)
- Metrics (loss, accuracy, validation accuracy)
- Supplementary information (device, PyTorch version)

**Model Artifacts:**
- Entire trained model serialized in PyTorch format
- Confusion matrix visualization
- Evaluation report JSON

### Accessing MLflow UI

```bash
# In project directory
mlflow ui

# Access at: http://localhost:5000
```

**UI Features:**
- Experiment browsing and comparison
- Run details and parameter values
- Metrics visualization over time
- Artifact download and inspection

---

## Inference and Prediction

### Prediction Module

Location: `src/inference/predict.py`

### Functionality

The `predict_image()` function:

1. Accepts image input (file path or PIL Image object)
2. Loads trained model from artifacts
3. Preprocesses input image (resize, normalize)
4. Performs forward pass through network
5. Computes probability scores for both classes
6. Returns structured prediction result

### Output Format

```python
{
    "label": "cat" | "dog",           # Predicted class
    "cat_probability": 0.0-1.0,       # Probability for cat class
    "dog_probability": 0.0-1.0        # Probability for dog class
}
```

### Error Handling

- Validates input image path existence
- Handles both file paths and PIL Image objects
- Raises TypeError for invalid input types
- Raises FileNotFoundError for missing files
- Graceful degradation for model loading failures

---

## FastAPI Inference Service

### Service Location

Implementation: `src/inference/service.py`

### Service Features

**Endpoints:**

1. **GET /health** - Service health check
   - Returns: `{"status": "healthy"}`
   - Purpose: Liveness probe for container orchestration

2. **POST /predict** - Single image prediction
   - Input: Image file upload
   - Output: Prediction with confidence scores
   - Error handling: Validation and exception catching

### Running the Service

```bash
# Local development
python -m src.inference.service

# With custom port
uvicorn src.inference.service:app --host 0.0.0.0 --port 8000

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Request/Response Examples

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Image Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

---

## Containerization with Docker

### Docker Configuration

**Dockerfile:**
- Base Image: `python:3.11-slim`
- Working Directory: `/app`
- Dependency Installation: Optimized layer caching
- Service Startup: FastAPI with Uvicorn

**Build Process:**

```bash
docker build -t cats-dogs-mlops:latest .
```

**Running Container:**

```bash
# Basic run
docker run -p 8000:8000 cats-dogs-mlops:latest

# With volume mount for data
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  cats-dogs-mlops:latest
```

### Docker Optimization

- Multi-stage builds (if applicable)
- Minimal base image (slim variants)
- Layer caching optimization
- Excluded large files with `.dockerignore`:
  - Virtual environments
  - Version control metadata
  - Raw datasets
  - MLflow run artifacts

---

## Testing Framework

### Test Structure

Location: `tests/`

**Test Configuration:** `conftest.py`
- Pytest initialization and fixtures
- Test environment setup
- Dummy model and dataset creation for CI/CD

### Test Coverage

**Prediction Tests** (`test_predict.py`):
1. Valid prediction structure validation
2. Probability range validation (0.0 to 1.0)
3. Probability sum validation (approximately 1.0)
4. Image size handling (arbitrary dimensions)
5. RGB conversion from grayscale
6. Extreme input handling (all-white/black images)
7. Invalid input error handling

**Preprocessing Tests** (`test_preprocess.py`):
1. Dataset directory structure validation
2. Output file count verification
3. Image format validation
4. Dimension consistency checking

### Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_predict.py -v

# Coverage report
pytest --cov=src tests/
```

---

## Continuous Integration/Continuous Deployment

### GitHub Actions Workflow

Location: `.github/workflows/ci.yml`

**Pipeline Stages:**

1. **Checkout**: Clone repository code
2. **Python Setup**: Install Python 3.11
3. **Dependency Installation**: Install from requirements.txt
4. **Linting** (optional): Code quality checks
5. **Unit Tests**: Run pytest suite
6. **Code Coverage**: Generate coverage reports
7. **Docker Build** (optional): Build and push container image

**Triggers:**
- Push to main branch
- Pull requests to main branch

**Status Checks:**
- All tests must pass
- Code coverage thresholds (if configured)
- Docker build success (if enabled)

### Workflow Configuration

```yaml
name: MLOps CI Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      # Configuration steps...
```

---

## Monitoring and Logging

### Application Logging

**Log Destinations:**
- Console output (stdout/stderr)
- Log files in `logs/` directory
- MLflow run logging for experiment metadata

**Log Levels:**
- INFO: General information, training progress
- WARNING: Potential issues, deprecated features
- ERROR: Recoverable errors, exceptions
- DEBUG: Detailed diagnostic information

### Training Monitoring

**Real-time Metrics:**
- Loss values per epoch
- Accuracy metrics during validation
- Resource utilization (GPU/CPU)
- Training time estimates

**Experiment Comparison:**
- Compare multiple runs in MLflow UI
- Identify best performing configurations
- Track metric progression across experiments

### Inference Monitoring

**Service Metrics:**
- Request count and rate
- Response times (latency)
- Error rates and exception counts
- Model inference duration
- Resource consumption per request

### Logging Best Practices

1. **Structured Logging**: Use consistent format for programmatic parsing
2. **Contextual Information**: Include run IDs, user IDs, timestamps
3. **Error Tracebacks**: Full stack traces for debugging
4. **Performance Metrics**: Log inference times, batch processing stats
5. **Audit Trail**: Track data access and model usage

---

## Setup and Usage Instructions

### Environment Setup

**1. Clone Repository:**
```bash
git clone <repository-url>
cd BitsAIML2025_Mlops_Assignment2
```

**2. Create Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Initialize DVC:**
```bash
dvc init
dvc remote add -d <remote-name> <remote-path>  # Optional
```

### Data Preparation

```bash
# Download and prepare data
# Place raw data in data/raw/

# Run preprocessing
python -m src.data.preprocess
```

### Training Workflow

```bash
# Train model with MLflow logging
python -m src.training.train

# View training experiments
mlflow ui
```

### Model Evaluation

```bash
# Evaluate trained model
python -m src.training.evaluate
```

### Inference

```bash
# Single image prediction
python -c "from src.inference.predict import predict_image; print(predict_image('path/to/image.jpg'))"

# Start inference service
python -m src.inference.service

# Test prediction via API
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

### Running Tests

```bash
# Execute all tests
pytest -v

# Generate coverage report
pytest --cov=src tests/
```

---

## Deliverables Summary

This comprehensive MLOps implementation delivers:

- Version-controlled code repository with Git
- Data versioning and management with DVC
- Reproducible ML pipeline with documented preprocessing
- Baseline CNN model with evaluation metrics
- Automated experiment tracking with MLflow
- Production-ready FastAPI inference service
- Containerized deployment with Docker
- Automated testing suite with pytest
- CI/CD pipeline with GitHub Actions
- Comprehensive monitoring and logging
- Complete documentation and usage guides

---

## Technical Stack

- **Language**: Python 3.11
- **Deep Learning**: PyTorch 2.10.0
- **Version Control**: Git, DVC
- **Experiment Tracking**: MLflow 3.9.0
- **Web Framework**: FastAPI 0.129.0
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: Pytest 9.0.2
- **Data Processing**: NumPy, Pillow
- **Visualization**: Matplotlib, Seaborn

---

## References and Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- MLflow Documentation: https://mlflow.org/docs/
- DVC Documentation: https://dvc.org/doc
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Docker Documentation: https://docs.docker.com/

