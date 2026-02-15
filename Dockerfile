# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements-docker.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the service
CMD ["uvicorn", "src.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
