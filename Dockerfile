FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install transformers fastapi uvicorn[standard] gunicorn torch

# Copy app files
WORKDIR /app
COPY . /app

# Expose FastAPI app via gunicorn with Uvicorn workers
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000"]
