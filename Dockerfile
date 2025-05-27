FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir torch transformers runpod

# Add app code
WORKDIR /app
COPY main.py .

# Run the job handler
CMD ["python", "main.py"]
