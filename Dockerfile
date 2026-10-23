FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and packaging tools first to support modern wheel metadata
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies using CPU-only PyTorch index from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (respecting .dockerignore)
COPY . .

# Create writable output directories
RUN mkdir -p uploads static/gradcam_output && chmod -R 777 uploads static/gradcam_output

ENV PORT=5050
EXPOSE 5050

# Start Flask web application
CMD ["python", "app.py"]
