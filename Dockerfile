FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-optimized PyTorch first for fast container builds
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create writable output directories for web uploads and generated CAM heatmaps
RUN mkdir -p uploads static/gradcam_output && chmod -R 777 uploads static/gradcam_output

# Expose Hugging Face Space default port
ENV PORT=7860
EXPOSE 7860

# Start the Flask web application
CMD ["python", "app.py"]
