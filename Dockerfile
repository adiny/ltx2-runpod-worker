FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    diffusers>=0.27.0 \
    transformers>=4.38.0 \
    accelerate>=0.27.0 \
    safetensors \
    imageio \
    imageio-ffmpeg \
    sentencepiece

# Copy handler
COPY src/handler.py /handler.py

# Start
CMD ["python", "-u", "/handler.py"]
