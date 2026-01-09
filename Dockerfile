FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install specific versions to avoid torch.xpu error
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod==1.6.0 \
    torch==2.1.0 \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    safetensors \
    imageio \
    imageio-ffmpeg \
    sentencepiece \
    protobuf

# Copy handler
COPY src/handler.py /handler.py

# Start
CMD ["python", "-u", "/handler.py"]
