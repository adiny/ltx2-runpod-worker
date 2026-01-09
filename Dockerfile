FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install runpod \
    diffusers>=0.32.0 \
    transformers>=4.46.0 \
    accelerate>=1.0.0 \
    sentencepiece \
    imageio-ffmpeg \
    huggingface_hub

COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
