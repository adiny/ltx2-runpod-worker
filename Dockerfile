FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    diffusers==0.24.0 \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    safetensors \
    imageio \
    imageio-ffmpeg

COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
