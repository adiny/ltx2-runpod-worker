FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    torch==2.1.0 \
    diffusers==0.24.0 \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    safetensors \
    imageio \
    imageio-ffmpeg

COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
