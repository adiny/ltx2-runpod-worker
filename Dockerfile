FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.24.0 \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    safetensors \
    imageio \
    imageio-ffmpeg \
    sentencepiece

COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
