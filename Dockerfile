FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install ninja packaging wheel && \
    pip install accelerate>=1.0.0 && \
    pip install git+https://github.com/huggingface/diffusers.git --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
