FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
ENV HF_HOME="/models/.cache"
ENV HUGGINGFACE_HUB_CACHE="/models/.cache"
ENV HF_HUB_ENABLE_HF_TRANSFER="0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install ninja packaging wheel && \
    pip install git+https://github.com/huggingface/diffusers.git --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /models/.cache /models/LTX-2

# Download LTX-2 model during build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lightricks/LTX-2', local_dir='/models/LTX-2', ignore_patterns=['*.md', '*.git*'])"

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
