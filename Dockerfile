FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
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
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Download model script
COPY download_model.py /download_model.py

# Download LTX-2 model during build
RUN python /download_model.py && rm /download_model.py

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
