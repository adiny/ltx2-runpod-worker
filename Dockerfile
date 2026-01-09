FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# הגדרת Shell ו-Path בצורה מפורשת
ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# התקנת Git ו-FFmpeg (עם וידוא שהם הותקנו)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    which git && which ffmpeg

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install ninja packaging wheel

# התקנת התלויות
RUN pip install --no-cache-dir -r requirements.txt

# התקנת Flash Attention
RUN pip install flash-attn --no-build-isolation --no-cache-dir

# העתקת הקוד
COPY handler.py /handler.py

CMD [ "python", "-u", "/handler.py" ]
