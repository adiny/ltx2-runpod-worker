FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# התקנת תלויות מערכת כולל FFmpeg לאודיו
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install ninja packaging wheel

# התקנת התלויות ללא Cache להקטנת גודל ה-Image
RUN pip install --no-cache-dir -r requirements.txt

# התקנת Flash Attention בנפרד למניעת שגיאות בנייה
RUN pip install flash-attn --no-build-isolation --no-cache-dir

# העתקת הקוד
COPY handler.py /handler.py

CMD [ "python", "-u", "/handler.py" ]
