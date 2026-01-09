# Base Image חזק ויציב
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# הגדרות סביבה למניעת שגיאות והבטחת זיהוי PATH
ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1. התקנת תלויות מערכת (כולל git ו-ffmpeg לאודיו)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. העתקת ה-Requirements
COPY requirements.txt .

# 3. שדרוג pip והתקנת כלי בסיס
RUN pip install --upgrade pip && \
    pip install ninja packaging wheel

# 4. התקנת Flash Attention מקובץ מוכן (הפתרון לקריסה!)
# זה חוסך את הקימפול שגורם לנפילה. הגרסה מותאמת ל-Torch 2.4 ו-Python 3.11
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-cache-dir

# 5. התקנת שאר התלויות
RUN pip install --no-cache-dir -r requirements.txt

# 6. העתקת הקוד
COPY handler.py /handler.py

# 7. הרצה
CMD [ "python", "-u", "/handler.py" ]
