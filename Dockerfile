FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install ninja packaging wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
```

### המבנה הסופי של ה-Repo צריך להיות:
```
ltx2-runpod-worker/
├── Dockerfile
├── requirements.txt
├── handler.py          ← בתיקייה הראשית!
├── .dockerignore
└── README.md
