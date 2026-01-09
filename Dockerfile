FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN pip install --no-cache-dir runpod diffusers transformers accelerate imageio

COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
