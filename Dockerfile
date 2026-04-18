FROM python:3.11-slim

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown -R user:user /app
USER user

ENV PATH="/home/user/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

COPY --chown=user:user requirements.txt ./

# Install CPU-only torch first so pip doesn't pull the 2+ GB CUDA wheels
RUN pip install --no-cache-dir --user \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . ./

EXPOSE 7860

CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
