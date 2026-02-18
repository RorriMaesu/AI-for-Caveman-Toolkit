FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    chromaprint-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/AI_FOR_CAVEMAN
COPY . /workspace/AI_FOR_CAVEMAN
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r dev-requirements.txt && \
    python3 setup_project.py --root /workspace/AI_FOR_CAVEMAN

ENTRYPOINT ["python3", "_scripts/controller.py"]
