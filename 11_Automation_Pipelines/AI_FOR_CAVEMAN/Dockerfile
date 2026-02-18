FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    chromaprint-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt dev-requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r dev-requirements.txt

COPY . .
RUN python3 setup_project.py --root /app

CMD ["python3", "_scripts/controller.py", "--mode=mock", "--topic", "Caveman launch trailer"]
