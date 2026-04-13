FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAN_MODEL_ID=Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    CIVITAI_MODEL_ID=1811313 \
    CIVITAI_MODEL_VERSION_ID=2553271 \
    CIVITAI_LORA_NAME=DR34ML4Y_I2V_14B_LOW_V2.safetensors \
    CIVITAI_LORA_DOWNLOAD_URL=https://civitai.com/api/download/models/2553271 \
    CIVITAI_LORA_SHA256=066EE4BFAFB685C85F08174C8283CD11BC6D36F4845347F20D633AB44581601F \
    CIVITAI_LORA_TARGET=transformer_2 \
    WAN_DEFAULT_RESOLUTION_PRESET=720p \
    WAN_DEFAULT_NUM_FRAMES=81 \
    WAN_DEFAULT_FPS=16 \
    WAN_DEFAULT_NUM_INFERENCE_STEPS=10 \
    WAN_DEFAULT_GUIDANCE_SCALE=5.0 \
    WAN_DEFAULT_GUIDANCE_SCALE_2=6.0 \
    WAN_DEFAULT_FLOW_SHIFT=8.0 \
    WAN_DEFAULT_LORA_SCALE=1.0 \
    WAN_ENABLE_MODEL_CPU_OFFLOAD=true \
    WAN_ENABLE_VAE_TILING=true \
    WAN_ENABLE_VAE_SLICING=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY wan_lora_inference.py /app/wan_lora_inference.py
COPY test_input.json /app/test_input.json
COPY runpod_test_client.py /app/runpod_test_client.py
COPY README.md /app/README.md

CMD ["python", "-u", "/app/handler.py"]
