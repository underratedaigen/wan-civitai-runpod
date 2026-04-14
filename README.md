# WAN 2.2 I2V Runpod Worker with Civitai LoRA

This repository is a separate Runpod Serverless worker for:

- official `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- one Civitai LoRA loaded on top of the base model
- Runpod queue-based inference with a simple JSON API

This setup is intentionally separate from your existing WAN endpoint so you can experiment without disturbing the current deployment.

## What This Worker Does

- downloads the official Wan 2.2 diffusers base from Hugging Face
- downloads the selected Civitai LoRA at worker startup the first time it is needed
- converts the Civitai safetensors LoRA into the format expected by Diffusers
- loads the LoRA into the low-noise denoiser by default
- accepts a simple request with a prompt and input image
- returns an MP4 as either:
  - base64 in the response
  - or a bucket URL if bucket env vars are configured

## Base Model and LoRA

- Base model: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- LoRA source: Civitai model `1811313`, version `2553271`
- LoRA file: `DR34ML4Y_I2V_14B_LOW_V2.safetensors`
- Default LoRA target: `transformer_2`

The `transformer_2` default is deliberate because this LoRA file is a `LOW` variant. That is an implementation choice based on the file naming and on Diffusers guidance that Wan 2.2 has two denoisers and low-noise LoRAs may need to load into `transformer_2`.

## Main Files

- `Dockerfile`
- `requirements.txt`
- `.env.example`
- `handler.py`
- `wan_lora_inference.py`
- `runpod_test_client.py`
- `test_input.json`

## Request Format

Minimal request:

```json
{
  "input": {
    "prompt": "YOUR_TRIGGER_WORD, cinematic image-to-video motion, stable framing, natural subject movement.",
    "image_url": "https://example.com/your-image.jpg"
  }
}
```

More controlled request:

```json
{
  "input": {
    "prompt": "YOUR_TRIGGER_WORD, cinematic image-to-video motion, stable framing, natural subject movement.",
    "negative_prompt": "blurry, low quality, artifacts, subtitles, watermark, static frame",
    "image_url": "https://example.com/your-image.jpg",
    "resolution_preset": "720p",
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 10,
    "guidance_scale": 5.0,
    "guidance_scale_2": 6.0,
    "flow_shift": 8.0,
    "lora_scale": 1.0,
    "seed": 42
  }
}
```

## Environment Variables

Use these in Runpod:

```text
WAN_MODEL_ID=Wan-AI/Wan2.2-I2V-A14B-Diffusers
HF_TOKEN=
CIVITAI_API_KEY=
CIVITAI_MODEL_ID=1811313
CIVITAI_MODEL_VERSION_ID=2553271
CIVITAI_LORA_NAME=DR34ML4Y_I2V_14B_LOW_V2.safetensors
CIVITAI_LORA_DOWNLOAD_URL=https://civitai.com/api/download/models/2553271
CIVITAI_LORA_SHA256=066EE4BFAFB685C85F08174C8283CD11BC6D36F4845347F20D633AB44581601F
CIVITAI_LORA_TARGET=transformer_2
CIVITAI_LORA_MAX_RETRIES=5
CIVITAI_LORA_RETRY_DELAY_S=15
CIVITAI_LORA_REQUEST_TIMEOUT_S=300
CIVITAI_LORA_CHUNK_SIZE_MB=8
WAN_DEFAULT_RESOLUTION_PRESET=720p
WAN_DEFAULT_NUM_FRAMES=81
WAN_DEFAULT_FPS=16
WAN_DEFAULT_NUM_INFERENCE_STEPS=10
WAN_DEFAULT_GUIDANCE_SCALE=5.0
WAN_DEFAULT_GUIDANCE_SCALE_2=6.0
WAN_DEFAULT_FLOW_SHIFT=8.0
WAN_DEFAULT_LORA_SCALE=1.0
WAN_ENABLE_MODEL_CPU_OFFLOAD=true
WAN_FORCE_CPU_OFFLOAD_ON_LARGE_GPU=false
WAN_FULL_GPU_MIN_VRAM_GB=70
WAN_SCHEDULER=auto
WAN_ENABLE_VAE_TILING=true
WAN_ENABLE_VAE_SLICING=true
WAN_INIT_TIMEOUT_S=3600
```

Optional bucket upload settings:

```text
BUCKET_ENDPOINT_URL=...
BUCKET_ACCESS_KEY_ID=...
BUCKET_SECRET_ACCESS_KEY=...
```

## Runpod Deployment

Use this repo as its own GitHub repo and set:

- Build context: `.` or `/`
- Dockerfile path: `Dockerfile`
- Endpoint type: `Queue`
- GPU: `A100 80GB` or better
- Active workers: `0`
- Max workers: `1`
- Execution timeout: `3600`
- Container disk: `120 GB` or more

Leave the Runpod `Model` field blank. This worker downloads the Wan base and the LoRA itself.

If you attach a network volume, also set Hugging Face cache env vars so the base model persists across cold starts:

```text
HF_HOME=/runpod-volume/hf-cache
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache/hub
```

## Local Browser Tester

Run locally:

```powershell
cd C:\Users\gusta\Documents\New project\wan-civitai-runpod
pip install -r requirements.txt
python runpod_test_client.py
```

Then open:

```text
http://127.0.0.1:7863
```

## Notes

- Replace `YOUR_TRIGGER_WORD` with the trigger wording from the LoRA model version page
- frame counts are rounded to valid Wan lengths (`4n + 1`)
- `guidance_scale_2` is exposed because Wan 2.2 uses a separate low-noise denoiser
- this worker uses Diffusers directly instead of ComfyUI because it is a cleaner fit for an official Wan base plus one external LoRA
- the Civitai LoRA download now retries automatically on transient HTTP/network failures; if you attach a network volume, the cached LoRA is reused on later cold starts
- on large GPUs, the worker skips CPU offload by default even if `WAN_ENABLE_MODEL_CPU_OFFLOAD=true`; override with `WAN_FORCE_CPU_OFFLOAD_ON_LARGE_GPU=true` if you really want the old behavior
- `WAN_SCHEDULER=auto` keeps UniPC on full-GPU runs but falls back to the model's default scheduler when CPU offload is active, which is safer for mixed-device execution

## Sources

- [Official Wan 2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [Wan diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/wan)
- [Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- [Diffusers loader docs](https://huggingface.co/docs/diffusers/main/en/api/loaders)
- [Diffusers issue about Wan2.2 LoRA loading](https://github.com/huggingface/diffusers/issues/12147)
