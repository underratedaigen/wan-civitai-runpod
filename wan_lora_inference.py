import base64
import hashlib
import logging
import math
import os
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import safetensors.torch
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from PIL import Image, ImageOps
from transformers import CLIPVisionModel


LOGGER = logging.getLogger("wan-civitai-worker")
logging.basicConfig(level=logging.INFO)

PIPELINE_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
PIPELINE: WanImageToVideoPipeline | None = None
PIPELINE_STATE: dict[str, Any] = {}

PRESET_PIXELS = {
    "480p": 832 * 480,
    "720p": 1280 * 720,
}

RESAMPLING_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def default_cache_root() -> Path:
    volume_root = Path("/runpod-volume")
    if volume_root.exists():
        return volume_root / "hf-cache"
    return Path.home() / ".cache" / "huggingface"


def default_lora_dir() -> Path:
    volume_root = Path("/runpod-volume")
    if volume_root.exists():
        return volume_root / "civitai-loras"
    return Path("/tmp/civitai-loras")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_to_multiple(value: int, multiple: int = 16) -> int:
    value = max(multiple, int(value))
    return max(multiple, int(round(value / multiple) * multiple))


def normalize_frame_count(value: int | str) -> int:
    frames = max(1, int(value))
    remainder = (frames - 1) % 4
    if remainder == 0:
        return frames
    return frames + (4 - remainder)


def coerce_seed(seed: int | str | None) -> int:
    if seed is None:
        return torch.seed() % (2**63 - 1)
    seed_value = int(seed)
    if seed_value < 0:
        return torch.seed() % (2**63 - 1)
    return seed_value


def get_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback).strip()


def strip_data_uri(data: str) -> str:
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        return data.split(",", 1)[1]
    return data


def image_source_to_pil(job_input: dict[str, Any]) -> Image.Image:
    image_value = job_input.get("image")
    image_base64 = job_input.get("image_base64")
    image_url = job_input.get("image_url")

    raw_bytes: bytes | None = None
    if image_base64:
        raw_bytes = base64.b64decode(strip_data_uri(image_base64))
    elif image_url:
        response = requests.get(str(image_url), timeout=120)
        response.raise_for_status()
        raw_bytes = response.content
    elif isinstance(image_value, str):
        if image_value.startswith(("http://", "https://")):
            response = requests.get(image_value, timeout=120)
            response.raise_for_status()
            raw_bytes = response.content
        else:
            raw_bytes = base64.b64decode(strip_data_uri(image_value))

    if raw_bytes is None:
        raise ValueError("Provide one of 'image', 'image_base64', or 'image_url'.")

    with Image.open(BytesIO(raw_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def preset_dimensions(original_width: int, original_height: int, preset: str) -> tuple[int, int]:
    preset_key = preset.lower()
    if preset_key not in PRESET_PIXELS:
        raise ValueError(f"Unsupported resolution_preset '{preset}'. Use one of: {', '.join(PRESET_PIXELS)}")

    target_pixels = PRESET_PIXELS[preset_key]
    aspect = original_width / original_height
    width = math.sqrt(target_pixels * aspect)
    height = width / aspect
    return round_to_multiple(int(width)), round_to_multiple(int(height))


def resolve_generation_dimensions(
    *,
    original_width: int,
    original_height: int,
    width: int | str | None,
    height: int | str | None,
    resolution_preset: str,
) -> tuple[int, int]:
    if width is None and height is None:
        return preset_dimensions(original_width, original_height, resolution_preset)

    if width is not None and height is not None:
        return round_to_multiple(int(width)), round_to_multiple(int(height))

    aspect = original_width / original_height
    if width is not None:
        final_width = round_to_multiple(int(width))
        final_height = round_to_multiple(int(final_width / aspect))
        return final_width, final_height

    final_height = round_to_multiple(int(height))
    final_width = round_to_multiple(int(final_height * aspect))
    return final_width, final_height


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_civitai_lora(
    *,
    download_url: str,
    api_key: str,
    filename: str,
    sha256: str | None,
    output_dir: Path,
) -> Path:
    ensure_directory(output_dir)
    destination = output_dir / filename

    if destination.exists():
        if sha256:
            existing_hash = compute_sha256(destination)
            if existing_hash == sha256.upper():
                LOGGER.info("Using cached LoRA %s", destination)
                return destination
            LOGGER.warning("Cached LoRA hash mismatch, re-downloading %s", destination)
            destination.unlink(missing_ok=True)
        else:
            LOGGER.info("Using cached LoRA %s", destination)
            return destination

    if not api_key:
        raise ValueError("Missing CIVITAI_API_KEY for LoRA download.")

    headers = {"Authorization": f"Bearer {api_key}"}
    LOGGER.info("Downloading LoRA to %s", destination)
    with requests.get(download_url, headers=headers, stream=True, timeout=300, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (128 * 1024 * 1024) < len(chunk):
                    LOGGER.info(
                        "Downloaded %.2f GB / %.2f GB",
                        downloaded / (1024**3),
                        total / (1024**3),
                    )

    if sha256:
        downloaded_hash = compute_sha256(destination)
        if downloaded_hash != sha256.upper():
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"LoRA SHA256 mismatch for {filename}: expected {sha256.upper()}, got {downloaded_hash}"
            )

    return destination


def convert_non_diffusers_wan_lora(path: Path) -> dict[str, torch.Tensor]:
    state_dict = safetensors.torch.load_file(str(path))
    stripped_state_dict = {key: value for key, value in state_dict.items() if not key.endswith(".alpha")}
    return _convert_non_diffusers_wan_lora_to_diffusers(stripped_state_dict)


def configure_lora_scale(pipe: WanImageToVideoPipeline, target: str, adapter_name: str, scale: float) -> None:
    if target == "transformer":
        pipe.transformer.set_adapters([adapter_name], weights=[scale])
    elif target == "transformer_2":
        if pipe.transformer_2 is None:
            raise ValueError("This pipeline has no transformer_2, but the LoRA target is transformer_2.")
        pipe.transformer_2.set_adapters([adapter_name], weights=[scale])
    elif target == "both":
        pipe.transformer.set_adapters([adapter_name], weights=[scale])
        if pipe.transformer_2 is not None:
            pipe.transformer_2.set_adapters([adapter_name], weights=[scale])
    else:
        raise ValueError(f"Unsupported CIVITAI_LORA_TARGET '{target}'.")


def load_pipeline() -> tuple[WanImageToVideoPipeline, dict[str, Any]]:
    global PIPELINE, PIPELINE_STATE

    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE, PIPELINE_STATE

        model_id = get_default("WAN_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        hf_token = os.environ.get("HF_TOKEN") or None
        cache_dir = ensure_directory(Path(os.environ.get("HF_HOME") or default_cache_root()))
        lora_dir = ensure_directory(default_lora_dir())
        lora_name = get_default("CIVITAI_LORA_NAME", "DR34ML4Y_I2V_14B_LOW_V2.safetensors")
        lora_url = get_default("CIVITAI_LORA_DOWNLOAD_URL", "https://civitai.com/api/download/models/2553271")
        lora_sha256 = os.environ.get("CIVITAI_LORA_SHA256") or None
        lora_target = get_default("CIVITAI_LORA_TARGET", "transformer_2").lower()
        lora_scale = float(get_default("WAN_DEFAULT_LORA_SCALE", "1.0"))
        api_key = os.environ.get("CIVITAI_API_KEY", "").strip()

        LOGGER.info("Loading Wan diffusers base %s", model_id)
        image_encoder = CLIPVisionModel.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
            cache_dir=str(cache_dir),
            token=hf_token,
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=str(cache_dir),
            token=hf_token,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            image_encoder=image_encoder,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=str(cache_dir),
            token=hf_token,
        )

        if getattr(pipe, "vae", None) is not None:
            if os.environ.get("WAN_ENABLE_VAE_TILING", "true").strip().lower() == "true" and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if os.environ.get("WAN_ENABLE_VAE_SLICING", "true").strip().lower() == "true" and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

        if os.environ.get("WAN_ENABLE_MODEL_CPU_OFFLOAD", "true").strip().lower() == "true":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        lora_path = download_civitai_lora(
            download_url=lora_url,
            api_key=api_key,
            filename=lora_name,
            sha256=lora_sha256,
            output_dir=lora_dir,
        )
        converted_lora = convert_non_diffusers_wan_lora(lora_path)
        adapter_name = "civitai-lora"
        LOGGER.info("Loading LoRA %s into %s", lora_path.name, lora_target)
        if lora_target in {"transformer", "both"}:
            pipe.transformer.load_lora_adapter(converted_lora, adapter_name=adapter_name)
        if lora_target in {"transformer_2", "both"}:
            if pipe.transformer_2 is None:
                raise ValueError("The Wan pipeline did not initialize transformer_2 required for this LoRA target.")
            pipe.transformer_2.load_lora_adapter(converted_lora, adapter_name=adapter_name)
        configure_lora_scale(pipe, lora_target, adapter_name, lora_scale)

        PIPELINE = pipe
        PIPELINE_STATE = {
            "model_id": model_id,
            "lora_name": lora_name,
            "lora_target": lora_target,
            "adapter_name": adapter_name,
            "cache_dir": str(cache_dir),
            "lora_path": str(lora_path),
        }
        return PIPELINE, PIPELINE_STATE


def run_generation(job_input: dict[str, Any]) -> dict[str, Any]:
    pipe, pipe_state = load_pipeline()

    with INFERENCE_LOCK:
        source_image = image_source_to_pil(job_input)
        original_width, original_height = source_image.size

        width, height = resolve_generation_dimensions(
            original_width=original_width,
            original_height=original_height,
            width=job_input.get("width"),
            height=job_input.get("height"),
            resolution_preset=str(
                job_input.get(
                    "resolution_preset",
                    get_default("WAN_DEFAULT_RESOLUTION_PRESET", "720p"),
                )
            ).strip().lower(),
        )

        image = source_image.resize((width, height), RESAMPLING_LANCZOS)
        num_frames = normalize_frame_count(job_input.get("num_frames", get_default("WAN_DEFAULT_NUM_FRAMES", "81")))
        fps = int(job_input.get("fps", get_default("WAN_DEFAULT_FPS", "16")))
        num_inference_steps = int(
            job_input.get("num_inference_steps", job_input.get("steps", get_default("WAN_DEFAULT_NUM_INFERENCE_STEPS", "10")))
        )
        guidance_scale = float(job_input.get("guidance_scale", get_default("WAN_DEFAULT_GUIDANCE_SCALE", "5.0")))
        guidance_scale_2 = float(
            job_input.get("guidance_scale_2", get_default("WAN_DEFAULT_GUIDANCE_SCALE_2", str(guidance_scale)))
        )
        flow_shift = float(job_input.get("flow_shift", get_default("WAN_DEFAULT_FLOW_SHIFT", "8.0")))
        lora_scale = float(job_input.get("lora_scale", get_default("WAN_DEFAULT_LORA_SCALE", "1.0")))
        seed = coerce_seed(job_input.get("seed"))
        prompt = str(job_input["prompt"]).strip()
        negative_prompt = str(job_input.get("negative_prompt", "")).strip() or None
        max_sequence_length = int(job_input.get("max_sequence_length", 512))

        configure_lora_scale(pipe, pipe_state["lora_target"], pipe_state["adapter_name"], lora_scale)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        LOGGER.info(
            "Generating video at %sx%s, frames=%s, steps=%s, guidance=(%s, %s), lora_scale=%s",
            width,
            height,
            num_frames,
            num_inference_steps,
            guidance_scale,
            guidance_scale_2,
            lora_scale,
        )

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            generator=generator,
            output_type="pil",
            max_sequence_length=max_sequence_length,
        )
        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_path = Path(temp_video.name)
        try:
            export_to_video(frames, str(temp_path), fps=fps)
            video_bytes = temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "video_bytes": video_bytes,
            "video_filename": f"wan_civitai_{seed}.mp4",
            "model": {
                "model_id": pipe_state["model_id"],
            },
            "lora": {
                "name": pipe_state["lora_name"],
                "target": pipe_state["lora_target"],
                "scale": lora_scale,
                "source": "civitai",
            },
            "input_image": {
                "width": original_width,
                "height": original_height,
            },
            "generation": {
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "flow_shift": flow_shift,
                "seed": seed,
            },
        }
