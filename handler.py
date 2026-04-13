import base64
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

import runpod
from runpod.serverless.utils import rp_upload

from wan_lora_inference import run_generation


def bucket_upload_enabled(job_input: dict[str, Any]) -> bool:
    if job_input.get("upload_to_bucket") is False:
        return False
    return bool(os.environ.get("BUCKET_ENDPOINT_URL"))


def upload_file_to_bucket(job_id: str, filename: str, payload: bytes) -> str:
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(payload)
        temp_path = temp_file.name
    try:
        return rp_upload.upload_image(job_id, temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def serialize_output(job_id: str, filename: str, payload: bytes, to_bucket: bool) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if to_bucket:
        return {
            "filename": filename,
            "type": "bucket_url",
            "data": upload_file_to_bucket(job_id, filename, payload),
            "mime_type": mime_type,
        }

    return {
        "filename": filename,
        "type": "base64",
        "data": base64.b64encode(payload).decode("utf-8"),
        "mime_type": mime_type,
    }


def validate_input(job_input: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job_input, dict):
        raise ValueError("Job input must be an object.")

    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Missing required 'prompt'.")

    has_image_source = any(job_input.get(key) for key in ("image", "image_base64", "image_url"))
    if not has_image_source:
        raise ValueError("Provide an input image using 'image', 'image_base64', or 'image_url'.")

    return job_input


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = validate_input(job.get("input", {}))
    job_id = job.get("id", "wan-civitai-job")

    result = run_generation(job_input)
    serialized_video = serialize_output(
        job_id=job_id,
        filename=result["video_filename"],
        payload=result["video_bytes"],
        to_bucket=bucket_upload_enabled(job_input),
    )

    return {
        "model": result["model"],
        "lora": result["lora"],
        "input_image": result["input_image"],
        "generation": result["generation"],
        "videos": [serialized_video],
    }


runpod.serverless.start({"handler": handle_job})
