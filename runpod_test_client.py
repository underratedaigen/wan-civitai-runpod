import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 7863
POLL_INTERVAL_SECONDS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "local_test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WAN Civitai Runpod Tester</title>
  <style>
    :root {
      --bg: #0d1017;
      --panel: #171b22;
      --panel-2: #1f2530;
      --text: #eef2ff;
      --muted: #a8b3c5;
      --accent: #7dd3fc;
      --accent-2: #86efac;
      --border: #2a3240;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(125,211,252,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(134,239,172,0.12), transparent 22%),
        linear-gradient(180deg, #0b1017 0%, #111825 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap {
      width: min(1100px, calc(100% - 32px));
      margin: 24px auto 40px;
      display: grid;
      gap: 20px;
    }
    .panel {
      background: rgba(23, 27, 34, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.26);
    }
    h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.1; }
    p { margin: 0; color: var(--muted); }
    form { display: grid; gap: 14px; margin-top: 18px; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    label { display: grid; gap: 8px; font-size: 13px; color: var(--muted); }
    input, textarea, select, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    textarea { min-height: 130px; resize: vertical; }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #081116;
      font-weight: 700;
      border: none;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .status { display: grid; gap: 10px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(125, 211, 252, 0.1);
      color: #bfeeff;
      border: 1px solid rgba(125, 211, 252, 0.25);
      font-weight: 600;
    }
    .meta, pre {
      background: #0d121a;
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 14px;
      overflow: auto;
    }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 320px; }
    video {
      width: 100%;
      max-width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #0b0f14;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>WAN + Civitai LoRA Tester</h1>
      <p>Upload an image, send it to your Runpod endpoint, and preview the returned MP4 when it is finished.</p>
      <form id="job-form">
        <div class="grid">
          <label>Runpod Endpoint ID
            <input name="endpoint_id" required>
          </label>
          <label>Runpod API Key
            <input name="api_key" type="password" required>
          </label>
        </div>
        <label>Prompt
          <textarea name="prompt" required>YOUR_TRIGGER_WORD, cinematic image-to-video motion, stable framing, natural subject movement, realistic lighting, detailed skin and fabric textures.</textarea>
        </label>
        <label>Negative Prompt
          <textarea name="negative_prompt" placeholder="Optional"></textarea>
        </label>
        <div class="grid">
          <label>Input Image
            <input name="image_file" type="file" accept="image/*" required>
          </label>
          <label>Resolution Preset
            <select name="resolution_preset">
              <option value="720p" selected>720p</option>
              <option value="480p">480p</option>
            </select>
          </label>
          <label>Frames
            <input name="num_frames" type="number" min="1" value="81">
          </label>
          <label>FPS
            <input name="fps" type="number" min="1" value="16">
          </label>
          <label>Inference Steps
            <input name="num_inference_steps" type="number" min="1" value="10">
          </label>
          <label>Guidance Scale
            <input name="guidance_scale" type="number" step="0.1" value="5.0">
          </label>
          <label>Guidance Scale 2
            <input name="guidance_scale_2" type="number" step="0.1" value="6.0">
          </label>
          <label>LoRA Scale
            <input name="lora_scale" type="number" step="0.05" value="1.0">
          </label>
          <label>Flow Shift
            <input name="flow_shift" type="number" step="0.1" value="8.0">
          </label>
          <label>Seed
            <input name="seed" type="number" value="42">
          </label>
        </div>
        <button id="submit-btn" type="submit">Generate Video</button>
      </form>
    </div>

    <div class="panel status">
      <div id="state-pill" class="pill">Idle</div>
      <div id="state-text" class="meta">Fill the form and submit a job.</div>
      <video id="video-output" class="hidden" controls playsinline></video>
      <pre id="json-output">{}</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById("job-form");
    const submitButton = document.getElementById("submit-btn");
    const statePill = document.getElementById("state-pill");
    const stateText = document.getElementById("state-text");
    const jsonOutput = document.getElementById("json-output");
    const videoOutput = document.getElementById("video-output");
    let pollHandle = null;

    function setState(label, text) {
      statePill.textContent = label;
      stateText.textContent = text;
    }

    function resetOutputs() {
      videoOutput.pause();
      videoOutput.removeAttribute("src");
      videoOutput.classList.add("hidden");
      jsonOutput.textContent = "{}";
    }

    function showResult(data) {
      jsonOutput.textContent = JSON.stringify(data, null, 2);
      const result = data.result || {};
      if (result.video_url) {
        videoOutput.src = result.video_url;
        videoOutput.classList.remove("hidden");
      }
    }

    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Failed to read the selected file."));
        reader.readAsDataURL(file);
      });
    }

    async function pollStatus(localJobId) {
      if (pollHandle) {
        clearInterval(pollHandle);
      }

      const runPoll = async () => {
        const response = await fetch(`/api/status?id=${encodeURIComponent(localJobId)}`);
        const data = await response.json();
        showResult(data);
        const label = data.state || "UNKNOWN";
        const remote = data.remote_status ? ` | Runpod: ${data.remote_status}` : "";
        setState(label, (data.message || "Waiting") + remote);
        if (["COMPLETED", "FAILED"].includes(label)) {
          clearInterval(pollHandle);
          pollHandle = null;
          submitButton.disabled = false;
        }
      };

      await runPoll();
      pollHandle = setInterval(runPoll, 3000);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submitButton.disabled = true;
      resetOutputs();
      setState("SUBMITTING", "Submitting job to the local proxy...");

      const formData = new FormData(form);
      const imageFile = formData.get("image_file");
      if (!(imageFile instanceof File) || !imageFile.size) {
        setState("FAILED", "Please choose an image file.");
        submitButton.disabled = false;
        return;
      }

      const payload = Object.fromEntries(formData.entries());
      payload.image_data_url = await readFileAsDataUrl(imageFile);
      delete payload.image_file;

      const response = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      showResult(data);

      if (!response.ok) {
        setState("FAILED", data.message || "Request failed.");
        submitButton.disabled = false;
        return;
      }

      setState(data.state || "SUBMITTED", data.message || "Job submitted.");
      await pollStatus(data.local_job_id);
    });
  </script>
</body>
</html>
"""


def _set_job(local_job_id: str, **updates) -> dict:
    with JOBS_LOCK:
        job = JOBS.setdefault(local_job_id, {})
        job.update(updates)
        return dict(job)


def _get_job(local_job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(local_job_id)
        return dict(job) if job else None


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _http_json(method: str, url: str, headers: dict | None = None, body: dict | None = None) -> dict:
    payload = None
    final_headers = dict(headers or {})
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        final_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, headers=final_headers, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _save_output_bytes(local_job_id: str, remote_file: dict) -> str | None:
    data = remote_file.get("data")
    output_type = remote_file.get("type")
    filename = remote_file.get("filename") or f"{local_job_id}.bin"
    if not data:
        return None

    if output_type == "bucket_url":
        return str(data)

    if output_type != "base64":
        return None

    output_path = OUTPUT_DIR / f"{local_job_id}_{Path(filename).name}"
    output_path.write_bytes(base64.b64decode(data))
    return f"/outputs/{output_path.name}"


def _strip_data_uri(data: str) -> str:
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        return data.split(",", 1)[1]
    return data


def _build_runpod_input(form_data: dict[str, str], image_base64: str) -> dict:
    payload: dict[str, object] = {
        "prompt": form_data["prompt"],
        "image_base64": _strip_data_uri(image_base64),
        "resolution_preset": form_data["resolution_preset"],
        "num_frames": int(form_data["num_frames"]),
        "fps": int(form_data["fps"]),
        "num_inference_steps": int(form_data["num_inference_steps"]),
        "guidance_scale": float(form_data["guidance_scale"]),
        "guidance_scale_2": float(form_data["guidance_scale_2"]),
        "lora_scale": float(form_data["lora_scale"]),
        "flow_shift": float(form_data["flow_shift"]),
        "seed": int(form_data["seed"]),
    }
    negative_prompt = form_data.get("negative_prompt", "").strip()
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload


def _process_job(local_job_id: str, endpoint_id: str, api_key: str, runpod_input: dict) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        submit_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        submit_response = _http_json("POST", submit_url, headers=headers, body={"input": runpod_input})
        remote_job_id = submit_response.get("id")
        if not remote_job_id:
            raise ValueError(f"Runpod response missing job id: {submit_response}")

        _set_job(
            local_job_id,
            state="SUBMITTED",
            message="Job accepted by Runpod.",
            remote_job_id=remote_job_id,
            remote_status=submit_response.get("status", "IN_QUEUE"),
            raw=submit_response,
        )

        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{remote_job_id}"
        while True:
            status_response = _http_json("GET", status_url, headers=headers)
            remote_status = str(status_response.get("status", "UNKNOWN"))
            _set_job(
                local_job_id,
                state="RUNNING" if remote_status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"} else remote_status,
                message="Waiting for Runpod to finish the job.",
                remote_status=remote_status,
                raw=status_response,
            )

            if remote_status == "COMPLETED":
                output = status_response.get("output", {})
                result: dict[str, str] = {}
                videos = output.get("videos") or []
                if videos:
                    video_url = _save_output_bytes(local_job_id, videos[0])
                    if video_url:
                        result["video_url"] = video_url
                _set_job(
                    local_job_id,
                    state="COMPLETED",
                    message="Video generation finished.",
                    remote_status=remote_status,
                    raw=status_response,
                    result=result,
                )
                return

            if remote_status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                error_text = status_response.get("error") or status_response.get("message") or "Runpod job failed."
                _set_job(
                    local_job_id,
                    state="FAILED",
                    message=str(error_text),
                    remote_status=remote_status,
                    raw=status_response,
                )
                return

            time.sleep(POLL_INTERVAL_SECONDS)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        _set_job(local_job_id, state="FAILED", message=f"HTTP {exc.code}: {details}", raw={"error": details})
    except Exception as exc:  # noqa: BLE001
        _set_job(local_job_id, state="FAILED", message=str(exc), raw={"error": str(exc)})


class WanTesterHandler(BaseHTTPRequestHandler):
    server_version = "WanCivitaiLocalTester/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/status"):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            local_job_id = params.get("id", [""])[0]
            job = _get_job(local_job_id)
            if not job:
                _json_response(self, {"message": "Unknown local job id."}, status=404)
                return
            _json_response(self, {"local_job_id": local_job_id, **job})
            return

        if self.path.startswith("/outputs/"):
            filename = self.path.removeprefix("/outputs/")
            file_path = OUTPUT_DIR / Path(filename).name
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Output file not found.")
                return
            payload = file_path.read_bytes()
            mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        if self.path != "/api/submit":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            endpoint_id = str(payload.get("endpoint_id", "")).strip()
            api_key = str(payload.get("api_key", "")).strip()
            prompt = str(payload.get("prompt", "")).strip()
            image_data_url = str(payload.get("image_data_url", "")).strip()

            if not endpoint_id:
                raise ValueError("Endpoint ID is required.")
            if not api_key:
                raise ValueError("API key is required.")
            if not prompt:
                raise ValueError("Prompt is required.")
            if not image_data_url:
                raise ValueError("Please choose an image file.")

            form_data = {
                "prompt": prompt,
                "negative_prompt": str(payload.get("negative_prompt", "")),
                "resolution_preset": str(payload.get("resolution_preset", "720p")),
                "num_frames": str(payload.get("num_frames", "81")),
                "fps": str(payload.get("fps", "16")),
                "num_inference_steps": str(payload.get("num_inference_steps", "10")),
                "guidance_scale": str(payload.get("guidance_scale", "5.0")),
                "guidance_scale_2": str(payload.get("guidance_scale_2", "6.0")),
                "lora_scale": str(payload.get("lora_scale", "1.0")),
                "flow_shift": str(payload.get("flow_shift", "8.0")),
                "seed": str(payload.get("seed", "42")),
            }

            runpod_input = _build_runpod_input(form_data, image_data_url)
            local_job_id = uuid.uuid4().hex
            _set_job(
                local_job_id,
                state="QUEUED",
                message="Local proxy accepted the job and is sending it to Runpod.",
                remote_status="PENDING",
                result={},
                raw={},
            )

            worker = threading.Thread(
                target=_process_job,
                args=(local_job_id, endpoint_id, api_key, runpod_input),
                daemon=True,
            )
            worker.start()

            _json_response(
                self,
                {
                    "local_job_id": local_job_id,
                    "state": "QUEUED",
                    "message": "Job queued locally. Polling will begin automatically.",
                    "result": {},
                    "raw": {},
                },
            )
        except json.JSONDecodeError:
            _json_response(self, {"message": "Invalid JSON payload."}, status=400)
        except ValueError as exc:
            _json_response(self, {"message": str(exc)}, status=400)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), WanTesterHandler)
    print(f"WAN Civitai tester running at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
