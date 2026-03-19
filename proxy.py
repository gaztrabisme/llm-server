"""LLM proxy — sampling presets, cold start, idle shutdown."""

import asyncio
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# --- Config ---

BACKEND_URL = os.getenv("LLM_PROXY_BACKEND", "http://localhost:8081")
PRESET_FILE = Path(__file__).parent / "presets.yaml"
IDLE_TIMEOUT = int(os.getenv("LLM_PROXY_IDLE_MINUTES", "30")) * 60
COMPOSE_DIR = Path(__file__).parent
COMPOSE_PROFILE = os.getenv("LLM_PROXY_PROFILE", "llama-cpp")
HEALTH_TIMEOUT = int(os.getenv("LLM_PROXY_HEALTH_TIMEOUT", "120"))
STARTUP_POLL_INTERVAL = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm-proxy")

# --- State ---

presets: dict[str, dict] = {}
last_request_time: float = 0.0
backend_running: bool = False
startup_lock = asyncio.Lock()


def load_presets() -> dict[str, dict]:
    if PRESET_FILE.exists():
        with open(PRESET_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


# --- Container lifecycle ---


def _compose(*args: str) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "--profile", COMPOSE_PROFILE, *args]
    log.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=COMPOSE_DIR, capture_output=True, text=True, timeout=300)


def is_backend_healthy() -> bool:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def ensure_backend() -> bool:
    """Start backend if not running. Returns True if healthy."""
    global backend_running

    if backend_running and is_backend_healthy():
        return True

    async with startup_lock:
        # Re-check after acquiring lock
        if is_backend_healthy():
            backend_running = True
            return True

        log.info("Backend not healthy, starting container...")
        result = _compose("up", "-d")
        if result.returncode != 0:
            log.error("Failed to start container: %s", result.stderr)
            return False

        # Poll for health
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(STARTUP_POLL_INTERVAL)
            if is_backend_healthy():
                backend_running = True
                log.info("Backend is healthy")
                return True

        log.error("Backend failed to become healthy within %ds", HEALTH_TIMEOUT)
        return False


def stop_backend():
    global backend_running
    log.info("Idle timeout reached, stopping container...")
    _compose("down")
    backend_running = False


# --- Idle monitor ---


async def idle_monitor():
    global last_request_time
    while True:
        await asyncio.sleep(60)
        if not backend_running:
            continue
        if IDLE_TIMEOUT <= 0:
            continue
        elapsed = time.monotonic() - last_request_time
        if elapsed > IDLE_TIMEOUT:
            stop_backend()


# --- Preset injection ---


PRESET_KEYS = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
               "frequency_penalty", "repetition_penalty", "max_tokens"}


def apply_preset(body: dict, mode: str) -> dict:
    preset = presets.get(mode)
    if not preset:
        return body

    for key, value in preset.items():
        if key in PRESET_KEYS and key not in body:
            body[key] = value

    return body


def extract_mode(request: Request) -> str | None:
    mode = request.query_params.get("mode")
    if not mode:
        mode = request.headers.get("x-mode")
    return mode


# --- App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    global presets, backend_running
    presets = load_presets()
    log.info("Loaded presets: %s", list(presets.keys()))
    backend_running = is_backend_healthy()
    if backend_running:
        log.info("Backend already running")
    task = asyncio.create_task(idle_monitor())
    yield
    task.cancel()


app = FastAPI(title="LLM Proxy", lifespan=lifespan)


@app.get("/health")
async def health():
    if backend_running and is_backend_healthy():
        return {"status": "ok", "backend": "running"}
    return JSONResponse({"status": "ok", "backend": "stopped"}, status_code=200)


@app.get("/presets")
async def list_presets():
    return presets


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    global last_request_time
    last_request_time = time.monotonic()

    # Cold start
    if not await ensure_backend():
        return JSONResponse(
            {"error": "Backend unavailable", "detail": "Container failed to start"},
            status_code=503,
            headers={"Retry-After": "30"},
        )

    # Build proxied request
    url = f"{BACKEND_URL}/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "x-mode", "content-length")}

    body = None
    mode = extract_mode(request)

    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        # Inject presets for JSON bodies on completions endpoints
        if mode and body and b"{" in body[:10]:
            try:
                body_dict = json.loads(body)
                body_dict = apply_preset(body_dict, mode)
                body = json.dumps(body_dict).encode()
                headers["content-type"] = "application/json"
            except (json.JSONDecodeError, ValueError):
                pass

    # Strip mode from query params
    params = {k: v for k, v in request.query_params.items() if k != "mode"}

    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

    # Check if this is a streaming request
    is_streaming = False
    if body and b'"stream"' in body and b"true" in body:
        is_streaming = True

    if is_streaming:
        req = client.build_request(
            method=request.method, url=url, headers=headers,
            content=body, params=params,
        )
        upstream = await client.send(req, stream=True)

        async def stream_body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            content=stream_body(),
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items()
                     if k.lower() not in ("content-length", "transfer-encoding")},
            media_type=upstream.headers.get("content-type"),
        )
    else:
        async with client:
            resp = await client.request(
                method=request.method, url=url, headers=headers,
                content=body, params=params,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
