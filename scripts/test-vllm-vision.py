#!/usr/bin/env python3
"""Test script for vLLM Qwen3.5-4B vision endpoint.

Sends an image to the vLLM OpenAI-compatible API and verifies vision works.
Uses sampling params from the Qwen3.5 model card (thinking mode + vision).

Usage:
    python scripts/test-vllm-vision.py [IMAGE_URL_OR_PATH]

If no image is provided, uses a default test image from the web.

Requirements:
    pip install openai   (or: .venv/bin/pip install openai)
"""

import argparse
import base64
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run:")
    print("  /home/bppc/projects/llm-server/.venv/bin/pip install openai")
    sys.exit(1)

VLLM_BASE_URL = "http://localhost:8090/v1"
MODEL_NAME = "qwen3.5-4b"

# Qwen3.5 model card recommended sampling for vision (thinking mode)
SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "max_tokens": 8192,  # reduced for test; production can use up to 81920
    "extra_body": {"top_k": 20},
}

DEFAULT_TEST_IMAGE = "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"


def encode_local_image(path: str) -> str:
    """Encode a local image file as a base64 data URL."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    suffix = p.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/jpeg")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def build_image_content(image_input: str) -> dict:
    """Build the image_url content block from a URL or local path."""
    if image_input.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": image_input}}
    else:
        data_url = encode_local_image(image_input)
        return {"type": "image_url", "image_url": {"url": data_url}}


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Qwen3.5-9B vision")
    parser.add_argument("image", nargs="?", default=DEFAULT_TEST_IMAGE,
                        help="Image URL or local file path (default: cat photo from Wikipedia)")
    parser.add_argument("--prompt", default="Describe this image in detail. What do you see?",
                        help="Text prompt to send with the image")
    parser.add_argument("--base-url", default=VLLM_BASE_URL,
                        help=f"vLLM API base URL (default: {VLLM_BASE_URL})")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Model name (default: {MODEL_NAME})")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # Check health first
    print(f"Connecting to {args.base_url} ...")
    try:
        models = client.models.list()
        print(f"Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM at {args.base_url}")
        print(f"  {e}")
        print("\nIs the vLLM service running?")
        print("  docker compose --profile vllm up -d")
        sys.exit(1)

    image_content = build_image_content(args.image)
    print(f"\nSending vision request...")
    print(f"  Image: {args.image[:80]}{'...' if len(args.image) > 80 else ''}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Model: {args.model}")
    print(f"  Sampling: temp={SAMPLING_PARAMS['temperature']}, "
          f"top_p={SAMPLING_PARAMS['top_p']}, "
          f"presence_penalty={SAMPLING_PARAMS['presence_penalty']}, "
          f"max_tokens={SAMPLING_PARAMS['max_tokens']}")
    print()

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        image_content,
                    ],
                }
            ],
            **SAMPLING_PARAMS,
        )
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)

    # Print response
    message = response.choices[0].message
    content = message.content or ""

    # Separate thinking from response if present
    if "<think>" in content and "</think>" in content:
        think_start = content.index("<think>")
        think_end = content.index("</think>") + len("</think>")
        thinking = content[think_start:think_end]
        answer = content[:think_start] + content[think_end:]
        answer = answer.strip()

        print("=== THINKING ===")
        print(thinking[7:-8].strip())  # Strip <think></think> tags
        print()
        print("=== RESPONSE ===")
        print(answer)
    else:
        print("=== RESPONSE ===")
        print(content)

    # Print usage stats
    if response.usage:
        u = response.usage
        print(f"\n--- Usage: {u.prompt_tokens} prompt + {u.completion_tokens} completion = {u.total_tokens} total tokens ---")

    print("\nVision test PASSED." if content.strip() else "\nWARNING: Empty response.")


if __name__ == "__main__":
    main()
