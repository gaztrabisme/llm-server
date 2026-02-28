"""Default TG prompts and workload definitions — ported from bench-matrix.sh."""

from __future__ import annotations

# Short prompt (~128 tokens) — simple question
SHORT_PROMPT = [
    {
        "role": "user",
        "content": (
            "Explain the concept of gradient descent in machine learning. "
            "Cover the basic idea, learning rate, and common variants like "
            "SGD, Adam, and RMSprop. Keep the explanation concise but thorough."
        ),
    }
]

# Medium prompt (~1024 tokens) — detailed technical question
MEDIUM_PROMPT = [
    {
        "role": "user",
        "content": (
            "Write a comprehensive guide to building a REST API with Python Flask. "
            "Cover the following topics in detail:\n\n"
            "1. Project setup and virtual environment\n"
            "2. Basic route definitions and HTTP methods (GET, POST, PUT, DELETE)\n"
            "3. Request parsing and validation\n"
            "4. Database integration with SQLAlchemy\n"
            "5. Error handling and custom error responses\n"
            "6. Authentication with JWT tokens\n"
            "7. Rate limiting and security best practices\n"
            "8. Testing with pytest\n"
            "9. Deployment considerations\n\n"
            "For each topic, provide code examples and explain the reasoning behind "
            "design decisions. Discuss trade-offs between different approaches where "
            "relevant. Include common pitfalls and how to avoid them.\n\n"
            "Also compare Flask with FastAPI and Django REST Framework, highlighting "
            "when each framework is most appropriate. Consider factors like performance, "
            "ease of use, community support, and ecosystem maturity.\n\n"
            "Finally, outline a production-ready project structure with proper separation "
            "of concerns, configuration management, logging, and monitoring."
        ),
    }
]

# Long prompt (~4096 tokens) — multi-part technical question
LONG_PROMPT = [
    {
        "role": "user",
        "content": (
            "I need a comprehensive technical deep-dive into modern GPU architectures "
            "and their impact on machine learning workloads. Please cover all of the "
            "following areas in detail:\n\n"
            "## Part 1: GPU Architecture Fundamentals\n"
            "Explain the evolution from graphics-focused GPUs to general-purpose compute "
            "GPUs (GPGPU). Cover the key architectural components: streaming multiprocessors "
            "(SMs), CUDA cores, tensor cores, memory hierarchy (registers, shared memory, "
            "L1/L2 cache, global memory), and the memory bus. Explain how warp scheduling "
            "works and why it matters for throughput.\n\n"
            "## Part 2: NVIDIA Architecture Generations\n"
            "Compare the following NVIDIA architectures in detail: Pascal (P100), Volta "
            "(V100), Ampere (A100), Hopper (H100), and Blackwell (B200). For each, discuss "
            "the number of SMs, CUDA cores, tensor cores, memory bandwidth, interconnect "
            "(NVLink generations), and key new features. Explain how each generation improved "
            "on the previous one for ML workloads specifically.\n\n"
            "## Part 3: Memory Systems and Bandwidth\n"
            "Deep dive into GPU memory systems. Explain HBM vs GDDR memory, their bandwidth "
            "characteristics, and why memory bandwidth is often the bottleneck in LLM "
            "inference. Cover memory coalescing, bank conflicts, and strategies for "
            "optimizing memory access patterns. Discuss the role of memory in batch "
            "processing vs single-request latency.\n\n"
            "## Part 4: Tensor Cores and Mixed Precision\n"
            "Explain how tensor cores work, from the original FP16 tensor cores in Volta "
            "to the FP8/FP4 capabilities in Blackwell. Cover mixed-precision training "
            "and inference, including techniques like automatic mixed precision (AMP), "
            "loss scaling, and when different precision levels are appropriate. Discuss "
            "the relationship between quantization and tensor core utilization.\n\n"
            "## Part 5: Multi-GPU and Distributed Computing\n"
            "Cover multi-GPU setups: NVLink, NVSwitch, InfiniBand, and RoCE. Explain "
            "data parallelism, model parallelism (tensor and pipeline), and expert "
            "parallelism for MoE models. Discuss the communication overhead and how "
            "it affects scaling efficiency. Include NCCL and its role in distributed "
            "training.\n\n"
            "## Part 6: Consumer vs Data Center GPUs for ML\n"
            "Compare consumer GPUs (RTX 4090, RTX 5090) with data center GPUs (A100, "
            "H100) for ML workloads. Discuss the trade-offs in terms of memory capacity, "
            "bandwidth, precision support, multi-GPU scaling, and cost effectiveness. "
            "When is it reasonable to use consumer hardware vs renting cloud GPUs?\n\n"
            "## Part 7: Software Stack\n"
            "Explain the CUDA software stack: CUDA runtime, cuBLAS, cuDNN, CUTLASS, "
            "and their roles. Cover alternative frameworks like ROCm for AMD GPUs "
            "and Metal for Apple Silicon. Discuss the compiler toolchain (nvcc, PTX) "
            "and profiling tools (Nsight, nvprof). How does the software ecosystem "
            "affect the choice between NVIDIA and alternatives?\n\n"
            "## Part 8: Inference Optimization Techniques\n"
            "Cover modern inference optimization: KV-cache, flash attention, paged "
            "attention, continuous batching, speculative decoding, and quantization "
            "(GPTQ, AWQ, GGML/GGUF formats). Explain how each technique improves "
            "throughput or latency and the trade-offs involved. Discuss serving "
            "frameworks like vLLM, TensorRT-LLM, and llama.cpp.\n\n"
            "For each section, include specific numbers, benchmarks, and practical "
            "recommendations. Where possible, explain the underlying physics and "
            "engineering constraints that drive the design decisions."
        ),
    }
]

# Multi-turn conversation follow-up prompts
MULTI_TURN_FOLLOWUPS = [
    "Can you elaborate on the most important point from your previous response? "
    "What are the practical implications?",

    "What are the common mistakes or misconceptions related to what you just explained? "
    "How can they be avoided?",

    "How does this compare to alternative approaches? What are the trade-offs?",

    "Can you provide a concrete example or case study that illustrates these concepts "
    "in a real-world scenario?",
]


def get_workload(name: str) -> list[dict]:
    """Get a workload by name. Returns the messages list."""
    workloads = {
        "short": SHORT_PROMPT,
        "medium": MEDIUM_PROMPT,
        "long": LONG_PROMPT,
    }
    if name not in workloads:
        raise ValueError(f"Unknown workload: {name}. Available: {list(workloads.keys())}")
    return workloads[name]


def get_workload_config(name: str) -> dict:
    """Get workload config (prompt_tokens estimate, max_tokens)."""
    configs = {
        "short": {"prompt_tokens_est": 128, "max_tokens": 256},
        "medium": {"prompt_tokens_est": 1024, "max_tokens": 256},
        "long": {"prompt_tokens_est": 4096, "max_tokens": 256},
    }
    return configs.get(name, {"prompt_tokens_est": 128, "max_tokens": 256})
