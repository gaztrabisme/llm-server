# Speculative Decoding for Qwen3-Next-80B-A3B with MoE Expert Offloading

Research date: 2026-02-25

## TL;DR

Speculative decoding is **feasible and promising** for our MoE-offloaded setup, but comes with important caveats. Two approaches exist: (1) external draft model using a small Qwen3 dense model (0.6B-4B), and (2) built-in Multi-Token Prediction (MTP) native to Qwen3-Next. The draft model approach is available in llama.cpp today. MTP is only supported in vLLM/SGLang, not llama.cpp yet. Expected speedup: **1.5-3x** for draft model approach if acceptance rates are good. Recommended next step: download Qwen3-1.7B Q8_0 GGUF (~1.83 GB) and test with `--model-draft` flag.

---

## 1. How Speculative Decoding Works

Speculative decoding uses a smaller, faster "draft" model to predict N tokens ahead, then the larger "target" model verifies all N tokens in a single batch forward pass. Tokens that match are accepted instantly; mismatches cause regeneration from the first disagreement point.

The key insight for our setup: **each verification step requires only one PCIe expert transfer round-trip** regardless of how many tokens are being verified. If the draft model proposes 5 tokens and the target model verifies them in one batch, we effectively amortize the PCIe bandwidth cost across 5 tokens instead of 1.

### Why This Matters for MoE Offloading

Our current bottleneck is PCIe 5.0 bandwidth (~64 GB/s) transferring ~1.51 GB of expert weights per token. At ~22 tok/s, each token generation cycle costs ~46ms (dominated by expert transfer latency). With speculative decoding:

- Draft model generates N candidate tokens on GPU (fast, no PCIe transfers)
- Target model verifies all N tokens in one batch (one set of expert transfers, processed together)
- If acceptance rate is R, effective throughput multiplier is approximately: `accepted_tokens / (1 + verification_cost)`

The SpecMoEOff paper (2025) demonstrated **2.5x average decode throughput improvement** on Mixtral-8x7B with MoE offloading by hiding PCIe latency behind speculative token generation.

## 2. Available Approaches

### Approach A: External Draft Model (Available Now in llama.cpp)

Use a small Qwen3 dense model as the draft, verified by Qwen3-Next-80B-A3B as target.

**llama-server flags:**
```bash
./llama-server \
  -m ./Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf \
  -md ./Qwen3-1.7B-Q8_0.gguf \       # draft model
  -c 32768 \
  -ngl 999 \
  -ngld 999 \                          # draft model layers on GPU
  -ot "exps=CPU" \
  -fa on \
  -t 16 \
  --draft-max 8 \                      # max tokens to draft per iteration
  --draft-min 2 \                      # min tokens to draft
  --draft-p-min 0.75 \                 # probability threshold for acceptance
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Key flags:**
| Flag | Purpose | Default |
|------|---------|---------|
| `-md, --model-draft` | Path to draft model GGUF | (none) |
| `--draft-max N` | Max tokens to draft per iteration | 16 |
| `--draft-min N` | Min tokens to draft per iteration | 0 |
| `--draft-p-min P` | Probability threshold for accepting draft tokens | 0.75 |
| `-ngld N` | Number of draft model layers to offload to GPU | 0 |
| `-cd N` | Context size for draft model | same as -c |

**Compatibility requirements:**
- Draft and target models must share **compatible tokenization** (same vocab)
- Architecture can differ — llama.cpp does not require identical architectures
- `--spec-replace` flag available for models with slightly different tokenizers
- llama.cpp test suite validates speculative decoding with MoE target models

### Approach B: Built-in Multi-Token Prediction (MTP) — NOT Available in llama.cpp

Qwen3-Next has native MTP heads that predict multiple tokens per forward pass. This is the most efficient approach since it requires no separate model and uses the target model's own predictions.

**Status in inference engines:**
- **vLLM**: Supported via `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`
- **SGLang**: Supported via `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1`
- **llama.cpp**: **NOT supported yet** — MTP for Qwen3-Next architecture not implemented

This approach is blocked for our setup since we rely on llama.cpp for MoE expert offloading (`-ot "exps=CPU"`), which is not available in vLLM/SGLang for single-GPU consumer hardware.

### Approach C: N-gram Speculative Decoding (Available, No Draft Model Needed)

llama.cpp supports draftless speculation using n-gram pattern matching from conversation history. Zero VRAM overhead, useful for repetitive content.

```bash
./llama-server [...] --spec-type ngram-simple --draft-max 64
```

Other variants: `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` (~16 MB overhead).

**Best for:** code iteration, reasoning models with repeated thinking, summarization. Less effective for diverse/creative generation. Worth testing as a free baseline.

## 3. Draft Model Candidates

All Qwen3 dense models share the same tokenizer (BPE, vocab_size = 151,936), matching Qwen3-Next's vocabulary. This makes them compatible draft model candidates.

| Model | Params | Q8_0 GGUF Size | VRAM (fully on GPU) | Speed vs Target | Recommendation |
|-------|--------|----------------|---------------------|-----------------|----------------|
| **Qwen3-0.6B** | 0.6B | 639 MB | ~700 MB | Very fast | Good: minimal VRAM, but lower acceptance rate due to weak model |
| **Qwen3-1.7B** | 1.7B | 1.83 GB | ~2.0 GB | Fast | **Best candidate**: good balance of speed and prediction quality |
| **Qwen3-4B** | 4B | 4.28 GB | ~4.5 GB | Moderate | Risky: tight VRAM budget (6.2 + 4.5 = 10.7 GB of 16 GB) |
| **Qwen3-8B** | 8B | 8.5 GB | ~9.0 GB | Slow | Too large: exceeds VRAM budget |

### VRAM Budget Analysis

Current main model GPU usage: ~6.2 GB (attention layers, shared experts, embeddings, norms, KV cache q8_0 at 32k ctx).

Available VRAM for draft model: **16 GB - 6.2 GB = ~9.8 GB**

- Qwen3-0.6B Q8_0: 0.7 GB on GPU -> total ~6.9 GB -- plenty of headroom
- **Qwen3-1.7B Q8_0: 2.0 GB on GPU -> total ~8.2 GB** -- comfortable fit
- Qwen3-4B Q8_0: 4.5 GB on GPU -> total ~10.7 GB -- tight but possible
- Qwen3-4B also needs its own KV cache, pushing VRAM higher

**Recommendation: Qwen3-1.7B Q8_0** — fits comfortably in VRAM, fast enough to not bottleneck generation, large enough for reasonable acceptance rates.

### Architecture Compatibility Note

Qwen3 dense models use standard transformer attention, while Qwen3-Next uses hybrid Gated DeltaNet + Gated Attention. This **does not** block draft model speculative decoding — llama.cpp requires only vocabulary compatibility, not architectural identity. The draft model just proposes token IDs; the target model verifies them through its own forward pass. Different internal architectures are fine.

However, the architectural difference (and likely different training data distributions) may reduce acceptance rates compared to a distilled draft model. The Qwen3 dense models were not specifically trained to mimic Qwen3-Next's output distribution.

### Acceptance Rate Expectations

From community benchmarks (Discussion #10466):
- **Same-family distilled models** (e.g., Qwen2.5 14B + 0.5B): 2.5x speedup on coding, 1.4x on reasoning
- **Non-distilled same-vocab models**: significantly lower acceptance rates
- **Content-dependent**: coding/structured output benefits most; creative/diverse text benefits least

For Qwen3-1.7B -> Qwen3-Next-80B-A3B (non-distilled, same vocab):
- **Optimistic estimate**: 40-60% acceptance rate, 1.5-2x speedup
- **Conservative estimate**: 20-40% acceptance rate, 1.2-1.5x speedup
- **Worst case**: <20% acceptance rate, no speedup (overhead from running draft model)

## 4. Expected Speedup Analysis

### Theoretical Model

Current baseline: ~22 tok/s (config A2)

With speculative decoding (draft_max=8, acceptance_rate=R):
- Average accepted tokens per step: R * draft_max
- Verification overhead: ~1 token generation time (one batch forward pass)
- Draft overhead: negligible if draft is fully on GPU

Effective speedup ≈ (1 + R * draft_max) / (1 + draft_overhead_ratio)

| Acceptance Rate | Avg Accepted | Estimated Speedup | Estimated tok/s |
|----------------|-------------|-------------------|-----------------|
| 20% | 1.6 | 1.1-1.3x | 24-29 |
| 40% | 3.2 | 1.5-2.0x | 33-44 |
| 60% | 4.8 | 2.0-2.5x | 44-55 |
| 80% | 6.4 | 2.5-3.0x | 55-66 |

### Reality Check

The SpecMoEOff paper achieved 2.5x on Mixtral-8x7B, but that used EAGLE (a purpose-trained draft model), not a generic small model from the same family. Our non-distilled Qwen3-1.7B will likely have lower acceptance rates.

**Realistic expectation: 1.3-2.0x speedup (29-44 tok/s)**, depending heavily on:
1. Task type (structured > creative)
2. Actual acceptance rate (unknown until tested)
3. Draft model overhead (should be small if fully on GPU)

## 5. Potential Issues and Gotchas

### Known Risks

1. **Acceptance rate uncertainty**: Qwen3 dense models are NOT distilled from Qwen3-Next. The architectural difference (standard attention vs Gated DeltaNet) and different training runs may result in low acceptance rates that negate any speedup.

2. **Draft model KV cache**: The draft model needs its own KV cache, consuming additional VRAM. At 32k context with Qwen3-1.7B, this could be significant. Use `-cd` to set a smaller context for the draft model.

3. **Verification batch size with MoE offloading**: When the target model verifies N draft tokens in a batch, it still needs to transfer experts from CPU for each token in the batch. The batch verification may not be as fast as standard prompt processing because each token may route to different experts. This is the key uncertainty.

4. **llama-server vs llama-speculative-simple**: GitHub issue #12968 reports that speculative decoding in llama-server is less performant than the standalone `llama-speculative-simple` binary. May need to benchmark both.

5. **First-token latency increase**: Draft model adds overhead to each generation step, even when speculation fails. This slightly increases latency for the first token.

### Mitigations

- Start with `--draft-max 4` (conservative) and tune up
- Set `--draft-p-min 0.8` to avoid wasting cycles on low-confidence drafts
- Use `-cd 4096` for draft model context (smaller than main model's 32k)
- Monitor acceptance rate in server logs

## 6. Comparison with Other Optimization Avenues

| Approach | Expected Speedup | Complexity | Available Now | VRAM Cost |
|----------|-----------------|------------|---------------|-----------|
| **Speculative decoding (draft model)** | 1.3-2.0x | Medium | Yes | +2 GB |
| **Expert caching** | 1.2-1.5x (est.) | High | Partial | +0 GB |
| **Thread sweep** | 1.0-1.2x | Low | Yes | +0 GB |
| **KV cache q4_0** | 1.0-1.1x (longer ctx) | Low | Yes | -1 GB |
| **N-gram speculation** | 1.0-1.5x (content-dependent) | Low | Yes | +0 GB |
| **Built-in MTP (vLLM)** | 1.5-2.5x (est.) | N/A | Not in llama.cpp | +0 GB |

## 7. Recommended Testing Plan

### Phase 1: Quick Feasibility Test
1. Download Qwen3-1.7B-Q8_0.gguf (~1.83 GB) from HuggingFace
2. Launch with `--model-draft` flag and conservative settings
3. Run the existing benchmark suite (`scripts/bench.sh`)
4. Check acceptance rate in logs — if <20%, speculative decoding may not help

### Phase 2: Parameter Tuning (if Phase 1 shows promise)
1. Test draft-max values: 4, 8, 12, 16
2. Test draft-p-min values: 0.5, 0.75, 0.9
3. Try Qwen3-0.6B as alternative (faster draft, possibly lower acceptance)
4. Try n-gram speculation as zero-overhead baseline

### Phase 3: Combined Optimizations
1. Test speculative decoding + optimal thread count from thread sweep
2. Test speculative decoding + expert caching (if available)
3. Compare `llama-server` vs `llama-speculative-simple` performance

### Download Commands
```bash
# Recommended: Qwen3-1.7B Q8_0
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir ./models/

# Alternative: Qwen3-0.6B Q8_0 (smaller, faster, lower quality)
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir ./models/

# If VRAM allows: Qwen3-4B Q8_0 (better predictions, tighter fit)
huggingface-cli download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q8_0.gguf --local-dir ./models/
```

## 8. References

- [llama.cpp speculative decoding docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [SpecMoEOff: Hiding Offloading Latency with Speculative Decoding](https://arxiv.org/html/2508.21706v1) — 2.5x speedup on MoE offloading
- [llama.cpp Discussion #10466: Speculative decoding for consumer GPUs](https://github.com/ggml-org/llama.cpp/discussions/10466) — community benchmarks
- [Qwen3-Next-80B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) — MTP and architecture details
- [Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) — recommended draft model
- [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) — alternative lightweight draft
- [MoE offloading guide for llama.cpp](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [DeepWiki: llama.cpp speculative decoding architecture](https://deepwiki.com/ggml-org/llama.cpp/8.1-build-system-and-configuration)
