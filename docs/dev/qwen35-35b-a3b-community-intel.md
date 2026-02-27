# Qwen3.5-35B-A3B — Community Intelligence

Compiled 2026-02-25 from r/LocalLLaMA posts on release day. For use in the next build session to inform model migration from Qwen3-Next-80B-A3B.

---

## Model Specs

| Spec | Qwen3-Next-80B-A3B (current) | Qwen3.5-35B-A3B (new) |
|------|------------------------------|------------------------|
| Total params | 80B | 35B |
| Active params | ~3.9B | ~3B |
| Architecture | `qwen3moe` | `qwen35moe` |
| Layers | 48 | 40 (10 blocks of 4) |
| Layer pattern | 36 Gated DeltaNet + 12 GQA | 3x(GatedDeltaNet+MoE) + 1x(GatedAttention+MoE), repeated 10x |
| Experts | 512 routed + 1 shared, top-10 | 256 routed + 1 shared, top-8 |
| Expert intermediate dim | ~1536 | 512 |
| Hidden dim | 2048 | 2048 |
| Q8_0 GGUF | 84.8 GB | 36.9 GB |
| Q4_K_M GGUF | ~42 GB | ~21 GB |
| MXFP4_MOE GGUF | N/A | ~18.4 GB |
| UD-Q4_K_XL GGUF | N/A | ~20 GB |
| Context | 32K native | 262K native, 1M with YaRN |
| Vision | No | Yes (image + video, mmproj file) |
| Vocab size | 152K | 248K |
| Thinking mode | No | Yes (default on, can disable) |
| Instruct variant | Separate `-Instruct` | IS the instruct model |
| Multi-token prediction | No | Yes (trained with MTP) |

## GGUF Sizes (Unsloth)

| Quant | Size | Notes |
|-------|------|-------|
| Q2_K | 12.8 GB | |
| UD-IQ2_M | ~15 GB | Unsloth dynamic, fits dual 3090s for 122B variant |
| Q3_K_M | 14.7 GB | |
| UD-Q3_K_XL | ~15 GB | Unsloth dynamic |
| Q4_0 | 19.7 GB | |
| Q4_K_M | ~21 GB | Used in the 5090 benchmark post |
| UD-Q4_K_XL | ~20 GB | 180 t/s on 5090 fully in VRAM |
| MXFP4_MOE | 18.4 GB | Unsloth, 100+ t/s on 3090 |
| Q6_K | 28.5 GB | |
| UD-Q6_K_XL | ~27 GB | |
| Q8_0 | 36.9 GB | |

## PCIe Transfer Budget (Our Setup)

Estimated per-token expert data transfer with CPU offloading:
- **Old (80B)**: 10 experts x ~3.15 MB x 48 layers = ~1.51 GB/token
- **New (35B)**: 8 experts x ~3 MB x 40 layers = ~0.96 GB/token
- **Reduction**: ~37% less PCIe bandwidth per token

At 64 GB/s PCIe 5.0: theoretical ceiling ~67 tok/s (up from ~42). Real-world with overhead: **30-50 tok/s** expected vs our current ~22.

---

## Benchmark Data (Community)

### Head-to-Head: Qwen3-30B-A3B vs Qwen3.5-35B-A3B (RTX 5090, Q4_K_M)

Source: u/3spky5u-oss, llama.cpp b8115, both fully in VRAM, 12 threads.

#### Raw Speed

| Test | 30B tok/s | 3.5 tok/s | Delta |
|------|-----------|-----------|-------|
| Short (8-9 tok) | 248.2 | 169.5 | -32% |
| Medium (73-78 tok) | 236.1 | 163.5 | -31% |
| Long-form (800 tok) | 232.6 | 116.3 | -50% |
| Code gen (298-400 tok) | 233.9 | 161.6 | -31% |
| Reasoning (200 tok) | 234.8 | 158.2 | -33% |
| **Average** | **237.1** | **153.8** | **-35%** |

**Note**: Both models fit fully in 5090 VRAM (32 GB). The 35B is slower because it has more total params despite fewer active — more data loaded per forward pass. This gap is specific to all-GPU scenarios. For **CPU offload** (our case) the gap should be much smaller or reversed since we're PCIe-bottlenecked and the 35B has less to transfer.

#### Context Scaling (THE killer feature)

| Context Tokens | 30B gen tok/s | 3.5 gen tok/s | 30B degradation | 3.5 degradation |
|----------------|---------------|---------------|-----------------|-----------------|
| 512 | 237.9 | 160.1 | baseline | baseline |
| 1,024 | 232.8 | 159.5 | -2.1% | -0.4% |
| 2,048 | 224.1 | 161.3 | -5.8% | +0.7% |
| 4,096 | 205.9 | 161.4 | -13.4% | +0.8% |
| 8,192 | 186.6 | 158.6 | -21.5% | **-0.9%** |

The 3.5 is **essentially flat** across context sizes. The 30B (and by extension 80B) degrades ~21% from 512 to 8K. This is the Gated DeltaNet hybrid attention architecture advantage.

#### Multi-Turn Degradation

| Turn | 30B tok/s | 3.5 tok/s |
|------|-----------|-----------|
| 1 | 234.4 | 161.0 |
| 5 | 215.8 | 160.0 |
| **Degradation** | **-7.9%** | **-0.6%** |

#### Thinking Mode

| Test | 30B think words | 3.5 think words | 30B tok/s | 3.5 tok/s |
|------|-----------------|-----------------|-----------|-----------|
| Sheep riddle | 585 | 223 | 229.5 | 95.6 |
| Bearing capacity | 2,100 | 1,240 | 222.8 | 161.4 |
| Logic puzzle | 943 | 691 | 226.2 | 161.2 |
| USCS classification | 1,949 | 1,563 | 221.7 | 160.7 |

3.5 thinks **more concisely** (fewer words) and reaches the answer more reliably within token budgets. But thinking mode slows it down on some prompts (95 tok/s on sheep riddle).

#### Quality

- Response quality: essentially a tie with slight 3.5 edge in structure/formatting
- RAG accuracy: 6/6 both
- JSON output: 4/4 both
- 3.5 generates wordier responses (~80% more tokens on average)
- Both correctly handle domain-specific tasks (engineering, coding, reasoning)

### Speed Reports by GPU (Fully in VRAM)

| GPU | Quant | tok/s | Context | Notes |
|-----|-------|-------|---------|-------|
| RTX 5090 (32 GB) | UD-Q4_K_XL | **180-185** | ? | u/Additional-Action566 |
| RTX 5090 (32 GB) | Q4_K_M | 153-169 | 32K | u/3spky5u-oss benchmark |
| RTX 5070 Ti + 5060 Ti | MXFP4_MOE | **70** | 131K | Vulkan, dual GPU, u/Corosus |
| RTX 4090 (24 GB) | UD-Q3_K_XL | **116** | 0 ctx | llama-bench, u/jiegec |
| RTX 3090 (24 GB) | MXFP4_MOE | **100+** | 131K | u/jslominski |
| RTX 3090 (24 GB) | MXFP4_MOE | 112 → 56 | 0-49K | llama-bench, u/jslominski |
| 2x RTX 3090 | UD-IQ2_M (122B!) | **~50** | 130K | 122B-A10B variant, 22.7 GB each |
| M4 Max (Mac) | ? | **60** | ? | LM Studio, u/zmanning |
| M4 Pro 48GB (Mac) | MXFP4 / Q4_K_L | **~30** | ? | u/kkb294 |
| P40 24GB | Q4_K_M | **37** | ? | u/l33t-Mt |
| 7800XT 16GB (ROCm/WSL2) | ? | 13 → 21 | 72K | After tuning, u/DeedleDumbDee |
| 7900XT 20GB (Vulkan) | ? | ~13 → 21 | 72K | u/Monad_Maya |
| RTX 4070 12GB + RAM | Q4 | **40** | 128K | 12 GB VRAM alloc, u/TheRealMasonMac |
| RTX 4070 6GB + disk | Q4 | **14** | 128K | Disk offload only, u/TheRealMasonMac |

### Speed with CPU Offload (Our Scenario)

| GPU | Quant | Config | tok/s | Context | Source |
|-----|-------|--------|-------|---------|--------|
| RTX 5080 16GB | UD-Q4_K_XL? | n-cpu-moe=16 (24/40 MoE on GPU) | **51** base, **31.7** at 233K tok | 256K | u/JoNike |

**This is the most relevant datapoint for us.** RTX 5080 16 GB, partial MoE offload. 51 t/s base, degrades gracefully to 31.7 at 180K words. Uses `--n-cpu-moe 16` to keep 24 of 40 MoE layers on GPU, q4_0 KV cache, flash attention.

---

## Critical Tips & Gotchas

### 1. KV Cache Quantization: Be Careful

> "this model might be sensitive to kv cache quantization. I had both K and V type set to q8_0 for the 35b moe model, but as the context grew to about 20-40K tokens, it kept making minor mistakes with LaTeX." — u/chickN00dle

> "you shouldn't need to quantize the k and v cache as the model is already really good at memory to kv cache ratio" — u/Odd-Ordinary-5922

**Decision needed**: Our current config uses `-ctk q8_0 -ctv q8_0`. The 35B may not benefit as much (or may degrade at long context). Test with and without KV quantization.

### 2. Chat Template Issue in GGUF

> "In llama.cpp, make sure to pass an explicit chat template from base model, not use the embedded one in gguf" — u/catplusplusok
> "One inside gguf is incomplete apparently"

**Action**: Download the chat template from the original HF model and pass via `--chat-template` flag. This fixes tool calling issues.

### 3. Tool Calling Requires Correct Setup

> "there's a difference between 'true openai tool calling' and whatever else people are doing. qwen3 needs the real one." — u/__SlimeQ__

> "I'm finding great success with qwen recommended values for thinking and precise coding in tool use: temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0" — u/guiopen

### 4. Thinking Mode Can Be Disabled

```bash
# Add to llama-server command:
--chat-template-kwargs '{"enable_thinking": false}'
```

When disabled, use different sampling params:
- **Thinking ON (coding)**: `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0`
- **Thinking ON (general)**: `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5`
- **Thinking OFF (general)**: `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5`
- **Thinking OFF (reasoning)**: `--temp 1.0 --top-p 1.0 --top-k 40 --min-p 0.0 --presence-penalty 1.5`

All modes: `--repeat-penalty 1.0`

### 5. `--n-cpu-moe` is Key for Partial Offload

u/JoNike on RTX 5080 16GB:
> "n-cpu-moe = 16 (24 of 40 MoE layers on GPU), 256K context, flash attention, q4_0 KV cache. VRAM: ~14.8 GB idle, ~15.2 GB peak at 180K word fill. Base: 51.1 t/s"

This is a huge finding. With 40 MoE layers (vs 48 in the 80B), keeping 24 on GPU means only 16 offloaded to CPU. Combined with smaller experts, this dramatically reduces PCIe traffic.

### 6. `--fit on` for Automatic VRAM Management

> "Use the `--fit on` argument with `--fit-target <mb>` which specifies how much VRAM you want to leave untouched (1024mb by default). By default, it loads from disk (mmap). But you can disable that with `--no-mmap`" — u/TheRealMasonMac

New llama.cpp feature that auto-determines GPU/CPU split. Worth testing.

### 7. Batch/UBatch Size Tuning (Critical for Prompt Processing)

> "You could try both [batch-size and ubatch-size] with some high values like 1024, 2048, 4096(max) for better t/s." — u/pmttyji

**Concrete llama-bench data** from u/Subject-Tea-5253 (build 2b6dfe824, MXFP4_MOE on 5090, `--n-cpu-moe 38`):

| batch | ubatch | pp1024 tok/s |
|-------|--------|-------------|
| 512 | 128 | 175 |
| 512 | 256 | 282 |
| 512 | 512 | 458 |
| 1024 | 512 | 458 |
| 1024 | 1024 | **706** |
| 2048 | 1024 | 702 |
| 2048 | 2048 | **707** |

**Key insight**: "Prompt processing speed is always high when batch and ubatch have the same value." Going from ubatch=128 to 512 gives **2.6x** pp speedup. Matching batch=ubatch=1024 gives **4x** speedup over 128.

**Implication for us**: We use `-b 4096 -ub 4096` which should be optimal. But this data is for prompt processing only — generation speed is PCIe-bottlenecked regardless of batch size in our offload scenario.

### 8. Larger Vocab = Slower Prompt Processing

The 3.5 has 248K vocab vs 152K in the 30B. This causes notably slower prompt processing (518 vs 774 tok/s in the benchmark). Not a concern for generation speed but affects TTFT.

### 9. Long Output Speed Drop

The 3.5 drops from ~160 to ~116 tok/s when generating 800+ tokens in a single response (in the 5090 all-VRAM test). Monitor this in our offload scenario.

### 10. Vision Requires mmproj File

The vision projector is a separate file: `Qwen3.5-35B-A3B-mmproj-F32.gguf` or `mmproj-BF16.gguf`. Pass via `--mmproj` flag. Some community reports of mmproj files being only 1KB (corrupt) — verify file size after download.

---

### 11. UD-Q4_K_XL Quality Issues Confirmed by Community

Multiple reports of poor quality from UD-Q4_K_XL on MoE models:
- HF discussion: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/1#699e0dd8a83362bde9a050a3
- u/Pristine-Woodpecker: "I'm getting bad results from the UD-Q4_K_XL as well. May switch to bartowski quants."
- u/Additional-Action566 (5090): MXFP4_MOE "ran 20-30 t/s slower" than UD-Q4_K_XL (i.e. ~155-165 vs 185 t/s) but implies better quality

**Our own PPL data confirms this** (Session 005, WikiText-2):
| Quant | PPL | vs Q8_0 |
|-------|-----|---------|
| Q8_0 | 6.5342 | baseline |
| Q4_K_M | 6.6688 | +2.1% |
| UD-Q4_K_XL | 7.1702 | **+9.7%** |

UD-Q4_K_XL is both larger AND worse quality than standard Q4_K_M on this MoE architecture.

### 12. MXFP4_MOE as Alternative to UD

u/Additional-Action566 reports MXFP4_MOE runs 20-30 t/s slower than UD-Q4_K_XL on 5090 (both fully in VRAM). MXFP4_MOE is 18.4 GB vs UD-Q4_K_XL's 20 GB. Quality comparison unknown but community sentiment suggests MXFP4 may be better for MoE models. Worth investigating as a Q4 alternative.

---

## Recommended Quant for Our Setup (RTX 5080 16GB + 128GB RAM)

| Quant | Model Size | Fits in VRAM? | Expected tok/s | Quality (PPL) | Rationale |
|-------|-----------|---------------|----------------|---------------|-----------|
| **Q8_0** | 36.9 GB | No (offload) | 30-50 | 6.53 (best) | Best quality, our RAM handles it easily |
| **Q4_K_M** | ~21 GB | Partial | 35-55 | 6.67 (+2.1%) | Best Q4 quality, well-tested standard quant |
| **MXFP4_MOE** | 18.4 GB | Partial | 40-60 | untested | Smallest, most VRAM-friendly. Quality TBD |
| ~~UD-Q4_K_XL~~ | ~20 GB | Partial | 40-60 | 7.17 (+9.7%) | **NOT RECOMMENDED** — worse than Q4_K_M on MoE |

Given our 128GB RAM, Q8_0 is affordable. **Q4_K_M** is the best lower-bit option — nearly half the size with only 2.1% PPL loss. The u/JoNike 5080 result (51 t/s with partial GPU offload) suggests Q4-class quants with `--n-cpu-moe` tuning could outperform Q8_0 with full CPU offload.

**Recommended test matrix** (updated with our findings):
1. Q8_0 + full expert offload (`-ot "exps=CPU"`) — quality baseline (**DONE: ~35 tok/s**)
2. Q4_K_M + `--n-cpu-moe 16` (24/40 on GPU) — speed target
3. Q4_K_M + full expert offload — compare with/without partial GPU
4. Q8_0 + `--n-cpu-moe 32` (8/40 on GPU) — partial offload for Q8

---

## Proposed Launch Command (Starting Point)

Based on community findings and our Session 002-004 config:

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q8_0.gguf \
  -c 65536 \
  -ngl 999 \
  -ot "exps=CPU" \
  -fa on \
  -t 20 \
  -b 4096 \
  -ub 4096 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Variants to test**:
- Replace `-ot "exps=CPU"` with `--n-cpu-moe 16` for partial GPU offload
- Try without `-ctk q8_0 -ctv q8_0` (community reports sensitivity)
- Try `--fit on` instead of manual layer assignment
- Add `--chat-template` from HF model (fix incomplete GGUF template)
- Add `--mmproj Qwen3.5-35B-A3B-mmproj-F32.gguf` for vision
- Context: start at 65K, test up to 262K

---

## Quality Assessment Summary

From community testing:
- **Comparable to Qwen3-235B-A22B** on many benchmarks (MMLU-Pro 85.3 vs 84.4)
- **Better than Qwen3-Next-80B-A3B** according to official Qwen claims
- Strong at agentic coding (passed mid-level dev recruitment test, Opencode workflows)
- Good tool use with correct sampling params
- Thinking mode is concise and efficient (reaches answers in fewer tokens than 30B)
- Quality is "a wash with slight 3.5 edge in structure/formatting" vs Qwen3-30B-A3B
- First open model to reliably complete complex coding tasks agentically on consumer hardware
- 201 languages supported

---

## What This Means for Our Project

1. **Model swap is a clear win**: Less than half the GGUF size, better quality, native vision, massive context
2. **All Session 002-004 optimizations still apply**: thread tuning, KV cache, expert offloading, spec decoding
3. **New optimization avenue**: `--n-cpu-moe` partial offload is highly effective (51 t/s on 5080!)
4. **New capability**: vision understanding for classification/analysis tasks
5. **Context window**: 262K native context unlocks RAG and long-conversation use cases
6. **Speed target**: 30-50+ tok/s is realistic (vs current 22 tok/s with the 80B)
7. **Thinking mode**: built-in chain-of-thought for reasoning tasks, can be toggled per-request

---

## Open Questions for Next Session

1. Does `qwen35moe` architecture work with our current Docker image / llama.cpp version?
2. What's the optimal `--n-cpu-moe` value for RTX 5080 16GB with Q8_0 vs Q4?
3. Does KV cache q8_0 help or hurt at long context (conflicting community reports)?
4. Is the GGUF chat template actually broken, or is that quant-specific?
5. What's the real-world speed difference between Q8_0 (quality) vs UD-Q4_K_XL (speed)?
6. Does vision (mmproj) work with our offloading setup?
7. Speculative decoding with Qwen3-1.7B draft — does it still apply to the 35B model?

---

## Sources

- [RTX 5090 Head-to-Head Benchmark](https://reddit.com/r/LocalLLaMA) — u/3spky5u-oss
- [Agentic Coding on RTX 3090](https://reddit.com/r/LocalLLaMA) — u/jslominski
- [Disable Thinking Mode](https://reddit.com/r/LocalLLaMA) — u/guiopen
- [Qwen3.5-35B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Unsloth GGUF Quants](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.5 Blog](https://qwen.ai/blog?id=qwen3.5)
