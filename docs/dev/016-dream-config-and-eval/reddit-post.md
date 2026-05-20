MTP (Multi-Token Prediction) just merged into mainline llama.cpp at b9190. I promised u/WarthogConfident4039 a Qwen3.6 benchmarking round. Three configs, tested at real coding-agent context lengths (not just 512 tokens). The main finding surprised me.

**TL;DR: 35B Q4_K_XL, no MTP, `--fit-target 1536`, 131k context. That's the config.** 56 tok/s generation, 1,584 tok/s prompt processing at 128k context. MTP doesn't help at 128k — both converge to the same speed. Skip the complexity. The 27B IQ3 is worth considering if 56k context is enough for you (or if you have a 12 GB card where the 35B won't fit).

## The Configs

| Config | 27B IQ3+MTP (A) | 35B Q4_K_XL+MTP (B) | 35B Q8_0+MTP (C) |
|---|---|---|---|
| Model | Qwen3.6-27B MTP-UD-IQ3_XXS | Qwen3.6-35B-A3B MTP-UD-Q4_K_XL | Qwen3.6-35B-A3B MTP-Q8_0 |
| Size | 12.45 GB | ~22 GB | ~36 GB |
| Source | Grafted (graft-mtp.py) | havenoammo | Grafted |
| GPU fit | Fully on GPU (66/66) | Partial offload | Heavy offload |

All tests on: **RTX 5080 16GB**, Ryzen 9 9950X, 128GB RAM, llama.cpp **b9204** (mainline).

Common MTP flags: `-np 1 --fit on -fa on -t 20 --no-mmap --jinja -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2`

## Results

### Speed — The MTP Surprise

### With MTP (mtp-bench, 9 prompt types)

| Metric | 27B IQ3 | **35B Q4_K_XL** | 35B Q8_0 |
|---|---|---|---|
| **Avg tok/s** | 73 | **74** | 46 |
| **Peak tok/s** | 83 (code) | **86 (translation)** | 51 |
| **MTP accept** | 74.4% | 79.5% | **80.1%** |
| **--fit-target** | 0 | 1536 | 1536 |

### The surprise: 35B is FASTER without MTP

| 35B Q4_K_XL config | --fit-target | MTP? | Avg tok/s | VRAM used |
|-----|-----------|------|-----------|-----------|
| Best (no MTP) | 0 | No | **97** | 15,815 MiB |
| Same VRAM budget | 1536 | No | 86 | 14,269 MiB |
| MTP enabled | 1536 | Yes | 74 | 14,623 MiB |

**MTP is 23% slower** for the 35B MoE on 16GB. Why?

1. MTP requires `--fit-target 1536` to reserve ~1.5 GB for the MTP compute buffer
2. That 1.5 GB pushes ~3 more MoE expert layers from GPU to CPU
3. CPU-bound expert layers are the bottleneck for MoE inference
4. MTP's multi-token speculation (~79% acceptance) doesn't compensate for the slower per-step speed

**For the 27B, MTP helps** because the model fits entirely on GPU (12.45 GB) — `--fit-target 0` works with and without MTP, so there's no VRAM penalty. The 27B goes from ~56 tok/s (no MTP, older builds) to 73 tok/s with MTP.

**Rule of thumb: MTP helps when your model fits on GPU. It hurts when the MTP compute buffer forces more layers to CPU.**

### Speed at Coding-Agent Context Lengths (the real test)

Everyone runs coding agents at 128k. Here's what actually happens as you fill the context window. Tested with synthetic prompts (Python classes, architecture docs, error stack traces — varied enough to prevent tokenizer compression), prompt cache disabled, 35B Q4_K_XL with `--fit-target 1536`:

| Context | PP (no MTP) | PP (MTP) | TG (no MTP) | TG (MTP) |
|---------|-------------|----------|-------------|----------|
| ~8k | 1,855 tok/s | 1,712 tok/s | 73 tok/s | 79 tok/s |
| ~32k | 1,810 tok/s | 1,674 tok/s | 74 tok/s | 70 tok/s |
| ~64k | 1,723 tok/s | 1,583 tok/s | 67 tok/s | 76 tok/s |
| **~128k** | **1,584 tok/s** | **1,437 tok/s** | **56 tok/s** | **56 tok/s** |

*8k/32k TG measured in a separate run from 64k/128k — expect ~5-10% variance between rows from measurement noise.*

**At 128k context, MTP and no-MTP converge to the same TG speed (~56 tok/s).** The KV cache fills VRAM at long context regardless of MTP, so the offload split ends up identical. MTP's multi-token speculation is offset by its compute overhead.

**PP degrades gracefully**: 1,855 → 1,584 tok/s from 8k to 128k (~15% decline). A 128k prompt processes in ~81 seconds.

**The "97 tok/s" only exists at short context** with `--fit-target 0`. At 64k+, `--fit-target 0` OOMs because there's no headroom for KV cache growth. You must use `--fit-target 1536` for long-context work, which brings speed down to ~73 tok/s at short context and ~56 tok/s at 128k.

**Bottom line for coding agents**: expect ~56 tok/s TG and ~1,500 tok/s PP at 128k context on 16GB. MTP is a wash — doesn't help or hurt at full context.

### VRAM Usage

| Config | VRAM used | VRAM free | Notes |
|--------|-----------|-----------|-------|
| A (27B IQ3+MTP) | 14,803 MiB | 1,039 MiB | Fully on GPU, fit-target 0 |
| B (35B Q4_K_XL+MTP) | 14,623 MiB | 1,219 MiB | Partial offload, fit-target 1536 |
| B (35B Q4_K_XL, no MTP) | 15,815 MiB | 27 MiB | Maximum GPU layers, fit-target 0 |
| C (35B Q8_0+MTP) | 14,567 MiB | 1,275 MiB | Heavy offload, fit-target 1536 |

### Context Limits (push to OOM)

| Limit | 27B IQ3 | **35B Q4_K_XL** | 35B Q8_0 |
|---|---|---|---|
| **Max ctx (q8_0 KV)** | 56k | **131k+** | **131k+** |
| **Max ctx (q4_0 KV)** | 110k | 131k+ | 131k+ |
| Speed at max ctx | 80.5 / 57.2 | **56** | 45 |

This is the **biggest differentiator**. The 35B MoE handles 131k context easily because its hybrid architecture (Gated DeltaNet + Attention) only has ~10 full-attention layers that need KV cache. The remaining SSM layers use a tiny recurrent state. The 27B dense model has KV on every layer, so it maxes out at 56k with q8_0 KV.

**Tip for 27B users**: switching from `-ctk q8_0 -ctv q8_0` to `-ctk q4_0 -ctv q4_0` extends your max context from 56k → 110k. Quality cost is minimal: q4_0 KV at 56k scores 218/220 CodeNeedle vs 220/220 with q8_0 KV (q4_0 at regular context: 219/220 — so most of the 2-line drop is from q4_0 itself, not the longer context).

The OOM at higher contexts is the **MTP compute buffer** (529 MiB fixed allocation), not the KV cache itself. This is a llama.cpp implementation detail that may improve in future versions.

### Quality — CodeNeedle (positional recall)

11 functions from Python's http.server, ~50k char corpus, testing exact line-level recall:

| Metric | **27B IQ3** | 35B Q4_K_XL | 35B Q8_0 |
|---|---|---|---|
| **Pass** | **11/11** | 11/11 | 11/11 |
| **Lines matched** | **220/220** | 217/220 | 216/220 |
| **Hallucinations** | **0** | 1 | 1 |

The 27B IQ3 has a **perfect score** — every line exact, zero hallucinations. The 35B models are close but not quite there. Interesting that Q8_0 doesn't beat Q4_K_XL here.

### Quality — GSM8K (grade school math, 100 cases)

| Metric | 27B IQ3 | **35B Q4_K_XL** | 35B Q8_0 |
|---|---|---|---|
| **Accuracy** | 89% | **91%** | 90% |
| **CI (95%, excl. truncated)** | [86.9%, 97.1%] | [84.9%, 95.8%] | [85.8%, 96.5%] |
| **Truncated** | 5 | **1** | 3 |
| **Wall time** | 106 min | **67 min** | 114 min |

All three overlap in confidence intervals — the quality difference is negligible. But the 35B Q4_K_XL is **37% faster** to evaluate (67 vs 106 min) with fewer truncations.

*Note: AIME2025 was also tested on the 27B — 50% overall but **100% on non-truncated cases**. Every failure was context exhaustion at 32k, not wrong reasoning. The 35B MoE with 131k context would likely score higher.*

### Ubatch PP Trick (coder543, May 18)

u/coder543 discovered that increasing `-ub` from 512→8192 gives **5.5x prompt processing speedup** for `--n-cpu-moe` partially offloaded models. I tested this on the 35B:

**Result: doesn't apply with `--fit on`.** The `-ub 2048+` OOMs because `--fit on` already maximizes VRAM for model layers — no headroom for larger batch buffers. If you use `--n-cpu-moe` manual offload instead, the trick works. But `--fit on` is simpler and handles the split automatically.

### Concurrency (-np sweep)

Tested `-np 1/2/4` on 10 GSM8K cases:

| -np | 27B tok/s | 27B throughput | 35B tok/s | 35B throughput |
|-----|-----------|---------------|-----------|---------------|
| 1 | 83.3 | 0.6 cases/min | 70.7 | 0.8 cases/min |
| **2** | 57.7 | **1.3 cases/min** | 49.7 | **1.1 cases/min** |
| 4 | 10.0 (CPU overflow) | 0.6 cases/min | 28 | failed |

**`-np 2` doubles batch throughput** at 30% slower per-request speed. `-np 4` pushes layers to CPU — 27B drops to 10 tok/s, 35B partially fails. Use `-np 1` for interactive chat, `-np 2` for batch evaluation.

## MTP Reference (for 27B / fully-on-GPU setups)

MTP is worth it when the model fits entirely on GPU (no offload penalty). For the 27B IQ3 on 12GB: 73 tok/s with MTP vs ~56 without. For the 35B on 16GB: skip it (see speed table above).

If you do use MTP:
1. **`--spec-type draft-mtp`** — not `mtp`. Mainline renamed it.
2. **`-np 1`** — b9204 defaults to 4 slots which pushes layers to CPU.
3. **`--spec-draft-n-max 2`** beats 3 (lower acceptance at 3 = slower overall).
4. **`--fit-target 1536`** for partial-offload models. `--fit-target 0` for fully-on-GPU.
5. **At 128k context, MTP gives no speedup** — KV cache dominates VRAM regardless.

Other notes:
- **Hadamard KV rotation (`-khad`)** is enabled by default since b8607 — no flag needed.
- **`-np 2`** doubles batch throughput at 30% slower per-request. Good for eval, bad for interactive.

## Recommendation

### The Config (just copy this)

```bash
./llama-server \
  -m Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  -c 131072 -np 1 --fit on --fit-target 1536 \
  -fa on -t 20 --no-mmap --jinja \
  -ctk q8_0 -ctv q8_0
```

No MTP. No special flags. `--fit-target 1536` is the key — it reserves VRAM headroom so the KV cache doesn't OOM at 128k. Load it, leave it running, point your coding agent at `localhost:8080/v1/chat/completions`.

**What you get**: 56 tok/s generation at 128k context. 1,584 tok/s prompt processing (81s to ingest 128k tokens). 131k max context. GSM8K 91%. Stable.

**Why no MTP?** At 128k context both MTP and no-MTP give the same 56 tok/s — the KV cache dominates VRAM either way. MTP adds 5 gotchas for zero benefit. Skip the complexity.

GGUF: [havenoammo/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF) (the MTP GGUF works fine without `--spec-type draft-mtp` — it just ignores the extra tensors).

### Other VRAM budgets (community data, not tested by us)

Everything above was tested on our RTX 5080 16GB. These estimates for other GPUs are from community reports:

| VRAM | Model | Speed | Source |
|------|-------|-------|--------|
| **8 GB** | 27B IQ3 (no MTP) | ~50 tok/s (est.) | Model is 12.45 GB, partial offload needed |
| **12 GB** | 27B IQ3+MTP | ~73-80 tok/s | janvitos (RTX 4070 Super 12GB, 635 upvotes) |
| **16 GB** | **35B Q4_K_XL** | **56 tok/s @ 128k** | **This post (RTX 5080)** |
| **24 GB** | 35B Q4_K_XL (no MTP) | ~90+ tok/s (est.) | Model is ~22 GB, fits fully on GPU with headroom for KV |

The 27B IQ3+MTP needs the MTP head grafted — `graft-mtp.py` in the repo.

### Why not the others?

**27B IQ3** — We tested it on our 16GB card where it fits fully on GPU (12.45 GB model). Perfect CodeNeedle (220/220), 73 tok/s with MTP. But it caps at 56k context (110k with q4_0 KV). If your coding agent needs 128k, it's out. Better suited for 12 GB cards where the 35B won't fit.

**35B Q8_0** — 38% slower (46 tok/s with MTP), negligible quality gain (GSM8K 90% vs 91%, overlapping CIs). Not worth the VRAM on 16 GB.

## Credits

This post exists because of the community:
- **am17an** — original MTP implementation (PR #22673), merged mainline b9190
- **havenoammo** — MTP GGUF variants + graft script
- **u/janvitos** — 80 tok/s MTP config on 12GB (635 upvotes), documented the flags
- **u/coder543** — ubatch PP trick for `--n-cpu-moe` (May 18)
- **u/OsmanthusBloom** — earlier ubatch discovery
- **u/Still-Notice8155** — GTX 1070 8GB MTP benchmarks proving it works everywhere
- **u/raketenkater** — run-time-repack, defrag-thold, -khad flags documentation
- **u/moflinCASIO** — 4060 Ti 16GB reference benchmarks
- **u/WarthogConfident4039** — requested this benchmarking round
- **ggerganov** — llama-eval, MTP mainline merge
- **u/Look_0ver_There** — corrected 24 GB VRAM table (Q8_0 is 36 GB, doesn't fit)
- **u/simracerman** — pushed for PP speed benchmarks ("your typical coding agent dumps 10k tokens")
- **u/danielhanchen** (Unsloth) — Dynamic quantization formula behind UD-Q4_K_XL
- **u/alexziskind1** — CodeNeedle positional recall benchmark

## What's Next

Session 017: **vLLM vs llama.cpp head-to-head**. vLLM >= 0.19.0 supports MTP natively with PagedAttention (dynamic KV allocation — no fixed compute buffer eating VRAM). Could make MTP actually faster for partial-offload models. Stay tuned.
