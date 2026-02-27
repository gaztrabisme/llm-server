# Follow-up: Qwen3.5-35B-A3B — 7 community-requested experiments on RTX 5080 16GB

**TL;DR**: Community asked great questions on [my original benchmarks post]. I ran every experiment you requested. The headline: **KV q8_0 is confirmed free lunch, Q4_K_M remains king, `--fit on` without batch flags hits 74.7 tok/s (+7% over my original config), and KL divergence confirms UD-Q4_K_XL is even worse than PPL suggested.** Full results and updated launch command below.

---

## Context

After posting [Qwen3.5-35B-A3B quantization quality + speed benchmarks on RTX 5080 16GB], you folks raised a bunch of great questions. Rather than hand-waving, I ran every experiment I could. Here's what I found.

**Hardware**: RTX 5080 16GB + 128GB DDR5 + Ryzen 9 9950X (32 threads)
**Software**: llama.cpp (built from source, CUDA 12.8, sm_120)
**Base model**: Qwen3.5-35B-A3B (MoE: 256 experts/layer, top-8 + 1 shared, ~3B active params/token)

---

## Experiment 1: KV Cache Quality — Is q8_0 really "free"?

**Requested by**: u/PhilippeEiffel (20 upvotes), u/MrMisterShin, u/llama-impersonator, u/WittyAmbassador7340, u/kreigiron, u/bartskol

Fair concern — I claimed KV q8_0 was free but didn't have PPL data to back it up. Here's the full matrix:

| Model Quant | KV f16 | KV q8_0 | KV q4_0 |
|-------------|--------|---------|---------|
| Q8_0 | 5.8831 | 5.8822 (-0.02%) | 5.8694 (-0.23%) |
| Q4_K_M | 6.0184 | 5.9997 (-0.31%) | 6.0422 (+0.40%) |

**Verdict**: KV q8_0 is genuinely free. PPL differences are within noise (< 0.4%). Even KV q4_0 is acceptable for most use cases. The "instant accuracy drops" some of you reported aren't reflected in PPL metrics — though I acknowledge PPL may not capture all degradation modes (more on that below).

**Recommendation unchanged**: Use `-ctk q8_0 -ctv q8_0` for +12-38% throughput at zero measurable quality cost.

---

## Experiment 2: KL Divergence — Does PPL tell the whole story?

**Requested by**: u/JermMX5 (12 upvotes), u/Embarrassed_Ad3189

u/JermMX5 cited the ["Accuracy is Not All You Need" paper](https://arxiv.org/abs/2407.09141) showing PPL can stay flat while token accuracy collapses. Great point. So I ran KLD against Q8_0 base logits (512 ctx, 80 chunks):

| Quant | Mean KLD | Max KLD | Same Top-1 Token % |
|-------|----------|---------|--------------------|
| Q4_K_M | 0.0282 | 0.1912 | 92.4% |
| UD-Q4_K_XL | 0.1087 | 1.2175 | 86.2% |

**Verdict**: KLD *confirms and amplifies* the PPL findings. UD-Q4_K_XL is **3.9x worse** than Q4_K_M by mean KLD and only preserves the top-1 token 86.2% of the time (vs 92.4%). PPL was not misleading here — it correctly ranked the quants, but KLD shows the gap is even larger than PPL suggested.

**Practical note**: Qwen3.5's 248K vocab makes full KLD evaluation produce enormous logit files (~19 GiB for 80 chunks). I used `--chunks 80` with uint16 storage which is feasible with 128GB RAM. If you have a smaller system, `--chunks 20-30` should give stable relative rankings.

---

## Experiment 3: Bartowski Q4_K_L — Is the imatrix quant worth it?

**Requested by**: u/bettertoknow (10 upvotes)

[bartowski's Q4_K_L](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) uses Q8_0 for embed/output tensors plus more q5_K and q6_K layers than Q4_K_M. Quality-wise, it's measurably better:

| Metric | Q4_K_M (Unsloth) | Q4_K_L (bartowski) | Q8_0 (reference) |
|--------|-------------------|--------------------|-------------------|
| PPL (WikiText-2) | 6.6688 | 6.6125 (-0.8%) | 6.5342 |
| Mean KLD | 0.0282 | 0.0181 (-36%) | — |
| Same top-1 % | 92.4% | 94.2% | — |
| File size | 20 GB (4.74 BPW) | 20.1 GB (4.98 BPW) | 36.9 GB |

But here's the problem — speed:

| Config | Short | Medium | Long | Multi-turn | VRAM |
|--------|-------|--------|------|------------|------|
| Q4_K_M fit-nobatch | 74.7 tok/s | 72.9 | 73.7 | 76.1 | 14559 MB |
| **Q4_K_L fit-nobatch** | **41.4 tok/s** | **41.4** | **40.8** | **41.8** | **14489 MB** |

Q4_K_L is **44% slower**. The larger q5_K/q6_K tensors (4.98 BPW vs 4.74) mean the model buffer is 8984 MiB vs Q4_K_M's 8556 MiB, causing `--fit` to overflow more expert layers to CPU (19/41 vs ~16/41). Manual `--n-cpu-moe 24` OOMs entirely because the model buffer alone exceeds what's available after compute buffer allocation.

**Verdict**: Q4_K_L has genuinely better quality (especially visible in KLD: -36%), but the speed penalty is massive on single-GPU setups where VRAM is the constraint. If your model fits fully in VRAM (5090 32GB), Q4_K_L is a strict upgrade. On 16GB cards, **Q4_K_M wins decisively**.

---

## Experiment 4: `--fit` Tuning — Can we close the gap with manual offload?

**Requested by**: u/Chromix_ (upvoted), u/guiopen, u/wisepal_app, u/DonkeyBonked

In my original post, `--fit on` was ~7% slower than manual `--n-cpu-moe 24`. u/Chromix_ suggested the issue might be that `-b 4096 -ub 4096` batch flags consume VRAM that `--fit` can't then use for expert layers. **Nailed it.**

| Config | Short | Medium | Long | Multi-turn | VRAM |
|--------|-------|--------|------|------------|------|
| C7 baseline (`--n-cpu-moe 24`, -b 4096) | 69.6 tok/s | 67.0 | 65.7 | 69.2 | 14874 MB |
| fit-default (`--fit on`, -b 4096) | 64.3 | 62.8 | 57.4\* | 54.2\* | 14595 MB |
| fit-256 (`--fit-target 256`, -b 4096) | 66.0 | 64.7 | 63.7 | 66.0 | 15321 MB |
| **fit-nobatch (`--fit on`, no -b/-ub)** | **74.7** | **72.9** | **73.7** | **76.1** | **14559 MB** |

\*high variance with outliers

**Verdict**: u/Chromix_ was right. Removing `-b 4096 -ub 4096` lets `--fit` allocate VRAM optimally for expert layers. **fit-nobatch is the new winner at ~74 tok/s** — simpler config AND faster than manual tuning. `--fit-target 256` alone doesn't close the gap; removing the batch flags is the key insight.

---

## Experiment 5: Speculative Decoding — Can we go faster?

**Requested by**: u/BreizhNode, plus our own optimization roadmap

**Bad news first**: No compatible draft model exists. Qwen3.5 has a 248K vocabulary, Qwen3 has 151K. The smallest Qwen3.5 model is 27B — there's no small Qwen3.5 that could serve as a draft. Draft-model speculation is a dead end for now.

**So I tried self-speculative methods** (no draft model needed):

| Config | Short | Medium | Long | Multi-turn | Status |
|--------|-------|--------|------|------------|--------|
| fit-nobatch baseline | 74.7 tok/s | 72.9 | 73.7 | 76.1 | — |
| ngram-simple | 44.9 | 43.4 | 42.4 | 51.3 | works |
| ngram-mod (m=64) | 44.6 | FAIL | FAIL | FAIL | crashes |
| ngram-simple-short (n=8, m=64) | 45.0 | 43.1 | 43.1 | FAIL | partial |

**Note**: ngram tests ran on a different llama.cpp build (`latest` vs `latest-fit`) that had a ~40% regression for unrelated reasons, so the absolute numbers aren't directly comparable. But even accounting for that, there's no speedup from ngram speculation on conversational workloads.

**Verdict**: Self-speculative ngram methods provide zero benefit for diverse conversational workloads. ngram-mod is unstable (crashes after first request). **Not recommended.** If Qwen releases a small Qwen3.5 model (1-3B), draft-model speculation could be huge — but that doesn't exist yet.

---

## Experiment 6: Qwen3.5-27B Dense — MoE vs Dense on single GPU

**Requested by**: u/moahmo88 (4 upvotes), u/Agreeable_Effect938

Some of you asked whether the dense 27B model might be a better fit for single-GPU setups. After all, it's simpler (no expert routing) and smaller (15.6 GB Q4_K_M).

| Metric | 35B-A3B Q4_K_M (MoE) | 27B Q4_K_M (dense) |
|--------|----------------------|--------------------|
| PPL (WikiText-2) | 6.6688 | 6.8573 (+2.8%) |
| Active params/token | ~3B | 27B |
| File size | 20 GB | 15.6 GB |

| Config | Short | Medium | Long | Multi-turn | VRAM |
|--------|-------|--------|------|------------|------|
| 35B-A3B Q4_K_M fit-nobatch | 74.7 tok/s | 72.9 | 73.7 | 76.1 | 14559 MB |
| **27B dense fit** | **7.4 tok/s** | **7.4** | **7.2** | **7.1** | **14075 MB** |

Yes, that's **10x slower**. And it has worse quality.

The dense model needs all 27B parameters computed per token vs only ~3B active for MoE. Even with `--fit` putting 54/65 layers on GPU, the remaining 11 layers on CPU create a massive bottleneck. Theoretical max even fully on GPU: ~61 tok/s (960 GB/s ÷ 15.6 GB model).

**Verdict**: The MoE architecture is the entire advantage on consumer hardware. Only ~3B active params per token means ~10x less memory bandwidth per token. **The 35B-A3B MoE dominates on both speed AND quality.** The 27B dense is only worth considering if you need a non-MoE model for compatibility reasons.

---

## Experiment 7: MXFP4_MOE — The Unsloth-recommended alternative

**Requested by**: u/ayylmaonade, u/jumpingcross, u/danielhanchen (Unsloth creator)

After u/danielhanchen confirmed UD-Q4_K_XL has issues and specifically recommended MXFP4 as the alternative, I ran both quality and speed benchmarks.

**Quality** (partial — MXFP4 dequant path has a memory leak that OOMs after ~40-50 chunks):

| Metric | Q4_K_M | MXFP4_MOE | UD-Q4_K_XL |
|--------|--------|-----------|------------|
| PPL (~40 chunks) | ~6.00 | 5.96 | ~7.17 |
| Mean KLD (31 chunks) | 0.028 | 0.037 | 0.109 |
| Same top-1 % | 92.4% | 91.0% | 86.2% |
| File size | 21.2 GB | 18.4 GB | 19.8 GB |

**Speed**:

| Config | Short | Medium | Long | Multi-turn | VRAM |
|--------|-------|--------|------|------------|------|
| Q4_K_M fit-nobatch | 74.7 tok/s | 72.9 | 73.7 | 76.1 | 14559 MB |
| **MXFP4_MOE fit-nobatch** | **49.5 tok/s** | **47.8** | **46.9** | **44.1** | **14531 MB** |

**Verdict**: MXFP4_MOE has marginally better PPL than Q4_K_M (5.96 vs 6.00) but is **34-42% slower** (~47 tok/s vs ~74 tok/s). Despite the smaller file size (18.4 vs 21.2 GB), it doesn't translate to more expert layers on GPU — VRAM usage is nearly identical. There's also a memory leak bug in the MXFP4 dequant path that prevents full perplexity evaluation. **Not recommended over Q4_K_M** — the quality gain is marginal while the speed loss is massive.

u/danielhanchen — if the Unsloth team has different results on MXFP4 speed, I'd love to compare notes. My build is llama.cpp b8149 with CUDA 12.8 on sm_120.

---

## Research Findings

A few questions didn't need experiments, just digging:

### Why is Ollama 3x slower? (u/InternationalNebula7)

**Ollama has no MoE expert offloading.** When a MoE model doesn't fit in VRAM, Ollama splits at the layer level — entire transformer blocks go to CPU or GPU. This means the GPU sits completely idle waiting for CPU layers. With expert-only offloading, attention/norms stay on GPU while only routed expert FFNs go to CPU — the GPU stays busy.

There's [an open PR (ollama/ollama#12333)](https://github.com/ollama/ollama/pull/12333) to add `num_moe_offload` but it hasn't merged yet. On top of that, Ollama defaults to KV cache f16 (we use q8_0, +20% throughput) and doesn't expose batch size or flash attention controls.

### Pre-built binaries vs source for Blackwell (u/wisepal_app)

For **RTX 50-series**: building from source matters. Release binaries use CUDA 12.4 which doesn't include sm_120 (Blackwell). You need CUDA 12.8+ for native support. Without it, PTX from sm_89 (Ada) gets JIT-compiled — slower first launch and you miss Blackwell-specific kernels.

For **RTX 30/40-series**: pre-built is fine (0-5% difference). Those architectures are already in the release builds.

### 8 GB VRAM recommendations (u/Qxz3)

Use Q4_K_M with full expert offload (`-ot "exps=CPU"`): ~7.2 GB VRAM, ~50 tok/s in our tests. Key flags: `-ctk q8_0 -ctv q8_0` (free lunch), `-fa on`, `--no-mmap`, and tune your thread count (try `physical_cores / 1.5` as starting point, sweep from there).

---

## Updated Launch Command

Based on everything above, here's the new recommended config. Simpler AND faster than my original post:

```
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 \
  --fit on \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**What changed from the original post**:
- Removed `-ngl 999 --n-cpu-moe 24` → replaced with `--fit on` (auto VRAM management)
- Removed `-b 4096 -ub 4096` → this was the key insight from u/Chromix_ — batch flags eat VRAM that `--fit` needs for expert layers
- Result: **74.7 tok/s** (up from 69.6), simpler config, and `--fit` adapts automatically to your available VRAM

---

## Summary Table

| What | Result | Verdict |
|------|--------|---------|
| KV q8_0 quality | < 0.4% PPL difference | **Free lunch. Use it.** |
| KLD: Q4_K_M vs UD-Q4_K_XL | 0.028 vs 0.109 (3.9x worse) | **UD-Q4_K_XL is bad for MoE** |
| Bartowski Q4_K_L | -0.8% PPL, -36% KLD, but 44% slower | **Not worth it on 16GB** |
| `--fit` without batch flags | 74.7 tok/s (+7% over manual) | **New best config** |
| ngram self-speculation | No speedup, unstable | **Don't bother** |
| 27B dense vs 35B-A3B MoE | 10x slower, worse quality | **MoE wins completely** |
| MXFP4_MOE | Marginal quality gain, 34-42% slower | **Q4_K_M still best** |

---

## Acknowledgments

Thanks to everyone who pushed for better data:

- u/PhilippeEiffel, u/MrMisterShin, u/llama-impersonator, u/WittyAmbassador7340, u/kreigiron, u/bartskol — KV cache quality concerns led to the full PPL matrix (E1)
- u/JermMX5, u/Embarrassed_Ad3189 — pushed for KLD over PPL, which revealed the UD-Q4_K_XL gap is worse than PPL showed (E2)
- u/bettertoknow — Bartowski Q4_K_L benchmark, good call even though it turned out too slow for our setup (E3)
- u/Chromix_, u/guiopen, u/wisepal_app, u/DonkeyBonked — `--fit` tuning, especially Chromix_'s insight about batch flags eating VRAM, which gave us the new fastest config (E4)
- u/BreizhNode — speculative decoding investigation, saved others the trouble (E5)
- u/moahmo88, u/Agreeable_Effect938 — 27B dense comparison, definitively answered "is MoE worth the complexity?" (E6)
- u/ayylmaonade, u/jumpingcross, u/danielhanchen — MXFP4_MOE testing, important to validate the Unsloth creator's recommendation (E7)
- u/InternationalNebula7 — Ollama performance gap explanation
- u/Qxz3 — 8GB VRAM config guidance
- u/JoNike — original RTX 5080 partial offload data that informed our testing
- u/3spky5u-oss — comprehensive RTX 5090 head-to-head benchmarks
- u/catplusplusok, u/__SlimeQ__, u/guiopen — chat template and tool calling tips
- u/chickN00dle, u/Odd-Ordinary-5922 — KV cache sensitivity reports at long context
- u/TheRealMasonMac — `--fit on` documentation and RTX 4070 results
- u/pmttyji, u/Subject-Tea-5253 — batch/ubatch tuning data
- u/Pristine-Woodpecker — independent confirmation of UD-Q4_K_XL quality issues
- u/jslominski, u/jiegec, u/Corosus, u/DeedleDumbDee, u/Monad_Maya, u/l33t-Mt, u/kkb294, u/zmanning, u/Additional-Action566 — speed reports across different GPUs

All raw data (benchmark JSONs, PPL logs, KLD logs, config files) is in [my llm-server repo] for anyone who wants to reproduce or verify.

---

**Edit**: Previous post [here]. This is a follow-up with all the experiments you requested.
