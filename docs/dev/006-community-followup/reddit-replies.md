# Reddit Reply Drafts — "Qwen3.5-35B-A3B quantization quality + speed benchmarks on RTX 5080 16GB"

---

## Reply 1 — u/JermMX5 (PPL methodology / KLD)

You're making a really good point, and the paper you cited (["Accuracy is Not All You Need"](https://arxiv.org/abs/2407.09141)) is exactly the right reference here. Their experiment is pretty damning for PPL-only evaluation — they showed PPL staying flat at 5.70 across noise levels while correct token selection dropped from 61.3% all the way down to 21.5%. The cancellation effect in log-probs is real.

That said, I think the nuance matters for how we used it. PPL's cancellation effect *hides* degradation (a bad quant can look the same as a good one), but it doesn't create false negatives — if a quant scores worse on PPL, it almost certainly *is* worse. The paper calls PPL "necessary but not sufficient," not useless. So when we see UD-Q4_K_XL at 7.17 vs Q4_K_M at 6.67, we can be fairly confident that's a real difference. What we *can't* say is that Q4_K_M is "nearly lossless" just because its PPL is close to Q8_0.

We actually tried running KLD, but hit a practical wall: Qwen3.5's 248K vocabulary produces a ~139 GiB logit file for WikiText-2 (`vocab_size × tokens × 2 bytes` for uint16 storage). That's... a lot.

Plan is to re-run with `--chunks 20-30` to limit it to ~23-37 GiB, which is feasible with 128 GB RAM. Relative rankings should be stable with fewer chunks. Will post KLD results as a follow-up — you've convinced me it's worth the effort.

---

## Reply 2 — u/InternationalNebula7 (Ollama 21.6 tok/s vs 70 tok/s)

The entire 3x gap comes down to one thing: **Ollama has no MoE expert offloading**.

When Ollama encounters a MoE model that doesn't fit in VRAM, it splits at the *layer* level — entire transformer blocks go to CPU or GPU. This is catastrophic for MoE because the GPU sits completely idle waiting for CPU layers to finish. It's the worst possible way to split these models.

With expert-only offloading (`-ot "exps=CPU"` or `--n-cpu-moe`), attention, norms, and shared experts stay on GPU while only the routed expert FFN weights transfer over PCIe. The GPU stays busy doing useful work while the CPU handles expert computation in parallel.

There *is* an [open PR (ollama/ollama#12333)](https://github.com/ollama/ollama/pull/12333) to add `num_moe_offload`, but it hasn't been merged yet. The maintainers seem to prefer automatic optimization, which is fair but means MoE performance suffers in the meantime.

You also can't pass custom llama.cpp flags through Ollama — it embeds llama.cpp as a library via CGO, not as a subprocess. So even if you know the right flags, there's no way to use them.

On top of the offloading gap, Ollama's defaults leave more performance on the table: KV cache at f16 (we use q8_0, which is ~20% faster *and* uses less VRAM), flash attention often disabled, no batch size control.

For MoE models on consumer GPUs right now, running llama.cpp directly is unfortunately the only way to get good performance. Hopefully that PR lands soon.

---

## Reply 3 — u/wisepal_app (pre-built binaries vs building from source)

Short answer: for **Blackwell GPUs (RTX 50-series), building from source matters**.

The release binaries on GitHub are built with CUDA 12.4, which does NOT include `sm_120` (Blackwell architecture). You need CUDA 12.8+ for native `120a-real` support. With CUDA 12.4 binaries on an RTX 5080, what happens is PTX code from sm_89 (Ada) gets JIT-compiled at first launch. This makes the first run noticeably slow, and you miss out on Blackwell-specific kernels and instructions entirely.

**For RTX 30/40-series GPUs, the difference is small** (0-5% steady-state) since those architectures (sm_86, sm_89) are included in the release builds. If you're on a 4090, pre-built is fine.

The biggest wins from building yourself on Blackwell:

- Newer CUDA toolkit (12.8+) with Blackwell-optimized cuBLAS routines
- `-DCMAKE_CUDA_ARCHITECTURES=120` for native Blackwell code generation
- `-DGGML_CUDA_FA_ALL_QUANTS=ON` for flash attention with quantized KV caches
- `-DGGML_NATIVE=ON` for host CPU optimizations (AVX-512 etc.)

Our Docker build uses CUDA 12.8.1 with exactly these flags. If you're not comfortable building from source, the Docker route is probably the easiest path — one `docker build` and you're done.

---

## Reply 4 — u/bettertoknow (Bartowski Q4_K_L vs Q4_K_M)

Good call — I looked into it and you're right that [bartowski's Q4_K_L](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) exists at 21.6 GB.

The difference is that Q4_K_L uses Q8_0 for the embedding layer (`token_embd.weight`) and the output head (`output.weight`) — essentially the model's "interface" with the token vocabulary. These are the layers where quantization errors map most directly to token prediction quality.

It's only 370 MB larger than Q4_K_M (21.60 vs 21.23 GB, ~1.7% increase), so it should fit in exactly the same configurations. And there should be zero speed difference — those embedding/output layers are accessed once per token, not during the heavy per-layer computation that dominates generation time.

We're planning to benchmark it: download, PPL test, and speed verification with our C7 config (`--n-cpu-moe 24`). Will share results in a follow-up. It's a good bet for "strictly better Q4_K_M at no cost."

---

## Reply 5 — u/danielhanchen (Unsloth creator — UD-Q4_K_XL issues)

Thanks for confirming this, and for the quick response! Always good to hear it directly from the source.

Our exact numbers for the record: **UD-Q4_K_XL PPL 7.1702 (+9.7% vs Q8_0 baseline 6.5342)** compared to **Q4_K_M PPL 6.6688 (+2.1%)**. So Q4_K_M is clearly the better pick for this model.

We've got MXFP4_MOE on the testing list as you recommended — curious to see how it compares both on quality and whether the slightly smaller size lets us fit more layers on GPU within the 16 GB VRAM budget.

One question: do you have any KLD data for Qwen3.5-35B-A3B quants? Since you folks advocate KLD as the proper eval metric (and rightly so — the PPL cancellation effect is real), it'd be great to see official numbers. We're working on running KLD ourselves with `--chunks 20-30` to manage the logit file size (248K vocab makes the full run produce ~139 GiB of logits), but if you already have data, that would save us some effort!

---

## Reply 6 — u/DonkeyBonked (--fit vs manual tuning)

Here's our data on this:

| Config | Strategy | tok/s range | VRAM |
|--------|----------|-------------|------|
| Q4_K_M | `--fit on` (auto) | ~62-67 | ~14.5 GB |
| Q4_K_M | `--n-cpu-moe 24` (manual) | ~67-70 | ~14.9 GB |
| Q8_0 | `--fit on` (auto) | ~40 | ~14.7 GB |

So manual tuning wins by about 7% for Q4_K_M. Not huge, but not nothing either.

A tip from u/Chromix_ that I haven't tested yet: `--fit-target 256` reduces the default VRAM headroom from 1 GB to 256 MB, which might close most of that gap. The idea is `--fit` is being overly conservative about how much VRAM to reserve.

Also worth noting: our `-b 4096 -ub 4096` batch settings may cause extra VRAM allocation that `--fit` doesn't account for when determining the split. Testing without those flags is on the list too.

TL;DR: `--fit on` is a great starting point and gets you 90%+ of the way there. If you want to squeeze out the last few tok/s, manual `--n-cpu-moe` tuning with trial-and-error is the way. Start with `--fit on`, note the VRAM usage, then try manual splits around that point.

---

## Reply 7 — u/Embarrassed_Ad3189 (quality evaluation beyond PPL)

Great question. PPL has known limitations for comparing quants — the ["Accuracy is Not All You Need" paper](https://arxiv.org/abs/2407.09141) demonstrated that PPL can stay flat while actual token prediction accuracy drops dramatically, because symmetric errors in log-probabilities cancel out when averaged.

The better metric is **KL Divergence (KLD)**, which captures the full distribution shift between the baseline model's output and the quantized model's output — not just the top-1 token. According to that same paper, KLD correlates at **0.981** with "flip rate" (the percentage of answers that actually change between baseline and quantized model), making it a much more reliable quality signal.

We're working on running KLD for our quants. It's non-trivial for this model because the 248K vocabulary makes the logit files enormous, but we have a plan to make it work with chunked evaluation.

Will post KLD results as a follow-up. It should give us a much clearer picture of whether Q4_K_M is truly "nearly lossless" or just appears that way through PPL's rose-tinted lens.

---

## Reply 8 — u/Qxz3 (settings for 8 GB VRAM)

For 8 GB VRAM with Q4_K_M, your best bet from our benchmarks:

**Full expert offload: `-ot "exps=CPU"`** — this puts only attention, norms, and shared experts on GPU, using about **7.2 GB VRAM**. We got ~50 tok/s with this config, which is solid. It should fit comfortably in 8 GB.

Key flags you definitely want regardless of the offload strategy:

- `-ctk q8_0 -ctv q8_0` — KV cache quantization. This is a **free lunch**: saves VRAM *and* speeds things up (~20% faster in our tests). No quality impact.
- `-fa on` — flash attention, required for the KV cache quantization to work
- `--no-mmap` — loads the full model into RAM upfront, gives consistent performance
- `-c` — set context length to what you actually need. 65K is the max but uses more VRAM for KV cache. If you only need 8K or 16K, set it there and save VRAM.

One thing to sweep yourself is **thread count**. We found 20 optimal on our 32-core system, but it's a U-shaped curve (not monotonically better with more threads). Try a few values around `physical_cores / 1.5` as a starting point.

If you want to try Q8_0 for better quality: `--n-cpu-moe 36` might work at 8 GB but it'll be tight. I'd stick with Q4_K_M + full offload for 8 GB — the performance-per-VRAM is hard to beat.

---

## Reply 9 — u/PhilippeEiffel (KV cache quantization quality concerns)

This is a fair concern and deserves a proper answer, not just "trust us."

We're running a full PPL matrix right now to get hard numbers:

| Model Quant | KV f16 | KV q8_0 | KV q4_0 |
|-------------|--------|---------|---------|
| Q8_0        | run 1  | run 2   | run 3   |
| Q4_K_M      | run 4  | run 5   | run 6   |

Six runs, ~25 minutes total. This should show exactly how much (if any) quality cost KV cache quantization adds on top of model quantization.

We're also going to test **short context vs long context PPL** specifically, since the community reports that KV cache quantization degradation shows up primarily at longer contexts — which makes sense since accumulated rounding errors in the KV cache compound over sequence length.

If KV q8_0 shows measurable quality cost, we'll update our recommendation. The performance win is real (~20% throughput + less VRAM), but not if it comes at a meaningful quality cost.

Will post results in a follow-up. Appreciate you pushing on this — it's exactly the kind of thing that's easy to hand-wave away but important to verify.
