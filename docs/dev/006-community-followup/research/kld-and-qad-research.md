# KL Divergence & QAD Research

Research date: 2026-02-26
Context: Follow-up on Reddit community feedback about our Qwen3.5-35B-A3B quant benchmarks.

---

## Topic 1: KL Divergence for GGUF Quant Quality Measurement

### Background

We posted PPL (perplexity) benchmarks for Qwen3.5-35B-A3B quants (Q8_0, Q4_K_M, UD-Q4_K_XL) on Reddit. User u/JermMX5 challenged the methodology, citing Unsloth's docs:

> "KL Divergence should be the gold standard for reporting quantization errors as per the research paper 'Accuracy is Not All You Need'. Using perplexity is incorrect since output token values can cancel out, so we must use KLD!"

### What Unsloth Claims (Source: unsloth.ai/docs)

Unsloth's Dynamic 2.0 GGUF docs state:

- PPL is fundamentally flawed for quantization benchmarking because "output token values can cancel out"
- KLD should be "one of the gold standards for reporting quantization errors"
- They cite the paper "Accuracy is Not All You Need" (arXiv:2407.09141) as their source
- They also recommend harder real-world benchmarks (e.g., Aider coding bench) as complementary evaluation
- They flag calibration bias: Wikipedia-only calibration datasets cause overfitting, especially for instruct models with chat templates

### The Paper: "Accuracy is Not All You Need" (arXiv:2407.09141)

**Authors**: Abhinav Dutta et al., Microsoft Research

**Core finding**: Accuracy and perplexity severely underestimate the true behavioral distance between original and compressed models.

**Key concepts**:

1. **Flips**: The percentage of answers that change between baseline and compressed model (correct-to-incorrect OR incorrect-to-correct). Example: Llama2-70b with 4-bit quantization showed only 0.78% accuracy loss but **8.1% flips** --- the model gives different answers on 1 in 12 questions even though aggregate accuracy barely changes.

2. **Why PPL fails**: The paper proves mathematically that adding symmetric Gaussian noise to log-probabilities preserves perplexity while degrading output quality. In their experiment, PPL stayed at 5.70 across different noise levels while correct token selection dropped from 61.3% to 21.5%. This is the "cancellation effect" Unsloth references --- positive and negative errors in log-probs average out, masking real degradation.

3. **KLD correlates with flips**: Spearman correlation of 0.981 on MMLU. KLD captures the full distribution shift, not just the top-1 token probability.

4. **Quantization methods tested**: BnB W8A8/W4A4, GPTQ W8A16/W4A16, AWQ W4A16, SmoothQuant W8A8.

5. **Recommendation**: Use distance metrics (KLD, flips) alongside accuracy when evaluating quantized models. Flips are "intuitive and inexpensive" as a proxy metric.

**Important nuance**: The paper does NOT say perplexity is useless --- it says perplexity is *necessary but not sufficient*. KLD and flips reveal problems that PPL hides, but PPL still provides useful signal about absolute quality.

### Running KLD in llama.cpp

**The workflow** (from PR #5076 by ikawrakow):

```bash
# Step 1: Save base model logits (FP16 or Q8_0 reference)
./llama-perplexity -m base_model.gguf -f wiki.test.raw \
    --kl-divergence-base base_logits.kld

# Step 2: Compute KLD against quantized model
./llama-perplexity -m quantized_model.gguf \
    --kl-divergence-base base_logits.kld \
    --kl-divergence
```

Note: In step 2, the dataset file (`-f`) is NOT needed --- tokens are loaded from the `.kld` file itself.

**Statistics computed**:
- PPL ratio (quantized/base) and its log
- Mean KL divergence with uncertainty
- Percentiles (KLD_01, KLD_05, KLD_95, KLD_99)
- Mean change in correct token probability
- Pearson correlation of token probabilities
- Top-1 token match frequency ("same top" %)

**File format**: Logits are stored as 16-bit unsigned integers (uint16_t) with scaling, not raw FP32. The implementation clamps min_logit to max_logit - 16 (probabilities below ~1e-7 are discarded). This roughly halves the file size compared to FP32 storage.

### The File Size Problem

The logit file is `vocab_size * n_tokens * 2 bytes` (uint16 storage):

| Model | Vocab Size | WikiText-2 Tokens | Logit File Size |
|-------|-----------|-------------------|-----------------|
| LLaMA 2 (7B) | 32,000 | ~297K | ~11 GiB |
| LLaMA 3 (8B) | 128,256 | ~297K | ~37 GiB |
| **Qwen3.5-35B-A3B** | **248,320** | **~297K** | **~139 GiB** |

The Qwen3.5 calculation: 248,320 * 297,000 * 2 bytes = ~139 GiB. Even with the uint16 optimization, this exceeds our available disk space.

(Note: Our original estimate of ~280 GB was for FP32 storage; the actual uint16 format is ~139 GiB. Still far too large.)

### Solutions to the File Size Problem

#### Option A: Use fewer chunks (`--chunks N`)

The `--chunks` flag limits how many context-window-sized chunks are processed. Default is all of WikiText-2 (~297K tokens). Comparative testing in llama.cpp has shown "the relative performance of different configs doesn't change for shorter perplexity tests."

**Math**: To fit in ~20 GiB, we need: 20 GiB / (248,320 * 2 bytes) = ~42,700 tokens, or roughly 8 chunks at 512-token stride. This is a very small sample but may be sufficient for relative ranking.

A more comfortable target of ~50 GiB allows ~107K tokens (~20 chunks).

```bash
# Example: limit to 20 chunks
./llama-perplexity -m base_model.gguf -f wiki.test.raw \
    --kl-divergence-base base_logits.kld \
    --chunks 20
```

#### Option B: Use a smaller/different corpus

Instead of WikiText-2 (297K tokens), use a smaller text file. The perplexity tool accepts any raw text via `-f`. A curated 50K-token file would produce a ~23 GiB logit file for Qwen3.5 vocab.

#### Option C: Pre-computed logits on Hugging Face

JohannesGaessler publishes pre-computed FP16 logits for popular models at `JohannesGaessler/llama.cpp_wikitext_logits` on Hugging Face. However, Qwen3.5-35B-A3B is unlikely to be there given its recency.

#### Option D: Compute KLD outside llama.cpp

A Python script (`llama_kl.py` gist by Ttl) can compute KLD from saved logits. More flexibility for chunking, streaming, or top-k only approaches. Could potentially compute KLD token-by-token without materializing the full logits file.

#### Recommended Approach

**Use `--chunks 20-30`** (approximately 50K-80K tokens). This produces a 23-37 GiB logit file --- tight but feasible on our system. The relative KLD ranking between quants should be stable even with reduced token count. Run the base model (Q8_0) first to generate the `.kld` file, then run each quant against it.

### Is Our PPL Finding Still Valid?

Our PPL benchmarks showed UD-Q4_K_XL performing worse than Q4_K_M on Qwen3.5-35B-A3B.

**Assessment**: The PPL cancellation effect described in the paper is about *hiding* degradation (a bad quant can look the same as a good one). It does NOT typically cause a *reversal* where a better quant appears worse. If UD-Q4_K_XL shows higher PPL than Q4_K_M, the most likely explanation is that it genuinely is worse for this specific model.

However, the paper's point stands: the *magnitude* of the PPL difference may understate the actual behavioral divergence. KLD would give a more accurate picture of how much worse it is.

**Bottom line**: Our directional finding (UD-Q4_K_XL < Q4_K_M for this model) is likely valid. KLD would strengthen the claim and quantify the difference more accurately.

### Community Response Strategy

For the Reddit reply:

1. **Acknowledge**: KLD is indeed a better metric than PPL alone --- the paper makes a compelling case
2. **Explain**: We tried KLD but hit the 139 GiB logit file problem with Qwen3.5's 248K vocab
3. **Plan**: We'll re-run with `--chunks 20-30` to produce manageable KLD measurements
4. **Defend PPL**: PPL's cancellation effect hides degradation but doesn't create false negatives --- if a quant looks worse on PPL, it almost certainly is worse. The paper says PPL is "necessary but not sufficient," not "useless"
5. **Share**: Post KLD results as a follow-up to strengthen the findings

---

## Topic 2: QAD (Quantization-Aware Distillation) from NVIDIA

### What Is QAD?

**Paper**: "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery" (arXiv:2601.20088, January 2026)
**Authors**: NVIDIA Research (Nemotron team)
**Project page**: https://research.nvidia.com/labs/nemotron/nemotron-qad/

QAD is a knowledge distillation technique specifically designed to recover accuracy lost during aggressive low-precision quantization (NVFP4 = 4-bit floating point).

### How QAD Works

**Teacher-student framework**:
1. **Teacher**: Frozen BF16 (full-precision) model
2. **Student**: NVFP4-quantized copy of the same model
3. **Loss**: KL divergence between teacher and student output logits: `L_QAD = D_KL(p_teacher || p_student)`
4. **Training**: Standard gradient descent on the quantized student, learning rate 1e-6 to 1e-5, temperature T=1

**Key insight**: QAD does NOT replay the original training pipeline (SFT, RLHF, etc.). It only trains the student to match the final teacher's behavior. This means:
- Works even when original training data is unavailable
- Stable for models that went through complex multi-stage post-training (SFT -> RL -> merging)
- Can use synthetic data, filtered data, or even partially random tokens
- Requires significantly less data than original training

### QAD vs Standard Quantization Methods

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **PTQ** (Post-Training Quantization) | Direct weight rounding, no training | Zero compute, instant | Significant accuracy loss at 4-bit |
| **GPTQ/AWQ** | Calibration-based weight optimization | Low compute, good at 4-bit int | Static, no learning |
| **QAT** (Quantization-Aware Training) | Retrain with quantization in the loop | Can recover accuracy | Requires original training pipeline/data, unstable for RL-trained models |
| **QAD** | Distill from frozen teacher to quantized student | Robust, works with any data, stable for RL models | Requires teacher model in memory during training |

### Results (from the paper)

QAD dramatically outperforms both PTQ and QAT, especially for RL-heavy models:

**Nemotron-3-Nano-30B on AIME25**:
- BF16 baseline: 89.1%
- NVFP4 PTQ: 85.0% (-4.1 pts)
- NVFP4 QAT: 83.3% (-5.8 pts, **worse** than PTQ --- QAT breaks RL capabilities)
- NVFP4 QAD: 87.9% (-1.2 pts, **recovers 95% of the gap**)

**AceReason-Nemotron-7B on AIME25**:
- BF16: 63.5%
- NVFP4 PTQ: 58.7%
- NVFP4 QAT: 46.1% (catastrophic --- RL reasoning destroyed)
- NVFP4 QAD: 62.0% (near-full recovery)

**Cross-domain robustness**: QAD trained with math-only data still recovers coding benchmarks, suggesting cross-domain knowledge transfer.

### Implementation & Code Availability

QAD is open source via NVIDIA's Model-Optimizer:

- **GitHub**: https://github.com/NVIDIA/Model-Optimizer
- **QAD examples**: `examples/llm_qad/` (Megatron, NeMo, HuggingFace Transformers versions)
- **QAT examples**: `examples/llm_qat/` in TensorRT-Model-Optimizer

The `QADTrainer` class handles the distillation loop with teacher/student setup.

### Hardware & Compute Requirements

**Training**:
- Minimum 2x 80GB GPUs per machine (stated for Llama3-8B QAT; QAD for 30B likely needs more)
- Data requirements vary by model size:
  - 7B model: ~0.8B tokens
  - 9B model: ~6B tokens
  - 12B model: ~0.5B tokens
  - 30B model: ~2.5B tokens
  - 49B model: ~0.3B tokens
- Described as "modest data and compute requirements compared to original post-training"
- Specific GPU hours not published

**Inference**:
- NVFP4 is an NVIDIA-specific format optimized for Blackwell GPUs (RTX 50-series, B-series data center)
- Claims 4x FLOPS improvement over BF16 and 1.7x memory savings vs FP8
- Supported in TensorRT-LLM and vLLM

### Applicability to Qwen3.5-35B-A3B

**Short answer**: Not directly applicable to our use case.

**Reasons**:

1. **Output format**: QAD produces NVFP4 checkpoints for TensorRT-LLM / vLLM, NOT GGUF. No GGUF export path exists.

2. **Inference engine**: We use llama.cpp for MoE expert offloading. TensorRT-LLM and vLLM don't support the CPU expert offloading strategy we depend on (84.8 GB model, 16 GB VRAM).

3. **NVFP4 hardware**: Full NVFP4 acceleration requires Blackwell GPUs. Our RTX 5080 IS Blackwell, so the hardware is compatible --- but only through TensorRT-LLM, not llama.cpp.

4. **Compute requirements**: Training QAD on a 35B model would require multiple 80GB GPUs (A100/H100 class). We have a single 16GB RTX 5080. Cloud compute would be needed.

5. **Model support**: The code supports Qwen2/2.5/3 dense models but Qwen3.5 MoE models are not listed. The MoE architecture adds complexity.

6. **The real question**: Even if QAD produced a perfectly distilled 4-bit model, would it be faster than our current Q8_0 with MoE offloading? Probably not --- our bottleneck is PCIe bandwidth for expert transfers, not model size. A smaller model helps VRAM but doesn't reduce expert transfer volume meaningfully.

### Could QAD Concepts Be Applied to GGUF?

The core idea (distill from full-precision teacher into quantized student using KLD loss) is format-agnostic. In principle:

1. Train a QAD-style student using HuggingFace Transformers (output: safetensors)
2. Convert to GGUF via `convert_hf_to_gguf.py`
3. Quantize to desired GGUF format

But this would require:
- Multi-GPU cloud compute for training
- Custom integration with Qwen3.5 MoE architecture
- Uncertain whether the safetensors-to-GGUF conversion preserves the QAD benefits at the target quant level

This is a research project, not a practical workflow for our use case.

### What the Reddit User Might Be Asking

u/datathe1st may be suggesting that QAD could produce better 4-bit quants than current GPTQ/AWQ/GGUF methods. The paper supports this claim for NVFP4 format, but:

1. The improvement over PTQ is real but NVIDIA-ecosystem-locked
2. For GGUF users, the closest equivalent is Unsloth Dynamic quants (which use calibration-aware mixed-precision, a simpler version of the same intuition)
3. No one has published QAD -> GGUF results yet

---

## Summary

### KLD

- **The criticism is valid**: KLD is a better metric than PPL alone for measuring quant quality. The "Accuracy is Not All You Need" paper provides solid evidence.
- **PPL has a real weakness**: symmetric errors in log-probs cancel out, hiding degradation. A model can maintain identical PPL while giving different answers 8-15% of the time.
- **Our PPL finding is likely still correct directionally**: PPL's cancellation hides degradation but doesn't create false reversals. If UD-Q4_K_XL shows higher PPL than Q4_K_M, it probably is worse.
- **We can run KLD**: Use `--chunks 20-30` to limit the logit file to ~23-37 GiB (feasible on our system with Qwen3.5's 248K vocab). The relative KLD ranking should be stable.
- **Action**: Re-run benchmarks with KLD and post results as a follow-up.

### QAD

- **Impressive technique**: Recovers 95%+ of BF16 accuracy at NVFP4 (4-bit), dramatically outperforms both PTQ and QAT.
- **Not applicable to us**: Locked to NVIDIA inference stack (TensorRT-LLM/vLLM), no GGUF support, requires multi-GPU training, and our bottleneck is PCIe bandwidth not model precision.
- **Conceptually interesting**: The teacher-student KLD distillation approach could theoretically improve any quantization format, but no one has built the GGUF pipeline.
- **For the Reddit reply**: Acknowledge QAD is impressive research but explain it's currently NVIDIA-ecosystem-only with no GGUF path. Suggest Unsloth Dynamic quants as the closest GGUF equivalent of calibration-aware quantization.

---

## References

1. Dutta, A. et al. "Accuracy is Not All You Need." arXiv:2407.09141. Microsoft Research, 2024.
2. NVIDIA. "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery." arXiv:2601.20088. January 2026.
3. ikawrakow. "KL-divergence." llama.cpp PR #5076. https://github.com/ggml-org/llama.cpp/pull/5076
4. llama.cpp perplexity tool README. https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
5. Unsloth Dynamic 2.0 GGUFs documentation. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
6. Rishiraj. "Why Maybe We're Measuring LLM Compression Wrong." HuggingFace blog. https://huggingface.co/blog/rishiraj/kld-guided-quantization
7. llama.cpp discussion #4110: "Perplexity as a quantization loss benchmark is inaccurate." https://github.com/ggml-org/llama.cpp/discussions/4110
8. NVIDIA Model-Optimizer. https://github.com/NVIDIA/Model-Optimizer
9. NVIDIA TensorRT-Model-Optimizer QAT examples. https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_qat
10. JohannesGaessler. Pre-computed llama.cpp wikitext logits. https://huggingface.co/JohannesGaessler/llama.cpp_wikitext_logits
