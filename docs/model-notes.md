# Gemma 4 E4B model investigation

This note records the selective investigation of
`model/gemma-4-E4B-it-Q8_0-dump.json`. The dump was queried as structured data;
the tokenizer vocabulary and complete tensor listing were not printed.

## GGUF metadata

| Property | Value |
|---|---:|
| GGUF version | 3 |
| Architecture | `gemma4` |
| Model name | `Gemma-4-E4B-It` |
| Tensor count | 720 |
| Vocabulary size | 262,144 (from embedding shapes) |
| Context length | 131,072 |
| Transformer blocks | 42 |
| Residual width | 2,560 |
| MLP width | 10,240 |
| Query heads | 8 |
| KV heads | 2 |
| Sliding attention head dimension | 256 |
| Global attention head dimension | 512 |
| Sliding window | 512 |
| KV-sharing suffix | final 18 layers (layers 24–41) |
| Per-layer embedding width | 256 |
| RMSNorm epsilon | 1e-6 |
| Final logit softcap | 30 |
| Sliding RoPE base | 10,000 |
| Global RoPE base | 1,000,000 |
| BOS / EOS / unknown / padding / mask | 2 / 106 / 3 / 0 / 4 |
| Tokenizer model | `gemma4` |

The embedded chat template is 18,808 bytes. For a simple text-only user turn it
serializes the visible boundary as:

```text
<BOS><|turn>user
What is the capital of France?<turn|>
<|turn>model
```

The implementation will keep chat serialization separate from tokenization.
The dump deliberately represents the large tokenizer arrays as null, so the
actual vocabulary and merges will be read from GGUF at load time.

## Tensor inventory

| Stored dtype | Count |
|---|---:|
| Q8_0 | 296 |
| F32 | 423 |
| BF16 | 1 |

The six non-block tensors are:

| Tensor | Type | GGUF shape |
|---|---|---|
| `token_embd.weight` | Q8_0 | `[2560, 262144]` |
| `per_layer_token_embd.weight` | Q8_0 | `[10752, 262144]` |
| `per_layer_model_proj.weight` | BF16 | `[2560, 10752]` |
| `per_layer_proj_norm.weight` | F32 | `[256]` |
| `rope_freqs.weight` | F32 | `[256]` |
| `output_norm.weight` | F32 | `[2560]` |

Every block has the same 17 tensor names: four attention projections, three
MLP projections, six normalization vectors, Q/K head-normalization vectors,
two Per-Layer Embedding projections, and one scalar layer output scale.

The shapes divide blocks into two structural classes:

- Sliding attention: layers 0–4, 6–10, 12–16, 18–22, 24–28, 30–34, 36–40.
  Q has shape `[2560, 2048]`, K/V `[2560, 512]`, and output
  `[2048, 2560]`.
- Global attention: layers 5, 11, 17, 23, 29, 35, and 41. Q has shape
  `[2560, 4096]`, K/V `[2560, 1024]`, and output `[4096, 2560]`.

All 294 block matrix weights are Q8_0. All 420 block normalization, PLE, and
layer-scale tensors are F32. Representative blocks 0, 1, 5, 20, 21, 35, 40,
and 41 were checked, covering both attention classes and the first, middle,
KV-shared, and final regions.

`blk.41.proj.weight` is not an attention projection. Every block has the same
F32 `[256, 2560]` tensor. It is the Per-Layer Embedding input projection paired
with `blk.N.inp_gate.weight` (`[2560, 256]`).

## Verified text architecture

The tensor interpretation was cross-checked against the current Hugging Face
Gemma 4 reference and llama.cpp's GGUF implementation.

1. Main token embeddings are scaled by `sqrt(2560)`.
2. Per-Layer Embeddings (PLE) combine a token-identity embedding with a BF16
   projection of the main embedding. The projection is scaled by
   `1/sqrt(2560)`, RMS-normalized, added to the token component, and scaled by
   `1/sqrt(2)`.
3. Layers use a five-sliding/one-global pattern. Sliding layers use 256-wide
   heads and ordinary RoPE with base 10,000. Global layers use 512-wide heads
   and proportional, quarter-partial RoPE with base 1,000,000.
4. Attention uses eight query heads, two KV heads, Q/K RMSNorm, scale 1.0, and
   unscaled RMS normalization of V. Sliding layers attend within 512 tokens;
   global layers use causal full attention.
5. Layers 24–41 calculate Q independently but reuse K/V from the most recent
   non-sharing layer of the same attention kind. Their stored K/V tensors are
   therefore not used by the reference forward path.
6. The dense MLP is parallel gated GELU: `down(gelu(gate(x)) * up(x))`.
7. After the ordinary attention and MLP residuals, each block applies its PLE
   branch: `proj(gelu(inp_gate(x)) * per_layer_input)`, RMSNorm, residual add,
   then the learned scalar.
8. The final RMS-normalized hidden state is projected with the tied main token
   embedding, then logits are softcapped as `30 * tanh(logits / 30)`.

This E4B checkpoint is dense; it contains no expert/router tensors.

## Numerical backend decision

Candle 0.11 is selected as the low-level tensor backend:

- Its GGUF v3 reader recognizes Q8_0, F32, and BF16—the exact stored types.
- `QMatMul::QTensor` retains Q8_0 blocks and dispatches activation × quantized
  weight operations through Candle's quantized kernels.
- It provides explicit tensors and primitive operations without requiring its
  existing high-level model implementations.
- CPU is the correctness baseline. Metal can be evaluated later behind a Cargo
  feature once the CPU logits match the reference.

Important limitation: Candle's current GGUF tensor reader seeks and copies each
stored tensor into owned storage. It does not retain an mmap-backed tensor view.
This uses approximately the quantized model size in resident weight memory but
does **not** dequantize the Q8_0 model at startup. A custom mmap-backed Candle
storage would be a later optimization and is not needed for the first correct
implementation.

## Sources

- Hugging Face `modular_gemma4.py` (reference PLE, attention, decoder, and logits)
- Hugging Face `configuration_gemma4.py` (layer pattern and RoPE configuration)
- llama.cpp `src/models/gemma4.cpp` (GGUF tensor mapping and graph semantics)
- Candle `quantized` and `gguf_file` modules (dtype and kernel/load behavior)

## Implemented execution path

The runnable path is now:

```text
src/main.rs
  -> Model::load
  -> GGUF validation and embedded Gemma 4 tokenizer
  -> quantized Gemma4Network weights
  -> exact text-only chat serialization
  -> prompt tokenization
  -> Model::prefill (clears and populates KvCache)
  -> explicit 42-block transformer
  -> tied Q8_0 output projection and logit softcap
  -> greedy/temperature sampling
  -> Model::decode (one token with existing KvCache)
```

Our code owns PLE, RMSNorm, RoPE, attention, shared-KV behavior, MLPs,
residuals, cache progression, logits, sampling, and the generation loop.
Candle provides tensors, elementary operations, GGUF parsing, and Q8_0
embedding/matrix kernels. `Model::logits_for_prompt` and `Model::prefill`
return the complete final-position logits vector. Future interpretability hook
locations are marked directly in `src/api/architecture.rs`.

Build and run the requested demo with:

```bash
cargo build --release
./target/release/gemma4-inference-engine
```
