# Gemma 4 E4B inference engine

A small, research-oriented Rust inference engine for the exact model:

```text
unsloth/gemma-4-E4B-it-GGUF:Q8_0
```

The project keeps the complete inference path visible:

```text
prompt -> chat template -> tokenizer -> embeddings -> Gemma 4 blocks
       -> logits -> sampling -> KV-cached decode
```

Our Rust code owns the Gemma 4 model logic, Per-Layer Embeddings, RoPE,
attention, shared KV behavior, transformer blocks, sampling, cache state, and
generation loop. Candle provides tensor primitives and Q8_0 numerical kernels.

## Model

Place the model at:

```text
model/gemma-4-E4B-it-Q8_0.gguf
```

The model file is intentionally excluded from Git because it is approximately
7.7 GB. The compact JSON metadata/tensor dump remains in the repository for
architecture inspection.

## Run

```bash
cargo run --release
```

The demo asks `What is the capital of France?`, generates up to 32 tokens, and
prints token IDs, timings, decode throughput, and the top first-token logits.

## Status

The included model has completed a successful CPU inference run. The demo
generated:

```text
The capital of France is **Paris**.
```

See [docs/model-notes.md](docs/model-notes.md) for the investigated architecture,
tensor inventory, backend boundary, and execution path.

