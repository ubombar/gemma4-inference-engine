# Apple Silicon performance

Build and run on the M2 Mac with 32 GB unified memory:

```bash
cargo build --release --features metal --examples
cargo run --release --features metal --example normal_inference -- --metal
cargo run --release --features metal --example benchmark -- --metal --verify
llama-bench -m model/gemma-4-E4B-it-Q8_0.gguf -p 32 -n 32 -r 2 -ngl 99
```

Run performance measurements sequentially, with other inference workloads idle.
Metal access is required; a restricted sandbox may fail to expose the GPU.

## Measurements

Initial measurements on 2026-10-09, using the same Q8_0 GGUF:

| Backend | Prefill tokens/sec | Decode tokens/sec |
| --- | ---: | ---: |
| llama.cpp Metal, build 11146 / 7fe450e | 275.61 ± 25.80 | 24.13 ± 0.70 |
| Rust Metal, corrected path, 3 warmed runs | 274.69–293.52 | 25.91–26.32 |

These are short-context throughput measurements, not a claim of a universal
speed advantage. llama-bench uses its own synthetic token workload; our example
prefills 32 BOS IDs and then processes token 100 for 32 decode steps, returning
all 262144 logits to the host each step. Thus our decode includes an existing
32-token context whereas llama-bench's standalone tg32 starts with a short
context. Load, tokenizer construction, sampling, and printing are excluded.
Both benchmark paths warm up. Our run 0 is printed but excluded from the table.
Metal compilation makes the first request slower; this is not hidden in the
normal inference example. Temperature and EOS do not affect fixed-work timing.

The original CPU chat demo measured 2.68 decode tokens/sec. Merely enabling
Metal measured 13.79; fused RMSNorm and decode attention produced the warmed
results above. These chat timings have a different workload from the table.

### Same serialized chat prompt

For `How are you doing?`, both engines consumed the same 14 prompt token IDs
and greedily generated the same 32-token response:

```text
I am doing well, thank you for asking! As a large language model, I don't experience feelings in the human sense, but I am functioning optimally
```

| Engine | Prefill | Decode (31 steps) | Decode tokens/sec |
| --- | ---: | ---: | ---: |
| Rust Metal | 151.89 ms | 1.18 s | 26.36 |
| llama.cpp Metal | 103.48 ms | 1.24145 s | 24.97 |

This single sequential comparison is about 6% faster decode for Rust, with
slower prefill. It does not establish a statistically significant speed lead.
Reproduce the llama.cpp side with the following command; `--escape` preserves
the final newline, which `-f` otherwise strips in this llama.cpp build:

```bash
llama-completion -m model/gemma-4-E4B-it-Q8_0.gguf -ngl 99 -c 256 \
  -p '<|turn>user\nHow are you doing?<turn|>\n<|turn>model\n' --escape \
  -n 32 -no-cnv --temp 0 --repeat-penalty 1 --verbose-prompt
```

## Numerical and implementation boundaries

The Rust code still owns Gemma 4's 42 blocks, per-layer embeddings, shared KV,
RoPE, chat serialization, sampling, and autoregressive generation. Candle 0.11
provides [QMatMul](https://docs.rs/candle-core/0.11.0/candle_core/quantized/enum.QMatMul.html)
and Metal kernels. Q8 weights stay quantized on the GPU; the numerical kernel
decodes quantization blocks while multiplying. No dense copy of the full model
is created. F32 activations and the existing F32 normalization weights remain.

Sliding-layer decode attention uses Candle SDPA with scale 1 and softcapping disabled (the
Candle sentinel is 1). It consumes the original KV heads without duplicating
them. Sliding layers attend only to the last window; global layers attend to
all keys. Global layers and multi-token prefill retain explicit attention.
Candle's F32 full-attention kernel does not support the 512-dimensional global
heads, and its F32 vector kernel failed a focused numerical comparison at 512
dimensions (max absolute error 0.421). We therefore do not use it. The enabled
256-dimensional vector kernel passed with max error 2.72e-7. An early 26.4 t/s
result used the failing global kernel and is not a valid performance result.

The corrected model passed cached/full forward logit parity (max error 0.00942,
RMS 0.00222). `GEMMA4_EXPLICIT_ATTENTION=1` selects the explicit attention path
for research/debugging, including on Metal. The focused GPU test can be run with:

```bash
cargo test --release --features metal metal_grouped_attention_matches_explicit -- --ignored --nocapture
```

`KvCache` still owns K/V and sequence position. Appending K/V currently copies
the cache, so long-context performance has further room for improvement.
`prefill`, `decode`, and `logits_for_prompt` expose complete host
logits. Residual and Q/K/V hook comments remain in `src/api/architecture.rs`.
To inspect attention scores, use the explicit attention branch; fused SDPA
does not materialize its intermediate probability matrix.

`benchmark --verify` checks cached decode against a full multi-token forward
pass. `--logits /tmp/logits.bin` writes the first prefill's complete little-endian
F32 vector for comparisons. CPU and GPU are not bit-identical: for the 32-BOS
probe, max absolute CPU/Metal difference was 0.699, RMS 0.153, with identical
top-five token IDs. This alone is not an independent architecture correctness
proof or a full-logit comparison with llama.cpp.

## Follow-up: targeting 30 tokens/sec

RoPE now prepares the sliding/global position tables once per forward call,
sharing them across every Q/K projection. V normalization uses the existing
fused RMS kernel with a constant unit scale. CPU autograd remains explicit.

The follow-up baseline measured 24.87–25.46 decode tokens/sec. Shared RoPE alone
measured 25.28–25.52. With fused V normalization and a trial command-batch size
of 10, the warmed range was 25.93–25.96, with cached/full logit max error 0.00936.
These small differences are close to run-to-run variation; 30 tokens/sec has
not been demonstrated. No command-batch override is enabled by default.

Rejected experiments: command batches of 200 did not improve throughput;
forcing the matrix kernel for decode reduced throughput to about 10 tokens/sec;
F16 dense projections measured 24.13–25.08. Neither alternative arithmetic path
is retained, and Q8_0 weights are unchanged.
