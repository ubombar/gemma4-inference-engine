You are an expert Rust developer who is writing a research purpose Gemma4 inference engine.

You are working in a new repository with this structure:

```text
gemma4-inference-engine/
└── model/
    ├── gemma-4-E4B-it-Q8_0-dump.json
    └── gemma-4-E4B-it-Q8_0.gguf
```

I want you to implement the first version of a small educational/research inference engine in Rust for this exact model:

`model/gemma-4-E4B-it-Q8_0.gguf`

The model is:

`unsloth/gemma-4-E4B-it-GGUF:Q8_0`

The goal is NOT to build a general-purpose inference framework and NOT to compete with llama.cpp.

I want a small, readable implementation where I can follow the path from:

```text
prompt
  ↓
tokens
  ↓
embeddings
  ↓
transformer blocks
  ↓
logits
  ↓
sampling
  ↓
generated token
```

I eventually want to use this project for mechanistic-interpretability experiments, so avoid abstractions that unnecessarily hide the model's internal computation.

# 1. Investigate the model first

Before writing code, inspect:

```text
model/gemma-4-E4B-it-Q8_0-dump.json
```

IMPORTANT: this JSON file is very large.

DO NOT `cat` it, read it completely into your context, or otherwise dump the entire file.

Treat it as a database that you query selectively.

Use tools such as:

```text
jq
grep
head
small Python scripts
```

to extract only the information you need.

First determine the top-level JSON structure.

Then selectively determine the model configuration, including:

- architecture
- embedding dimension
- number of transformer blocks
- attention configuration
- number of attention heads
- number of KV heads
- head dimensions
- RoPE configuration
- normalization parameters
- context length
- tokenizer type
- BOS/EOS/special token IDs
- chat-template information

Do NOT dump the entire tokenizer vocabulary.

For tensors, first produce compact summaries programmatically.

Determine:

- total tensor count
- counts by dtype
- unique tensor-name patterns/suffixes
- tensor structure of each block
- whether all blocks have identical tensor structures
- which tensors are Q8_0
- which tensors remain F32/F16/etc.

Then inspect only representative tensors.

At minimum inspect:

```text
embedding tensors
output/final normalization tensors

blk.0.*
blk.1.*

several representative middle blocks

final block tensors

any tensors whose naming/shape differs from the normal block pattern

any non-Q8_0 tensors
```

For example, if hundreds of blocks/tensors follow the same pattern, summarize that pattern rather than printing all entries.

Pay particular attention to unusual tensors such as:

```text
blk.41.proj.weight
```

Do NOT assume that a tensor is a normal attention or MLP projection merely because its name contains `proj`.

If the dump is insufficient to understand a Gemma 4-specific operation, inspect the relevant upstream implementation or documentation.

Do not guess architecture semantics.

# 2. Scope of this implementation

Support ONLY this model architecture.

Do not build:

- a generic model registry
- Llama support
- Mistral support
- arbitrary GGUF architecture support
- an HTTP server
- an OpenAI API
- async infrastructure
- a plugin system
- complicated trait hierarchies

Generalization can happen later.

For this version I want:

```text
GGUF
 ↓
model weights
 ↓
tokenizer
 ↓
prompt tokens
 ↓
prefill
 ↓
transformer
 ↓
logits
 ↓
sampling
 ↓
KV-cached decode
 ↓
next token
```

The implementation should be small enough that I can read the source and understand this complete path.

# 3. Project structure

Prefer approximately:

```text
Cargo.toml

src/
├── api/***
└── main.rs
```

You may add a small number of additional modules if they substantially improve clarity.

Do not fragment the project unnecessarily.

`main.rs` should be a minimal demonstration program.

# 4. Dependencies

You MAY use existing Rust libraries for:

- GGUF parsing
- memory mapping
- tokenizer handling
- tensor allocation
- Q8_0 representation
- Q8_0 matrix operations
- matrix multiplication
- numerical kernels
- CPU acceleration
- Metal acceleration

Do NOT implement optimized:

- SIMD kernels
- BLAS
- Metal kernels
- Q8_0 assembly kernels

from scratch.

Investigate the current Rust ecosystem before choosing dependencies.

Candle is one candidate, but do NOT choose it automatically.

Verify whether the chosen library correctly supports the GGUF tensor types present in THIS file, particularly Q8_0.

Avoid llama.cpp FFI if possible.

The purpose of this project is to own and understand the inference implementation rather than merely wrapping llama.cpp.

# 5. Architectural boundary

Our code should own the MODEL LOGIC.

Ideally our implementation contains understandable implementations of concepts such as:

```text
Gemma4Model
Gemma4Block

RMSNorm
Attention
RoPE
MLP

KVCache

forward()
prefill()
decode()

sampling
generation loop
```

The numerical dependency should primarily provide:

```text
Tensor

matmul
elementwise operations
softmax primitives

Q8_0 operations

CPU/Metal numerical kernels
```

Do NOT solve the task by doing something equivalent to:

```rust
external_library_model.generate(prompt)
```

I specifically want OUR code to own the autoregressive generation loop.

Similarly, avoid using an entire pre-existing Gemma inference implementation unless there is a very strong technical reason.

If you must delegate a Gemma-specific operation to an existing implementation, isolate it and explain why.

# 6. Quantization

The model is primarily Q8_0.

Do NOT dequantize the entire model to FP32 at startup.

Keep quantized weights quantized wherever reasonably possible.

The conceptual boundary should remain visible:

```text
activation
    ×
Q8_0 weight
    ↓
quantized numerical kernel
    ↓
activation
```

Activations may use F32/F16/BF16 as appropriate.

If the selected numerical library temporarily dequantizes blocks during multiplication, that is acceptable.

Explain where this happens.

# 7. Public API

Design a small API.

I want usage roughly like:

```rust
let mut model = Model::load(
    "model/gemma-4-E4B-it-Q8_0.gguf"
)?;

let result = model.generate(GenerationRequest {
    prompt: "What is the capital of France?".into(),
    max_tokens: 32,
    temperature: 0.0,
})?;

println!("{}", result.text);
```

The exact types are your decision.

Something conceptually similar to this would be reasonable:

```rust
pub struct Model {
    // ...
}

pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub steps: Vec<TokenStep>,
}

pub struct TokenStep {
    pub token_id: u32,
    pub text: String,
    pub top_logits: Vec<TokenLogit>,
}

pub struct TokenLogit {
    pub token_id: u32,
    pub text: String,
    pub logit: f32,
}
```

Keep it small.

Do not over-engineer this API.

# 8. Keep the generation loop visible

I want our code to visibly perform something conceptually like:

```text
prompt
   ↓
tokenize
   ↓
prefill(prompt_tokens)
   ↓
logits
   ↓
sample
   ↓
token
   ↓
decode(token, cache)
   ↓
logits
   ↓
sample
   ↓
...
```

Do not hide this behind a library's high-level generation API.

I want to be able to put a breakpoint at each stage.

# 9. KV cache

KV caching is required.

Clearly separate:

```text
PREFILL
```

from:

```text
DECODE
```

During prefill:

```text
entire prompt
     ↓
transformer
     ↓
populate K/V cache for every layer
     ↓
last-position logits
```

During decode:

```text
one new token
     ↓
embedding
     ↓
transformer
     +
existing KV cache
     ↓
append new K/V
     ↓
next logits
```

Do NOT recompute the entire sequence after every generated token.

The implementation should make it understandable:

- what K and V contain
- their shapes
- how they are organized per layer
- where they are stored
- how sequence position is tracked
- how new K/V entries are appended
- how attention uses cached K/V

# 10. Tokenizer and chat template

Use the correct tokenizer for this GGUF.

Keep these stages conceptually separate:

```text
user message
     ↓
chat template
     ↓
serialized prompt
     ↓
tokenizer
     ↓
token IDs
```

Do not silently bury chat-template processing inside unrelated code.

For the demo, use the model's proper instruction/chat format so that the result can be compared against llama.cpp.

If supporting the embedded generic Jinja chat template would require a large dependency, it is acceptable to implement the exact Gemma 4 IT template required by THIS model.

Do not build a general chat-template engine.

# 11. Sampling

Initially implement only:

```text
temperature == 0
    → greedy decoding

temperature > 0
    → temperature sampling
```

Use deterministic/random-seed-controlled sampling where appropriate.

Do not add:

- beam search
- speculative decoding
- repetition penalties
- min-p
- complicated sampling chains

unless something is strictly necessary for correctness.

# 12. Interpretability

This project will later become an interpretability playground.

I eventually want access to things such as:

```text
token embeddings

residual stream before block
residual stream after attention
residual stream after MLP

Q
K
V

attention scores
attention probabilities
attention output

MLP intermediate activations

final hidden state

logits
```

DO NOT implement a complicated hook framework now.

For this first version:

- expose final logits/top logits
- keep transformer computations explicit
- avoid abstractions that make intermediate tensors inaccessible
- add concise comments marking useful future hook points

For example, I eventually want it to be straightforward to do something conceptually like:

```rust
let x = self.attn_norm.forward(&x)?;

let q = self.q_proj.forward(&x)?;
let k = self.k_proj.forward(&x)?;
let v = self.v_proj.forward(&x)?;

// future hook: inspect/modify Q/K/V

let attention = self.attention(q, k, v, cache)?;

// future hook: inspect/modify attention output
```

Do not blindly implement this pseudocode if Gemma 4 differs.

Follow the actual architecture.

# 13. main.rs

`main.rs` should be very small.

It should load:

```text
model/gemma-4-E4B-it-Q8_0.gguf
```

Then print basic information such as:

```text
model
architecture
layers
embedding dimension
quantization information
load time
```

Run a simple demonstration prompt such as:

```text
What is the capital of France?
```

Generate approximately 32 tokens.

Print:

```text
Prompt:
What is the capital of France?

Generated:
...

Prompt tokens: ...
Generated tokens: ...

Load time: ...
Prefill time: ...
Decode time: ...
Decode speed: ... tokens/sec
```

Also print the top few logits for the FIRST generated token.

For example:

```text
Top predictions:

token               logit
--------------------------
Paris               ...
The                 ...
France              ...
...
```

# 14. Correctness strategy

Correctness is much more important than optimization initially.

I want this implementation to be comparable against llama.cpp.

The most useful initial correctness test is:

```text
EXACT SAME SERIALIZED PROMPT

          ┌──────────────────┐
          │                  │
          ▼                  ▼

our Rust engine          llama.cpp

          │                  │
          ▼                  ▼

first-token logits      first-token logits

          └────────┬─────────┘
                   ↓
                 compare
```

Before worrying about long generation, verify that the first forward pass produces plausible logits.

If possible, provide a debug mode or API that allows us to obtain the complete final logits vector for a prompt.

This will later allow us to compare numerically against llama.cpp.

# 15. Performance

Do not optimize prematurely.

A slow but understandable and correct implementation is acceptable.

However, avoid obviously pathological behavior such as:

- loading the GGUF repeatedly
- dequantizing the entire model for every token
- recomputing the complete prompt during every decode step
- copying multi-gigabyte tensors unnecessarily

Use memory mapping if appropriate.

CPU-only operation is acceptable for the first implementation.

If Metal acceleration is straightforward through the chosen numerical backend, supporting it is useful, but correctness takes priority.

# 16. Verification

Do not merely write the code.

Actually build it:

```bash
cargo build --release
```

Then run it against:

```text
model/gemma-4-E4B-it-Q8_0.gguf
```

Fix compilation errors.

Verify progressively rather than trying to debug everything simultaneously.

A sensible progression is:

```text
1. parse GGUF
2. print model metadata
3. locate tensors
4. initialize tokenizer
5. tokenize prompt
6. load embeddings
7. execute one transformer block
8. execute complete forward pass
9. inspect first-token logits
10. generate one token
11. add/use KV cache
12. generate several tokens
```

At minimum verify:

- GGUF loads
- expected tensors exist
- tokenizer works
- prompt tokenization works
- forward pass completes
- logits contain finite values
- generated token is plausible
- EOS detection works
- KV cache is actually reused during decode

If complete generation is prohibitively slow, verifying one or a few generated tokens is acceptable.

# 17. Do not fake unsupported functionality

This is important.

If you discover that the selected Rust tensor library cannot correctly perform some required Gemma 4 or Q8_0 operation:

STOP and explain the specific limitation.

Do not silently substitute a different architecture.

Do not pretend Gemma 3 is Gemma 4.

Do not invent tensor semantics.

Do not return code that compiles but mathematically implements the wrong model.

If necessary, implement the missing model-level operation ourselves using lower-level tensor primitives.

# 18. Keep context usage under control

Do not ingest huge files unnecessarily.

In particular:

```text
model/gemma-4-E4B-it-Q8_0-dump.json
```

should always be queried selectively.

Prefer commands that produce compact summaries.

For example, instead of dumping hundreds of tensor entries, write a short script that reports something like:

```text
Tensor count: 720

Types:
Q8_0    ...
F32     ...
F16     ...

Block patterns:
blocks 0-...     pattern A
blocks ...       pattern B

Exceptional tensors:
...
```

Similarly, do not print the entire tokenizer vocabulary.

When inspecting upstream source code, search for the relevant Gemma 4 structures/functions rather than loading giant source trees into context.

# 19. Final explanation

After the implementation works, give me a concise explanation of the actual execution path:

```text
main.rs
   ↓
Model::load
   ↓
GGUF metadata
   ↓
quantized tensors
   ↓
tokenizer/chat template
   ↓
prefill
   ↓
Gemma 4 transformer blocks
   ↓
Q8_0 numerical kernels
   ↓
logits
   ↓
sampling
   ↓
KV-cached decode
```

Explicitly identify:

1. what code WE implemented,
2. what functionality dependencies provide,
3. where Q8_0 computation occurs,
4. where the KV cache lives,
5. where model state/sequence position lives,
6. where I can inspect the complete logits vector,
7. where future residual-stream hooks should go,
8. where future Q/K/V and attention hooks should go.

# Guiding principle

This is an educational and research inference engine.

Prefer:

```text
slightly more explicit code
that exposes the computation
```

over:

```text
one convenient library call
that hides the computation
```

But do not waste time reimplementing optimized numerical kernels.

I want to understand the model, not write a BLAS library.
