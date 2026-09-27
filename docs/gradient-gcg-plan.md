# Gradient-guided GCG implementation plan

## Goal

Replace the current random replacement proposal step with the gradient-guided
proposal used by Greedy Coordinate Gradient (GCG), while keeping exact discrete
candidate evaluation and the existing Q8_0 inference path.

For suffix token IDs `s` and target IDs `t`, optimize:

```text
L(s) = -sum_i log P(t_i | prompt + s + t_<i)
```

The gradient does not make token IDs continuous permanently. It is used only to
rank promising discrete replacements. Every proposed suffix is still scored by
the real quantized model with teacher forcing, and only an actual loss decrease
is accepted.

## Important current limitation

Candle's `QMatMul::QTensor` invokes `apply_op1_no_bwd`. Consequently, the
current forward pass deliberately severs Candle's autograd graph at every Q8_0
linear operation. Calling `loss.backward()` on the existing model cannot
produce a suffix-embedding gradient.

The gradient path is allowed to use F32. In particular, activations, losses,
adjoints, and the temporary dequantized weight used by a linear backward step
will all be F32. The production inference path and exact candidate scorer
remain Q8_0.

This permission does not require keeping a complete second F32 copy of the
model. Such a copy would need roughly four times the quantized matrix storage
and would substantially increase peak memory. The default implementation will
therefore dequantize one frozen matrix to F32 when its backward operation needs
it, compute the F32 activation gradient, and release that matrix. An optional
eager F32 gradient model can be considered later for machines with sufficient
memory, but is not part of the first version.

The implementation should add activation-only backward support at the
quantized linear boundary. Model parameters remain frozen; only suffix-input
gradients are required.

## Dtype policy

The two paths deliberately use different numerical policies:

| Operation | Weight storage | Activations / gradients |
|---|---|---|
| Normal inference | Q8_0 where stored in GGUF | Existing F32 path |
| Exact candidate scoring | Q8_0 where stored in GGUF | Existing F32 path |
| Gradient-guidance forward | Q8_0 forward kernels | F32 activations |
| Gradient-guidance backward | One matrix temporarily dequantized to F32 | F32 adjoints |
| Vocabulary-gradient projection | Q8_0 embedding projection | F32 input and output scores |

The F32 backward uses the values represented by the Q8_0 blocks after
dequantization. Candle's optimized Q8_0 forward may internally quantize or
accumulate activations differently from dense F32 matmul, so this is a
straight-through surrogate gradient rather than a literal derivative of every
discrete kernel operation. It does not compute or apply gradients to the
quantized parameters. Exact Q8_0 candidate rescoring remains the authority for
accepting a token replacement.

## Algorithm

For a suffix of length `n`, represent each position temporarily as a one-hot
row over the vocabulary. If `E` is the embedding table and `X` is the one-hot
suffix, the embedded suffix is `X E`. Backpropagation gives:

```text
dL/dX = (dL/d(XE)) E^T
```

For each selected coordinate, take the token IDs with the smallest gradient
values as replacement proposals, optionally filter them, score the resulting
discrete suffixes with the existing teacher-forced loss, and retain the best
one.

Gemma 4 E4B requires a model-specific extension to the usual formula. Token
identity enters through both tables:

```text
main token embedding:       [vocab, 2560]
per-layer token embedding:  [vocab, 42 * 256]
```

Therefore the correct vocabulary gradient is:

```text
dL/dX = G_main E_main^T + G_PLE E_PLE^T
```

where `G_PLE` concatenates the gradient from every layer's 256-wide PLE token
component. Ignoring the PLE term would optimize a different model.

## Proposed public API

Keep the current API small and add proposal controls without exposing Candle
tensors:

```rust
pub struct GradientSuffixOptimizationRequest {
    pub prompt: String,
    pub target: String,
    pub suffix_length: usize,
    pub iterations: usize,
    pub candidate_count: usize,
    pub coordinates_per_iteration: usize,
    pub seed: u64,
}

pub struct GradientSuffixOptimizationStep {
    pub iteration: usize,
    pub loss: f32,
    pub suffix: String,
    pub suffix_tokens: Vec<u32>,
    pub selected_coordinate: Option<usize>,
    pub gradient_candidates: Vec<u32>,
}

pub fn optimize_suffix_with_gradients(
    &mut self,
    request: GradientSuffixOptimizationRequest,
) -> Result<GradientSuffixOptimizationResult>;
```

`evaluate_suffix` remains the final reporting API for top-k first-token
predictions, per-target-token conditional probabilities, and the joint target
probability.

## Implementation phases

### 1. Establish a small autograd feasibility test

Before changing the model, construct a tiny quantized matrix and verify:

1. the existing Q8_0 forward result;
2. a custom quantized-linear operation with the same forward result;
3. the activation gradient against a dense F32 reference;
4. a central finite-difference check against the dense F32 surrogate for
   several activation elements.

This spike determines the exact QTensor orientation needed for
`grad_input = grad_output * weight`. Do not compare finite differences against
the non-smooth Q8_0 kernel; compare against the declared dequantized F32
surrogate. Do not proceed until its numerical error is under a documented
tolerance.

### 2. Add an F32-backward Q8_0 linear boundary

Replace the internal `Linear` wrapper with a model-local custom Candle unary
operation that owns an `Arc<QTensor>`:

- forward delegates to the existing Q8_0 CPU kernel;
- backward dequantizes that operation's frozen weight to F32 and computes only
  the F32 activation gradient;
- no weight gradient is allocated because all weights are frozen;
- the Q8_0 tensor remains the canonical stored weight.

The correctness-first implementation will dequantize one matrix temporarily,
cast `grad_output` to F32 if necessary, multiply it by the F32 weight, and
release the dense tensor when that operation's backward step finishes. Add a
memory test proving dense weights are not retained between steps. This removes
the need to implement a Q8_0 transpose-gradient kernel in the first version.

If Candle retains temporary tensors in the graph, perform the dequantization
inside the custom operation's backward method and ensure the result does not
capture the dense weight. If peak memory remains excessive, add layer-wise
backward recomputation/checkpointing before considering a new numerical kernel.

### 3. Make the Gemma 4 forward pass differentiable

Audit every operation between suffix embeddings and target loss:

- Q8_0 linear layers;
- RMSNorm and Q/K normalization;
- RoPE;
- causal/sliding attention masking and softmax;
- GELU and elementwise gating;
- PLE projection, normalization, and residual path;
- final normalization, tied output projection, and logit softcap;
- log-softmax/NLL.

Add `forward_from_embeddings` internally so suffix embeddings can be variables
while prompt, chat boundary, and target embeddings still come from token IDs.
Do not add a second transformer implementation. The token-ID forward path and
the differentiable path must converge immediately after embedding lookup.

Use a differentiable tensor NLL for the gradient pass. Retain the existing
stable scalar NLL for exact candidate scoring and reporting.

### 4. Represent both suffix embedding inputs as variables

For the suffix positions, create two F32 leaf tensors:

```text
suffix_main: [n, 2560]
suffix_ple:  [n, 10752]
```

Initialize them by gathering the current discrete suffix IDs from their Q8_0
tables. Split `suffix_ple` into 42 slices of width 256 at the same places the
ordinary model consumes PLE inputs. After backward, collect both leaf
gradients.

Prompt, closing-template, and teacher-forced target token embeddings are
constants. Only suffix embedding leaves require gradients.

### 5. Convert embedding gradients into vocabulary scores

Compute the score matrix without materializing a `[n, vocab, hidden]` tensor:

```text
scores_main = grad_main projected by tied main embedding
scores_ple  = grad_ple projected by per-layer embedding
scores      = scores_main + scores_ple
```

Both operations can reuse the existing activation-times-Q8_0 kernels: they have
the same shape as projecting hidden states to vocabulary logits. Their inputs
and output scores are F32, while the large embedding tables stay quantized.
This produces only `[n, 262144]` F32 scores (about 1 MiB per suffix position;
about 4 MiB for the demo's four-token suffix).

For each chosen coordinate:

1. select the lowest-scoring token IDs;
2. remove the current ID, special/control tokens, and invalid candidates;
3. optionally require that a token round-trips through decode/encode;
4. sample `candidate_count` replacements from the retained top set using the
   request seed.

Keep filtering policy explicit because it changes the attack search space.

### 6. Preserve the exact discrete candidate scorer

Do not use the differentiable loss to decide which suffix wins. Continue to:

1. prefill the invariant prompt once;
2. clone the prompt `KvCache` for each candidate;
3. forward the discrete suffix and closing chat boundary;
4. score every target token with cached teacher-forced decode;
5. accept only the candidate with the lowest exact Q8_0 loss.

Initially score candidates sequentially. Batched candidate caches can be a
later optimization after parity is established.

### 7. Define cache and graph ownership clearly

Use two cache modes rather than making the public cache generic:

- `KvCache` remains the detached inference/candidate-scoring cache.
- A private gradient-pass cache holds graph-connected K/V for the suffix and
  target trajectory.

The invariant prompt K/V may be detached and reused because the prompt does not
depend on suffix embeddings. However, attention over cached prompt values must
still propagate gradients to suffix queries. Suffix K/V must remain connected
to the graph, including the Gemma 4 shared-KV layers.

Clear the autograd graph, leaf variables, and gradient cache after every GCG
iteration. Add an iteration-level memory check to catch accidental graph
retention.

### 8. Integrate the optimizer loop

Each iteration should visibly perform:

```text
current discrete suffix
  -> differentiable teacher-forced forward
  -> backward to main + PLE suffix embeddings
  -> quantized projection to vocabulary gradient scores
  -> discrete replacement proposals
  -> exact Q8_0 candidate scoring with cached prompt
  -> accept best candidate
```

Progress output should include iteration, exact loss, suffix text, chosen
coordinate, and whether a proposal improved the loss. Avoid printing full
vocabulary gradients.

### 9. Verification gates

Implement and pass these checks in order:

1. custom Q8_0 linear forward parity with current `QMatMul`;
2. custom Q8_0 activation gradient parity with dense matmul;
3. finite-difference gradient for a tiny linear loss;
4. finite-difference suffix-main gradient through one Gemma block;
5. finite-difference suffix-PLE gradient through one Gemma block;
6. full-model directional derivative check on a short sequence;
7. equality of differentiable and scalar teacher-forced losses;
8. equality of old and new discrete candidate losses;
9. fixed-seed optimizer determinism;
10. at least one gradient-proposed candidate lowers loss on the harmless
    `"ACCESS GRANTED"` demo;
11. all suffix gradients and adjoints are F32;
12. no persistent dense copy of all Q8_0 weights;
13. release-mode run against the real GGUF.

Finite differences should perturb embedding activations, not token IDs. Use
directional derivatives for the full model to keep the number of 42-layer
forwards manageable.

### 10. Compare against a reference GCG implementation

For the same serialized prompt, suffix IDs, and target IDs, export:

- scalar target loss;
- gradient norms for main and PLE suffix leaves;
- top replacement token IDs at each suffix coordinate;
- exact losses for a small fixed candidate set.

A reference that lacks Gemma 4 PLE is insufficient. The comparison must run a
Gemma 4 implementation that exposes gradients for both embedding paths.

## Expected file changes

Keep changes localized:

```text
src/api/architecture.rs  differentiable Q8 linear and embedding-input forward
src/api/cache.rs         private graph-connected cache support
src/api/optimizer.rs     gradient extraction, vocabulary ranking, GCG loop
src/api/mod.rs           small public request/result exports
examples/gcg_search.rs   gradient-GCG demonstration
```

Add a separate module only if the custom quantized backward operation makes
`architecture.rs` materially harder to read.

## Performance expectations

The first correct version will be substantially slower than inference because
each iteration includes a full backward pass and several exact candidate
forwards. The priorities are:

1. mathematical correctness;
2. F32 gradient correctness without complete persistent dequantization;
3. bounded graph/cache memory;
4. prompt-cache reuse for discrete candidates;
5. candidate batching and a native Q8_0 transpose kernel only afterward.

## Definition of done

The feature is complete when the real Q8_0 model can run a seeded
gradient-guided suffix search where:

- gradients include both main and per-layer token embeddings;
- gradient activations and adjoints are F32;
- Q8_0 weights remain canonical and only the current backward matrix is
  temporarily dequantized to F32;
- proposed replacements come from measured vocabulary gradients;
- exact teacher-forced Q8_0 loss selects the winner;
- the final suffix is independently evaluated with `evaluate_suffix`;
- finite-difference and dense-reference checks pass;
- memory does not grow across iterations;
- the complete gradient and candidate-selection path remains readable in this
  repository.
