//! Explicit Gemma 4 E4B model computation.

use super::{Gemma4Config, cache::LayerKv};
use anyhow::{Context, Result, bail};
use candle_core::{
    CpuStorage, CustomOp1, D, Device, Layout, Module, Shape, Tensor, Var,
    quantized::{GgmlDType, QMatMul, QTensor, gguf_file::Content},
};
use std::{fs::File, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Sliding,
    Global,
}

impl AttentionKind {
    pub fn for_layer(layer: usize) -> Self {
        if (layer + 1).is_multiple_of(6) {
            Self::Global
        } else {
            Self::Sliding
        }
    }
}

#[derive(Clone)]
struct Linear {
    weight: QMatMul,
    quantized_weight: Arc<QTensor>,
    dense_gradient_weight: Option<Tensor>,
}

impl Linear {
    fn load(content: &Content, file: &mut File, name: &str, device: &Device) -> Result<Self> {
        let tensor = Arc::new(
            content
                .tensor(file, name, device)
                .with_context(|| format!("loading tensor {name}"))?,
        );
        let dense_gradient_weight = match tensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => Some(
                tensor
                    .dequantize(device)?
                    .to_dtype(candle_core::DType::F32)?,
            ),
            _ => None,
        };
        Ok(Self {
            weight: QMatMul::from_arc(tensor.clone())?,
            quantized_weight: tensor,
            dense_gradient_weight,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.weight.forward(x)?)
    }

    fn forward_gradient(&self, x: &Tensor) -> Result<Tensor> {
        match &self.dense_gradient_weight {
            Some(weight) => Ok(x.matmul(&weight.t()?)?),
            None => Ok(x.apply_op1(F32BackwardQMatMul {
                weight: self.quantized_weight.clone(),
            })?),
        }
    }

    fn forward_mode(&self, x: &Tensor, differentiable: bool) -> Result<Tensor> {
        if differentiable {
            self.forward_gradient(x)
        } else {
            self.forward(x)
        }
    }

    fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.weight.embedding(ids)?)
    }
}

/// Q8_0 forward with an F32 activation-only backward. The frozen weight is
/// dequantized only while this operation's backward calculation is running.
#[derive(Clone)]
struct F32BackwardQMatMul {
    weight: Arc<QTensor>,
}

impl CustomOp1 for F32BackwardQMatMul {
    fn name(&self) -> &'static str {
        "qmatmul-f32-activation-backward"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        self.weight.cpu_fwd(storage, layout)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let weight = self
            .weight
            .dequantize(arg.device())?
            .to_dtype(candle_core::DType::F32)?;
        let gradient = grad_result
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?
            .matmul(&weight)?;
        // Prevent the returned first-order gradient from retaining the large
        // temporary F32 weight through a second-order autograd graph.
        Ok(Some(gradient.detach()))
    }
}

#[derive(Clone)]
struct RmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl RmsNorm {
    fn load(
        content: &Content,
        file: &mut File,
        name: &str,
        device: &Device,
        epsilon: f64,
    ) -> Result<Self> {
        let weight = content
            .tensor(file, name, device)
            .with_context(|| format!("loading tensor {name}"))?
            .dequantize(device)?;
        Ok(Self { weight, epsilon })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_metal() {
            return Ok(candle_nn::ops::rms_norm(
                &x.contiguous()?,
                &self.weight,
                self.epsilon as f32,
            )?);
        }
        let normalized = rms_no_scale(x, self.epsilon)?;
        Ok(normalized.broadcast_mul(&self.weight)?)
    }
}

struct Gemma4Block {
    kind: AttentionKind,
    head_dim: usize,
    attn_norm: RmsNorm,
    q_proj: Linear,
    q_norm: RmsNorm,
    k_proj: Option<Linear>,
    k_norm: Option<RmsNorm>,
    v_proj: Option<Linear>,
    v_norm: RmsNorm,
    out_proj: Linear,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_gate: Linear,
    ffn_up: Linear,
    ffn_down: Linear,
    post_ffn_norm: RmsNorm,
    ple_gate: Linear,
    ple_proj: Linear,
    post_ple_norm: RmsNorm,
    output_scale: Tensor,
}

impl Gemma4Block {
    fn load(
        content: &Content,
        file: &mut File,
        layer: usize,
        config: &Gemma4Config,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let norm = |suffix: &str, file: &mut File| {
            RmsNorm::load(
                content,
                file,
                &format!("{prefix}.{suffix}"),
                device,
                config.rms_norm_epsilon,
            )
        };
        let linear = |suffix: &str, file: &mut File| {
            Linear::load(content, file, &format!("{prefix}.{suffix}"), device)
        };
        let shared = layer >= config.layer_count - config.shared_kv_layers;
        let (k_proj, k_norm, v_proj) = if shared {
            (None, None, None)
        } else {
            (
                Some(linear("attn_k.weight", file)?),
                Some(norm("attn_k_norm.weight", file)?),
                Some(linear("attn_v.weight", file)?),
            )
        };
        let output_scale = content
            .tensor(file, &format!("{prefix}.layer_output_scale.weight"), device)?
            .dequantize(device)?;
        let kind = AttentionKind::for_layer(layer);
        let head_dim = match kind {
            AttentionKind::Sliding => config.sliding_head_dim,
            AttentionKind::Global => config.global_head_dim,
        };
        Ok(Self {
            kind,
            head_dim,
            attn_norm: norm("attn_norm.weight", file)?,
            q_proj: linear("attn_q.weight", file)?,
            q_norm: norm("attn_q_norm.weight", file)?,
            k_proj,
            k_norm,
            v_proj,
            // V normalization has no learned scale. Ones let the numerical
            // backend execute the same operation with its fused RMS kernel.
            v_norm: RmsNorm {
                weight: Tensor::ones(head_dim, candle_core::DType::F32, device)?,
                epsilon: config.rms_norm_epsilon,
            },
            out_proj: linear("attn_output.weight", file)?,
            post_attn_norm: norm("post_attention_norm.weight", file)?,
            ffn_norm: norm("ffn_norm.weight", file)?,
            ffn_gate: linear("ffn_gate.weight", file)?,
            ffn_up: linear("ffn_up.weight", file)?,
            ffn_down: linear("ffn_down.weight", file)?,
            post_ffn_norm: norm("post_ffw_norm.weight", file)?,
            ple_gate: linear("inp_gate.weight", file)?,
            ple_proj: linear("proj.weight", file)?,
            post_ple_norm: norm("post_norm.weight", file)?,
            output_scale,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        per_layer_input: &Tensor,
        layer_cache: &mut Option<LayerKv>,
        shared_kv: Option<&LayerKv>,
        start_position: usize,
        config: &Gemma4Config,
        rope: &PreparedRope,
        differentiable: bool,
    ) -> Result<Tensor> {
        let sequence = x.dim(0)?;
        let residual = x;
        let normalized = self.attn_norm.forward(x)?;

        let q = self
            .q_proj
            .forward_mode(&normalized, differentiable)?
            .reshape((sequence, config.attention_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        // Future hook: inspect or modify normalized Q before RoPE.
        let q = rope
            .apply(&q, self.kind, differentiable)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let current_kv = if let (Some(k_proj), Some(k_norm), Some(v_proj)) =
            (&self.k_proj, &self.k_norm, &self.v_proj)
        {
            let k = k_proj.forward_mode(&normalized, differentiable)?.reshape((
                sequence,
                config.kv_heads,
                self.head_dim,
            ))?;
            let k = k_norm.forward(&k)?;
            let k = rope.apply(&k, self.kind, differentiable)?.transpose(0, 1)?;
            let v = v_proj.forward_mode(&normalized, differentiable)?.reshape((
                sequence,
                config.kv_heads,
                self.head_dim,
            ))?;
            let v = self.v_norm.forward(&v)?.transpose(0, 1)?;
            // Future hook: inspect or modify K/V before they enter the cache.
            let combined = match layer_cache.as_ref() {
                Some(previous) => LayerKv {
                    key: Tensor::cat(&[&previous.key, &k], 1)?,
                    value: Tensor::cat(&[&previous.value, &v], 1)?,
                },
                None => LayerKv { key: k, value: v },
            };
            *layer_cache = Some(combined.clone());
            combined
        } else {
            shared_kv
                .context("shared-KV layer ran before its source state was available")?
                .clone()
        };

        let key = current_kv.key.unsqueeze(0)?;
        let value = current_kv.value.unsqueeze(0)?;
        // Candle's fused Metal attention consumes grouped KV heads directly.
        // Gemma 4 uses unit attention scale after Q/K normalization.
        // Keep global attention explicit: Candle 0.11's F32 SDPA vector kernel
        // also fails numerical parity at head_dim=512 on this M2.
        let attention = if x.device().is_metal()
            && !differentiable
            && sequence == 1
            && self.head_dim == 256
            && std::env::var_os("GEMMA4_EXPLICIT_ATTENTION").is_none()
        {
            let total_keys = key.dim(2)?;
            let keep = if self.kind == AttentionKind::Sliding {
                total_keys.min(config.sliding_window)
            } else {
                total_keys
            };
            let key = key.narrow(2, total_keys - keep, keep)?.contiguous()?;
            let value = value.narrow(2, total_keys - keep, keep)?.contiguous()?;
            candle_nn::ops::sdpa(&q.contiguous()?, &key, &value, None, false, 1.0, 1.0)?
        } else {
            let repetitions = config.attention_heads / config.kv_heads;
            let key = repeat_kv(&key, repetitions)?;
            let value = repeat_kv(&value, repetitions)?;
            let total_keys = key.dim(2)?;

            let scores = q.matmul(&key.transpose(2, 3)?)?;
            let mask = attention_mask(
                sequence,
                total_keys,
                start_position,
                self.kind,
                config.sliding_window,
                x.device(),
            )?;
            let scores = scores.broadcast_add(&mask)?;
            let probabilities = if differentiable {
                candle_nn::ops::softmax(&scores, D::Minus1)?
            } else {
                candle_nn::ops::softmax_last_dim(&scores)?
            };
            // Future hook: inspect attention scores/probabilities here.
            probabilities.matmul(&value)?
        };
        let attention = attention
            .transpose(1, 2)?
            .contiguous()?
            .reshape((sequence, config.attention_heads * self.head_dim))?;
        let attention = self.out_proj.forward_mode(&attention, differentiable)?;
        let attention = self.post_attn_norm.forward(&attention)?;
        let after_attention = (residual + attention)?;
        // Future hook: residual stream after attention.

        let mlp_input = self.ffn_norm.forward(&after_attention)?;
        let gate = self
            .ffn_gate
            .forward_mode(&mlp_input, differentiable)?
            .gelu()?;
        let up = self.ffn_up.forward_mode(&mlp_input, differentiable)?;
        let mlp = self.ffn_down.forward_mode(&(gate * up)?, differentiable)?;
        // Future hook: inspect gate/up/product and MLP output here.
        let mlp = self.post_ffn_norm.forward(&mlp)?;
        let after_mlp = (after_attention + mlp)?;

        let ple = self
            .ple_gate
            .forward_mode(&after_mlp, differentiable)?
            .gelu()?;
        let ple = ple.broadcast_mul(per_layer_input)?;
        let ple = self.ple_proj.forward_mode(&ple, differentiable)?;
        let ple = self.post_ple_norm.forward(&ple)?;
        let output = (after_mlp + ple)?.broadcast_mul(&self.output_scale)?;
        // Future hook: residual stream at block output.
        Ok(output)
    }
}

struct Rope {
    sliding_inv_freq: Vec<f32>,
    global_inv_freq: Vec<f32>,
    device: Device,
}

impl Rope {
    fn load(
        content: &Content,
        file: &mut File,
        config: &Gemma4Config,
        device: &Device,
    ) -> Result<Self> {
        let sliding_inv_freq = (0..config.sliding_head_dim)
            .step_by(2)
            .map(|index| 1.0 / 10_000f32.powf(index as f32 / config.sliding_head_dim as f32))
            .collect();
        // Proportional RoPE: 0.25 of the 512 dimensions rotate, but the
        // frequency exponent denominator remains the complete head dimension.
        let factors = content
            .tensor(file, "rope_freqs.weight", device)?
            .dequantize(device)?
            .to_vec1::<f32>()?;
        if factors.len() != config.global_head_dim / 2 {
            bail!("rope_freqs.weight has unexpected length {}", factors.len());
        }
        let global_inv_freq = (0..config.global_head_dim / 2)
            .map(|pair| {
                let base_frequency =
                    1.0 / 1_000_000f32.powf((2 * pair) as f32 / config.global_head_dim as f32);
                base_frequency / factors[pair]
            })
            .collect();
        Ok(Self {
            sliding_inv_freq,
            global_inv_freq,
            device: device.clone(),
        })
    }

    // All Q/K projections in a forward call share the same position tables.
    // Build two tables once, rather than uploading positions and launching
    // matmul/cos/sin separately for every Q and K in all 42 layers.
    fn prepare(&self, start: usize, sequence: usize) -> Result<PreparedRope> {
        let positions: Vec<f32> = (start..start + sequence).map(|x| x as f32).collect();
        let positions = Tensor::from_vec(positions, (sequence, 1), &self.device)?;
        let table = |inverse: &[f32]| -> Result<(Tensor, Tensor)> {
            let inverse = Tensor::from_slice(inverse, (1, inverse.len()), &self.device)?;
            let angles = positions.matmul(&inverse)?;
            Ok((angles.cos()?, angles.sin()?))
        };
        Ok(PreparedRope {
            sliding: table(&self.sliding_inv_freq)?,
            global: table(&self.global_inv_freq)?,
        })
    }
}

struct PreparedRope {
    sliding: (Tensor, Tensor),
    global: (Tensor, Tensor),
}

impl PreparedRope {
    fn apply(&self, x: &Tensor, kind: AttentionKind, differentiable: bool) -> Result<Tensor> {
        let (cos, sin) = match kind {
            AttentionKind::Sliding => &self.sliding,
            AttentionKind::Global => &self.global,
        };
        let x = x.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let rotated = if differentiable {
            candle_nn::rotary_emb::rope_slow(&x, &cos, &sin)?
        } else {
            candle_nn::rotary_emb::rope(&x, &cos, &sin)?
        };
        Ok(rotated.squeeze(0)?.transpose(0, 1)?)
    }
}

pub(crate) struct Gemma4Network {
    token_embedding: Linear,
    per_layer_token_embedding: Linear,
    per_layer_model_projection: Linear,
    per_layer_projection_norm: RmsNorm,
    blocks: Vec<Gemma4Block>,
    final_norm: RmsNorm,
    output: Linear,
    rope: Rope,
    config: Gemma4Config,
    device: Device,
}

pub(crate) struct SuffixGradient {
    pub(crate) loss: f32,
    pub(crate) vocabulary_scores: Vec<Vec<f32>>,
}

impl Gemma4Network {
    pub(crate) fn load(
        content: &Content,
        file: &mut File,
        config: &Gemma4Config,
        device: Device,
    ) -> Result<Self> {
        let token_tensor = Arc::new(content.tensor(file, "token_embd.weight", &device)?);
        let token_weight = QMatMul::from_arc(token_tensor.clone())?;
        let token_embedding = Linear {
            weight: token_weight.clone(),
            quantized_weight: token_tensor.clone(),
            dense_gradient_weight: None,
        };
        let output = Linear {
            weight: token_weight,
            quantized_weight: token_tensor,
            dense_gradient_weight: None,
        };
        let mut blocks = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            blocks.push(Gemma4Block::load(content, file, layer, config, &device)?);
        }
        Ok(Self {
            token_embedding,
            per_layer_token_embedding: Linear::load(
                content,
                file,
                "per_layer_token_embd.weight",
                &device,
            )?,
            per_layer_model_projection: Linear::load(
                content,
                file,
                "per_layer_model_proj.weight",
                &device,
            )?,
            per_layer_projection_norm: RmsNorm::load(
                content,
                file,
                "per_layer_proj_norm.weight",
                &device,
                config.rms_norm_epsilon,
            )?,
            blocks,
            final_norm: RmsNorm::load(
                content,
                file,
                "output_norm.weight",
                &device,
                config.rms_norm_epsilon,
            )?,
            output,
            rope: Rope::load(content, file, config, &device)?,
            config: config.clone(),
            device,
        })
    }

    pub(crate) fn forward(&self, tokens: &[u32], cache: &mut super::KvCache) -> Result<Tensor> {
        if tokens.is_empty() {
            bail!("forward requires at least one token");
        }
        let start = cache.sequence_position();
        if start + tokens.len() > self.config.context_length {
            bail!("context length exceeds {}", self.config.context_length);
        }
        let ids = Tensor::from_slice(tokens, tokens.len(), &self.device)?;
        let embedded = self.token_embedding.embedding(&ids)?;
        let token_ple = self.per_layer_token_embedding.embedding(&ids)?;
        self.forward_embeddings(&embedded, &token_ple, cache, false, false)
    }

    fn forward_embeddings(
        &self,
        embedded: &Tensor,
        token_ple: &Tensor,
        cache: &mut super::KvCache,
        differentiable: bool,
        all_logits: bool,
    ) -> Result<Tensor> {
        let tokens = embedded.dim(0)?;
        let start = cache.sequence_position();
        let mut hidden = (embedded * (self.config.hidden_size as f64).sqrt())?;

        let token_ple = (token_ple * (self.config.per_layer_embedding_size as f64).sqrt())?
            .reshape((
                tokens,
                self.config.layer_count,
                self.config.per_layer_embedding_size,
            ))?;
        let projected_ple = self
            .per_layer_model_projection
            .forward_mode(&hidden, differentiable)?;
        let projected_ple = (projected_ple * (self.config.hidden_size as f64).sqrt().recip())?
            .reshape((
                tokens,
                self.config.layer_count,
                self.config.per_layer_embedding_size,
            ))?;
        let projected_ple = self.per_layer_projection_norm.forward(&projected_ple)?;
        let per_layer_inputs = ((projected_ple + token_ple)? * 2f64.sqrt().recip())?;

        let mut shared_sliding: Option<LayerKv> = None;
        let mut shared_global: Option<LayerKv> = None;
        let rope = self.rope.prepare(start, tokens)?;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            // Future hook: residual stream before block.
            let ple = per_layer_inputs.narrow(1, layer_index, 1)?.squeeze(1)?;
            let shared = match block.kind {
                AttentionKind::Sliding => shared_sliding.as_ref(),
                AttentionKind::Global => shared_global.as_ref(),
            };
            hidden = block.forward(
                &hidden,
                &ple,
                &mut cache.layers[layer_index],
                shared,
                start,
                &self.config,
                &rope,
                differentiable,
            )?;
            if layer_index == self.config.layer_count - self.config.shared_kv_layers - 2 {
                shared_sliding = cache.layers[layer_index].clone();
            }
            if layer_index == self.config.layer_count - self.config.shared_kv_layers - 1 {
                shared_global = cache.layers[layer_index].clone();
            }
        }
        cache.advance(tokens);

        let hidden = self.final_norm.forward(&hidden)?;
        // Future hook: final hidden state before the tied LM head.
        let output_hidden = if all_logits {
            hidden
        } else {
            hidden.narrow(0, tokens - 1, 1)?
        };
        let logits = self.output.forward_mode(&output_hidden, differentiable)?;
        let cap = self.config.final_logit_softcap;
        Ok(((logits / cap)?.tanh()? * cap)?)
    }

    /// Compute an F32 one-hot vocabulary gradient for every suffix position.
    /// The full sequence is recomputed so suffix K/V remains graph-connected.
    pub(crate) fn suffix_gradient(
        &self,
        input_tokens: &[u32],
        suffix_start: usize,
        suffix_length: usize,
        target_start: usize,
        target_tokens: &[u32],
    ) -> Result<SuffixGradient> {
        if !self.device.is_cpu() {
            bail!("suffix gradients currently require Model::load (CPU); Metal is inference-only");
        }
        if suffix_length == 0 || target_tokens.is_empty() {
            bail!("suffix gradient requires non-empty suffix and target");
        }
        if suffix_start + suffix_length > input_tokens.len() || target_start == 0 {
            bail!("invalid suffix or target range for gradient input");
        }
        if input_tokens.len() > self.config.context_length {
            bail!("context length exceeds {}", self.config.context_length);
        }
        if target_start + target_tokens.len() - 1 > input_tokens.len() {
            bail!("gradient input does not contain the teacher-forced target prefix");
        }

        let ids = Tensor::from_slice(input_tokens, input_tokens.len(), &self.device)?;
        let embedded = self.token_embedding.embedding(&ids)?;
        let token_ple = self.per_layer_token_embedding.embedding(&ids)?;
        let suffix_main = Var::from_tensor(&embedded.narrow(0, suffix_start, suffix_length)?)?;
        let suffix_ple = Var::from_tensor(&token_ple.narrow(0, suffix_start, suffix_length)?)?;
        let embedded = replace_rows(
            &embedded,
            suffix_main.as_tensor(),
            suffix_start,
            suffix_length,
        )?;
        let token_ple = replace_rows(
            &token_ple,
            suffix_ple.as_tensor(),
            suffix_start,
            suffix_length,
        )?;

        let mut cache = super::KvCache::new(self.config.layer_count);
        let logits = self.forward_embeddings(&embedded, &token_ple, &mut cache, true, true)?;
        let prediction_positions: Vec<u32> = (0..target_tokens.len())
            .map(|index| (target_start - 1 + index) as u32)
            .collect();
        let prediction_positions =
            Tensor::from_vec(prediction_positions, target_tokens.len(), &self.device)?;
        let selected_logits = logits.index_select(&prediction_positions, 0)?;
        let log_probabilities = candle_nn::ops::log_softmax(&selected_logits, D::Minus1)?;
        let targets =
            Tensor::from_slice(target_tokens, target_tokens.len(), &self.device)?.unsqueeze(1)?;
        let loss = log_probabilities.gather(&targets, 1)?.sum_all()?.neg()?;
        let gradients = loss.backward()?;
        let main_gradient = gradients
            .get(suffix_main.as_tensor())
            .context("missing main suffix embedding gradient")?;
        let ple_gradient = gradients
            .get(suffix_ple.as_tensor())
            .context("missing per-layer suffix embedding gradient")?;
        if main_gradient.dtype() != candle_core::DType::F32
            || ple_gradient.dtype() != candle_core::DType::F32
        {
            bail!("suffix embedding gradients must be F32");
        }

        // dL/d(one-hot) includes both Gemma 4 token-identity paths.
        let main_scores = self.output.forward(main_gradient)?;
        let ple_scores = self.per_layer_token_embedding.forward(ple_gradient)?;
        let vocabulary_scores = (main_scores + ple_scores)?.to_vec2::<f32>()?;
        if vocabulary_scores
            .iter()
            .flatten()
            .any(|score| !score.is_finite())
        {
            bail!("model produced non-finite suffix gradients");
        }
        let loss = loss.to_scalar::<f32>()?;
        if !loss.is_finite() {
            bail!("model produced a non-finite differentiable loss");
        }
        Ok(SuffixGradient {
            loss,
            vocabulary_scores,
        })
    }
}

fn replace_rows(
    source: &Tensor,
    replacement: &Tensor,
    start: usize,
    length: usize,
) -> Result<Tensor> {
    let total = source.dim(0)?;
    let mut parts = Vec::with_capacity(3);
    if start > 0 {
        parts.push(source.narrow(0, 0, start)?);
    }
    parts.push(replacement.clone());
    if start + length < total {
        parts.push(source.narrow(0, start + length, total - start - length)?);
    }
    let references: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&references, 0)?)
}

fn rms_no_scale(x: &Tensor, epsilon: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inverse = (variance + epsilon)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inverse)?)
}

fn repeat_kv(x: &Tensor, repetitions: usize) -> Result<Tensor> {
    if repetitions == 1 {
        return Ok(x.clone());
    }
    let (batch, heads, sequence, dimension) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((batch, heads, repetitions, sequence, dimension))?
        .reshape((batch, heads * repetitions, sequence, dimension))?)
}

fn attention_mask(
    query_count: usize,
    key_count: usize,
    query_start: usize,
    kind: AttentionKind,
    sliding_window: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = vec![0f32; query_count * key_count];
    for query in 0..query_count {
        let absolute_query = query_start + query;
        for key in 0..key_count {
            let future = key > absolute_query;
            let too_old = kind == AttentionKind::Sliding
                && key.saturating_add(sliding_window) <= absolute_query;
            if future || too_old {
                values[query * key_count + key] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(
        values,
        (1, 1, query_count, key_count),
        device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    #[test]
    #[cfg(feature = "metal")]
    #[ignore = "requires an Apple GPU"]
    fn metal_grouped_attention_matches_explicit() -> Result<()> {
        let device = Device::new_metal(0)?;
        // Only the 256-dimensional kernel is enabled in the model. The 512
        // kernel failed this check (max error 0.421), so global attention uses
        // explicit matmul/softmax. Full-model parity covers that fallback.
        for dim in [256] {
            let q = Tensor::arange(0f32, (8 * dim) as f32, &device)?
                .sin()?
                .reshape((1, 8, 1, dim))?;
            let k = Tensor::arange(0f32, (2 * 33 * dim) as f32, &device)?
                .cos()?
                .reshape((1, 2, 33, dim))?;
            let v = (&k * 0.3)?.sin()?;
            let fused = candle_nn::ops::sdpa(&q, &k, &v, None, false, 1.0, 1.0)?;
            let scores = q.matmul(&repeat_kv(&k, 4)?.transpose(2, 3)?)?;
            let explicit = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&repeat_kv(&v, 4)?)?;
            let error = (&fused - &explicit)?.abs()?.max_all()?.to_scalar::<f32>()?;
            println!("head_dim={dim}, max_error={error}");
            assert!(error < 1e-4, "SDPA head_dim={dim} error={error}");
        }
        Ok(())
    }

    #[test]
    fn masks_are_causal_and_sliding() -> Result<()> {
        let global = attention_mask(3, 3, 0, AttentionKind::Global, 2, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(global[0], 0.0);
        assert!(global[1].is_infinite() && global[1].is_sign_negative());
        assert_eq!(global[7], 0.0);

        let sliding = attention_mask(1, 4, 3, AttentionKind::Sliding, 2, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(sliding[0].is_infinite());
        assert!(sliding[1].is_infinite());
        assert_eq!(&sliding[2..], &[0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn quantized_linear_f32_backward_matches_dense_reference() -> Result<()> {
        let device = Device::Cpu;
        let weight_values: Vec<f32> = (0..3 * 32)
            .map(|index| (index as f32 - 40.0) / 37.0)
            .collect();
        let dense_weight = Tensor::from_vec(weight_values, (3, 32), &device)?;
        let quantized = Arc::new(QTensor::quantize(&dense_weight, GgmlDType::Q8_0)?);
        let effective_weight = quantized.dequantize(&device)?;
        let input_values: Vec<f32> = (0..2 * 32)
            .map(|index| (index as f32 - 20.0) / 29.0)
            .collect();
        let input = Var::from_vec(input_values.clone(), (2, 32), &device)?;
        let output = input.apply_op1(F32BackwardQMatMul {
            weight: quantized.clone(),
        })?;

        let forward_reference = input.matmul(&effective_weight.t()?)?;
        let forward_error = (&output - forward_reference)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        // The Q8 kernel and dense dequantized matmul accumulate in a different
        // order, so forward parity is approximate rather than bit-identical.
        assert!(forward_error < 5e-2, "forward error {forward_error}");

        let output_gradient =
            Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 1.5, 0.25, -0.75], (2, 3), &device)?;
        let loss = (&output * &output_gradient)?.sum_all()?;
        let gradients = loss.backward()?;
        let actual = gradients.get(input.as_tensor()).context("input gradient")?;
        let expected = output_gradient.matmul(&effective_weight)?;
        let gradient_error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(gradient_error < 1e-5, "gradient error {gradient_error}");

        let element = 7;
        let epsilon = 1e-2;
        let mut plus = input_values.clone();
        let mut minus = input_values;
        plus[element] += epsilon;
        minus[element] -= epsilon;
        let finite_loss = |values: Vec<f32>| -> candle_core::Result<f32> {
            let value = Tensor::from_vec(values, (2, 32), &device)?;
            // The custom backward deliberately follows this dequantized F32
            // surrogate, not the non-smooth activation quantization inside
            // the optimized Q8 forward kernel.
            let output = value.matmul(&effective_weight.t()?)?;
            (&output * &output_gradient)?.sum_all()?.to_scalar::<f32>()
        };
        let numerical = (finite_loss(plus)? - finite_loss(minus)?) / (2.0 * epsilon);
        let analytic = actual.flatten_all()?.to_vec1::<f32>()?[element];
        assert!(
            (numerical - analytic).abs() < 1e-2,
            "finite difference {numerical}, analytic {analytic}"
        );
        Ok(())
    }
}
