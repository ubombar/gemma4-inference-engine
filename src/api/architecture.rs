//! Explicit Gemma 4 E4B model computation.

use super::{Gemma4Config, cache::LayerKv};
use anyhow::{Context, Result, bail};
use candle_core::{
    D, Device, Module, Tensor,
    quantized::{QMatMul, gguf_file::Content},
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
}

impl Linear {
    fn load(content: &Content, file: &mut File, name: &str, device: &Device) -> Result<Self> {
        let tensor = content
            .tensor(file, name, device)
            .with_context(|| format!("loading tensor {name}"))?;
        Ok(Self {
            weight: QMatMul::from_qtensor(tensor)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.weight.forward(x)?)
    }

    fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.weight.embedding(ids)?)
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
        Ok(Self {
            kind,
            head_dim: match kind {
                AttentionKind::Sliding => config.sliding_head_dim,
                AttentionKind::Global => config.global_head_dim,
            },
            attn_norm: norm("attn_norm.weight", file)?,
            q_proj: linear("attn_q.weight", file)?,
            q_norm: norm("attn_q_norm.weight", file)?,
            k_proj,
            k_norm,
            v_proj,
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
        rope: &Rope,
    ) -> Result<Tensor> {
        let sequence = x.dim(0)?;
        let residual = x;
        let normalized = self.attn_norm.forward(x)?;

        let q = self.q_proj.forward(&normalized)?.reshape((
            sequence,
            config.attention_heads,
            self.head_dim,
        ))?;
        let q = self.q_norm.forward(&q)?;
        // Future hook: inspect or modify normalized Q before RoPE.
        let q = rope
            .apply(&q, start_position, self.kind)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let current_kv = if let (Some(k_proj), Some(k_norm), Some(v_proj)) =
            (&self.k_proj, &self.k_norm, &self.v_proj)
        {
            let k =
                k_proj
                    .forward(&normalized)?
                    .reshape((sequence, config.kv_heads, self.head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = rope.apply(&k, start_position, self.kind)?.transpose(0, 1)?;
            let v =
                v_proj
                    .forward(&normalized)?
                    .reshape((sequence, config.kv_heads, self.head_dim))?;
            let v = rms_no_scale(&v, config.rms_norm_epsilon)?.transpose(0, 1)?;
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
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        // Future hook: inspect attention scores/probabilities here.
        let attention = probabilities.matmul(&value)?;
        let attention = attention
            .transpose(1, 2)?
            .contiguous()?
            .reshape((sequence, config.attention_heads * self.head_dim))?;
        let attention = self.out_proj.forward(&attention)?;
        let attention = self.post_attn_norm.forward(&attention)?;
        let after_attention = (residual + attention)?;
        // Future hook: residual stream after attention.

        let mlp_input = self.ffn_norm.forward(&after_attention)?;
        let gate = self.ffn_gate.forward(&mlp_input)?.gelu()?;
        let up = self.ffn_up.forward(&mlp_input)?;
        let mlp = self.ffn_down.forward(&(gate * up)?)?;
        // Future hook: inspect gate/up/product and MLP output here.
        let mlp = self.post_ffn_norm.forward(&mlp)?;
        let after_mlp = (after_attention + mlp)?;

        let ple = self.ple_gate.forward(&after_mlp)?.gelu()?;
        let ple = ple.broadcast_mul(per_layer_input)?;
        let ple = self.ple_proj.forward(&ple)?;
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

    fn apply(&self, x: &Tensor, start: usize, kind: AttentionKind) -> Result<Tensor> {
        let sequence = x.dim(0)?;
        let inverse = match kind {
            AttentionKind::Sliding => &self.sliding_inv_freq,
            AttentionKind::Global => &self.global_inv_freq,
        };
        let positions: Vec<f32> = (start..start + sequence).map(|x| x as f32).collect();
        let positions = Tensor::from_vec(positions, (sequence, 1), &self.device)?;
        let inverse = Tensor::from_slice(inverse, (1, inverse.len()), &self.device)?;
        let angles = positions.matmul(&inverse)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        let x = x.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let rotated = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
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

impl Gemma4Network {
    pub(crate) fn load(content: &Content, file: &mut File, config: &Gemma4Config) -> Result<Self> {
        let device = Device::Cpu;
        let token_tensor = content.tensor(file, "token_embd.weight", &device)?;
        let token_weight = QMatMul::from_arc(Arc::new(token_tensor))?;
        let token_embedding = Linear {
            weight: token_weight.clone(),
        };
        let output = Linear {
            weight: token_weight,
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
        let mut hidden = (&embedded * (self.config.hidden_size as f64).sqrt())?;

        let token_ple = self.per_layer_token_embedding.embedding(&ids)?;
        let token_ple = (token_ple * (self.config.per_layer_embedding_size as f64).sqrt())?
            .reshape((
                tokens.len(),
                self.config.layer_count,
                self.config.per_layer_embedding_size,
            ))?;
        let projected_ple = self.per_layer_model_projection.forward(&hidden)?;
        let projected_ple = (projected_ple * (self.config.hidden_size as f64).sqrt().recip())?
            .reshape((
                tokens.len(),
                self.config.layer_count,
                self.config.per_layer_embedding_size,
            ))?;
        let projected_ple = self.per_layer_projection_norm.forward(&projected_ple)?;
        let per_layer_inputs = ((projected_ple + token_ple)? * 2f64.sqrt().recip())?;

        let mut shared_sliding: Option<LayerKv> = None;
        let mut shared_global: Option<LayerKv> = None;
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
                &self.rope,
            )?;
            if layer_index == self.config.layer_count - self.config.shared_kv_layers - 2 {
                shared_sliding = cache.layers[layer_index].clone();
            }
            if layer_index == self.config.layer_count - self.config.shared_kv_layers - 1 {
                shared_global = cache.layers[layer_index].clone();
            }
        }
        cache.advance(tokens.len());

        let hidden = self.final_norm.forward(&hidden)?;
        // Future hook: final hidden state before the tied LM head.
        let last = hidden.narrow(0, tokens.len() - 1, 1)?;
        let logits = self.output.forward(&last)?;
        let cap = self.config.final_logit_softcap;
        Ok(((logits / cap)?.tanh()? * cap)?)
    }
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
}
