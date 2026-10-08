//! Small public API and explicit model components.

mod architecture;
mod cache;
mod optimizer;
mod sampling;
mod tokenizer;

use anyhow::{Context, Result, bail};
use architecture::Gemma4Network;
use candle_core::quantized::GgmlDType;
use candle_core::quantized::gguf_file::{Content as GgufContent, Value};
use sampling::Sampler;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokenizer::Gemma4Tokenizer;

pub use architecture::AttentionKind;
pub use cache::KvCache;
pub use optimizer::{
    GradientSuffixOptimizationRequest, GradientSuffixOptimizationResult,
    GradientSuffixOptimizationStep, SuffixEvaluation, SuffixOptimizationRequest,
    SuffixOptimizationResult, SuffixOptimizationStep, TargetTokenProbability, TokenProbability,
};
pub use sampling::SamplingConfig;
pub use tokenizer::{ChatMessage, serialize_user_prompt};

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub architecture: &'static str,
    pub vocabulary_size: usize,
    pub context_length: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub sliding_head_dim: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub per_layer_embedding_size: usize,
    pub rms_norm_epsilon: f64,
    pub final_logit_softcap: f64,
}

impl Gemma4Config {
    pub const E4B: Self = Self {
        architecture: "gemma4",
        vocabulary_size: 262_144,
        context_length: 131_072,
        hidden_size: 2_560,
        intermediate_size: 10_240,
        layer_count: 42,
        attention_heads: 8,
        kv_heads: 2,
        sliding_head_dim: 256,
        global_head_dim: 512,
        sliding_window: 512,
        shared_kv_layers: 18,
        per_layer_embedding_size: 256,
        rms_norm_epsilon: 1e-6,
        final_logit_softcap: 30.0,
    };

    pub fn is_global_layer(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(6)
    }
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub serialized_prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub steps: Vec<TokenStep>,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

#[derive(Debug, Clone)]
pub struct TokenStep {
    pub token_id: u32,
    pub text: String,
    pub top_logits: Vec<TokenLogit>,
}

#[derive(Debug, Clone)]
pub struct TokenLogit {
    pub token_id: u32,
    pub text: String,
    pub logit: f32,
}

pub struct Model {
    path: PathBuf,
    config: Gemma4Config,
    tokenizer: Gemma4Tokenizer,
    network: Gemma4Network,
    cache: KvCache,
    load_time: Duration,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(path, candle_core::Device::Cpu)
    }

    /// Select the numerical backend explicitly; model logic and Q8 weights are shared.
    pub fn load_on_device(path: impl AsRef<Path>, device: candle_core::Device) -> Result<Self> {
        let started = Instant::now();
        let path = path.as_ref();
        let mut file =
            File::open(path).with_context(|| format!("opening model {}", path.display()))?;
        let content = GgufContent::read(&mut file).context("parsing GGUF header")?;
        validate_gguf(&content)?;
        let config = Gemma4Config::E4B;
        let tokenizer = Gemma4Tokenizer::from_gguf(&content)?;
        let network = Gemma4Network::load(&content, &mut file, &config, device)?;
        Ok(Self {
            path: path.to_path_buf(),
            config: config.clone(),
            tokenizer,
            network,
            cache: KvCache::new(config.layer_count),
            load_time: started.elapsed(),
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_time(&self) -> Duration {
        self.load_time
    }

    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    pub fn serialized_prompt(&self, prompt: &str) -> String {
        serialize_user_prompt(prompt)
    }

    pub fn tokenize_serialized(&self, serialized: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode_prompt(serialized)
    }

    /// Clear prior state, process the complete prompt, populate K/V caches, and
    /// return the complete final-position logits vector.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.cache.clear();
        self.forward_logits(tokens)
    }

    /// Process exactly one new token using existing cached K/V states.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.cache.sequence_position() == 0 {
            bail!("decode requires a preceding prefill");
        }
        self.forward_logits(&[token])
    }

    /// Debug API used for first-token comparison with llama.cpp.
    pub fn logits_for_prompt(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let serialized = self.serialized_prompt(prompt);
        let tokens = self.tokenize_serialized(&serialized)?;
        self.prefill(&tokens)
    }

    pub fn generate(&mut self, request: GenerationRequest) -> Result<GenerationResult> {
        let serialized_prompt = self.serialized_prompt(&request.prompt);
        let prompt_tokens = self.tokenize_serialized(&serialized_prompt)?;
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: request.temperature,
            seed: request.seed,
        })?;

        let prefill_started = Instant::now();
        let mut logits = self.prefill(&prompt_tokens)?;
        let prefill_time = prefill_started.elapsed();
        let mut decode_time = Duration::ZERO;
        let mut generated_tokens = Vec::with_capacity(request.max_tokens);
        let mut steps = Vec::with_capacity(request.max_tokens);

        for step_index in 0..request.max_tokens {
            let token_id = sampler.sample(&logits)?;
            let piece = self.tokenizer.token_piece(token_id)?;
            steps.push(TokenStep {
                token_id,
                text: piece,
                top_logits: self.top_logits(&logits, 5)?,
            });
            generated_tokens.push(token_id);
            if token_id == self.tokenizer.eos_id() {
                break;
            }
            if step_index + 1 < request.max_tokens {
                let decode_started = Instant::now();
                logits = self.decode(token_id)?;
                decode_time += decode_started.elapsed();
            }
        }

        let text = self.tokenizer.decode(&generated_tokens, true)?;
        Ok(GenerationResult {
            text,
            serialized_prompt,
            prompt_tokens,
            generated_tokens,
            steps,
            prefill_time,
            decode_time,
        })
    }

    fn forward_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let logits = self.network.forward(tokens, &mut self.cache)?;
        let logits = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        let values = logits.to_vec1::<f32>()?;
        if values.len() != self.config.vocabulary_size {
            bail!(
                "expected {} logits, received {}",
                self.config.vocabulary_size,
                values.len()
            );
        }
        if values.iter().any(|x| !x.is_finite()) {
            bail!("model produced non-finite logits");
        }
        Ok(values)
    }

    fn top_logits(&self, logits: &[f32], count: usize) -> Result<Vec<TokenLogit>> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        let keep = count.min(indexed.len());
        if keep > 0 {
            indexed.select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
        }
        indexed.truncate(keep);
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed
            .into_iter()
            .map(|(token_id, logit)| {
                Ok(TokenLogit {
                    token_id: token_id as u32,
                    text: self.tokenizer.token_piece(token_id as u32)?,
                    logit,
                })
            })
            .collect()
    }
}

fn validate_gguf(content: &GgufContent) -> Result<()> {
    expect_string(content, "general.architecture", "gemma4")?;
    if content.tensor_infos.len() != 720 {
        bail!("expected 720 tensors, found {}", content.tensor_infos.len());
    }
    expect_u64(content, "gemma4.block_count", 42)?;
    expect_u64(content, "gemma4.embedding_length", 2560)?;
    expect_u64(content, "gemma4.feed_forward_length", 10240)?;
    expect_u64(content, "gemma4.attention.head_count", 8)?;
    expect_u64(content, "gemma4.attention.head_count_kv", 2)?;
    expect_u64(content, "gemma4.attention.key_length", 512)?;
    expect_u64(content, "gemma4.attention.key_length_swa", 256)?;
    expect_u64(content, "gemma4.attention.shared_kv_layers", 18)?;
    for required in [
        "token_embd.weight",
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        "per_layer_proj_norm.weight",
        "output_norm.weight",
    ] {
        if !content.tensor_infos.contains_key(required) {
            bail!("required GGUF tensor is absent: {required}");
        }
    }
    expect_tensor(
        content,
        "token_embd.weight",
        &[262_144, 2_560],
        GgmlDType::Q8_0,
    )?;
    expect_tensor(
        content,
        "per_layer_token_embd.weight",
        &[262_144, 10_752],
        GgmlDType::Q8_0,
    )?;
    expect_tensor(
        content,
        "per_layer_model_proj.weight",
        &[10_752, 2_560],
        GgmlDType::BF16,
    )?;
    for layer in 0usize..42 {
        let global = (layer + 1).is_multiple_of(6);
        let head_dim = if global { 512 } else { 256 };
        let q_width = 8 * head_dim;
        let kv_width = 2 * head_dim;
        let prefix = format!("blk.{layer}");
        expect_tensor(
            content,
            &format!("{prefix}.attn_q.weight"),
            &[q_width, 2_560],
            GgmlDType::Q8_0,
        )?;
        expect_tensor(
            content,
            &format!("{prefix}.attn_k.weight"),
            &[kv_width, 2_560],
            GgmlDType::Q8_0,
        )?;
        expect_tensor(
            content,
            &format!("{prefix}.attn_v.weight"),
            &[kv_width, 2_560],
            GgmlDType::Q8_0,
        )?;
        expect_tensor(
            content,
            &format!("{prefix}.attn_output.weight"),
            &[2_560, q_width],
            GgmlDType::Q8_0,
        )?;
    }
    Ok(())
}

fn expect_tensor(
    content: &GgufContent,
    name: &str,
    shape: &[usize],
    dtype: GgmlDType,
) -> Result<()> {
    let info = content
        .tensor_infos
        .get(name)
        .with_context(|| format!("missing tensor {name}"))?;
    if info.shape.dims() != shape || info.ggml_dtype != dtype {
        bail!(
            "tensor {name} mismatch: expected {shape:?}/{dtype:?}, found {:?}/{:?}",
            info.shape.dims(),
            info.ggml_dtype
        );
    }
    Ok(())
}

fn expect_string(content: &GgufContent, key: &str, expected: &str) -> Result<()> {
    match content.metadata.get(key) {
        Some(Value::String(actual)) if actual == expected => Ok(()),
        actual => bail!("expected metadata {key}={expected:?}, found {actual:?}"),
    }
}

fn expect_u64(content: &GgufContent, key: &str, expected: u64) -> Result<()> {
    let actual = content
        .metadata
        .get(key)
        .with_context(|| format!("missing metadata {key}"))?
        .to_u64()?;
    if actual != expected {
        bail!("expected metadata {key}={expected}, found {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_gguf_header_and_tokenizer_are_supported() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/gemma-4-E4B-it-Q8_0.gguf");
        let mut file = File::open(path)?;
        let content = GgufContent::read(&mut file)?;
        validate_gguf(&content)?;
        let tokenizer = Gemma4Tokenizer::from_gguf(&content)?;
        let serialized = serialize_user_prompt("What is the capital of France?");
        let ids = tokenizer.encode_prompt(&serialized)?;
        assert_eq!(ids.first(), Some(&2));
        assert!(ids.len() > 5);
        assert!(tokenizer.decode(&ids[1..], false)?.contains("France"));
        let rope = content
            .tensor(&mut file, "rope_freqs.weight", &candle_core::Device::Cpu)?
            .dequantize(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?;
        assert_eq!(rope.len(), 256);
        assert!(rope.iter().all(|value| value.is_finite()));
        assert!((rope[0] - 1.0).abs() < 1e-6);
        assert_eq!(rope[63], 1.0);
        assert!(rope[64] > 1e20);
        Ok(())
    }
}
