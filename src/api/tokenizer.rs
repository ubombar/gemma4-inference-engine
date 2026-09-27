//! Gemma 4 tokenizer and exact text-only chat serialization boundary.

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file::{Content as GgufContent, Value};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence},
    models::bpe::{BPE, Vocab},
    normalizers::Replace,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Exact no-tools, no-thinking Gemma 4 IT template used by the demo.
pub fn serialize_user_prompt(prompt: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt.trim())
}

pub(crate) struct Gemma4Tokenizer {
    inner: Tokenizer,
    bos_id: u32,
    eos_id: u32,
}

impl Gemma4Tokenizer {
    pub(crate) fn from_gguf(content: &GgufContent) -> Result<Self> {
        let model = string(content, "tokenizer.ggml.model")?;
        if model != "gemma4" {
            bail!("expected Gemma 4 tokenizer, found {model:?}");
        }
        let tokens = strings(content, "tokenizer.ggml.tokens")?;
        let merges = strings(content, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|merge| {
                merge
                    .split_once(' ')
                    .map(|(left, right)| (left.to_owned(), right.to_owned()))
                    .with_context(|| format!("invalid BPE merge {merge:?}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let unknown_id = number(content, "tokenizer.ggml.unknown_token_id")?;
        let unknown = tokens
            .get(unknown_id as usize)
            .context("unknown token id is outside the vocabulary")?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token(unknown.clone())
            .fuse_unk(true)
            .byte_fallback(true)
            .build()
            .map_err(|error| anyhow::anyhow!("building Gemma 4 BPE tokenizer: {error}"))?;
        let mut inner = Tokenizer::new(bpe);
        inner
            .with_normalizer(Some(Replace::new(" ", "▁").map_err(|error| {
                anyhow::anyhow!("building Gemma normalizer: {error}")
            })?));
        inner.with_decoder(Some(DecoderSequence::new(vec![
            Replace::new("▁", " ")
                .map_err(|error| anyhow::anyhow!("building Gemma decoder: {error}"))?
                .into(),
            ByteFallback::new().into(),
            Fuse::new().into(),
        ])));

        // Register control/user-defined pieces so turn markers are never split.
        let token_types = numbers(content, "tokenizer.ggml.token_type")?;
        let mut controls = Vec::new();
        let mut user_defined = Vec::new();
        for (token, token_type) in tokens.iter().zip(token_types) {
            match token_type {
                3 => controls.push(AddedToken::from(token.clone(), true)),
                4 => user_defined.push(AddedToken::from(token.clone(), false)),
                _ => {}
            }
        }
        inner.add_special_tokens(&controls);
        inner.add_tokens(&user_defined);

        Ok(Self {
            inner,
            bos_id: number(content, "tokenizer.ggml.bos_token_id")?,
            eos_id: number(content, "tokenizer.ggml.eos_token_id")?,
        })
    }

    pub(crate) fn encode_prompt(&self, serialized: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(serialized, true)
            .map_err(|error| anyhow::anyhow!("tokenizing prompt: {error}"))?;
        let mut ids = Vec::with_capacity(encoding.len() + 1);
        ids.push(self.bos_id);
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    pub(crate) fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|error| anyhow::anyhow!("decoding tokens: {error}"))
    }

    pub(crate) fn token_piece(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }

    pub(crate) fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

fn value<'a>(content: &'a GgufContent, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .with_context(|| format!("missing GGUF metadata {key}"))
}

fn string(content: &GgufContent, key: &str) -> Result<String> {
    Ok(value(content, key)?.to_string()?.clone())
}

fn strings(content: &GgufContent, key: &str) -> Result<Vec<String>> {
    value(content, key)?
        .to_vec()?
        .iter()
        .map(|item| Ok(item.to_string()?.clone()))
        .collect()
}

fn number(content: &GgufContent, key: &str) -> Result<u32> {
    match value(content, key)? {
        Value::U8(x) => Ok(*x as u32),
        Value::I8(x) => Ok(*x as u32),
        Value::U16(x) => Ok(*x as u32),
        Value::I16(x) => Ok(*x as u32),
        Value::U32(x) => Ok(*x),
        Value::I32(x) => Ok(*x as u32),
        Value::U64(x) => Ok(*x as u32),
        Value::I64(x) => Ok(*x as u32),
        other => bail!("metadata {key} is not an integer: {other:?}"),
    }
}

fn numbers(content: &GgufContent, key: &str) -> Result<Vec<u32>> {
    value(content, key)?
        .to_vec()?
        .iter()
        .map(|item| match item {
            Value::U8(x) => Ok(*x as u32),
            Value::I8(x) => Ok(*x as u32),
            Value::U16(x) => Ok(*x as u32),
            Value::I16(x) => Ok(*x as u32),
            Value::U32(x) => Ok(*x),
            Value::I32(x) => Ok(*x as u32),
            other => bail!("metadata {key} contains a non-integer: {other:?}"),
        })
        .collect()
}
