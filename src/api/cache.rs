//! Per-layer K/V cache and sequence-position ownership.

use candle_core::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct LayerKv {
    /// `[kv_heads, sequence, head_dim]`, after K normalization and RoPE.
    pub(crate) key: Tensor,
    /// `[kv_heads, sequence, head_dim]`, after unscaled RMS normalization.
    pub(crate) value: Tensor,
}

/// Autoregressive state owned by a loaded model.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub(crate) layers: Vec<Option<LayerKv>>,
    sequence_position: usize,
}

impl KvCache {
    pub(crate) fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![None; layer_count],
            sequence_position: 0,
        }
    }

    pub fn sequence_position(&self) -> usize {
        self.sequence_position
    }

    pub fn cached_layer_count(&self) -> usize {
        self.layers.iter().filter(|entry| entry.is_some()).count()
    }

    pub fn clear(&mut self) {
        self.layers.fill(None);
        self.sequence_position = 0;
    }

    pub(crate) fn advance(&mut self, tokens: usize) {
        self.sequence_position += tokens;
    }
}
