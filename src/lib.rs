//! Educational inference engine for the exact Gemma 4 E4B GGUF in `model/`.
//!
//! The crate deliberately owns the model and generation logic. Candle is the
//! tensor and quantized-kernel layer, not a high-level model implementation.

pub mod api;
