//! Fixed-work benchmark: excludes load, tokenization, sampling, and printing.
//! Each timed call returns host logits, so GPU work is synchronized.
use anyhow::{Result, ensure};
use gemma4_inference_engine::api::Model;
use std::{io::Write, time::Instant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = if args.iter().any(|a| a == "--metal") {
        candle_core::Device::new_metal(0)?
    } else {
        candle_core::Device::Cpu
    };
    println!("Backend: {device:?}");
    let mut model = Model::load_on_device("model/gemma-4-E4B-it-Q8_0.gguf", device)?;
    // Identical IDs for all backends. EOS does not stop this fixed-work benchmark.
    let tokens = vec![2u32; 32];
    if args.iter().any(|a| a == "--verify") {
        model.prefill(&tokens)?;
        let cached = model.decode(100)?;
        let mut complete = tokens.clone();
        complete.push(100);
        let reference = model.prefill(&complete)?;
        let max_error = cached
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms_error = (cached
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / cached.len() as f32)
            .sqrt();
        println!("cached/full logits: max_error={max_error:.6} rms_error={rms_error:.6}");
        ensure!(max_error < 0.05, "cached/full logits diverged");
    }
    for run in 0..4 {
        let started = Instant::now();
        let mut logits = model.prefill(&tokens)?;
        let prefill = started.elapsed();
        if run == 0 {
            if let Some(index) = args.iter().position(|a| a == "--logits") {
                let path = args.get(index + 1).expect("--logits requires a path");
                let mut file = std::fs::File::create(path)?;
                for value in &logits {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
        }
        let started = Instant::now();
        for _ in 0..32 {
            logits = model.decode(100)?;
        }
        let decode = started.elapsed();
        ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
        ensure!(model.cache().sequence_position() == 64, "cache position");
        println!(
            "run={run} warmup={} pp32={:.2} t/s tg32={:.2} t/s cached_layers={}",
            run == 0,
            32. / prefill.as_secs_f64(),
            32. / decode.as_secs_f64(),
            model.cache().cached_layer_count()
        );
    }
    Ok(())
}
