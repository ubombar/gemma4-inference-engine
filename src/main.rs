//! Minimal one-turn terminal chat for the Gemma 4 E4B model.

use anyhow::Result;
use gemma4_inference_engine::api::{GenerationRequest, Model};
use std::io::{self, BufRead, Write};

const MODEL_PATH: &str = "model/gemma-4-E4B-it-Q8_0.gguf";

fn main() -> Result<()> {
    let use_metal = std::env::args().any(|arg| arg == "--metal");
    let enable_thinking = std::env::args().any(|arg| arg == "--thinking");
    let device = if use_metal {
        candle_core::Device::new_metal(0)?
    } else {
        candle_core::Device::Cpu
    };

    eprintln!("Loading {MODEL_PATH} on {device:?}...");
    let mut model = Model::load_on_device(MODEL_PATH, device)?;
    eprintln!("Ready. Type /exit to quit. Thinking: {enable_thinking}.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let message = line?;
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        if matches!(message, "/exit" | "/quit") {
            break;
        }

        write!(stdout, "Gemma: ")?;
        stdout.flush()?;
        let mut in_thinking = false;
        let response = model.generate_streaming(
            GenerationRequest {
                prompt: message.into(),
                max_tokens: 256,
                temperature: 0.0,
                seed: 42,
                enable_thinking,
            },
            |step| {
                match step.text.as_str() {
                    "<|channel>thought" | "<|think|>" => {
                        in_thinking = true;
                        write!(stdout, "\n\n[thinking]\n")?;
                    }
                    "<channel|>" => {
                        if in_thinking {
                            in_thinking = false;
                            write!(stdout, "\n[/thinking]\n\nGemma: ")?;
                        }
                    }
                    piece if piece.starts_with("<|") && piece.ends_with("|>") => {}
                    piece => write!(stdout, "{piece}")?,
                }
                stdout.flush()?;
                Ok(())
            },
        )?;
        let decoded_tokens = response.generated_tokens.len().saturating_sub(1);
        let tokens_per_second = if response.decode_time.is_zero() {
            0.0
        } else {
            decoded_tokens as f64 / response.decode_time.as_secs_f64()
        };
        writeln!(
            stdout,
            "\n\n[{} generated tokens; prefill {:.2?}; decode {:.2?}; {:.2} tok/s]\n",
            response.generated_tokens.len(),
            response.prefill_time,
            response.decode_time,
            tokens_per_second,
        )?;
        stdout.flush()?;
    }
    Ok(())
}
