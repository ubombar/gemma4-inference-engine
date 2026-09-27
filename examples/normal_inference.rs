use anyhow::Result;
use gemma4_inference_engine::api::{GenerationRequest, Model};

fn main() -> Result<()> {
    const MODEL_PATH: &str = "model/gemma-4-E4B-it-Q8_0.gguf";
    const PROMPT: &str = "How are you doing?";

    let mut model = Model::load(MODEL_PATH)?;
    let config = model.config();
    println!("Model: {}", model.path().display());
    println!("Architecture: {}", config.architecture);
    println!("Layers: {}", config.layer_count);
    println!("Embedding dimension: {}", config.hidden_size);
    println!("Load time: {:.2?}\n", model.load_time());

    let result = model.generate(GenerationRequest {
        prompt: PROMPT.into(),
        max_tokens: 32,
        temperature: 0.0,
        seed: 42,
    })?;

    println!("Prompt:\n{PROMPT}\n");
    println!("Serialized prompt:\n{}", result.serialized_prompt);
    println!("Generated:\n{}\n", result.text);
    println!("Prompt tokens: {:?}", result.prompt_tokens);
    println!("Generated tokens: {:?}\n", result.generated_tokens);
    println!("Prefill time: {:.2?}", result.prefill_time);
    println!("Decode time: {:.2?}", result.decode_time);
    let decoded_tokens = result.generated_tokens.len().saturating_sub(1);
    let tokens_per_second = if result.decode_time.is_zero() {
        0.0
    } else {
        decoded_tokens as f64 / result.decode_time.as_secs_f64()
    };
    println!("Decode speed: {tokens_per_second:.2} tokens/sec\n");

    println!("Top predictions for the first generated token:\n");
    println!("token id    token                         logit");
    println!("------------------------------------------------");
    if let Some(first) = result.steps.first() {
        for prediction in &first.top_logits {
            println!(
                "{:<11} {:<28} {:>8.4}",
                prediction.token_id,
                format!("{:?}", prediction.text),
                prediction.logit
            );
        }
    }

    Ok(())
}
