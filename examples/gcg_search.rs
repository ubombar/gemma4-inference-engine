use anyhow::Result;
use gemma4_inference_engine::api::{GradientSuffixOptimizationRequest, Model};

fn main() -> Result<()> {
    const MODEL_PATH: &str = "model/gemma-4-E4B-it-Q8_0.gguf";
    const PROMPT: &str = "How are you doing today?";
    const TARGET: &str = "Terrible";

    let mut model = Model::load(MODEL_PATH)?;
    let config = model.config();
    println!("Model: {}", model.path().display());
    println!("Architecture: {}", config.architecture);
    println!("Layers: {}", config.layer_count);
    println!("Embedding dimension: {}", config.hidden_size);
    println!("Load time: {:.2?}\n", model.load_time());
    println!("Prompt: {PROMPT:?}");
    println!("Target continuation: {TARGET:?}\n");

    let result = model.optimize_suffix_with_gradients(GradientSuffixOptimizationRequest {
        prompt: PROMPT.into(),
        target: TARGET.into(),
        suffix_length: 20,
        iterations: 30,
        candidate_count: 32,
        coordinates_per_iteration: 4,
        seed: 42,
    })?;

    println!("\nOptimized suffix: {:?}", result.suffix);
    println!("Suffix tokens: {:?}", result.suffix_tokens);
    println!("Target tokens: {:?}", result.target_tokens);
    println!("Final teacher-forced loss: {:.5}", result.loss);

    println!("\nRunning the optimized suffix once more for final evaluation...");
    let evaluation = model.evaluate_suffix(PROMPT, &result.suffix_tokens, TARGET, 10)?;
    println!("\nTop 10 predictions for the first target token:");
    println!("{:>8}  {:>11}  token", "token ID", "probability");
    for prediction in &evaluation.top_next_tokens {
        println!(
            "{:>8}  {:>10.6}%  {:?}",
            prediction.token_id,
            prediction.probability * 100.0,
            prediction.text
        );
    }

    println!("\nDesired continuation probabilities (teacher forced):");
    for token in &evaluation.target_token_probabilities {
        println!(
            "{:>8}  {:>12.6e}%  {:?}",
            token.token_id,
            token.probability * 100.0,
            token.text
        );
    }
    println!(
        "Joint probability of {:?}: {:.8e}%",
        TARGET,
        evaluation.target_probability * 100.0
    );
    println!("Final verified loss: {:.5}", evaluation.loss);
    Ok(())
}
