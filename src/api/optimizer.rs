//! Minimal discrete and gradient-guided GCG-style suffix optimization.
//!
//! The invariant user-prompt prefix is prefilled once. Each iteration chooses
//! one suffix coordinate, scores discrete replacements with teacher forcing,
//! and greedily retains the lowest-loss candidate.

use super::{Model, cache::KvCache, tokenizer::adversarial_prompt_parts};
use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

#[derive(Debug, Clone)]
pub struct SuffixOptimizationRequest {
    pub prompt: String,
    pub target: String,
    pub suffix_length: usize,
    pub iterations: usize,
    pub candidate_count: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct SuffixOptimizationStep {
    pub iteration: usize,
    pub loss: f32,
    pub suffix: String,
    pub suffix_tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct SuffixOptimizationResult {
    pub suffix: String,
    pub suffix_tokens: Vec<u32>,
    pub target_tokens: Vec<u32>,
    pub loss: f32,
    pub steps: Vec<SuffixOptimizationStep>,
}

#[derive(Debug, Clone)]
pub struct GradientSuffixOptimizationRequest {
    pub prompt: String,
    pub target: String,
    pub suffix_length: usize,
    pub iterations: usize,
    /// Total number of discrete proposals scored per iteration.
    pub candidate_count: usize,
    pub coordinates_per_iteration: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct GradientSuffixOptimizationStep {
    pub iteration: usize,
    /// Exact discrete Q8_0 teacher-forced loss.
    pub loss: f32,
    /// Loss from the differentiable F32 graph before proposing replacements.
    pub gradient_loss: f32,
    pub suffix: String,
    pub suffix_tokens: Vec<u32>,
    /// Serialized context visible before the first target token is predicted.
    pub model_input: String,
    pub model_input_tokens: Vec<u32>,
    pub selected_coordinate: Option<usize>,
    pub gradient_candidates: Vec<u32>,
    pub improved: bool,
}

#[derive(Debug, Clone)]
pub struct GradientSuffixOptimizationResult {
    pub suffix: String,
    pub suffix_tokens: Vec<u32>,
    pub target_tokens: Vec<u32>,
    pub loss: f32,
    pub steps: Vec<GradientSuffixOptimizationStep>,
}

#[derive(Debug, Clone)]
pub struct TokenProbability {
    pub token_id: u32,
    pub text: String,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct TargetTokenProbability {
    pub token_id: u32,
    pub text: String,
    /// P(this token | prompt + suffix + preceding target tokens).
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct SuffixEvaluation {
    pub loss: f32,
    /// Product of the teacher-forced conditional target-token probabilities.
    pub target_probability: f64,
    /// Most likely tokens at the first position after prompt + suffix.
    pub top_next_tokens: Vec<TokenProbability>,
    pub target_token_probabilities: Vec<TargetTokenProbability>,
}

impl Model {
    /// Gradient-guided GCG. F32 activation gradients rank discrete token
    /// replacements; the unchanged Q8_0 teacher-forced scorer picks winners.
    pub fn optimize_suffix_with_gradients(
        &mut self,
        request: GradientSuffixOptimizationRequest,
    ) -> Result<GradientSuffixOptimizationResult> {
        validate_gradient_request(&request)?;
        let (prefix_text, closing_text) = adversarial_prompt_parts(&request.prompt);
        let prefix_tokens = self.tokenizer.encode_prefix(&prefix_text)?;
        let closing_tokens = self.tokenizer.encode_text(closing_text)?;
        let target_tokens = self.tokenizer.encode_text(&request.target)?;
        if target_tokens.is_empty() {
            bail!("target must tokenize to at least one token");
        }

        self.prefill(&prefix_tokens)?;
        let prefix_cache = self.cache.clone();
        let initial_token = self.tokenizer.initial_suffix_token()?;
        let mut suffix = vec![initial_token; request.suffix_length];
        let allowed_tokens = self.tokenizer.ordinary_token_ids();
        if allowed_tokens.is_empty() {
            bail!("tokenizer contains no candidate suffix tokens");
        }
        let mut rng = StdRng::seed_from_u64(request.seed);
        let mut loss = self.teacher_forced_suffix_loss(
            &prefix_cache,
            &suffix,
            &closing_tokens,
            &target_tokens,
        )?;
        let mut steps = Vec::with_capacity(request.iterations + 1);
        record_gradient_progress(
            self,
            &mut steps,
            0,
            loss,
            loss,
            &suffix,
            &prefix_text,
            closing_text,
            &prefix_tokens,
            &closing_tokens,
            &request.target,
            None,
            Vec::new(),
            false,
        )?;

        for iteration in 1..=request.iterations {
            let suffix_start = prefix_tokens.len();
            let target_start = suffix_start + suffix.len() + closing_tokens.len();
            let mut gradient_input = Vec::with_capacity(target_start + target_tokens.len() - 1);
            gradient_input.extend_from_slice(&prefix_tokens);
            gradient_input.extend_from_slice(&suffix);
            gradient_input.extend_from_slice(&closing_tokens);
            gradient_input.extend_from_slice(&target_tokens[..target_tokens.len() - 1]);
            let gradient = self.network.suffix_gradient(
                &gradient_input,
                suffix_start,
                suffix.len(),
                target_start,
                &target_tokens,
            )?;

            let coordinate_count = request.coordinates_per_iteration.min(suffix.len());
            let mut coordinates: Vec<usize> = (0..suffix.len()).collect();
            coordinates.shuffle(&mut rng);
            coordinates.truncate(coordinate_count);
            let proposals = gradient_proposals(
                &gradient.vocabulary_scores,
                &suffix,
                &allowed_tokens,
                &coordinates,
                request.candidate_count,
            );
            let candidate_ids = proposals.iter().map(|&(_, token)| token).collect();

            let previous_loss = loss;
            let mut best_loss = loss;
            let mut best_coordinate = None;
            let mut best_token = None;
            for &(coordinate, token) in &proposals {
                let mut proposal = suffix.clone();
                proposal[coordinate] = token;
                let proposal_loss = self.teacher_forced_suffix_loss(
                    &prefix_cache,
                    &proposal,
                    &closing_tokens,
                    &target_tokens,
                )?;
                if proposal_loss < best_loss {
                    best_loss = proposal_loss;
                    best_coordinate = Some(coordinate);
                    best_token = Some(token);
                }
            }
            if let (Some(coordinate), Some(token)) = (best_coordinate, best_token) {
                suffix[coordinate] = token;
                loss = best_loss;
            }
            record_gradient_progress(
                self,
                &mut steps,
                iteration,
                loss,
                gradient.loss,
                &suffix,
                &prefix_text,
                closing_text,
                &prefix_tokens,
                &closing_tokens,
                &request.target,
                best_coordinate,
                candidate_ids,
                loss < previous_loss,
            )?;
        }

        self.cache.clear();
        Ok(GradientSuffixOptimizationResult {
            suffix: self.tokenizer.decode(&suffix, false)?,
            suffix_tokens: suffix,
            target_tokens,
            loss,
            steps,
        })
    }

    /// Baseline greedy coordinate search over randomly proposed token suffixes
    /// using teacher-forced target negative log-likelihood.
    pub fn optimize_suffix(
        &mut self,
        request: SuffixOptimizationRequest,
    ) -> Result<SuffixOptimizationResult> {
        validate_request(&request)?;
        let (prefix_text, closing_text) = adversarial_prompt_parts(&request.prompt);
        let prefix_tokens = self.tokenizer.encode_prefix(&prefix_text)?;
        let closing_tokens = self.tokenizer.encode_text(closing_text)?;
        let target_tokens = self.tokenizer.encode_text(&request.target)?;
        if target_tokens.is_empty() {
            bail!("target must tokenize to at least one token");
        }

        // Reused for every candidate: prompt prefix -> populated K/V cache.
        self.prefill(&prefix_tokens)?;
        let prefix_cache = self.cache.clone();
        let initial_token = self.tokenizer.initial_suffix_token()?;
        let mut suffix = vec![initial_token; request.suffix_length];
        let candidates = self.tokenizer.ordinary_token_ids();
        if candidates.is_empty() {
            bail!("tokenizer contains no candidate suffix tokens");
        }
        let mut rng = StdRng::seed_from_u64(request.seed);
        let mut loss = self.teacher_forced_suffix_loss(
            &prefix_cache,
            &suffix,
            &closing_tokens,
            &target_tokens,
        )?;
        let mut steps = Vec::with_capacity(request.iterations + 1);
        record_progress(self, &mut steps, 0, loss, &suffix)?;

        for iteration in 1..=request.iterations {
            let coordinate = (iteration - 1) % request.suffix_length;
            let mut best_loss = loss;
            let mut best_token = suffix[coordinate];

            for _ in 0..request.candidate_count {
                let token = candidates[rng.random_range(0..candidates.len())];
                if token == suffix[coordinate] {
                    continue;
                }
                let mut proposal = suffix.clone();
                proposal[coordinate] = token;
                let proposal_loss = self.teacher_forced_suffix_loss(
                    &prefix_cache,
                    &proposal,
                    &closing_tokens,
                    &target_tokens,
                )?;
                if proposal_loss < best_loss {
                    best_loss = proposal_loss;
                    best_token = token;
                }
            }

            suffix[coordinate] = best_token;
            loss = best_loss;
            record_progress(self, &mut steps, iteration, loss, &suffix)?;
        }
        self.cache.clear();
        let suffix_text = self.tokenizer.decode(&suffix, false)?;
        Ok(SuffixOptimizationResult {
            suffix: suffix_text,
            suffix_tokens: suffix,
            target_tokens,
            loss,
            steps,
        })
    }

    /// Evaluate an optimized suffix with a fresh cache. The complete target
    /// probability is the teacher-forced joint probability, exp(-loss).
    pub fn evaluate_suffix(
        &mut self,
        prompt: &str,
        suffix: &[u32],
        target: &str,
        top_k: usize,
    ) -> Result<SuffixEvaluation> {
        if prompt.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        if suffix.is_empty() {
            bail!("suffix must contain at least one token");
        }
        if target.is_empty() {
            bail!("target must not be empty");
        }

        let (prefix_text, closing_text) = adversarial_prompt_parts(prompt);
        let prefix_tokens = self.tokenizer.encode_prefix(&prefix_text)?;
        let closing_tokens = self.tokenizer.encode_text(closing_text)?;
        let target_tokens = self.tokenizer.encode_text(target)?;
        if target_tokens.is_empty() {
            bail!("target must tokenize to at least one token");
        }

        self.prefill(&prefix_tokens)?;
        let mut tail = Vec::with_capacity(suffix.len() + closing_tokens.len());
        tail.extend_from_slice(suffix);
        tail.extend_from_slice(&closing_tokens);
        let mut logits = self.forward_logits(&tail)?;
        let top_next_tokens = top_probabilities(self, &logits, top_k)?;
        let mut loss = 0.0;
        let mut target_token_probabilities = Vec::with_capacity(target_tokens.len());

        for (index, &token_id) in target_tokens.iter().enumerate() {
            let token_loss = negative_log_probability(&logits, token_id as usize)
                .with_context(|| format!("scoring target token {index} ({token_id})"))?;
            loss += token_loss;
            target_token_probabilities.push(TargetTokenProbability {
                token_id,
                text: self.tokenizer.token_piece(token_id)?,
                probability: (-token_loss).exp(),
            });
            if index + 1 < target_tokens.len() {
                logits = self.decode(token_id)?;
            }
        }

        Ok(SuffixEvaluation {
            loss,
            target_probability: (-(loss as f64)).exp(),
            top_next_tokens,
            target_token_probabilities,
        })
    }

    fn teacher_forced_suffix_loss(
        &mut self,
        prefix_cache: &KvCache,
        suffix: &[u32],
        closing: &[u32],
        target: &[u32],
    ) -> Result<f32> {
        self.cache = prefix_cache.clone();
        let mut tail = Vec::with_capacity(suffix.len() + closing.len());
        tail.extend_from_slice(suffix);
        tail.extend_from_slice(closing);
        let mut logits = self.forward_logits(&tail)?;
        let mut loss = 0f32;
        for (index, &target_id) in target.iter().enumerate() {
            loss += negative_log_probability(&logits, target_id as usize)
                .with_context(|| format!("scoring target token {index} ({target_id})"))?;
            if index + 1 < target.len() {
                logits = self.decode(target_id)?;
            }
        }
        Ok(loss)
    }
}

fn gradient_proposals(
    scores: &[Vec<f32>],
    suffix: &[u32],
    allowed_tokens: &[u32],
    coordinates: &[usize],
    candidate_count: usize,
) -> Vec<(usize, u32)> {
    if coordinates.is_empty() || candidate_count == 0 {
        return Vec::new();
    }
    let per_coordinate = candidate_count.div_ceil(coordinates.len());
    let mut ranked = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        let mut tokens: Vec<u32> = allowed_tokens
            .iter()
            .copied()
            .filter(|&token| token != suffix[coordinate])
            .collect();
        tokens.sort_unstable_by(|&left, &right| {
            scores[coordinate][left as usize].total_cmp(&scores[coordinate][right as usize])
        });
        tokens.truncate(per_coordinate);
        ranked.push((coordinate, tokens));
    }

    let mut proposals = Vec::with_capacity(candidate_count);
    for rank in 0..per_coordinate {
        for (coordinate, tokens) in &ranked {
            if let Some(&token) = tokens.get(rank) {
                proposals.push((*coordinate, token));
                if proposals.len() == candidate_count {
                    return proposals;
                }
            }
        }
    }
    proposals
}

#[allow(clippy::too_many_arguments)]
fn record_gradient_progress(
    model: &Model,
    steps: &mut Vec<GradientSuffixOptimizationStep>,
    iteration: usize,
    loss: f32,
    gradient_loss: f32,
    suffix: &[u32],
    prefix_text: &str,
    closing_text: &str,
    prefix_tokens: &[u32],
    closing_tokens: &[u32],
    target: &str,
    selected_coordinate: Option<usize>,
    gradient_candidates: Vec<u32>,
    improved: bool,
) -> Result<()> {
    let text = model.tokenizer.decode(suffix, false)?;
    let model_input = format!("<BOS>{prefix_text}{text}{closing_text}");
    let mut model_input_tokens =
        Vec::with_capacity(prefix_tokens.len() + suffix.len() + closing_tokens.len());
    model_input_tokens.extend_from_slice(prefix_tokens);
    model_input_tokens.extend_from_slice(suffix);
    model_input_tokens.extend_from_slice(closing_tokens);
    let coordinate = selected_coordinate
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into());
    println!(
        "GCG iteration {iteration:>3} | loss {loss:>10.5} | coordinate {coordinate:>2} | improved {improved:<5} | suffix {text:?}"
    );
    println!("--- raw model input before first target token ---");
    println!("{model_input}");
    println!("--- raw input token IDs ---");
    println!("{model_input_tokens:?}");
    println!("--- teacher-forced target ---");
    println!("{target:?}\n");
    steps.push(GradientSuffixOptimizationStep {
        iteration,
        loss,
        gradient_loss,
        suffix: text,
        suffix_tokens: suffix.to_vec(),
        model_input,
        model_input_tokens,
        selected_coordinate,
        gradient_candidates,
        improved,
    });
    Ok(())
}

fn top_probabilities(model: &Model, logits: &[f32], count: usize) -> Result<Vec<TokenProbability>> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    model
        .top_logits(logits, count)?
        .into_iter()
        .map(|entry| {
            Ok(TokenProbability {
                token_id: entry.token_id,
                text: entry.text,
                probability: (((entry.logit - max) as f64).exp() / sum_exp) as f32,
            })
        })
        .collect()
}

fn validate_request(request: &SuffixOptimizationRequest) -> Result<()> {
    if request.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if request.target.is_empty() {
        bail!("target must not be empty");
    }
    if request.suffix_length == 0 {
        bail!("suffix_length must be greater than zero");
    }
    if request.candidate_count == 0 {
        bail!("candidate_count must be greater than zero");
    }
    Ok(())
}

fn validate_gradient_request(request: &GradientSuffixOptimizationRequest) -> Result<()> {
    if request.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if request.target.is_empty() {
        bail!("target must not be empty");
    }
    if request.suffix_length == 0 {
        bail!("suffix_length must be greater than zero");
    }
    if request.candidate_count == 0 {
        bail!("candidate_count must be greater than zero");
    }
    if request.coordinates_per_iteration == 0 {
        bail!("coordinates_per_iteration must be greater than zero");
    }
    Ok(())
}

fn negative_log_probability(logits: &[f32], target: usize) -> Result<f32> {
    let target_logit = *logits
        .get(target)
        .context("target token is outside logits")?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    Ok((max - target_logit) + sum_exp.ln() as f32)
}

fn record_progress(
    model: &Model,
    steps: &mut Vec<SuffixOptimizationStep>,
    iteration: usize,
    loss: f32,
    suffix: &[u32],
) -> Result<()> {
    let text = model.tokenizer.decode(suffix, false)?;
    println!("GCG iteration {iteration:>3} | loss {loss:>10.5} | suffix {text:?}");
    steps.push(SuffixOptimizationStep {
        iteration,
        loss,
        suffix: text,
        suffix_tokens: suffix.to_vec(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_forced_nll_matches_known_softmax() -> Result<()> {
        let logits = [0.0, 0.0];
        let loss = negative_log_probability(&logits, 1)?;
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn invalid_search_shape_is_rejected() {
        let request = SuffixOptimizationRequest {
            prompt: "prompt".into(),
            target: "target".into(),
            suffix_length: 0,
            iterations: 1,
            candidate_count: 1,
            seed: 0,
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn gradient_proposals_follow_lowest_allowed_scores() {
        let scores = vec![vec![9.0, 3.0, -2.0, -1.0], vec![9.0, -4.0, 2.0, 1.0]];
        let proposals = gradient_proposals(&scores, &[1, 2], &[1, 2, 3], &[0, 1], 4);
        assert_eq!(proposals, vec![(0, 2), (1, 1), (0, 3), (1, 3)]);
    }

    #[test]
    fn invalid_gradient_search_shape_is_rejected() {
        let request = GradientSuffixOptimizationRequest {
            prompt: "prompt".into(),
            target: "target".into(),
            suffix_length: 1,
            iterations: 1,
            candidate_count: 1,
            coordinates_per_iteration: 0,
            seed: 0,
        };
        assert!(validate_gradient_request(&request).is_err());
    }
}
