//! Intentionally small sampling implementation.

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 42,
        }
    }
}

pub(crate) struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub(crate) fn new(config: SamplingConfig) -> Result<Self> {
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            bail!("temperature must be finite and non-negative");
        }
        Ok(Self {
            config,
            rng: StdRng::seed_from_u64(config.seed),
        })
    }

    pub(crate) fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            bail!("cannot sample empty or non-finite logits");
        }
        if self.config.temperature == 0.0 {
            return Ok(argmax(logits) as u32);
        }

        let inv_temperature = self.config.temperature.recip();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = logits
            .iter()
            .map(|&x| (((x - max) * inv_temperature) as f64).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            bail!("invalid temperature-sampling probability mass");
        }
        let mut needle = self.rng.random::<f64>() * total;
        for (id, weight) in weights.into_iter().enumerate() {
            needle -= weight;
            if needle <= 0.0 {
                return Ok(id as u32);
            }
        }
        Ok((logits.len() - 1) as u32)
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded_sampling_are_deterministic() -> Result<()> {
        let logits = [0.1, 2.0, 1.0];
        let mut greedy = Sampler::new(SamplingConfig::default())?;
        assert_eq!(greedy.sample(&logits)?, 1);

        let config = SamplingConfig {
            temperature: 0.8,
            seed: 7,
        };
        let mut first = Sampler::new(config)?;
        let mut second = Sampler::new(config)?;
        assert_eq!(first.sample(&logits)?, second.sample(&logits)?);
        Ok(())
    }
}
