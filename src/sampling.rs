//! Reproducible temperature / top-k / top-p sampling for the autoregressive path.
use crate::{Result, invalid};

/// Seed used when a request does not provide one.
pub(crate) fn random_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    time ^ COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0 selects greedy decoding (the only mode MTP verification supports).
    pub temperature: f32,
    /// Nucleus mass in (0, 1]; 1 disables the filter.
    pub top_p: f32,
    /// Number of highest-logit candidates kept; 0 disables the filter.
    pub top_k: usize,
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_p: 1.,
            top_k: 0,
            seed: 0,
        }
    }
}
impl Sampling {
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.
    }
    pub fn validate(&self) -> Result<()> {
        if !(0. ..=2.).contains(&self.temperature) {
            return Err(invalid("temperature must be between 0 and 2"));
        }
        if !(self.top_p > 0. && self.top_p <= 1.) {
            return Err(invalid("top_p must be greater than 0 and at most 1"));
        }
        Ok(())
    }
}

/// xoshiro256** seeded through SplitMix64: small, fast and fully deterministic.
pub struct Sampler {
    config: Sampling,
    state: [u64; 4],
}
impl Sampler {
    pub fn new(config: Sampling) -> Result<Self> {
        config.validate()?;
        let mut x = config.seed;
        let mut state = [0; 4];
        for s in &mut state {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *s = z ^ (z >> 31);
        }
        Ok(Self { config, state })
    }
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    /// Uniform in [0, 1) with 53 bits of precision.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(invalid("empty logits"));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite logits"));
        }
        let c = self.config;
        // Highest logit first; ties prefer the lower token ID.
        let order = |a: &u32, b: &u32| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        };
        if c.is_greedy() {
            return Ok((0..logits.len() as u32).min_by(order).unwrap());
        }
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        if c.top_k > 0 && c.top_k < ids.len() {
            ids.select_nth_unstable_by(c.top_k - 1, order);
            ids.truncate(c.top_k);
        }
        let sorted = c.top_p < 1.;
        if sorted {
            ids.sort_unstable_by(order);
        }
        let max = ids
            .iter()
            .map(|&i| logits[i as usize])
            .fold(f32::NEG_INFINITY, f32::max) as f64;
        let temperature = c.temperature as f64;
        let mut weights: Vec<f64> = ids
            .iter()
            .map(|&i| ((logits[i as usize] as f64 - max) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if sorted {
            // Keep the smallest high-probability prefix whose mass reaches top_p.
            let target = c.top_p as f64 * total;
            let mut mass = 0.;
            let keep = weights
                .iter()
                .position(|w| {
                    mass += w;
                    mass >= target
                })
                .map_or(weights.len(), |i| i + 1);
            weights.truncate(keep);
        }
        let total: f64 = weights.iter().sum();
        let mut r = self.uniform() * total;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return Ok(ids[i]);
            }
            r -= w;
        }
        Ok(ids[weights.len() - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Sampling {
        Sampling {
            temperature,
            top_p,
            top_k,
            seed,
        }
    }
    fn draw(c: Sampling, logits: &[f32], n: usize) -> Vec<u32> {
        let mut s = Sampler::new(c).unwrap();
        (0..n).map(|_| s.sample(logits).unwrap()).collect()
    }
    #[test]
    fn fixed_seed_is_reproducible() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 / 10.).collect();
        let c = config(0.8, 0.95, 40, 1234);
        let a = draw(c, &logits, 200);
        assert_eq!(a, draw(c, &logits, 200));
        assert_ne!(a, draw(config(0.8, 0.95, 40, 1235), &logits, 200));
        // Pin the generator so an accidental algorithm change is caught.
        let mut s = Sampler::new(config(1., 1., 0, 0)).unwrap();
        assert_eq!(s.next_u64(), 0x99ec_5f36_cb75_f2b4);
    }
    #[test]
    fn greedy_and_filters() {
        let logits = [1., 5., 5., -2.];
        // Temperature 0 is argmax with lower-ID tie breaking.
        assert_eq!(draw(config(0., 1., 0, 1), &logits, 3), [1, 1, 1]);
        // top_k=1 always selects the argmax regardless of temperature.
        assert!(
            draw(config(2., 1., 1, 9), &[0., 3., 1.], 50)
                .iter()
                .all(|&t| t == 1)
        );
        // top_p keeps only the dominant token here (p(1) ~ 0.88 > 0.5).
        assert!(
            draw(config(1., 0.5, 0, 3), &[0., 3., 1.], 50)
                .iter()
                .all(|&t| t == 1)
        );
        // Without filters every token with real mass appears.
        let mut seen = draw(config(1., 1., 0, 5), &[0., 0., 0.], 300);
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, [0, 1, 2]);
        // top_k=2 never emits the third-ranked token.
        assert!(
            draw(config(1.5, 1., 2, 11), &[1., 0.9, 0.8], 300)
                .iter()
                .all(|&t| t < 2)
        );
    }
    #[test]
    fn empirical_distribution_follows_softmax() {
        let logits = [0f32, (2f32).ln(), (3f32).ln()];
        let samples = draw(config(1., 1., 0, 42), &logits, 60_000);
        for (token, expected) in [(0, 1. / 6.), (1, 2. / 6.), (2, 3. / 6.)] {
            let p = samples.iter().filter(|&&t| t == token).count() as f64 / 60_000.;
            assert!(
                (p - expected).abs() < 0.01,
                "token {token}: {p} vs {expected}"
            );
        }
    }
    #[test]
    fn rejects_invalid_configuration_and_logits() {
        assert!(Sampler::new(config(-0.1, 1., 0, 0)).is_err());
        assert!(Sampler::new(config(2.5, 1., 0, 0)).is_err());
        assert!(Sampler::new(config(1., 0., 0, 0)).is_err());
        assert!(Sampler::new(config(1., 1.5, 0, 0)).is_err());
        assert!(Sampler::new(config(f32::NAN, 1., 0, 0)).is_err());
        let mut s = Sampler::new(config(1., 1., 0, 0)).unwrap();
        assert!(s.sample(&[]).is_err());
        assert!(s.sample(&[0., f32::NAN]).is_err());
    }
}
