//! Coverage / support heuristics (classical baselines).
//!
//! These are “orientation tools” that connect the `fingerprints` unseen/coverage story to classical
//! toy problems:
//! - German tank: estimate a finite support maximum from a sample without replacement.
//! - Coupon collector: expected time to cover all types under uniform sampling with replacement.

use crate::{PropEstError, Result};

/// German tank problem (classic): serial numbers are `1..=N` sampled **without replacement**.
///
/// Given `max_seen = m` and `k = sample_size`, an unbiased estimator of `N` is:
/// \[
/// \hat N = m \cdot \frac{k+1}{k} - 1.
/// \]
///
/// Returns a floating estimate (it is not generally an integer).
pub fn german_tank_unbiased_1_indexed(max_seen: u64, sample_size: u64) -> Result<f64> {
    if sample_size == 0 {
        return Err(PropEstError::Invalid("sample_size must be >= 1"));
    }
    if max_seen == 0 {
        return Err(PropEstError::Invalid("max_seen must be >= 1 (1-indexed)"));
    }
    let m = max_seen as f64;
    let k = sample_size as f64;
    Ok(m * ((k + 1.0) / k) - 1.0)
}

/// German tank variant for 0-indexed IDs: true support is `0..N` (size `N+1`), sampled without replacement.
///
/// If the observed maximum is `m0` (0-indexed), convert to 1-indexed by `m=m0+1` and return
/// an estimate of the **support size** `(N+1)`.
pub fn german_tank_unbiased_support_size_0_indexed(
    max_seen_0: u64,
    sample_size: u64,
) -> Result<f64> {
    german_tank_unbiased_1_indexed(max_seen_0 + 1, sample_size)
}

/// Expected draws to collect all `n` coupons (uniform, with replacement): `n * H_n`.
#[must_use]
pub fn coupon_collector_expected_draws(n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut h = 0.0;
    for i in 1..=n {
        h += 1.0 / (i as f64);
    }
    (n as f64) * h
}

/// Expected number of distinct types observed after `t` uniform draws from `n` types.
///
/// \[
/// \mathbb{E}[D_t] = n\left(1 - \left(1 - \frac1n\right)^t\right).
/// \]
#[must_use]
pub fn expected_distinct_uniform(n: u64, t: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let p_not_seen = (1.0 - 1.0 / n_f).powf(t as f64);
    n_f * (1.0 - p_not_seen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupon_collector_small() {
        assert!((coupon_collector_expected_draws(1) - 1.0).abs() < 1e-12);
        assert!((coupon_collector_expected_draws(2) - 3.0).abs() < 1e-12);
        assert!((coupon_collector_expected_draws(3) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn expected_distinct_sane_bounds() {
        let n = 10;
        for t in [0, 1, 5, 100] {
            let d = expected_distinct_uniform(n, t);
            assert!(d >= -1e-12);
            assert!(d <= n as f64 + 1e-9);
        }
    }
}
