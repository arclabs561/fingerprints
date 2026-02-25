//! Coverage / support heuristics (classical baselines).
//!
//! These are “orientation tools” that connect the `fingerprints` unseen/coverage story to classical
//! toy problems:
//!
//! - **German tank problem**: estimate the maximum of a finite discrete uniform support from
//!   a sample drawn without replacement.
//! - **Coupon collector problem**: expected number of draws (with replacement) to observe all
//!   distinct types under a uniform distribution.
//! - **Expected distinct types**: expected number of distinct types after \(t\) uniform draws
//!   from \(n\) types.
//!
//! These provide useful baselines and sanity checks for more sophisticated estimators in
//! [`crate::pml`] and [`crate::vv`].

use crate::{PropEstError, Result};

/// German tank problem (classic): serial numbers are `1..=N` sampled **without replacement**.
///
/// Given the observed maximum \(m\) and sample size \(k\), the minimum-variance unbiased
/// estimator (MVUE) of \(N\) is:
///
/// \[
/// \hat N = m \cdot \frac{k+1}{k} - 1
/// \]
///
/// Returns a floating-point estimate (it is not generally an integer).
///
/// # Domain
///
/// - `sample_size` must be at least 1.
/// - `max_seen` must be at least 1 (1-indexed serial numbers).
///
/// # Examples
///
/// ```
/// use fingerprints::coverage::german_tank_unbiased_1_indexed;
///
/// // Saw serial numbers {3, 7, 12}; max = 12, k = 3.
/// let n_hat = german_tank_unbiased_1_indexed(12, 3).unwrap();
/// assert!((n_hat - 15.0).abs() < 1e-12);
/// ```
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

/// German tank variant for 0-indexed IDs: true support is `0..N` (size `N+1`), sampled
/// without replacement.
///
/// Converts the 0-indexed maximum `m0` to 1-indexed (`m = m0 + 1`) and returns an
/// estimate of the **support size** (i.e., \(N + 1\), not \(N\)).
///
/// # Examples
///
/// ```
/// use fingerprints::coverage::german_tank_unbiased_support_size_0_indexed;
///
/// // 0-indexed IDs: saw {0, 2, 5}; max_0 = 5, k = 3.
/// let s_hat = german_tank_unbiased_support_size_0_indexed(5, 3).unwrap();
/// // Equivalent to german_tank_unbiased_1_indexed(6, 3).
/// assert!((s_hat - 7.0).abs() < 1e-12);
/// ```
pub fn german_tank_unbiased_support_size_0_indexed(
    max_seen_0: u64,
    sample_size: u64,
) -> Result<f64> {
    german_tank_unbiased_1_indexed(max_seen_0 + 1, sample_size)
}

/// Expected draws to collect all `n` coupons (uniform, with replacement).
///
/// \[
/// \mathbb{E}[T] = n \cdot H_n = n \sum_{i=1}^{n} \frac{1}{i}
/// \]
///
/// where \(H_n\) is the \(n\)-th harmonic number. For large \(n\), this is approximately
/// \(n \ln n + \gamma n\) where \(\gamma \approx 0.5772\) is the Euler--Mascheroni constant.
///
/// Returns 0 when `n == 0`.
///
/// # Examples
///
/// ```
/// use fingerprints::coverage::coupon_collector_expected_draws;
///
/// assert!((coupon_collector_expected_draws(1) - 1.0).abs() < 1e-12);
/// assert!((coupon_collector_expected_draws(2) - 3.0).abs() < 1e-12);
/// ```
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
/// \mathbb{E}[D_t] = n\left(1 - \left(1 - \frac{1}{n}\right)^t\right)
/// \]
///
/// This is exact for sampling with replacement from a uniform distribution. It gives a
/// useful baseline: if the observed distinct count is much lower than this, the
/// distribution is likely non-uniform.
///
/// Returns 0 when `n == 0`.
///
/// # Examples
///
/// ```
/// use fingerprints::coverage::expected_distinct_uniform;
///
/// // After 1 draw from 10 types, expect exactly 1 distinct type.
/// assert!((expected_distinct_uniform(10, 1) - 1.0).abs() < 1e-12);
/// // After many draws, approach n.
/// assert!(expected_distinct_uniform(10, 1000) > 9.99);
/// ```
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
