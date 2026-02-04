//! `propest`: property estimation from samples (fingerprints/profiles).
//!
//! This crate is for the “unseen regime”: estimating properties of an unknown discrete distribution
//! from a sample when the support may be large and the empirical plug-in estimator is biased.
//!
//! Design intent:
//! - Keep `logp` as the *definition* layer (entropy/divergence functionals on known distributions).
//! - Put estimation policy here (bias correction, sample-size regimes, solver-backed methods).
//!
//! References (orientation):
//! - Valiant & Valiant (2013/2017): “Estimating the Unseen…”
//! - Orlitsky line: profile / PML estimators (future).
//!
//! ## Quick example
//!
//! ```rust
//! use propest::{Fingerprint, entropy_miller_madow_nats, unseen_mass_good_turing, support_chao1};
//!
//! let counts = [3usize, 3, 2, 1, 1];
//! let fp = Fingerprint::from_counts(counts).unwrap();
//!
//! let h_mm = entropy_miller_madow_nats(&fp);
//! let p0 = unseen_mass_good_turing(&fp);
//! let s_hat = support_chao1(&fp);
//!
//! assert!(h_mm >= 0.0);
//! assert!((0.0..=1.0).contains(&p0));
//! assert!(s_hat >= fp.observed_support() as f64);
//! ```

#![forbid(unsafe_code)]

use core::num::NonZeroUsize;
use thiserror::Error;

pub mod pml;
pub mod vv;

/// Errors for sample-based estimators.
#[derive(Debug, Error)]
pub enum PropEstError {
    #[error("empty sample")]
    EmptySample,

    #[error("invalid input: {0}")]
    Invalid(&'static str),

    #[error(transparent)]
    Logp(#[from] logp::Error),
}

pub type Result<T> = core::result::Result<T, PropEstError>;

/// Convert per-symbol counts into an empirical distribution on the observed support.
///
/// Returns a probability vector `p` with `p.len() == counts.len()` and `Σ p = 1`.
pub fn empirical_simplex_from_counts(counts: &[usize]) -> Result<Vec<f64>> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    let n: usize = counts.iter().sum();
    let n = NonZeroUsize::new(n).ok_or(PropEstError::Invalid("sum(counts) == 0"))?;
    Ok(counts.iter().map(|&c| (c as f64) / (n.get() as f64)).collect())
}

/// A fingerprint/profile: `F[i]` is the number of domain elements seen exactly `i` times.
///
/// Conventions:
/// - `F[0]` is unused and always 0 (we store from 0 for indexing convenience).
/// - Sample size is `n = Σ_i i * F[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// Counts-of-counts (index = multiplicity).
    pub f: Vec<usize>,
}

impl Fingerprint {
    /// Compute fingerprint from per-symbol counts.
    ///
    /// Example: counts = [5,3,1,1] => F[1]=2, F[3]=1, F[5]=1.
    pub fn from_counts<I>(counts: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        let counts: Vec<usize> = counts.into_iter().collect();
        if counts.is_empty() {
            return Err(PropEstError::EmptySample);
        }
        let max_c = *counts.iter().max().unwrap_or(&0);
        if max_c == 0 {
            return Err(PropEstError::Invalid("all counts are zero"));
        }
        let mut f = vec![0usize; max_c + 1];
        for c in counts {
            if c == 0 {
                continue;
            }
            f[c] += 1;
        }
        Ok(Self { f })
    }

    /// Total sample size `n = Σ_i i * F[i]`.
    #[must_use]
    pub fn sample_size(&self) -> usize {
        self.f
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &fi)| i * fi)
            .sum()
    }

    /// Number of observed distinct elements: `S_obs = Σ_{i>=1} F[i]`.
    #[must_use]
    pub fn observed_support(&self) -> usize {
        self.f.iter().skip(1).sum()
    }

    /// `F1`: the number of singletons.
    #[must_use]
    pub fn singletons(&self) -> usize {
        self.f.get(1).copied().unwrap_or(0)
    }

    /// `F2`: the number of doubletons.
    #[must_use]
    pub fn doubletons(&self) -> usize {
        self.f.get(2).copied().unwrap_or(0)
    }
}

/// Plug-in (empirical) entropy estimator (nats) from a fingerprint.
///
/// This treats the observed histogram as if it were the true distribution.
#[must_use]
pub fn entropy_plugin_nats(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for (i, &fi) in fp.f.iter().enumerate().skip(1) {
        if fi == 0 {
            continue;
        }
        let p = (i as f64) / n;
        // There are `fi` symbols with probability `p`.
        h -= (fi as f64) * p * p.ln();
    }
    h
}

/// Plug-in (empirical) entropy estimator (nats) from per-symbol counts.
///
/// This is a convenience wrapper that routes through `logp`'s entropy implementation.
pub fn entropy_plugin_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let p = empirical_simplex_from_counts(counts)?;
    Ok(logp::entropy_unchecked(&p))
}

/// Miller–Madow bias-corrected entropy estimator (nats).
///
/// Classical correction: H_MM = H_plugin + (S_obs - 1)/(2n).
#[must_use]
pub fn entropy_miller_madow_nats(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return 0.0;
    }
    let s_obs = fp.observed_support() as f64;
    entropy_plugin_nats(fp) + (s_obs - 1.0) / (2.0 * n)
}

/// Jackknife (delete-1) entropy estimator (nats) from a fingerprint.
///
/// Bias reduction via:
/// \[
/// H_{JK} = n H_n - (n-1) \mathbb{E}[H_{n-1}]
/// \]
/// where \(H_{n-1}\) is the plug-in entropy of the sample with one observation removed uniformly
/// at random.
///
/// This is a classical, solver-free improvement over the plug-in estimator.
#[must_use]
pub fn entropy_jackknife_nats(fp: &Fingerprint) -> f64 {
    let n_usize = fp.sample_size();
    if n_usize <= 1 {
        return 0.0;
    }
    let n = n_usize as f64;
    let n1 = (n_usize - 1) as f64;

    // Plug-in entropy for size n.
    let h_n = entropy_plugin_nats(fp);

    // S = Σ c ln c across observed symbols, expressed via fingerprint:
    // each multiplicity r contributes F_r symbols of count r.
    let mut s = 0.0;
    for (r, &fr) in fp.f.iter().enumerate().skip(1) {
        if fr == 0 {
            continue;
        }
        let r_f = r as f64;
        s += (fr as f64) * r_f * r_f.ln();
    }

    // Expected leave-one-out plug-in entropy.
    // Removing a uniformly random observation hits a symbol of count r with probability (r F_r)/n.
    let ln_n1 = n1.ln();
    let mut e_h_n1 = 0.0;
    for (r, &fr) in fp.f.iter().enumerate().skip(1) {
        if fr == 0 {
            continue;
        }
        let r_f = r as f64;
        let prob_pick = (r_f * fr as f64) / n;

        // Update S for decrementing one symbol of count r.
        let s_minus = if r > 1 {
            let rm1 = (r - 1) as f64;
            s - (r_f * r_f.ln()) + (rm1 * rm1.ln())
        } else {
            // r == 1 => symbol disappears; contribution becomes 0.
            s - (r_f * r_f.ln())
        };

        let h_minus = ln_n1 - (s_minus / n1);
        e_h_n1 += prob_pick * h_minus;
    }

    n * h_n - (n - 1.0) * e_h_n1
}

/// Jackknife (delete-1) entropy estimator (nats) from per-symbol counts.
pub fn entropy_jackknife_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let fp = Fingerprint::from_counts(counts.iter().copied())?;
    Ok(entropy_jackknife_nats(&fp))
}

/// Good–Turing coverage estimate: the estimated **unseen probability mass**.
///
/// Classical estimate: \(\hat p_0 \approx F_1 / n\).
#[must_use]
pub fn unseen_mass_good_turing(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return 0.0;
    }
    (fp.singletons() as f64 / n).clamp(0.0, 1.0)
}

/// Chao1 support-size estimator from the fingerprint.
///
/// \(\hat S = S_{obs} + \frac{F_1^2}{2F_2}\).
/// If \(F_2 = 0\), use the usual bias-corrected fallback:
/// \(\hat S = S_{obs} + \frac{F_1(F_1-1)}{2}\).
#[must_use]
pub fn support_chao1(fp: &Fingerprint) -> f64 {
    let s_obs = fp.observed_support() as f64;
    let f1 = fp.singletons() as f64;
    let f2 = fp.doubletons() as f64;
    if f1 <= 0.0 {
        return s_obs;
    }
    if f2 > 0.0 {
        s_obs + (f1 * f1) / (2.0 * f2)
    } else {
        s_obs + (f1 * (f1 - 1.0)) / 2.0
    }
}

/// A very small helper: entropy in bits for convenience.
#[must_use]
pub fn entropy_plugin_bits(fp: &Fingerprint) -> f64 {
    entropy_plugin_nats(fp) / logp::LN_2
}

#[must_use]
pub fn entropy_miller_madow_bits(fp: &Fingerprint) -> f64 {
    entropy_miller_madow_nats(fp) / logp::LN_2
}

#[must_use]
pub fn entropy_jackknife_bits(fp: &Fingerprint) -> f64 {
    entropy_jackknife_nats(fp) / logp::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fp_sample_size_matches_counts(counts in prop::collection::vec(1usize..50, 1..100)) {
            // Interpret `counts` as multiplicities for distinct symbols.
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let n_from_fp = fp.sample_size();
            let n_from_counts: usize = counts.iter().sum();
            prop_assert_eq!(n_from_fp, n_from_counts);
        }

        #[test]
        fn plugin_entropy_nonnegative(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let h = entropy_plugin_nats(&fp);
            prop_assert!(h >= -1e-12);
        }

        #[test]
        fn miller_madow_ge_plugin(counts in prop::collection::vec(1usize..50, 2..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let h0 = entropy_plugin_nats(&fp);
            let hmm = entropy_miller_madow_nats(&fp);
            // For n>=2, the (S-1)/(2n) term is >= 0, so MM >= plugin.
            prop_assert!(hmm + 1e-12 >= h0);
        }

        #[test]
        fn unseen_mass_is_in_unit_interval(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let p0 = unseen_mass_good_turing(&fp);
            prop_assert!(p0 >= -1e-12);
            prop_assert!(p0 <= 1.0 + 1e-12);
        }

        #[test]
        fn chao1_support_is_at_least_observed(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let s_hat = support_chao1(&fp);
            prop_assert!(s_hat + 1e-12 >= fp.observed_support() as f64);
        }

        #[test]
        fn plugin_entropy_matches_logp_on_counts(counts in prop::collection::vec(1usize..50, 1..200)) {
            // Compare the fingerprint-based computation against building the explicit simplex
            // and delegating to `logp`.
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let h_fp = entropy_plugin_nats(&fp);
            let h_logp = entropy_plugin_nats_from_counts(&counts).unwrap();
            prop_assert!((h_fp - h_logp).abs() < 1e-12);
        }

        #[test]
        fn jackknife_entropy_is_finite_and_nonnegative(counts in prop::collection::vec(1usize..50, 2..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let h = entropy_jackknife_nats(&fp);
            prop_assert!(h.is_finite());
            prop_assert!(h >= -1e-12);
        }

        #[test]
        fn jackknife_counts_matches_fingerprint(counts in prop::collection::vec(1usize..50, 2..200)) {
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let h1 = entropy_jackknife_nats(&fp);
            let h2 = entropy_jackknife_nats_from_counts(&counts).unwrap();
            prop_assert!((h1 - h2).abs() < 1e-12);
        }
    }
}

