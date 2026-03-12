//! `fingerprints`: property estimation from samples (fingerprints/profiles).
//!
//! This crate is for the “unseen regime”: estimating properties of an unknown discrete distribution
//! from a sample when the support may be large and the empirical plug-in estimator is biased.
//!
//! Design intent:
//! - Rely on a *definition* layer for entropy/divergence on known distributions.
//! - Put estimation policy here (bias correction, sample-size regimes, solver-backed methods).
//!
//! ## Estimator hierarchy
//!
//! The entropy estimators form a bias-correction hierarchy from cheapest to most principled:
//!
//! 1. **Plug-in** -- maximum-likelihood; O(1) per fingerprint entry; negatively biased.
//! 2. **Miller-Madow** -- adds `(S_obs - 1) / 2n`; corrects the leading O(1/n) bias term.
//! 3. **Jackknife** -- delete-1 resampling; removes bias to higher order without a parametric model.
//! 4. **Pitman-Yor (DPYM)** -- Bayesian nonparametric; models the unseen tail explicitly via the
//!    Pitman-Yor process. Formally posterior-consistent under mild conditions (Hashino & Tsukuda, 2026).
//!
//! For support estimation, Chao1 provides a nonparametric lower bound on the true support size.
//! For unseen mass, the Good-Turing estimator is the classical baseline.
//!
//! ## References (orientation)
//!
//! - Valiant & Valiant (2013/2017): "Estimating the Unseen" (JACM)
//! - Orlitsky line: profile / PML estimators (see [`pml`] module)
//! - Han, Jiao, Weissman (2025): "Besting Good-Turing: Optimality of NPMLE" -- establishes
//!   that the nonparametric MLE achieves minimax-optimal rates for symmetric functionals,
//!   providing theoretical motivation for the PML direction on our roadmap
//! - Hashino & Tsukuda (2026): "Estimating the Shannon Entropy Using the Pitman-Yor Process" --
//!   posterior consistency of the PY estimator implemented here
//!
//! ## Quick example
//!
//! ```rust
//! use fingerprints::{Fingerprint, entropy_miller_madow_nats, unseen_mass_good_turing, support_chao1};
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
#![warn(missing_docs)]

use core::cmp::Ordering;
use core::num::NonZeroUsize;
use thiserror::Error;

pub mod coverage;
pub mod pml;
#[cfg(feature = "lp")]
pub mod vv;

/// Errors for sample-based estimators.
#[derive(Debug, Error)]
pub enum EstimationError {
    /// The input sample was empty.
    #[error("empty sample")]
    EmptySample,

    /// Invalid input (with reason).
    #[error("invalid input: {0}")]
    Invalid(&'static str),

    /// Error propagated from logp.
    #[error(transparent)]
    Logp(#[from] logp::Error),
}

/// Deprecated alias for [`EstimationError`].
#[deprecated(since = "0.2.0", note = "renamed to EstimationError")]
pub type PropEstError = EstimationError;

/// Result type for estimation operations.
pub type Result<T> = core::result::Result<T, EstimationError>;

/// Convert per-symbol counts into an empirical distribution on the observed support.
///
/// Given counts \(c_1, \dots, c_m\) with sample size \(n = \sum_i c_i\), returns the
/// maximum-likelihood probability vector:
///
/// \[
/// \hat p_i = \frac{c_i}{n}, \quad i = 1, \dots, m.
/// \]
///
/// The returned vector satisfies `p.len() == counts.len()` and \(\sum_i \hat p_i = 1\).
///
/// # Errors
///
/// Returns [`EstimationError::EmptySample`] if `counts` is empty, or
/// [`EstimationError::Invalid`] if all counts are zero.
///
/// # Examples
///
/// ```
/// use fingerprints::empirical_simplex_from_counts;
///
/// let p = empirical_simplex_from_counts(&[3, 1]).unwrap();
/// assert!((p[0] - 0.75).abs() < 1e-12);
/// assert!((p[1] - 0.25).abs() < 1e-12);
/// ```
pub fn empirical_simplex_from_counts(counts: &[usize]) -> Result<Vec<f64>> {
    if counts.is_empty() {
        return Err(EstimationError::EmptySample);
    }
    let n: usize = counts.iter().sum();
    let n = NonZeroUsize::new(n).ok_or(EstimationError::Invalid("sum(counts) == 0"))?;
    Ok(counts
        .iter()
        .map(|&c| (c as f64) / (n.get() as f64))
        .collect())
}

/// A fingerprint (also called a "profile" or "pattern"): the sufficient statistic for
/// symmetric properties of an unknown distribution.
///
/// `F[i]` is the number of domain elements that appear exactly `i` times in the sample.
///
/// Conventions:
/// - `F[0]` is unused and always 0 (stored from index 0 for convenience).
/// - Sample size: \(n = \sum_{i \ge 1} i \cdot F_i\).
/// - Observed support: \(S_{\text{obs}} = \sum_{i \ge 1} F_i\).
///
/// # References
///
/// - Orlitsky, Suresh, Wu (2016): "Optimal prediction of the number of unseen species"
/// - Valiant & Valiant (2013/2017): "Estimating the Unseen"
///
/// # Examples
///
/// ```
/// use fingerprints::Fingerprint;
///
/// // Four symbols with counts [5, 3, 1, 1]:
/// let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
/// assert_eq!(fp.sample_size(), 10);
/// assert_eq!(fp.observed_support(), 4);
/// assert_eq!(fp.singletons(), 2);   // F_1 = 2 (two symbols seen once)
/// assert_eq!(fp.doubletons(), 0);   // F_2 = 0
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// Counts-of-counts (index = multiplicity).
    f: Vec<usize>,
}

/// `ln(n!)` via Stirling's approximation (exact for n <= 20 via lookup).
pub(crate) fn ln_factorial(n: usize) -> f64 {
    // Exact values for small n (avoids rounding issues in combinatorics).
    const EXACT: [f64; 21] = [
        0.0,                     // 0!
        0.0,                     // 1!
        core::f64::consts::LN_2, // 2! = 2
        1.791_759_469_228_055,   // ln(6)
        3.178_053_830_347_946_5, // ln(24)
        4.787_491_742_782_046,   // ln(120)
        6.579_251_212_010_101,   // ln(720)
        8.525_161_361_065_415,   // ln(5040)
        10.604_602_902_745_251,  // ln(40320)
        12.801_827_480_081_469,  // ln(362880)
        15.104_412_573_075_516,  // ln(3628800)
        17.502_307_845_873_887,  // ln(39916800)
        19.987_214_495_661_885,  // ln(479001600)
        22.552_163_853_123_42,   // ln(6227020800)
        25.191_221_182_738_68,   // ln(87178291200)
        27.899_271_383_840_89,   // ln(1307674368000)
        30.671_860_128_818_843,  // ln(20922789888000)
        33.505_073_450_136_89,   // ln(355687428096000)
        36.395_445_208_033_05,   // ln(6402373705728000)
        39.339_884_187_199_49,   // ln(121645100408832000)
        42.335_616_460_753_485,  // ln(2432902008176640000)
    ];
    if n < EXACT.len() {
        return EXACT[n];
    }
    // Stirling: ln(n!) ~ n ln(n) - n + 0.5 ln(2 pi n) + 1/(12n) - 1/(360 n^3)
    let nf = n as f64;
    nf * nf.ln() - nf + 0.5 * (2.0 * core::f64::consts::PI * nf).ln() + 1.0 / (12.0 * nf)
        - 1.0 / (360.0 * nf * nf * nf)
}

/// Convert nats to bits.
#[must_use]
pub fn to_bits(nats: f64) -> f64 {
    nats / core::f64::consts::LN_2
}

impl Fingerprint {
    /// Compute a fingerprint from per-symbol counts.
    ///
    /// Each element of the input is the number of times a distinct symbol was observed.
    /// Zero counts are silently ignored (they correspond to unseen symbols).
    ///
    /// Example: counts = \[5,3,1,1\] produces F\[1\]=2, F\[3\]=1, F\[5\]=1.
    ///
    /// # Errors
    ///
    /// Returns [`EstimationError::EmptySample`] if the iterator is empty, or
    /// [`EstimationError::Invalid`] if all counts are zero.
    pub fn from_counts<I>(counts: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        let counts: Vec<usize> = counts.into_iter().collect();
        if counts.is_empty() {
            return Err(EstimationError::EmptySample);
        }
        let max_c = *counts.iter().max().unwrap_or(&0);
        if max_c == 0 {
            return Err(EstimationError::Invalid("all counts are zero"));
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

    /// Build a fingerprint directly from a counts-of-counts vector.
    ///
    /// `f[i]` is the number of symbols seen exactly `i` times. `f[0]` must be 0
    /// (unseen symbols are not tracked). The vector must contain at least one
    /// non-zero entry at index >= 1.
    ///
    /// Use this when you already have the fingerprint representation (e.g., from
    /// serialized data or an external tool).
    ///
    /// # Errors
    ///
    /// Returns [`EstimationError::Invalid`] if `f[0] != 0` or if all entries are zero.
    /// Returns [`EstimationError::EmptySample`] if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use fingerprints::Fingerprint;
    ///
    /// // F[1]=2, F[3]=1 means: 2 singletons, 1 symbol seen 3 times.
    /// let fp = Fingerprint::from_frequency_counts(&[0, 2, 0, 1]).unwrap();
    /// assert_eq!(fp.singletons(), 2);
    /// assert_eq!(fp.observed_support(), 3); // 2 + 0 + 1
    /// assert_eq!(fp.sample_size(), 5);      // 1*2 + 3*1
    /// ```
    pub fn from_frequency_counts(f: &[usize]) -> Result<Self> {
        if f.is_empty() {
            return Err(EstimationError::EmptySample);
        }
        if f.first().copied().unwrap_or(0) != 0 {
            return Err(EstimationError::Invalid(
                "f[0] must be 0 (unseen count is not stored)",
            ));
        }
        if f.iter().skip(1).all(|&fi| fi == 0) {
            return Err(EstimationError::Invalid("all frequency counts are zero"));
        }
        let mut v = f.to_vec();
        // Strip trailing zeros so semantically-identical fingerprints are equal.
        while v.len() > 1 && v.last() == Some(&0) {
            v.pop();
        }
        Ok(Self { f: v })
    }

    /// Total sample size: \(n = \sum_{i \ge 1} i \cdot F_i\).
    #[must_use]
    pub fn sample_size(&self) -> usize {
        self.f
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &fi)| i * fi)
            .sum()
    }

    /// Number of observed distinct elements: \(S_{\text{obs}} = \sum_{i \ge 1} F_i\).
    #[must_use]
    pub fn observed_support(&self) -> usize {
        self.f.iter().skip(1).sum()
    }

    /// \(F_1\): the number of singletons (symbols seen exactly once).
    ///
    /// Singletons are the primary driver of Good--Turing coverage correction and Chao1
    /// support estimation.
    #[must_use]
    pub fn singletons(&self) -> usize {
        self.f.get(1).copied().unwrap_or(0)
    }

    /// \(F_2\): the number of doubletons (symbols seen exactly twice).
    ///
    /// Doubletons appear in the denominator of the Chao1 estimator.
    #[must_use]
    pub fn doubletons(&self) -> usize {
        self.f.get(2).copied().unwrap_or(0)
    }

    /// \(F_r\): the number of symbols seen exactly `r` times.
    ///
    /// Returns 0 for any `r` beyond the stored fingerprint length, and for `r = 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fingerprints::Fingerprint;
    ///
    /// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
    /// assert_eq!(fp.count_at(1), fp.singletons());
    /// assert_eq!(fp.count_at(2), fp.doubletons());
    /// assert_eq!(fp.count_at(100), 0);
    /// ```
    #[must_use]
    pub fn count_at(&self, r: usize) -> usize {
        self.f.get(r).copied().unwrap_or(0)
    }

    /// Number of species observed exactly `j` times (0-indexed: f[0] is always 0).
    #[must_use]
    pub fn count(&self, j: usize) -> usize {
        self.f.get(j).copied().unwrap_or(0)
    }

    /// Maximum frequency observed.
    #[must_use]
    pub fn max_freq(&self) -> usize {
        self.f.len().saturating_sub(1)
    }

    /// Iterator over (frequency, count) pairs, skipping zeros.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.f.iter().copied().enumerate().filter(|&(_, c)| c > 0)
    }

    /// Raw fingerprint vector (0-indexed).
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.f
    }

    /// Number of distinct species observed (alias for [`observed_support`]).
    #[must_use]
    pub fn observed_species(&self) -> usize {
        self.f.iter().skip(1).sum()
    }
}

/// Plug-in (empirical) entropy estimator (nats) from a fingerprint.
///
/// Treats the observed histogram as the true distribution and computes:
///
/// \[
/// \hat H_{\text{plug}} = -\sum_{i \ge 1} F_i \cdot \frac{i}{n} \ln\!\left(\frac{i}{n}\right)
/// \]
///
/// where \(n\) is the sample size. This is the maximum-likelihood entropy estimator.
/// It is **negatively biased**: \(\mathbb{E}[\hat H_{\text{plug}}] \le H(p)\) for any \(p\).
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, entropy_plugin_nats};
///
/// let fp = Fingerprint::from_counts([4, 4]).unwrap();
/// // Uniform over 2 symbols => H = ln(2).
/// assert!((entropy_plugin_nats(&fp) - 2.0_f64.ln()).abs() < 1e-12);
/// ```
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

/// Miller--Madow bias-corrected entropy estimator (nats).
///
/// Applies a first-order bias correction to the plug-in estimator:
///
/// \[
/// \hat H_{\text{MM}} = \hat H_{\text{plug}} + \frac{S_{\text{obs}} - 1}{2n}
/// \]
///
/// where \(S_{\text{obs}}\) is the observed support size. The correction term
/// compensates for the leading \(O(1/n)\) negative bias of the plug-in estimator.
///
/// # Caution
///
/// The correction is first-order only. When \(S_{\text{obs}} \approx n\) (many singletons,
/// undersampled regime), higher-order bias terms dominate and Miller--Madow can be *less*
/// accurate than the plug-in estimator. In that regime, prefer [`entropy_pitman_yor_nats`]
/// or the jackknife.
///
/// # References
///
/// - Miller (1955), "Note on the bias of information estimates"
/// - Paninski (2003), "Estimation of entropy and mutual information" -- analyzes failure
///   of the first-order correction when alphabet size grows with sample size
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, entropy_miller_madow_nats, entropy_plugin_nats};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// assert!(entropy_miller_madow_nats(&fp) >= entropy_plugin_nats(&fp));
/// ```
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
/// H_{JK} = n H_n - (n-1) \mathbb{E}\[H_{n-1}\]
/// \]
/// where \(H_{n-1}\) is the plug-in entropy of the sample with one observation removed uniformly
/// at random.
///
/// This is a classical, solver-free improvement over the plug-in estimator. It removes bias to
/// O(1/n^2) without requiring a parametric model, sitting between Miller-Madow (O(1/n) correction)
/// and the full Bayesian nonparametric approach (Pitman-Yor). The tradeoff is higher variance
/// than Miller-Madow for very small samples.
///
/// # References
///
/// - Zahl (1977), "Jackknifing an index of diversity"
/// - Efron (1979), "Bootstrap methods: another look at the jackknife"
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

/// Good--Turing coverage estimate: the estimated **unseen probability mass**.
///
/// Classical estimate:
///
/// \[
/// \hat p_0 = \frac{F_1}{n}
/// \]
///
/// where \(F_1\) is the singleton count and \(n\) is the sample size. The result is
/// clamped to \([0, 1]\). When \(F_1 = 0\), no unseen mass is estimated.
///
/// # References
///
/// - Good (1953), "The population frequencies of species and the estimation of population parameters"
/// - Painsky (2023), "Generalized Good-Turing improves missing mass estimation" (JASA) --
///   shows a generalized GT estimator that dominates classical GT in MSE; a potential
///   upgrade path for this function
/// - Chang, Liu, Zheng (2025), "Confidence Intervals Using Turing's Estimator" -- provides
///   non-asymptotic CIs for the missing mass; relevant for future CI support
///
/// For the general frequency formula (`r >= 1`), see [`good_turing_estimate`].
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, unseen_mass_good_turing};
///
/// let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
/// let p0 = unseen_mass_good_turing(&fp);
/// // F_1 = 2, n = 10, so p0 = 0.2.
/// assert!((p0 - 0.2).abs() < 1e-12);
/// ```
#[must_use]
pub fn unseen_mass_good_turing(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return 0.0;
    }
    (fp.singletons() as f64 / n).clamp(0.0, 1.0)
}

/// Minimal-bias unseen mass estimator using the full fingerprint.
///
/// An alternating-sign linear combination of all fingerprint entries:
///
/// \[
/// \hat M_0^{\text{MB}} = \sum_{i=1}^{r_{\max}} (-1)^{i-1} \frac{F_i}{\binom{n}{i}}
/// \]
///
/// This uses all available frequency classes to exponentially reduce bias compared to
/// the Good--Turing estimator (which uses only `F_1`). The first term `F_1/n` equals
/// the Good--Turing estimate; the remaining terms are its exact bias correction.
///
/// # Tradeoffs
///
/// - **Bias**: exponentially smaller than Good--Turing (O(S * p_max^n) vs O(1/n)).
/// - **Variance**: can be higher than Good--Turing for skewed distributions where
///   p_max >= 0.5. Use Good--Turing when the distribution is known to be highly skewed.
/// - **Numerical**: the alternating signs and large binomial coefficients require
///   careful computation; the implementation uses iterative binomial coefficient
///   evaluation to avoid overflow.
///
/// # References
///
/// - Lee & Bohme (2025), "How Much is Unseen Depends Chiefly on Information About
///   the Seen" (ICLR) -- derives the exact bias decomposition of Good--Turing and
///   the minimal-bias estimator
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, unseen_mass_minimal_bias, unseen_mass_good_turing};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// let p0_mb = unseen_mass_minimal_bias(&fp);
/// let p0_gt = unseen_mass_good_turing(&fp);
/// assert!((0.0..=1.0).contains(&p0_mb));
/// // Both are estimates of the same quantity.
/// assert!(p0_mb.is_finite());
/// ```
#[must_use]
pub fn unseen_mass_minimal_bias(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 1..fp.f.len() {
        let f_i = fp.count_at(i) as f64;
        if f_i == 0.0 {
            continue;
        }
        let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
        let binom = binom_f64(n, i);
        if binom > 0.0 {
            sum += sign * f_i / binom;
        }
    }
    sum.clamp(0.0, 1.0)
}

/// Binomial coefficient C(n, k) as f64, computed iteratively for stability.
fn binom_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Good--Turing sample coverage: the estimated fraction of the distribution observed.
///
/// \[
/// \hat C = 1 - \frac{F_1}{n}
/// \]
///
/// Coverage is the complement of unseen mass: `coverage_good_turing(fp) == 1.0 -
/// unseen_mass_good_turing(fp)`. When coverage is 1.0 (no singletons), all observed
/// species have been seen at least twice, suggesting the sample may be adequate.
///
/// For a bias-corrected variant using doubletons, see [`coverage_chao_shen`].
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, coverage_good_turing};
///
/// let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
/// let c = coverage_good_turing(&fp);
/// // F_1 = 2, n = 10, coverage = 1 - 2/10 = 0.8.
/// assert!((c - 0.8).abs() < 1e-12);
/// ```
#[must_use]
pub fn coverage_good_turing(fp: &Fingerprint) -> f64 {
    1.0 - unseen_mass_good_turing(fp)
}

/// Good--Turing frequency estimate for items seen exactly `r` times.
///
/// For `r >= 1`, estimates the true probability of any item observed exactly `r` times:
///
/// \[
/// \hat\theta(r) = \frac{r+1}{n} \cdot \frac{F_{r+1}}{F_r}
/// \]
///
/// Returns `None` when `r = 0` (use [`unseen_mass_good_turing`] for the total unseen mass),
/// when `F_r = 0` (undefined), or when the sample is empty.
///
/// # Properties
///
/// - **Normalization**: `unseen_mass + sum_r theta_hat(r) * F_r = 1` for all fingerprints.
/// - **Max-count pathology**: `theta_hat(r_max) = 0` because `F_{r_max+1} = 0`.
///   The most frequent items are systematically underweighted.
///
/// # References
///
/// - Good (1953), "The population frequencies of species and the estimation of
///   population parameters"
/// - Gale & Sampson (1995), "Good-Turing frequency estimation without tears" --
///   log-linear smoothing of the frequency spectrum to handle zero `F_r` entries
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, good_turing_estimate};
///
/// // counts [3, 3, 2, 1, 1]: F_1=2, F_2=1, F_3=2
/// let fp = Fingerprint::from_counts([3, 3, 2, 1, 1]).unwrap();
/// let n = fp.sample_size() as f64; // 10
///
/// // theta_hat(1) = 2/10 * F_2/F_1 = 2/10 * 1/2 = 0.1
/// assert!((good_turing_estimate(&fp, 1).unwrap() - 0.1).abs() < 1e-12);
///
/// // r=0: returns None (use unseen_mass_good_turing instead).
/// assert_eq!(good_turing_estimate(&fp, 0), None);
/// ```
#[must_use]
pub fn good_turing_estimate(fp: &Fingerprint, r: usize) -> Option<f64> {
    if r == 0 {
        return None;
    }
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return None;
    }
    let f_r = fp.count_at(r);
    if f_r == 0 {
        return None;
    }
    let f_r1 = fp.count_at(r + 1) as f64;
    Some(((r + 1) as f64 / n) * (f_r1 / f_r as f64))
}

/// Chao--Shen bias-corrected sample coverage.
///
/// An improved coverage estimator that uses both singletons and doubletons:
///
/// \[
/// \hat C_{\text{CS}} = 1 - \frac{F_1}{n} \cdot \frac{(n-1) F_1}{(n-1) F_1 + 2 F_2}
/// \]
///
/// When `F_2 > 0`, the correction factor is strictly less than 1, so
/// `coverage_chao_shen >= coverage_good_turing`. When `F_2 = 0`, the correction
/// factor equals 1 and both estimators coincide.
///
/// # References
///
/// - Chao & Shen (2003), "Nonparametric estimation of Shannon's index of diversity
///   when there are unseen species in sample" (Environmental and Ecological Statistics)
/// - Chao, Wang, Jost (2013), "Coverage-based rarefaction and extrapolation" --
///   derives the coverage estimator from the Good--Turing frequency formula
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, coverage_chao_shen, coverage_good_turing};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// let c_cs = coverage_chao_shen(&fp);
/// let c_gt = coverage_good_turing(&fp);
/// // Chao-Shen is always at least as high as basic Good-Turing.
/// assert!(c_cs >= c_gt - 1e-12);
/// ```
#[must_use]
pub fn coverage_chao_shen(fp: &Fingerprint) -> f64 {
    let n = fp.sample_size() as f64;
    if n <= 0.0 {
        return 1.0;
    }
    let f1 = fp.singletons() as f64;
    if f1 <= 0.0 {
        return 1.0;
    }
    let f2 = fp.doubletons() as f64;
    let denom = (n - 1.0) * f1 + 2.0 * f2;
    if denom <= 0.0 {
        return (1.0 - f1 / n).clamp(0.0, 1.0);
    }
    let correction = (n - 1.0) * f1 / denom;
    (1.0 - (f1 / n) * correction).clamp(0.0, 1.0)
}

/// Chao1 lower-bound estimator of the true support size.
///
/// \[
/// \hat S = S_{\text{obs}} + \frac{F_1^2}{2 F_2}
/// \]
///
/// If \(F_2 = 0\), the bias-corrected fallback is used:
///
/// \[
/// \hat S = S_{\text{obs}} + \frac{F_1(F_1 - 1)}{2}
/// \]
///
/// The estimator satisfies \(\hat S \ge S_{\text{obs}}\). When \(F_1 = 0\) (no singletons),
/// the estimate equals \(S_{\text{obs}}\) (no unseen species are predicted).
///
/// The Chao1 estimator is a nonparametric lower bound: it is guaranteed to never overestimate
/// the true support size (in expectation), making it safe for conservative decisions.
///
/// # References
///
/// - Chao (1984), "Nonparametric estimation of the number of classes in a population" --
///   the foundational reference; derives the estimator as a lower bound on species richness
///   using only singletons and doubletons
/// - Chao & Jost (2012), "Coverage-based rarefaction and extrapolation" -- extends the
///   framework to abundance-based coverage
///
/// For variance and confidence intervals, see [`support_chao1_with_ci`].
/// For a reduced-bias variant using `F_3` and `F_4`, see [`support_ichao1`].
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, support_chao1};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// let s_hat = support_chao1(&fp);
/// assert!(s_hat >= fp.observed_support() as f64);
/// ```
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

/// Point estimate, variance, and 95% confidence interval for the Chao1 estimator.
///
/// Returned by [`support_chao1_with_ci`]. The confidence interval uses the
/// log-transformation from Chao (1987), which guarantees the lower bound
/// is at least `S_obs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chao1Estimate {
    /// Point estimate (same value as [`support_chao1`]).
    pub point: f64,
    /// Estimated variance of the point estimate.
    pub variance: f64,
    /// Lower bound of the 95% CI.
    pub ci_lower: f64,
    /// Upper bound of the 95% CI.
    pub ci_upper: f64,
}

/// Chao1 lower-bound estimator with variance and 95% confidence interval.
///
/// Returns the same point estimate as [`support_chao1`], plus the estimated
/// variance (delta method) and a log-transformation 95% CI.
///
/// ## Variance
///
/// When \(F_2 > 0\):
///
/// \[
/// \widehat{\text{Var}} = F_2 \left[
///   \frac{q^2}{2} + q^3 + \frac{q^4}{4}
/// \right], \quad q = \frac{F_1}{F_2}
/// \]
///
/// When \(F_2 = 0\) (bias-corrected fallback):
///
/// \[
/// \widehat{\text{Var}} = \frac{F_1(F_1-1)}{2}
///     + \frac{F_1(2F_1-1)^2}{4}
///     - \frac{F_1^4}{4\hat S}
///     \]
///
/// ## Confidence interval
///
/// Uses the log-transformation CI from Chao (1987):
///
/// \[
/// S_{\text{obs}} + \frac{T}{R} \;\le\; S \;\le\; S_{\text{obs}} + T \cdot R
/// \]
///
/// where \(T = \hat S - S_{\text{obs}}\) and
/// \(R = \exp\!\bigl(1.96 \sqrt{\ln(1 + V / T^2)}\bigr)\).
///
/// # References
///
/// - Chao (1984, 1987): variance formula and log-transformation CI
/// - Chao & Colwell (2017), "Thirty years of progeny from Chao's inequality" (SORT) --
///   comprehensive review of variance and CI methods
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, support_chao1_with_ci};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// let est = support_chao1_with_ci(&fp);
/// assert!(est.ci_lower <= est.point);
/// assert!(est.point <= est.ci_upper);
/// assert!(est.variance >= 0.0);
/// assert!(est.ci_lower >= fp.observed_support() as f64 - 1e-12);
/// ```
#[must_use]
pub fn support_chao1_with_ci(fp: &Fingerprint) -> Chao1Estimate {
    let point = support_chao1(fp);
    let s_obs = fp.observed_support() as f64;
    let f1 = fp.singletons() as f64;
    let f2 = fp.doubletons() as f64;

    if f1 <= 0.0 {
        return Chao1Estimate {
            point,
            variance: 0.0,
            ci_lower: s_obs,
            ci_upper: s_obs,
        };
    }

    let variance = if f2 > 0.0 {
        let q = f1 / f2;
        f2 * (q * q / 2.0 + q * q * q + q * q * q * q / 4.0)
    } else {
        let term1 = f1 * (f1 - 1.0) / 2.0;
        let term2 = f1 * (2.0 * f1 - 1.0).powi(2) / 4.0;
        let term3 = if point > 0.0 {
            f1.powi(4) / (4.0 * point)
        } else {
            0.0
        };
        (term1 + term2 - term3).max(0.0)
    };

    let t = point - s_obs;
    let (ci_lower, ci_upper) = if t > 0.0 && variance > 0.0 {
        let log_arg = (1.0 + variance / (t * t)).ln();
        if log_arg > 0.0 {
            let r = (1.96 * log_arg.sqrt()).exp();
            (s_obs + t / r, s_obs + t * r)
        } else {
            (point, point)
        }
    } else {
        (s_obs, s_obs)
    };

    Chao1Estimate {
        point,
        variance,
        ci_lower,
        ci_upper,
    }
}

/// Improved Chao1 (iChao1) lower-bound estimator of the true support size.
///
/// Uses `F_3` and `F_4` in addition to `F_1` and `F_2` to reduce bias:
///
/// \[
/// \hat S_{\text{iChao1}} = \hat S_{\text{Chao1}} + \frac{F_3}{4 F_4}
///   \max\!\left(F_1 - \frac{F_2 F_3}{2 F_4},\; 0\right)
/// \]
///
/// Falls back to [`support_chao1`] when `F_4 = 0` (correction undefined)
/// or `F_3 = 0` (correction is zero).
///
/// # References
///
/// - Chiu, Wang, Walther, Chao (2014), "An improved nonparametric lower bound
///   of species richness via a modified Good--Turing frequency formula"
///   (Biometrics)
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, support_ichao1, support_chao1};
///
/// let fp = Fingerprint::from_counts([10, 5, 3, 2, 1, 1, 1, 1]).unwrap();
/// let s_ichao = support_ichao1(&fp);
/// let s_chao = support_chao1(&fp);
/// // iChao1 is always at least Chao1.
/// assert!(s_ichao >= s_chao - 1e-12);
/// ```
#[must_use]
pub fn support_ichao1(fp: &Fingerprint) -> f64 {
    let s_chao1 = support_chao1(fp);
    let f3 = fp.count_at(3) as f64;
    let f4 = fp.count_at(4) as f64;
    if f4 <= 0.0 || f3 <= 0.0 {
        return s_chao1;
    }
    let f1 = fp.singletons() as f64;
    let f2 = fp.doubletons() as f64;
    let correction = (f3 / (4.0 * f4)) * (f1 - f2 * f3 / (2.0 * f4)).max(0.0);
    s_chao1 + correction
}

/// Selected hyperparameters for the Pitman–Yor entropy estimator.
///
/// Notation matches Hashino & Tsukuda (2026): discount `d ∈ [0,1)` and concentration `alpha > -d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitmanYorParams {
    /// Discount parameter `d ∈ [0,1)`.
    pub d: f64,
    /// Concentration parameter `alpha > -d`.
    pub alpha: f64,
}

#[inline]
fn gt0(x: f64) -> bool {
    matches!(x.partial_cmp(&0.0), Some(Ordering::Greater))
}

fn pitman_yor_upper_bound(n: f64, t: f64, c01: f64, big_f: f64, d: f64, alpha: f64) -> Option<f64> {
    if !(n.is_finite() && t.is_finite() && c01.is_finite() && big_f.is_finite()) {
        return None;
    }
    if !(d.is_finite() && alpha.is_finite()) {
        return None;
    }
    if !(0.0..1.0).contains(&d) {
        return None;
    }
    // Domain: N + alpha > 0, alpha + T d > 0, 1 - d > 0.
    let n_plus_alpha = n + alpha;
    if !gt0(n_plus_alpha) {
        return None;
    }
    let u = alpha + t * d;
    if !gt0(u) {
        return None;
    }
    let one_minus_d = 1.0 - d;
    if !gt0(one_minus_d) {
        return None;
    }

    let term1 = n_plus_alpha.ln();
    let term2 = -c01 * one_minus_d.ln();
    let term3 = if big_f > 0.0 {
        // log((u+1)/u) computed as ln(u+1) - ln(u) for stability.
        big_f * ((u + 1.0).ln() - u.ln())
    } else {
        0.0
    };
    Some(term1 + term2 + term3)
}

fn pitman_yor_alpha_star(n: f64, t: f64, big_f: f64, d: f64) -> Option<f64> {
    if !(n.is_finite() && t.is_finite() && big_f.is_finite() && d.is_finite()) {
        return None;
    }
    if !(0.0..1.0).contains(&d) {
        return None;
    }
    // If the upper-bound term `F` is non-positive, the minimizer in alpha is at the boundary;
    // in our usage `F > 0` whenever there is at least one singleton, but keep this defensive.
    if !gt0(big_f) {
        return Some(-t * d);
    }
    let nd = n - t * d;
    // Discriminant: (1-F)^2 + 4F(N - Td) >= 0 when F>=0 and N-Td >= 0.
    let disc = (1.0 - big_f).mul_add(1.0 - big_f, 4.0 * big_f * nd);
    let sqrt_disc = disc.max(0.0).sqrt();
    // u = alpha + T d.
    let u = (big_f - 1.0 + sqrt_disc) / 2.0;
    Some(u - t * d)
}

fn entropy_mpy_nats(d: f64, alpha: f64) -> f64 {
    // Shannon entropy (nats) of the marginal Pitman–Yor distribution MPY(d, alpha).
    //
    // - For d = 0: MPY becomes geometric with parameter 1/(alpha+1) and has a closed form.
    // - For d > 0: we compute a prefix by recurrence and close the tail by an asymptotic
    //   integral approximation (power-law tail with exponent 1/d).
    if !gt0(alpha) {
        // Degenerate/invalid alpha shouldn't happen for the estimator; return a safe value.
        return 0.0;
    }
    if d <= 0.0 {
        // Geometric distribution entropy (nats): (1+α) ln(1+α) − α ln α.
        if alpha == 0.0 {
            return 0.0;
        }
        return (1.0 + alpha) * (1.0 + alpha).ln() - alpha * alpha.ln();
    }

    let one_minus_d = 1.0 - d;
    if !gt0(one_minus_d) {
        return 0.0;
    }

    // p1 = (1-d)/(alpha+1)
    let mut p = (one_minus_d / (alpha + 1.0)).max(0.0);
    if p == 0.0 {
        return 0.0;
    }

    let mut mass = p;
    let mut h = -p * p.ln();

    // Tuned for "library default" cost/accuracy; callers can wrap if they need tighter bounds.
    const MAX_TERMS: usize = 200_000;
    const TAIL_MASS_TOL: f64 = 1e-14;
    const TAIL_REL_TOL: f64 = 1e-8;
    const X_MIN: f64 = 50.0; // require alpha + k d to be moderately large before trusting tail closure

    // Track k as the current index of p (= p_k). Tail approximation starts from p_{k+1}.
    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        let num = alpha + kf * d;
        let den = num + 1.0;
        if !gt0(den) {
            break;
        }
        let p_next = p * (num / den);

        // True remaining mass after k terms (tail starts at k+1).
        let tail_mass_true = (1.0 - mass).max(0.0);
        if tail_mass_true <= TAIL_MASS_TOL {
            return h;
        }

        // Asymptotic tail mass estimate from the next term p_{k+1}.
        // Using: tail_mass ≈ p_{k+1} * (alpha + (k+1)d) / (1-d).
        let x0 = alpha + (kf + 1.0) * d;
        let tail_mass_est = if x0.is_finite() && x0 > 0.0 {
            p_next * (x0 / one_minus_d)
        } else {
            f64::INFINITY
        };

        let rel_err = if tail_mass_true > 0.0 && tail_mass_est.is_finite() {
            ((tail_mass_est - tail_mass_true).abs() / tail_mass_true).abs()
        } else {
            f64::INFINITY
        };

        // If the tail mass estimate matches the (exact-by-definition) remainder, close the tail.
        if x0 >= X_MIN && rel_err <= TAIL_REL_TOL && p_next > 0.0 {
            let tail_entropy = tail_mass_true * (-p_next.ln() + 1.0 / one_minus_d);
            return h + tail_entropy;
        }

        // Accumulate p_{k+1} and continue.
        p = p_next;
        mass += p;
        // If we lose all remaining mass to underflow, just stop.
        if !gt0(p) {
            break;
        }
        if p.is_finite() {
            h -= p * p.ln();
        } else {
            break;
        }
        // If numerical drift pushes mass over 1, clamp for tail logic.
        if mass > 1.0 {
            mass = 1.0;
        }
    }

    // Fallback tail closure using the last computed p as a scale proxy.
    let tail_mass_true = (1.0 - mass).max(0.0);
    if tail_mass_true <= 0.0 {
        h
    } else {
        let p0 = p.max(1e-300);
        h + tail_mass_true * (-p0.ln() + 1.0 / one_minus_d)
    }
}

fn entropy_dpym_nats(fp: &Fingerprint, d: f64, alpha: f64) -> f64 {
    // DPYM predictive entropy estimator:
    // H(q) = H(q*) + q_new * H(MPY(d, alpha + T d)),
    // where q* includes the "unseen bucket" q_new = (alpha + T d)/(N + alpha).
    let n = fp.sample_size() as f64;
    if !gt0(n) {
        return 0.0;
    }
    let t = fp.observed_support() as f64;
    let one_minus_d = 1.0 - d;
    if !gt0(one_minus_d) {
        return 0.0;
    }
    let denom = n + alpha;
    if !gt0(denom) {
        return 0.0;
    }

    let mut h_qstar = 0.0;
    for (r, &fr) in fp.f.iter().enumerate().skip(1) {
        if fr == 0 {
            continue;
        }
        let q = ((r as f64) - d) / denom;
        if q > 0.0 && q.is_finite() {
            h_qstar -= (fr as f64) * q * q.ln();
        }
    }

    let q_new = (alpha + t * d) / denom;
    if q_new > 0.0 && q_new.is_finite() {
        h_qstar -= q_new * q_new.ln();
    }

    // H(pi) with alpha' = alpha + T d.
    let alpha_prime = alpha + t * d;
    if !gt0(q_new) || !gt0(alpha_prime) {
        return h_qstar;
    }
    h_qstar + q_new * entropy_mpy_nats(d, alpha_prime)
}

/// Select `(d, alpha)` for the Pitman–Yor entropy estimator from a fingerprint.
///
/// Policy:
/// - If there are **no singletons**, return `(0, 0)` (large-sample regime ⇒ plug-in).
/// - Otherwise, minimize the Hashino–Tsukuda cross-entropy upper bound (estimated by plug-ins)
///   over `d ∈ [0, 1)` with `alpha` set to the (per-`d`) minimizer.
#[must_use]
pub fn pitman_yor_params_hat(fp: &Fingerprint) -> PitmanYorParams {
    let n_usize = fp.sample_size();
    if n_usize == 0 {
        return PitmanYorParams { d: 0.0, alpha: 0.0 };
    }
    let n = n_usize as f64;
    let t = fp.observed_support() as f64;
    let m1 = fp.singletons() as f64;

    // Large-sample regime heuristic from the paper: if there are no singletons, corrections vanish.
    if m1 <= 0.0 {
        return PitmanYorParams { d: 0.0, alpha: 0.0 };
    }

    // Plug-in estimates (Good–Turing style) used by Hashino & Tsukuda.
    let c0 = (m1 / n).clamp(0.0, 1.0);
    let c1 = (1.0 - c0) * (m1 / n);
    let c01 = c0 + c1;

    // K-hat from the paper: N / (1 - C0_hat).
    // Guard the denominator to avoid inf when C0_hat ≈ 1 (all singletons).
    let one_minus_c0 = (1.0 - c0).max(1.0 / n.max(1.0));
    let k_hat = n / one_minus_c0;
    let big_f = 0.5 * c0 * (k_hat - t + 1.0).max(0.0);

    let d_max = 1.0 - 1e-9;
    let eps_alpha = 1e-12;

    // Coarse-to-fine grid search over d. This is cheap, robust, and avoids brittle quadratic cases.
    let mut best_d = 0.0;
    let mut best_alpha = 0.0;
    let mut best_val = f64::INFINITY;

    let coarse_steps: usize = 1024;
    let coarse_step = d_max / (coarse_steps as f64);
    for i in 0..=coarse_steps {
        let d = (i as f64) * coarse_step;
        let mut alpha = match pitman_yor_alpha_star(n, t, big_f, d) {
            Some(a) if a.is_finite() => a,
            _ => continue,
        };
        alpha = alpha.max(-d + eps_alpha);
        if let Some(v) = pitman_yor_upper_bound(n, t, c01, big_f, d, alpha) {
            if v.is_finite() && v < best_val {
                best_val = v;
                best_d = d;
                best_alpha = alpha;
            }
        }
    }

    // Local refinement around the best coarse d.
    let lo = (best_d - 2.0 * coarse_step).max(0.0);
    let hi = (best_d + 2.0 * coarse_step).min(d_max);
    let fine_steps: usize = 2048;
    let fine_step = (hi - lo) / (fine_steps as f64);
    if fine_step > 0.0 {
        for i in 0..=fine_steps {
            let d = lo + (i as f64) * fine_step;
            let mut alpha = match pitman_yor_alpha_star(n, t, big_f, d) {
                Some(a) if a.is_finite() => a,
                _ => continue,
            };
            alpha = alpha.max(-d + eps_alpha);
            if let Some(v) = pitman_yor_upper_bound(n, t, c01, big_f, d, alpha) {
                if v.is_finite() && v < best_val {
                    best_val = v;
                    best_d = d;
                    best_alpha = alpha;
                }
            }
        }
    }

    // Final clamp to the model domain.
    let d = best_d.clamp(0.0, d_max);
    let alpha = best_alpha.max(-d + eps_alpha);
    PitmanYorParams { d, alpha }
}

/// Pitman--Yor entropy estimator (nats) from a fingerprint.
///
/// Implements the estimator described in:
/// - Takato Hashino & Koji Tsukuda (2026), “Estimating the Shannon Entropy Using the Pitman--Yor Process”.
///
/// The method approximates the unknown population distribution by the DPYM predictive
/// distribution \(q\), including an explicit “unseen mass” bucket, and returns \(H(q)\).
/// Parameters \((d, \alpha)\) are selected by minimizing a cross-entropy upper bound
/// (see [`pitman_yor_params_hat`]).
///
/// The PY process provides a power-law tail model for the unseen portion of the distribution,
/// which is more realistic than the geometric tail of the Dirichlet process (d=0 case).
/// Hashino & Tsukuda (2026) prove posterior consistency: as sample size grows, the DPYM
/// estimator converges to the true entropy under mild regularity conditions.
///
/// When there are no singletons (\(F_1 = 0\)), the estimator reduces to the plug-in
/// estimator (no unseen-mass correction is applied).
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, entropy_pitman_yor_nats, entropy_plugin_nats};
///
/// let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
/// let h_py = entropy_pitman_yor_nats(&fp);
/// let h_plug = entropy_plugin_nats(&fp);
/// // PY corrects upward when singletons are present.
/// assert!(h_py >= h_plug - 1e-12);
/// ```
#[must_use]
pub fn entropy_pitman_yor_nats(fp: &Fingerprint) -> f64 {
    let params = pitman_yor_params_hat(fp);
    entropy_dpym_nats(fp, params.d, params.alpha)
}

/// Opinionated default entropy estimator (nats).
///
/// A single-call “good default” for the **unseen regime**:
/// - Uses the Pitman--Yor / DPYM estimator when singletons are present.
/// - Reduces to the plug-in estimator when there are no singletons.
///
/// Currently delegates to [`entropy_pitman_yor_nats`]. The routing policy may
/// evolve in future versions as new estimators are added.
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, entropy_default_nats};
///
/// let fp = Fingerprint::from_counts([10, 8, 5, 3, 1]).unwrap();
/// let h = entropy_default_nats(&fp);
/// assert!(h.is_finite() && h >= 0.0);
/// ```
#[must_use]
pub fn entropy_default_nats(fp: &Fingerprint) -> f64 {
    entropy_pitman_yor_nats(fp)
}

/// Plug-in **sample code length** (nats) for the observed sample.
///
/// Concretely: if you encode each observation using the plug-in (empirical) model \(\hat p\),
/// the expected per-symbol code length is \(H(\hat p)\) (in nats). For a sample of size `n`,
/// the total codelength is approximately:
///
/// \[
/// L \approx n \cdot H(\hat p)
/// \]
///
/// This is a useful *scalar* for MDL-style comparisons (representation/model selection) and for
/// turning “entropy” into an auditable quantity measured in total nats.
#[must_use]
pub fn sample_codelen_plugin_nats(fp: &Fingerprint) -> f64 {
    (fp.sample_size() as f64) * entropy_plugin_nats(fp)
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
        fn minimal_bias_unseen_mass_in_unit_interval(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let p0 = unseen_mass_minimal_bias(&fp);
            prop_assert!(p0 >= -1e-12, "minimal-bias unseen mass {} < 0", p0);
            prop_assert!(p0 <= 1.0 + 1e-12, "minimal-bias unseen mass {} > 1", p0);
        }

        #[test]
        fn chao1_support_is_at_least_observed(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let s_hat = support_chao1(&fp);
            prop_assert!(s_hat + 1e-12 >= fp.observed_support() as f64);
        }

        #[test]
        fn good_turing_reindexing_identity(counts in prop::collection::vec(1usize..50, 2..200)) {
            // The Good-Turing normalization identity (algebraic):
            //   F_1/n + sum_{r=1}^{r_max} (r+1)/n * F_{r+1} = 1
            // This is the re-indexing identity that proves GT estimates sum to 1.
            let fp = Fingerprint::from_counts(counts).unwrap();
            let n = fp.sample_size() as f64;
            let mut total = fp.singletons() as f64 / n;
            for r in 1..fp.f.len() {
                total += (r + 1) as f64 / n * fp.count_at(r + 1) as f64;
            }
            prop_assert!((total - 1.0).abs() < 1e-12,
                "GT re-indexing identity failed: total = {}", total);
        }

        #[test]
        fn good_turing_agrees_with_formula(counts in prop::collection::vec(1usize..50, 2..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let n = fp.sample_size() as f64;
            for r in 1..fp.f.len() {
                let f_r = fp.count_at(r);
                if f_r > 0 {
                    let theta = good_turing_estimate(&fp, r).unwrap();
                    let f_r1 = fp.count_at(r + 1) as f64;
                    let expected = ((r + 1) as f64 / n) * (f_r1 / f_r as f64);
                    prop_assert!((theta - expected).abs() < 1e-14,
                        "GT at r={}: got {} expected {}", r, theta, expected);
                } else {
                    prop_assert!(good_turing_estimate(&fp, r).is_none(),
                        "GT at r={} should be None (F_r=0)", r);
                }
            }
        }

        #[test]
        fn coverage_chao_shen_ge_basic(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let c_cs = coverage_chao_shen(&fp);
            let c_gt = coverage_good_turing(&fp);
            prop_assert!(c_cs >= c_gt - 1e-12,
                "Chao-Shen {} < basic GT {}", c_cs, c_gt);
            prop_assert!(c_cs >= 0.0 - 1e-12);
            prop_assert!(c_cs <= 1.0 + 1e-12);
        }

        #[test]
        fn chao1_ci_contains_point(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let est = support_chao1_with_ci(&fp);
            prop_assert!(est.ci_lower <= est.point + 1e-12,
                "CI lower {} > point {}", est.ci_lower, est.point);
            prop_assert!(est.ci_upper >= est.point - 1e-12,
                "CI upper {} < point {}", est.ci_upper, est.point);
            prop_assert!(est.variance >= -1e-12,
                "variance {} < 0", est.variance);
            prop_assert!(est.ci_lower >= fp.observed_support() as f64 - 1e-12,
                "CI lower {} < S_obs {}", est.ci_lower, fp.observed_support());
        }

        #[test]
        fn ichao1_ge_chao1(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let s_ichao = support_ichao1(&fp);
            let s_chao = support_chao1(&fp);
            prop_assert!(s_ichao >= s_chao - 1e-12,
                "iChao1 {} < Chao1 {}", s_ichao, s_chao);
        }

        #[test]
        fn plugin_entropy_matches_logp_on_counts(counts in prop::collection::vec(1usize..50, 1..200)) {
            // Compare the fingerprint-based computation against building the explicit simplex
            // and delegating to `logp`.
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let h_fp = entropy_plugin_nats(&fp);
            let p = empirical_simplex_from_counts(&counts).unwrap();
            let h_logp = logp::entropy_unchecked(&p);
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
            let fp2 = Fingerprint::from_counts(counts.iter().copied()).unwrap();
            let h2 = entropy_jackknife_nats(&fp2);
            prop_assert!((h1 - h2).abs() < 1e-12);
        }

        #[test]
        fn pitman_yor_params_are_in_domain(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let params = pitman_yor_params_hat(&fp);

            let n = fp.sample_size() as f64;
            let t = fp.observed_support() as f64;
            let f1 = fp.singletons();

            prop_assert!((0.0..1.0).contains(&params.d));
            // DPYM + MPY domain bits we rely on downstream:
            prop_assert!((n + params.alpha).is_finite());
            prop_assert!((n + params.alpha) > 0.0);
            prop_assert!((params.alpha + t * params.d).is_finite());

            // Our policy uses the boundary point (0,0) to mean “no unseen correction”.
            if f1 == 0 {
                prop_assert!(params.d == 0.0);
                prop_assert!(params.alpha == 0.0);
                prop_assert!((params.alpha + t * params.d) >= 0.0);
            } else {
                // When we *do* apply a PYP correction, we stay in the strict domain.
                prop_assert!(params.alpha > -params.d);
                prop_assert!((params.alpha + t * params.d) > 0.0);
            }

            // The implied DPYM predictive distribution sums to 1.
            let denom = n + params.alpha;
            let mut sum = 0.0;
            for (r, &fr) in fp.f.iter().enumerate().skip(1) {
                if fr == 0 {
                    continue;
                }
                let q = ((r as f64) - params.d) / denom;
                prop_assert!(q.is_finite());
                prop_assert!(q > 0.0);
                sum += (fr as f64) * q;
            }
            let q_new = (params.alpha + t * params.d) / denom;
            prop_assert!(q_new.is_finite());
            prop_assert!(q_new >= 0.0);
            prop_assert!(q_new <= 1.0 + 1e-12);
            sum += q_new;
            prop_assert!((sum - 1.0).abs() < 1e-10);
        }

        #[test]
        fn pitman_yor_from_counts_matches_fingerprint(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let h_fp = entropy_pitman_yor_nats(&fp);
            let fp2 = Fingerprint::from_counts(counts.iter().copied()).unwrap();
            let h_counts = entropy_pitman_yor_nats(&fp2);
            prop_assert!((h_fp - h_counts).abs() < 1e-12);
        }

        #[test]
        fn pitman_yor_scale_invariance_via_no_singletons(counts in prop::collection::vec(1usize..50, 1..200)) {
            // Scaling all counts by a constant doesn't change the empirical distribution, and it
            // removes singletons when the factor >= 2. Our policy then reduces to plug-in.
            let fp = Fingerprint::from_counts(counts.clone()).unwrap();
            let h_plugin = entropy_plugin_nats(&fp);

            let scaled: Vec<usize> = counts.into_iter().map(|c| c.saturating_mul(2)).collect();
            let fp2 = Fingerprint::from_counts(scaled).unwrap();
            prop_assert_eq!(fp2.singletons(), 0);

            let h_py2 = entropy_pitman_yor_nats(&fp2);
            prop_assert!((h_py2 - h_plugin).abs() < 1e-12);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

        #[test]
        fn pitman_yor_entropy_is_finite_and_nonnegative(counts in prop::collection::vec(1usize..50, 1..200)) {
            let fp = Fingerprint::from_counts(counts).unwrap();
            let h = entropy_pitman_yor_nats(&fp);
            prop_assert!(h.is_finite());
            prop_assert!(h >= -1e-12);

            // to_bits is consistent with nats.
            let hb = to_bits(h);
            prop_assert!((hb - h / core::f64::consts::LN_2).abs() < 1e-12);
        }
    }

    #[test]
    fn pitman_yor_reduces_to_plugin_without_singletons() {
        let counts = [2usize, 2, 2, 2];
        let fp = Fingerprint::from_counts(counts).unwrap();
        assert_eq!(fp.singletons(), 0);
        let h_py = entropy_pitman_yor_nats(&fp);
        let h_plug = entropy_plugin_nats(&fp);
        assert!((h_py - h_plug).abs() < 1e-12);
    }

    #[test]
    fn pitman_yor_entropy_is_finite_and_nonnegative_for_all_singletons() {
        let counts = [1usize, 1, 1, 1, 1, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        assert_eq!(fp.sample_size(), 8);
        assert_eq!(fp.singletons(), 8);
        let params = pitman_yor_params_hat(&fp);
        assert!((0.0..1.0).contains(&params.d));
        assert!(params.alpha > -params.d);
        let h_py = entropy_pitman_yor_nats(&fp);
        assert!(h_py.is_finite());
        assert!(h_py >= -1e-12);
    }

    #[test]
    fn entropy_default_is_pitman_yor() {
        let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let h1 = entropy_default_nats(&fp);
        let h2 = entropy_pitman_yor_nats(&fp);
        assert!((h1 - h2).abs() < 1e-12);
    }

    fn mpy_entropy_bruteforce_nats(d: f64, alpha: f64, max_terms: usize) -> (f64, f64) {
        // Returns (entropy_prefix, remaining_mass).
        let one_minus_d = 1.0 - d;
        let mut p = (one_minus_d / (alpha + 1.0)).max(0.0);
        let mut mass = p;
        let mut h = if p > 0.0 { -p * p.ln() } else { 0.0 };
        for k in 1..=max_terms {
            let kf = k as f64;
            let num = alpha + kf * d;
            let den = num + 1.0;
            if !(den.is_finite() && den > 0.0) {
                break;
            }
            let p_next = p * (num / den);
            p = p_next;
            if !(p.is_finite() && p > 0.0) {
                break;
            }
            mass += p;
            h -= p * p.ln();
            if mass >= 1.0 - 1e-14 {
                mass = 1.0;
                break;
            }
        }
        (h, (1.0 - mass).max(0.0))
    }

    #[test]
    fn mpy_entropy_geometric_matches_direct_sum() {
        let alpha = 7.0;
        let d = 0.0;
        let h_closed = entropy_mpy_nats(d, alpha);

        // Direct sum of the geometric pmf.
        let p1 = 1.0 / (alpha + 1.0);
        let r = alpha / (alpha + 1.0);
        let mut p = p1;
        let mut mass = p;
        let mut h = -p * p.ln();
        for _k in 1..=2_000_000usize {
            if 1.0 - mass < 1e-14 {
                break;
            }
            p *= r;
            mass += p;
            h -= p * p.ln();
        }
        assert!((mass - 1.0).abs() < 1e-12);
        assert!((h - h_closed).abs() < 1e-10);
    }

    #[test]
    fn mpy_entropy_light_tail_matches_bruteforce_prefix() {
        // Pick a light tail (small d) so brute-force prefix captures essentially all mass.
        let d = 0.2;
        let alpha = 5.0;
        let h = entropy_mpy_nats(d, alpha);
        let (h_ref, rem) = mpy_entropy_bruteforce_nats(d, alpha, 1_000_000);
        assert!(rem < 1e-12);
        assert!((h - h_ref).abs() < 1e-6);
    }

    #[test]
    fn mpy_entropy_heavy_tail_is_finite() {
        // Heavy tail (d close to 1) should still yield finite Shannon entropy.
        let d = 0.95;
        let alpha = 0.25;
        let h = entropy_mpy_nats(d, alpha);
        assert!(h.is_finite());
        assert!(h >= 0.0);
    }

    // ---- Edge case tests ----

    #[test]
    fn single_symbol_distribution() {
        // A single symbol seen many times: zero entropy, no unseen mass, support = 1.
        let fp = Fingerprint::from_counts([100]).unwrap();
        assert_eq!(fp.sample_size(), 100);
        assert_eq!(fp.observed_support(), 1);
        assert_eq!(fp.singletons(), 0);
        assert_eq!(fp.doubletons(), 0);

        assert!((entropy_plugin_nats(&fp)).abs() < 1e-12);
        assert!((entropy_miller_madow_nats(&fp)).abs() < 1e-12);
        assert!((entropy_jackknife_nats(&fp)).abs() < 1e-12);
        assert!((entropy_pitman_yor_nats(&fp)).abs() < 1e-12);
        assert!((unseen_mass_good_turing(&fp)).abs() < 1e-12);
        assert!((support_chao1(&fp) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn from_counts_rejects_empty() {
        let result = Fingerprint::from_counts(Vec::<usize>::new());
        assert!(result.is_err());
    }

    #[test]
    fn from_counts_rejects_all_zeros() {
        let result = Fingerprint::from_counts(vec![0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn empirical_simplex_rejects_empty() {
        let result = empirical_simplex_from_counts(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn empirical_simplex_rejects_all_zeros() {
        let result = empirical_simplex_from_counts(&[0, 0]);
        assert!(result.is_err());
    }

    // ---- Cross-module: fingerprint -> estimators consistency ----

    #[test]
    fn all_estimators_consistent_on_uniform() {
        // Uniform distribution over 4 symbols, each seen 10 times.
        let counts = [10usize, 10, 10, 10];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let h_true = 4.0_f64.ln();

        // Plugin is exact for uniform.
        assert!((entropy_plugin_nats(&fp) - h_true).abs() < 1e-12);
        // MM adds a positive correction.
        assert!(entropy_miller_madow_nats(&fp) >= h_true - 1e-12);
        // All estimators should be close to the true value for large n.
        assert!((entropy_jackknife_nats(&fp) - h_true).abs() < 0.1);
        // No singletons => PY = plugin.
        assert!((entropy_pitman_yor_nats(&fp) - h_true).abs() < 1e-12);

        // Bits conversion consistency.
        let h_bits_nats = entropy_plugin_nats(&fp) / core::f64::consts::LN_2;
        assert!((to_bits(entropy_plugin_nats(&fp)) - h_bits_nats).abs() < 1e-12);
    }

    #[test]
    fn estimators_ordered_on_heavy_singletons() {
        // Many singletons: unseen regime. PY should correct upward.
        let counts = [10usize, 5, 3, 1, 1, 1, 1, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let h_plug = entropy_plugin_nats(&fp);
        let h_mm = entropy_miller_madow_nats(&fp);
        let h_py = entropy_pitman_yor_nats(&fp);

        assert!(h_mm >= h_plug - 1e-12, "MM >= plugin");
        // PY should be >= plugin in the unseen regime (it models unseen mass).
        assert!(h_py >= h_plug - 1e-12, "PY >= plugin");
    }

    #[test]
    fn codelen_equals_n_times_entropy() {
        let counts = [5usize, 3, 2, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let n = fp.sample_size() as f64;
        let h = entropy_plugin_nats(&fp);
        let cl = sample_codelen_plugin_nats(&fp);
        assert!((cl - n * h).abs() < 1e-12);

        let cl_bits = to_bits(cl);
        assert!((cl_bits - cl / core::f64::consts::LN_2).abs() < 1e-12);
    }

    // (Cross-crate logp tests removed -- they belong in the logp crate.)

    // ---- from_frequency_counts constructor ----

    #[test]
    fn from_frequency_counts_roundtrip() {
        let fp1 = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        let fp2 = Fingerprint::from_frequency_counts(fp1.as_slice()).unwrap();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn from_frequency_counts_rejects_nonzero_f0() {
        let err = Fingerprint::from_frequency_counts(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, EstimationError::Invalid(_)));
    }

    #[test]
    fn from_frequency_counts_rejects_all_zero() {
        let err = Fingerprint::from_frequency_counts(&[0, 0, 0]).unwrap_err();
        assert!(matches!(err, EstimationError::Invalid(_)));
    }

    #[test]
    fn from_frequency_counts_rejects_empty() {
        let err = Fingerprint::from_frequency_counts(&[]).unwrap_err();
        assert!(matches!(err, EstimationError::EmptySample));
    }

    // ---- coverage_good_turing ----

    #[test]
    fn coverage_complements_unseen_mass() {
        let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        let c = coverage_good_turing(&fp);
        let p0 = unseen_mass_good_turing(&fp);
        assert!((c + p0 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn coverage_no_singletons_is_one() {
        let fp = Fingerprint::from_counts([4, 4, 4]).unwrap();
        assert!((coverage_good_turing(&fp) - 1.0).abs() < 1e-15);
    }

    // ---- good_turing_estimate ----

    #[test]
    fn good_turing_max_count_is_zero() {
        let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        let r_max = fp.max_freq(); // = 5
        let theta = good_turing_estimate(&fp, r_max).unwrap();
        assert!(theta.abs() < 1e-15, "max-count GT should be 0, got {theta}");
    }

    #[test]
    fn good_turing_zero_fr_returns_none() {
        // counts [5, 3, 1, 1]: F_2 = 0, F_4 = 0.
        let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        assert!(good_turing_estimate(&fp, 2).is_none());
        assert!(good_turing_estimate(&fp, 4).is_none());
        assert!(good_turing_estimate(&fp, 0).is_none());
    }

    #[test]
    fn good_turing_known_values() {
        // counts [3, 3, 2, 1, 1]: n=10, F_1=2, F_2=1, F_3=2
        let fp = Fingerprint::from_counts([3, 3, 2, 1, 1]).unwrap();
        assert_eq!(fp.sample_size(), 10);
        // theta_hat(1) = 2/10 * F_2/F_1 = 2/10 * 1/2 = 0.1
        assert!((good_turing_estimate(&fp, 1).unwrap() - 0.1).abs() < 1e-12);
        // theta_hat(2) = 3/10 * F_3/F_2 = 3/10 * 2/1 = 0.6
        assert!((good_turing_estimate(&fp, 2).unwrap() - 0.6).abs() < 1e-12);
    }

    // ---- unseen_mass_minimal_bias ----

    #[test]
    fn minimal_bias_first_term_is_good_turing() {
        // For a sample with only singletons (F_1 = n, F_i = 0 for i > 1),
        // minimal-bias = F_1/C(n,1) = F_1/n = Good-Turing.
        let fp = Fingerprint::from_counts([1, 1, 1, 1, 1]).unwrap();
        let mb = unseen_mass_minimal_bias(&fp);
        let gt = unseen_mass_good_turing(&fp);
        assert!(
            (mb - gt).abs() < 1e-12,
            "minimal-bias {} != Good-Turing {} for all-singletons",
            mb,
            gt
        );
    }

    #[test]
    fn minimal_bias_no_singletons_is_zero() {
        // No singletons: all F_i with odd i are 0, so positive terms vanish.
        // But F_2 contributes a negative term: -F_2/C(n,2).
        // For counts [4, 4, 4]: F_4=3, n=12, mb = -3/C(12,4) = -3/495.
        // Clamped to 0.
        let fp = Fingerprint::from_counts([4, 4, 4]).unwrap();
        let mb = unseen_mass_minimal_bias(&fp);
        assert!(
            mb.abs() < 1e-12,
            "no-singletons minimal-bias should be ~0, got {}",
            mb
        );
    }

    #[test]
    fn minimal_bias_known_value() {
        // counts [3, 2, 1]: n=6, F_1=1, F_2=1, F_3=1
        // mb = F_1/C(6,1) - F_2/C(6,2) + F_3/C(6,3)
        //    = 1/6 - 1/15 + 1/20
        //    = 10/60 - 4/60 + 3/60 = 9/60 = 3/20 = 0.15
        let fp = Fingerprint::from_counts([3, 2, 1]).unwrap();
        assert_eq!(fp.sample_size(), 6);
        let mb = unseen_mass_minimal_bias(&fp);
        assert!((mb - 0.15).abs() < 1e-12, "expected 0.15, got {}", mb);
    }

    // ---- coverage_chao_shen ----

    #[test]
    fn coverage_chao_shen_no_singletons_is_one() {
        let fp = Fingerprint::from_counts([4, 4, 4]).unwrap();
        assert!((coverage_chao_shen(&fp) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn coverage_chao_shen_known_value() {
        // counts [5, 3, 2, 1, 1]: n=12, F_1=2, F_2=1
        let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
        let n = 12.0;
        let f1 = 2.0;
        let f2 = 1.0;
        let expected = 1.0 - (f1 / n) * ((n - 1.0) * f1 / ((n - 1.0) * f1 + 2.0 * f2));
        assert!((coverage_chao_shen(&fp) - expected).abs() < 1e-12);
    }

    // ---- support_chao1_with_ci ----

    #[test]
    fn chao1_ci_no_singletons() {
        let fp = Fingerprint::from_counts([4, 4, 4]).unwrap();
        let est = support_chao1_with_ci(&fp);
        let s_obs = fp.observed_support() as f64;
        assert!((est.point - s_obs).abs() < 1e-12);
        assert!(est.variance.abs() < 1e-12);
        assert!((est.ci_lower - s_obs).abs() < 1e-12);
        assert!((est.ci_upper - s_obs).abs() < 1e-12);
    }

    #[test]
    fn chao1_ci_f2_zero_fallback() {
        // All singletons: f1=5, f2=0, S_hat = 5 + 5*4/2 = 15
        let fp = Fingerprint::from_counts([1, 1, 1, 1, 1]).unwrap();
        let est = support_chao1_with_ci(&fp);
        assert!((est.point - 15.0).abs() < 1e-12);
        assert!(est.variance > 0.0);
        assert!(est.ci_lower <= est.point);
        assert!(est.ci_upper >= est.point);
        assert!(est.ci_lower >= 5.0 - 1e-12);
    }

    // ---- support_ichao1 ----

    #[test]
    fn ichao1_equals_chao1_when_f4_zero() {
        let fp = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        assert_eq!(fp.count_at(4), 0);
        assert!((support_ichao1(&fp) - support_chao1(&fp)).abs() < 1e-12);
    }

    #[test]
    fn ichao1_known_value() {
        // counts [10, 5, 3, 2, 1, 1, 1, 1]: F_1=4, F_2=1, F_3=1, F_4=0, F_5=1, F_10=1
        // f4=0, so iChao1 == Chao1
        let fp = Fingerprint::from_counts([10, 5, 3, 2, 1, 1, 1, 1]).unwrap();
        assert!((support_ichao1(&fp) - support_chao1(&fp)).abs() < 1e-12);

        // Build a case where f3 > 0 and f4 > 0:
        // counts [4, 4, 3, 3, 2, 2, 1, 1]: F_1=2, F_2=2, F_3=2, F_4=2
        let fp2 = Fingerprint::from_counts([4, 4, 3, 3, 2, 2, 1, 1]).unwrap();
        let f1 = 2.0_f64;
        let f2 = 2.0;
        let f3 = 2.0;
        let f4 = 2.0;
        let chao1 = support_chao1(&fp2);
        let correction = (f3 / (4.0 * f4)) * (f1 - f2 * f3 / (2.0 * f4)).max(0.0);
        let expected = chao1 + correction;
        assert!((support_ichao1(&fp2) - expected).abs() < 1e-12);
    }

    // ---- count_at ----

    #[test]
    fn count_at_matches_singletons_doubletons() {
        let fp = Fingerprint::from_counts([5, 3, 2, 1, 1]).unwrap();
        assert_eq!(fp.count_at(1), fp.singletons());
        assert_eq!(fp.count_at(2), fp.doubletons());
        assert_eq!(fp.count_at(0), 0);
        assert_eq!(fp.count_at(100), 0);
    }

    // ---- all-singletons edge case for all estimators ----

    #[test]
    fn all_singletons_all_estimators() {
        // Every symbol seen exactly once. Maximum uncertainty about unseen mass.
        let fp = Fingerprint::from_counts([1, 1, 1, 1, 1]).unwrap();
        assert_eq!(fp.singletons(), 5);
        assert_eq!(fp.doubletons(), 0);

        let h_plugin = entropy_plugin_nats(&fp);
        let h_mm = entropy_miller_madow_nats(&fp);
        let h_jk = entropy_jackknife_nats(&fp);
        let h_py = entropy_pitman_yor_nats(&fp);

        assert!(h_plugin >= 0.0);
        assert!(h_mm >= h_plugin); // MM adds positive correction
        assert!(h_jk.is_finite());
        assert!(h_py.is_finite() && h_py >= 0.0);

        // Good-Turing: coverage = 0 (all singletons -> f_1/n = 1 -> unseen mass = 1)
        assert!((unseen_mass_good_turing(&fp) - 1.0).abs() < 1e-12);
        assert!((coverage_good_turing(&fp)).abs() < 1e-12);

        // Chao1 with f_2=0 fallback: S_obs + f_1*(f_1-1)/2 = 5 + 5*4/2 = 15
        let s_hat = support_chao1(&fp);
        assert!((s_hat - 15.0).abs() < 1e-12);
    }

    // ---- f_1=0 edge case (no singletons) ----

    #[test]
    fn no_singletons_all_estimators() {
        // All symbols seen >= 2 times.
        let fp = Fingerprint::from_counts([4, 3, 2, 2]).unwrap();
        assert_eq!(fp.singletons(), 0);

        let h_plugin = entropy_plugin_nats(&fp);
        let h_mm = entropy_miller_madow_nats(&fp);
        let h_jk = entropy_jackknife_nats(&fp);
        let h_py = entropy_pitman_yor_nats(&fp);

        assert!(h_plugin >= 0.0);
        assert!(h_mm >= h_plugin);
        assert!(h_jk.is_finite() && h_jk >= 0.0);
        assert!(h_py.is_finite() && h_py >= 0.0);

        // Good-Turing: unseen mass = 0 (no singletons)
        assert!((unseen_mass_good_turing(&fp)).abs() < 1e-12);
        assert!((coverage_good_turing(&fp) - 1.0).abs() < 1e-12);

        // Chao1: no singletons -> S_hat = S_obs
        assert!((support_chao1(&fp) - fp.observed_support() as f64).abs() < 1e-12);
    }

    // ---- fingerprint sufficiency: identical fingerprints -> identical estimates ----

    #[test]
    fn fingerprint_sufficiency_invariant() {
        // Two different count vectors that produce the same fingerprint.
        // [5, 3, 1, 1] and [1, 5, 1, 3] have the same fingerprint: F[1]=2, F[3]=1, F[5]=1.
        let fp1 = Fingerprint::from_counts([5, 3, 1, 1]).unwrap();
        let fp2 = Fingerprint::from_counts([1, 5, 1, 3]).unwrap();
        assert_eq!(fp1, fp2);

        assert_eq!(entropy_plugin_nats(&fp1), entropy_plugin_nats(&fp2));
        assert_eq!(
            entropy_miller_madow_nats(&fp1),
            entropy_miller_madow_nats(&fp2)
        );
        assert_eq!(entropy_jackknife_nats(&fp1), entropy_jackknife_nats(&fp2));
        assert_eq!(entropy_pitman_yor_nats(&fp1), entropy_pitman_yor_nats(&fp2));
        assert_eq!(unseen_mass_good_turing(&fp1), unseen_mass_good_turing(&fp2));
        assert_eq!(support_chao1(&fp1), support_chao1(&fp2));
    }

    // ---- convergence test: entropy estimates should improve with more data ----

    #[test]
    fn entropy_convergence_on_zipf() {
        // Draw samples from Zipf(s=1.5) on support size S=20.
        // True entropy = sum_{i=1}^{S} p_i * ln(1/p_i).
        // Use a deterministic LCG to draw samples at N=100, N=1000, N=5000.
        let s = 20usize;
        let alpha = 1.5_f64;

        // Compute Zipf probabilities.
        let unnorm: Vec<f64> = (1..=s).map(|i| 1.0 / (i as f64).powf(alpha)).collect();
        let z: f64 = unnorm.iter().sum();
        let probs: Vec<f64> = unnorm.iter().map(|u| u / z).collect();
        let h_true: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        let mut errors = Vec::new();
        for &n in &[100usize, 1000, 5000] {
            let counts = deterministic_zipf_sample(s, &probs, n, 42 + n as u64);
            let fp = Fingerprint::from_counts(counts).unwrap();
            let h_py = entropy_pitman_yor_nats(&fp);
            errors.push((h_py - h_true).abs());
        }

        // Error at N=5000 should be smaller than at N=100.
        assert!(
            errors[2] < errors[0],
            "PY estimate should converge: err@100={:.4}, err@5000={:.4}",
            errors[0],
            errors[2]
        );
    }

    /// Draw `n` samples from a discrete distribution with given probabilities.
    /// Uses a deterministic LCG for reproducibility.
    fn deterministic_zipf_sample(s: usize, probs: &[f64], n: usize, seed: u64) -> Vec<usize> {
        let mut state: u64 = seed;
        let mut next_uniform = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        // Build CDF.
        let mut cdf = vec![0.0; s + 1];
        for i in 0..s {
            cdf[i + 1] = cdf[i] + probs[i];
        }

        let mut counts = vec![0usize; s];
        for _ in 0..n {
            let u = next_uniform();
            // Binary search for the bin.
            let mut lo = 0;
            let mut hi = s;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid + 1] <= u {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo < s {
                counts[lo] += 1;
            }
        }
        counts
    }

    // ---- Chao1 lower-bound property test ----

    #[test]
    fn chao1_is_lower_bound_on_average() {
        // Chao1 is a lower bound in expectation: E[Chao1] <= S_true.
        // Individual samples can exceed S (it is a biased estimator), but
        // the average over many samples should not exceed the true support.
        let s = 100usize;
        let alpha = 1.5_f64;

        let unnorm: Vec<f64> = (1..=s).map(|i| 1.0 / (i as f64).powf(alpha)).collect();
        let z: f64 = unnorm.iter().sum();
        let probs: Vec<f64> = unnorm.iter().map(|u| u / z).collect();

        let n = 500usize;
        let n_trials = 50;
        let mut sum_chao1 = 0.0;
        for trial in 0..n_trials {
            let seed = 0xDEAD_BEEF_u64
                .wrapping_add(trial as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let counts = deterministic_zipf_sample(s, &probs, n, seed);
            let fp = Fingerprint::from_counts(counts).unwrap();
            let s_hat = support_chao1(&fp);
            // Chao1 is always >= S_obs.
            assert!(s_hat >= fp.observed_support() as f64 - 1e-9);
            sum_chao1 += s_hat;
        }
        let mean_chao1 = sum_chao1 / n_trials as f64;
        // Mean Chao1 should be <= S_true (lower bound property).
        // Allow small slack for finite-sample variance.
        assert!(
            mean_chao1 <= s as f64 + 5.0,
            "mean Chao1 {mean_chao1:.1} >> true S={s}"
        );
    }

    // ---- from_frequency_counts trailing zero normalization ----

    #[test]
    fn from_frequency_counts_strips_trailing_zeros() {
        let fp1 = Fingerprint::from_frequency_counts(&[0, 2, 0, 1]).unwrap();
        let fp2 = Fingerprint::from_frequency_counts(&[0, 2, 0, 1, 0, 0]).unwrap();
        assert_eq!(fp1, fp2, "trailing zeros should be stripped for equality");
    }
}
