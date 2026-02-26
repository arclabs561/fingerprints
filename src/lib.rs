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

use core::cmp::Ordering;
use core::num::NonZeroUsize;
use thiserror::Error;

pub mod coverage;
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
/// Returns [`PropEstError::EmptySample`] if `counts` is empty, or
/// [`PropEstError::Invalid`] if all counts are zero.
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
        return Err(PropEstError::EmptySample);
    }
    let n: usize = counts.iter().sum();
    let n = NonZeroUsize::new(n).ok_or(PropEstError::Invalid("sum(counts) == 0"))?;
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
    pub f: Vec<usize>,
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
    /// Returns [`PropEstError::EmptySample`] if the iterator is empty, or
    /// [`PropEstError::Invalid`] if all counts are zero.
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

/// Plug-in (empirical) entropy estimator (nats) from per-symbol counts.
///
/// Convenience wrapper: builds the empirical simplex from `counts` and delegates to
/// `logp::entropy_unchecked`. Equivalent to [`entropy_plugin_nats`] applied to the
/// fingerprint derived from the same counts.
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
///
/// # Examples
///
/// ```
/// use fingerprints::entropy_plugin_nats_from_counts;
///
/// let h = entropy_plugin_nats_from_counts(&[3, 3]).unwrap();
/// assert!((h - 2.0_f64.ln()).abs() < 1e-12);
/// ```
pub fn entropy_plugin_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let p = empirical_simplex_from_counts(counts)?;
    Ok(logp::entropy_unchecked(&p))
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
/// # References
///
/// - Miller (1955), "Note on the bias of information estimates"
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

/// Jackknife (delete-1) entropy estimator (nats) from per-symbol counts.
///
/// Convenience wrapper: builds a [`Fingerprint`] from `counts` and delegates to
/// [`entropy_jackknife_nats`].
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
///
/// # Examples
///
/// ```
/// use fingerprints::entropy_jackknife_nats_from_counts;
///
/// let h = entropy_jackknife_nats_from_counts(&[5, 3, 2]).unwrap();
/// assert!(h >= 0.0);
/// ```
pub fn entropy_jackknife_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let fp = Fingerprint::from_counts(counts.iter().copied())?;
    Ok(entropy_jackknife_nats(&fp))
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

/// Pitman--Yor entropy estimator (nats) from per-symbol counts.
///
/// Convenience wrapper: builds a [`Fingerprint`] and delegates to
/// [`entropy_pitman_yor_nats`].
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
pub fn entropy_pitman_yor_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let fp = Fingerprint::from_counts(counts.iter().copied())?;
    Ok(entropy_pitman_yor_nats(&fp))
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

/// Opinionated default entropy estimator (nats) from per-symbol counts.
///
/// Convenience wrapper: builds a [`Fingerprint`] and delegates to
/// [`entropy_default_nats`].
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
pub fn entropy_default_nats_from_counts(counts: &[usize]) -> Result<f64> {
    entropy_pitman_yor_nats_from_counts(counts)
}

/// Plug-in entropy estimator in **bits** (\(\log_2\)).
///
/// Equivalent to `entropy_plugin_nats(fp) / ln(2)`.
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, entropy_plugin_bits};
///
/// let fp = Fingerprint::from_counts([4, 4]).unwrap();
/// assert!((entropy_plugin_bits(&fp) - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn entropy_plugin_bits(fp: &Fingerprint) -> f64 {
    entropy_plugin_nats(fp) / logp::LN_2
}

/// Miller--Madow bias-corrected entropy estimator in **bits**.
///
/// Equivalent to `entropy_miller_madow_nats(fp) / ln(2)`.
#[must_use]
pub fn entropy_miller_madow_bits(fp: &Fingerprint) -> f64 {
    entropy_miller_madow_nats(fp) / logp::LN_2
}

/// Jackknife (delete-1) entropy estimator in **bits**.
///
/// Equivalent to `entropy_jackknife_nats(fp) / ln(2)`.
#[must_use]
pub fn entropy_jackknife_bits(fp: &Fingerprint) -> f64 {
    entropy_jackknife_nats(fp) / logp::LN_2
}

/// Pitman--Yor entropy estimator in **bits**.
///
/// Equivalent to `entropy_pitman_yor_nats(fp) / ln(2)`.
#[must_use]
pub fn entropy_pitman_yor_bits(fp: &Fingerprint) -> f64 {
    entropy_pitman_yor_nats(fp) / logp::LN_2
}

/// Pitman--Yor entropy estimator (bits) from per-symbol counts.
///
/// Convenience wrapper: equivalent to `entropy_pitman_yor_nats_from_counts(counts)? / ln(2)`.
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
pub fn entropy_pitman_yor_bits_from_counts(counts: &[usize]) -> Result<f64> {
    Ok(entropy_pitman_yor_nats_from_counts(counts)? / logp::LN_2)
}

/// Opinionated default entropy estimator in **bits**.
///
/// Equivalent to `entropy_default_nats(fp) / ln(2)`.
#[must_use]
pub fn entropy_default_bits(fp: &Fingerprint) -> f64 {
    entropy_default_nats(fp) / logp::LN_2
}

/// Opinionated default entropy estimator (bits) from per-symbol counts.
///
/// Convenience wrapper: equivalent to `entropy_default_nats_from_counts(counts)? / ln(2)`.
///
/// # Errors
///
/// Returns [`PropEstError::EmptySample`] if `counts` is empty.
pub fn entropy_default_bits_from_counts(counts: &[usize]) -> Result<f64> {
    Ok(entropy_default_nats_from_counts(counts)? / logp::LN_2)
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

/// Plug-in sample code length in **bits**.
///
/// Equivalent to `sample_codelen_plugin_nats(fp) / ln(2)`.
#[must_use]
pub fn sample_codelen_plugin_bits(fp: &Fingerprint) -> f64 {
    sample_codelen_plugin_nats(fp) / logp::LN_2
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
            let h_counts = entropy_pitman_yor_nats_from_counts(&counts).unwrap();
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

            // Bits wrapper is consistent with nats.
            let hb = entropy_pitman_yor_bits(&fp);
            prop_assert!((hb - h / logp::LN_2).abs() < 1e-12);
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
        let h_bits_nats = entropy_plugin_nats(&fp) / logp::LN_2;
        assert!((entropy_plugin_bits(&fp) - h_bits_nats).abs() < 1e-12);
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

        let cl_bits = sample_codelen_plugin_bits(&fp);
        assert!((cl_bits - cl / logp::LN_2).abs() < 1e-12);
    }

    // ---- logp cross-crate: total_bregman_divergence normalization ----

    #[test]
    fn total_bregman_divergence_normalization_selfinfo() {
        // For any p, total_bregman_divergence(p, p, F) = 0 (since Bregman D_F(x,x) = 0).
        let gen = logp::SquaredL2;
        let p = [0.3, 0.2, 0.5];
        let mut grad = [0.0; 3];
        let d = logp::total_bregman_divergence(&gen, &p, &p, &mut grad).unwrap();
        assert!(d.abs() < 1e-12, "total Bregman self-divergence should be zero, got {d}");
    }

    #[test]
    fn total_bregman_divergence_nonneg() {
        let gen = logp::SquaredL2;
        let p = [0.4, 0.3, 0.3];
        let q = [0.5, 0.2, 0.3];
        let mut grad = [0.0; 3];
        let d = logp::total_bregman_divergence(&gen, &p, &q, &mut grad).unwrap();
        assert!(d >= -1e-12, "total Bregman divergence should be non-negative, got {d}");
    }

    #[test]
    fn total_bregman_divergence_le_bregman() {
        // Total Bregman <= Bregman (normalization divides by >= 1).
        let gen = logp::SquaredL2;
        let p = [1.0, 3.0];
        let q = [2.0, 5.0];
        let mut grad_tb = [0.0; 2];
        let mut grad_b = [0.0; 2];
        let tb = logp::total_bregman_divergence(&gen, &p, &q, &mut grad_tb).unwrap();
        let b = logp::bregman_divergence(&gen, &p, &q, &mut grad_b).unwrap();
        assert!(tb <= b + 1e-12, "total Bregman {tb} > Bregman {b}");
    }

    // ---- logp cross-crate: rho_alpha ----

    #[test]
    fn rho_alpha_self_is_one() {
        // rho_alpha(p, p, alpha) = sum p_i^alpha * p_i^{1-alpha} = sum p_i = 1.
        let p = [0.25, 0.25, 0.5];
        for alpha in [0.0, 0.5, 1.0, 2.0, -1.0] {
            let r = logp::rho_alpha(&p, &p, alpha, 1e-9).unwrap();
            assert!((r - 1.0).abs() < 1e-12, "rho_alpha(p,p,{alpha}) = {r}, expected 1.0");
        }
    }

    #[test]
    fn rho_alpha_bounded_by_one() {
        // For alpha in (0,1), rho_alpha(p, q) <= 1 by Holder's inequality.
        let p = [0.6, 0.4];
        let q = [0.3, 0.7];
        let r = logp::rho_alpha(&p, &q, 0.5, 1e-9).unwrap();
        assert!(r <= 1.0 + 1e-12, "rho_alpha should be <= 1 for alpha in (0,1)");
        assert!(r >= 0.0 - 1e-12, "rho_alpha should be non-negative");
    }

    // ---- logp cross-crate: hellinger triangle inequality ----

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

        #[test]
        fn hellinger_triangle_inequality(
            a1 in 0.01f64..10.0,
            a2 in 0.01f64..10.0,
            a3 in 0.01f64..10.0,
            b1 in 0.01f64..10.0,
            b2 in 0.01f64..10.0,
            b3 in 0.01f64..10.0,
            c1 in 0.01f64..10.0,
            c2 in 0.01f64..10.0,
            c3 in 0.01f64..10.0,
        ) {
            // Normalize to simplices.
            let sa = a1 + a2 + a3;
            let p = [a1/sa, a2/sa, a3/sa];
            let sb = b1 + b2 + b3;
            let q = [b1/sb, b2/sb, b3/sb];
            let sc = c1 + c2 + c3;
            let r = [c1/sc, c2/sc, c3/sc];

            let h_pq = logp::hellinger(&p, &q, 1e-9).unwrap();
            let h_qr = logp::hellinger(&q, &r, 1e-9).unwrap();
            let h_pr = logp::hellinger(&p, &r, 1e-9).unwrap();

            // Hellinger distance satisfies the triangle inequality.
            prop_assert!(h_pr <= h_pq + h_qr + 1e-9,
                "triangle ineq violated: H(p,r)={h_pr} > H(p,q)={h_pq} + H(q,r)={h_qr}");
        }
    }
}
