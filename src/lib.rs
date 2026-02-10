//! `fingerprints`: property estimation from samples (fingerprints/profiles).
//!
//! This crate is for the “unseen regime”: estimating properties of an unknown discrete distribution
//! from a sample when the support may be large and the empirical plug-in estimator is biased.
//!
//! Design intent:
//! - Keep `logp` as the *definition* layer (entropy/divergence functionals on known distributions).
//! - Put estimation policy here (bias correction, sample-size regimes, solver-backed methods).
//! - Use `infogeom` when you have explicit simplex vectors and want geometry-aware distances.
//!
//! References (orientation):
//! - Valiant & Valiant (2013/2017): “Estimating the Unseen…”
//! - Orlitsky line: profile / PML estimators (future).
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
/// Returns a probability vector `p` with `p.len() == counts.len()` and `Σ p = 1`.
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

/// Pitman–Yor entropy estimator (nats) from a fingerprint.
///
/// This implements the estimator described in:
/// - Takato Hashino & Koji Tsukuda (2026), “Estimating the Shannon Entropy Using the Pitman–Yor Process”.
///
/// High-level: approximate the unknown population distribution by the DPYM predictive distribution `q`,
/// including an explicit “unseen mass” bucket, and return `H(q)`.
#[must_use]
pub fn entropy_pitman_yor_nats(fp: &Fingerprint) -> f64 {
    let params = pitman_yor_params_hat(fp);
    entropy_dpym_nats(fp, params.d, params.alpha)
}

/// Pitman–Yor entropy estimator (nats) from per-symbol counts.
pub fn entropy_pitman_yor_nats_from_counts(counts: &[usize]) -> Result<f64> {
    let fp = Fingerprint::from_counts(counts.iter().copied())?;
    Ok(entropy_pitman_yor_nats(&fp))
}

/// Opinionated default entropy estimator (nats).
///
/// This is a single-call “good default” for the **unseen regime**:
/// - Uses the Pitman–Yor / DPYM estimator when there are singletons.
/// - Reduces to the plug-in estimator when there are no singletons.
#[must_use]
pub fn entropy_default_nats(fp: &Fingerprint) -> f64 {
    entropy_pitman_yor_nats(fp)
}

/// Opinionated default entropy estimator (nats) from per-symbol counts.
pub fn entropy_default_nats_from_counts(counts: &[usize]) -> Result<f64> {
    entropy_pitman_yor_nats_from_counts(counts)
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

/// Pitman–Yor entropy estimator (bits).
#[must_use]
pub fn entropy_pitman_yor_bits(fp: &Fingerprint) -> f64 {
    entropy_pitman_yor_nats(fp) / logp::LN_2
}

/// Pitman–Yor entropy estimator (bits) from per-symbol counts.
pub fn entropy_pitman_yor_bits_from_counts(counts: &[usize]) -> Result<f64> {
    Ok(entropy_pitman_yor_nats_from_counts(counts)? / logp::LN_2)
}

/// Opinionated default entropy estimator (bits).
#[must_use]
pub fn entropy_default_bits(fp: &Fingerprint) -> f64 {
    entropy_default_nats(fp) / logp::LN_2
}

/// Opinionated default entropy estimator (bits) from per-symbol counts.
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

/// Plug-in sample code length in bits.
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
}
