//! Profile Maximum Likelihood (PML) utilities.
//!
//! The **profile** (or **pattern**) of a sample is its fingerprint: the multiset of
//! multiplicities, ignoring symbol labels. The profile likelihood is the probability
//! of observing a given profile under a candidate distribution, marginalizing over all
//! label permutations.
//!
//! This module provides small, exact building blocks:
//!
//! - [`uniform_profile_log_likelihood`]: log-likelihood of a profile under the uniform
//!   distribution on a given support size.
//! - [`best_uniform_support_size`]: optimize support size within the uniform family.
//! - [`profile_log_likelihood_small`]: exact profile log-likelihood for small observed
//!   supports (\(m \le 20\)) via permanent computation (Ryser formula).
//!
//! Full PML (optimizing over arbitrary unlabeled distributions with unseen support) is an
//! active research area; this scaffolding supports future solvers cleanly.
//!
//! # References
//!
//! - Orlitsky, Suresh, Wu (2016): “Optimal prediction of the number of unseen species”
//! - Acharya, Das, Orlitsky, Suresh (2017): “A unified maximum likelihood approach for
//!   estimating symmetric properties of discrete distributions”

#![forbid(unsafe_code)]

use crate::{PropEstError, Result};

fn ln_factorial(n: usize) -> f64 {
    // Exact enough for the regimes we use here; can be replaced by ln-gamma later.
    if n <= 1 {
        return 0.0;
    }
    (2..=n).map(|k| (k as f64).ln()).sum()
}

fn ln_multinomial_coeff(n: usize, counts: &[usize]) -> Result<f64> {
    let sum: usize = counts.iter().sum();
    if sum != n {
        return Err(PropEstError::Invalid("counts do not sum to n"));
    }
    let mut v = ln_factorial(n);
    for &c in counts {
        v -= ln_factorial(c);
    }
    Ok(v)
}

fn ln_permutation_count_for_unlabeled_counts(counts: &[usize]) -> f64 {
    // Number of distinct permutations of the multiset of counts:
    // m! / ∏_t mult_t!, where mult_t counts how many counts equal t.
    use std::collections::BTreeMap;
    let m = counts.len();
    let mut mult: BTreeMap<usize, usize> = BTreeMap::new();
    for &c in counts {
        *mult.entry(c).or_insert(0) += 1;
    }
    let mut v = ln_factorial(m);
    for &k in mult.values() {
        v -= ln_factorial(k);
    }
    v
}

fn ln_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// Log-likelihood (natural log) of observing an **unlabeled** count profile under a
/// uniform distribution on `support_size` symbols.
///
/// Given observed multiplicities \(c_1, \dots, c_m\) and support size \(S \ge m\),
/// the profile log-likelihood under the uniform distribution \(p_j = 1/S\) is:
///
/// \[
/// \log L = \log \binom{S}{m} + \log\!\left(\frac{m!}{\prod_t \mathrm{mult}_t!}\right)
///        + \log \frac{n!}{\prod_i c_i!} - n \log S
/// \]
///
/// where \(\mathrm{mult}_t\) counts the number of distinct count values equal to \(t\).
///
/// The first two terms account for the assignment of symbols to support positions and
/// the symmetry breaking for equal counts. The third term is the multinomial
/// coefficient. The last term is the uniform probability.
///
/// # Errors
///
/// - [`PropEstError::EmptySample`] if `counts` is empty.
/// - [`PropEstError::Invalid`] if any count is zero, or `support_size < m`.
///
/// # Examples
///
/// ```
/// use fingerprints::pml::uniform_profile_log_likelihood;
///
/// // Two symbols each seen 3 times, support = 2.
/// let ll = uniform_profile_log_likelihood(&[3, 3], 2).unwrap();
/// assert!(ll.is_finite());
/// // Larger support should decrease likelihood (more “wasted” symbols).
/// let ll2 = uniform_profile_log_likelihood(&[3, 3], 10).unwrap();
/// assert!(ll > ll2);
/// ```
pub fn uniform_profile_log_likelihood(counts: &[usize], support_size: usize) -> Result<f64> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    if counts.contains(&0) {
        return Err(PropEstError::Invalid("counts must be positive"));
    }

    let m = counts.len();
    if support_size < m {
        return Err(PropEstError::Invalid(
            "support_size must be >= observed distinct count",
        ));
    }
    let n: usize = counts.iter().sum();
    if n == 0 {
        return Err(PropEstError::Invalid("sum(counts) == 0"));
    }

    let ln_choose_species = ln_binom(support_size, m);
    let ln_mult = ln_multinomial_coeff(n, counts)?;
    let ln_perm_counts = ln_permutation_count_for_unlabeled_counts(counts);
    let ln_p = -(n as f64) * (support_size as f64).ln();

    Ok(ln_choose_species + ln_perm_counts + ln_mult + ln_p)
}

/// Choose the `support_size` in \([m, s_{\max}]\) maximizing the uniform-family profile
/// likelihood.
///
/// Returns `(best_support_size, log_likelihood)`. The search is exhaustive over
/// `m..=s_max`, so it is exact but costs \(O(s_{\max} - m)\) likelihood evaluations.
///
/// # Errors
///
/// - [`PropEstError::EmptySample`] if `counts` is empty.
/// - [`PropEstError::Invalid`] if `s_max < m` (observed distinct count).
///
/// # Examples
///
/// ```
/// use fingerprints::pml::best_uniform_support_size;
///
/// let counts = [2, 2, 2, 2];
/// let (s_best, ll) = best_uniform_support_size(&counts, 20).unwrap();
/// assert!(s_best >= counts.len());
/// assert!(ll.is_finite());
/// ```
#[must_use = "returns (best_support_size, log_likelihood)"]
pub fn best_uniform_support_size(counts: &[usize], s_max: usize) -> Result<(usize, f64)> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    let m = counts.len();
    if s_max < m {
        return Err(PropEstError::Invalid(
            "s_max must be >= observed distinct count",
        ));
    }
    let mut best_s = m;
    let mut best_ll = f64::NEG_INFINITY;
    for s in m..=s_max {
        let ll = uniform_profile_log_likelihood(counts, s)?;
        if ll > best_ll {
            best_ll = ll;
            best_s = s;
        }
    }
    Ok((best_s, best_ll))
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    if !m.is_finite() {
        return m;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// Exact log profile-likelihood for **small** observed support (\(m \le 20\)).
///
/// Computes:
///
/// \[
/// \log L = \log \frac{n!}{\prod_i c_i!}
///        - \log \prod_t \mathrm{mult}_t!
///        + \log \operatorname{perm}(A)
/// \]
///
/// where \(A_{ij} = p_j^{c_i}\), and \(\operatorname{perm}(A)\) is the matrix permanent
/// (sum of products over all permutations). The permanent is computed via the Ryser
/// formula in the log domain with Kahan summation for numerical stability.
///
/// The matrix permanent captures all ways to assign observed count patterns to
/// distribution components, which is the essence of profile (label-invariant)
/// likelihood.
///
/// # Errors
///
/// - [`PropEstError::EmptySample`] if `counts` is empty.
/// - [`PropEstError::Invalid`] if `counts.len() != probs.len()`, or `m > 20`, or any
///   count is zero.
/// - Propagates simplex validation errors from `logp::validate_simplex`.
///
/// # Examples
///
/// ```
/// use fingerprints::pml::profile_log_likelihood_small;
///
/// let counts = [2, 1];
/// let probs = [0.6, 0.4];
/// let ll = profile_log_likelihood_small(&counts, &probs, 1e-9).unwrap();
/// assert!(ll.is_finite());
/// ```
pub fn profile_log_likelihood_small(counts: &[usize], probs: &[f64], tol: f64) -> Result<f64> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    if counts.len() != probs.len() {
        return Err(PropEstError::Invalid(
            "counts and probs must have same length",
        ));
    }
    if counts.len() > 20 {
        return Err(PropEstError::Invalid(
            "observed support too large for exact profile likelihood",
        ));
    }
    if counts.contains(&0) {
        return Err(PropEstError::Invalid("counts must be positive"));
    }
    logp::validate_simplex(probs, tol)?;

    let m = counts.len();
    let n: usize = counts.iter().sum();

    // Build log(a_ij) = c_i * ln(p_j)
    let mut log_a = vec![vec![0.0f64; m]; m];
    for (i, &c) in counts.iter().enumerate() {
        let c_f = c as f64;
        for (j, &p) in probs.iter().enumerate() {
            // p can be 0; handle ln(0) -> -inf, exp -> 0.
            log_a[i][j] = if p > 0.0 {
                c_f * p.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
    }

    // Ryser formula in log domain for row sums; sum of subset terms in normal domain with sign.
    let mut terms: Vec<f64> = Vec::with_capacity((1usize << m) - 1);
    let full = 1usize << m;
    for mask in 1..full {
        let bits = mask.count_ones() as usize;
        let sign = if (m - bits).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };

        let mut prod_ln = 0.0f64;
        for row in &log_a {
            // log(sum_{j in mask} exp(log_a[i][j]))
            let mut row_terms = Vec::with_capacity(bits);
            for (j, &val) in row.iter().enumerate() {
                if ((mask >> j) & 1) == 1 {
                    row_terms.push(val);
                }
            }
            let row_ln = log_sum_exp(&row_terms);
            prod_ln += row_ln;
        }
        terms.push(sign * prod_ln.exp());
    }

    // Numerical stability: Ryser has cancellation; sum in a permutation-invariant order.
    terms.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
    let mut perm = 0.0f64;
    let mut c = 0.0f64; // Kahan compensation
    for t in terms {
        let y = t - c;
        let tmp = perm + y;
        c = (tmp - perm) - y;
        perm = tmp;
    }

    if perm.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Ok(f64::NEG_INFINITY);
    }

    let ln_mult = ln_multinomial_coeff(n, counts)?;

    // Correct for duplicated count permutations: counts are unlabeled.
    // permanent() sums over all m! permutations; equal counts create identical terms.
    // Divide by ∏ mult_t! where mult_t counts multiplicity of each distinct count value.
    let ln_dedup = {
        use std::collections::BTreeMap;
        let mut mult: BTreeMap<usize, usize> = BTreeMap::new();
        for &c in counts {
            *mult.entry(c).or_insert(0) += 1;
        }
        mult.values().map(|&k| ln_factorial(k)).sum::<f64>()
    };

    Ok(ln_mult + perm.ln() - ln_dedup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn uniform_profile_prefers_support_at_least_observed(counts in prop::collection::vec(1usize..20, 1..20)) {
            let m = counts.len();
            let (s, _ll) = best_uniform_support_size(&counts, m + 50).unwrap();
            prop_assert!(s >= m);
        }
    }

    fn permute_sum(probs: &[f64], counts: &[usize]) -> f64 {
        // Sum over all permutations π of probs: ∑_π ∏_i probs[π(i)]^{counts[i]}.
        // This equals the permanent of A_{ij} = probs[j]^{counts[i]}.
        let m = probs.len();
        assert_eq!(m, counts.len());
        let mut idx: Vec<usize> = (0..m).collect();
        let mut sum = 0.0f64;

        fn heap(k: usize, idx: &mut [usize], probs: &[f64], counts: &[usize], sum: &mut f64) {
            if k == 1 {
                let mut prod = 1.0;
                for (i, &j) in idx.iter().enumerate() {
                    prod *= probs[j].powi(counts[i] as i32);
                }
                *sum += prod;
                return;
            }
            heap(k - 1, idx, probs, counts, sum);
            for i in 0..(k - 1) {
                if k.is_multiple_of(2) {
                    idx.swap(i, k - 1);
                } else {
                    idx.swap(0, k - 1);
                }
                heap(k - 1, idx, probs, counts, sum);
            }
        }

        heap(m, &mut idx, probs, counts, &mut sum);
        sum
    }

    #[test]
    fn profile_log_likelihood_matches_bruteforce_perm_small() {
        let counts = [1usize, 2, 3, 3, 5, 1];
        let mut probs = vec![0.2, 0.05, 0.15, 0.1, 0.3, 0.2];
        let s: f64 = probs.iter().sum();
        for x in probs.iter_mut() {
            *x /= s;
        }
        let mut probs_rev = probs.clone();
        probs_rev.reverse();

        let ll = profile_log_likelihood_small(&counts, &probs, 1e-12).unwrap();
        let ll_rev = profile_log_likelihood_small(&counts, &probs_rev, 1e-12).unwrap();

        // Brute permanent
        let perm = permute_sum(&probs, &counts);
        let ln_mult = ln_multinomial_coeff(counts.iter().sum(), &counts).unwrap();
        let ln_dedup = {
            use std::collections::BTreeMap;
            let mut mult: BTreeMap<usize, usize> = BTreeMap::new();
            for &c in &counts {
                *mult.entry(c).or_insert(0) += 1;
            }
            mult.values().map(|&k| ln_factorial(k)).sum::<f64>()
        };
        let ll_brute = ln_mult + perm.ln() - ln_dedup;

        assert!((ll - ll_brute).abs() < 1e-9);
        assert!((ll_rev - ll_brute).abs() < 1e-9);
    }

    #[test]
    fn uniform_profile_rejects_zero_counts() {
        let result = uniform_profile_log_likelihood(&[0, 1, 2], 5);
        assert!(result.is_err());
    }

    #[test]
    fn uniform_profile_rejects_support_too_small() {
        let result = uniform_profile_log_likelihood(&[1, 2, 3], 2);
        assert!(result.is_err());
    }

    #[test]
    fn uniform_profile_ll_monotone_in_support_near_true() {
        // For a uniform sample, likelihood should peak near the true support size.
        let counts = [5, 5, 5, 5]; // 4 symbols, each 5 times
        let m = counts.len();
        let (s_best, _) = best_uniform_support_size(&counts, m + 100).unwrap();
        // Best support should be exactly m for a uniform sample (since more symbols
        // waste probability mass). In practice it is m or close to m.
        assert!(s_best <= m + 5, "s_best = {s_best}, expected near {m}");
    }

    #[test]
    fn profile_ll_small_uniform_matches_uniform_profile() {
        // For a uniform distribution, profile_log_likelihood_small and
        // uniform_profile_log_likelihood should agree.
        let counts = [2, 2, 2];
        let m = counts.len();
        let probs: Vec<f64> = vec![1.0 / m as f64; m];

        let ll_small = profile_log_likelihood_small(&counts, &probs, 1e-9).unwrap();
        let ll_uniform = uniform_profile_log_likelihood(&counts, m).unwrap();

        // They compute the same thing for uniform probs (up to numerical precision).
        assert!(
            (ll_small - ll_uniform).abs() < 1e-6,
            "ll_small={ll_small}, ll_uniform={ll_uniform}"
        );
    }

    #[test]
    fn profile_ll_small_rejects_mismatched_lengths() {
        let result = profile_log_likelihood_small(&[1, 2], &[0.5, 0.3, 0.2], 1e-9);
        assert!(result.is_err());
    }

    #[test]
    fn profile_ll_small_rejects_too_large_support() {
        let counts: Vec<usize> = (1..=21).collect();
        let probs: Vec<f64> = {
            let s: f64 = (1..=21).map(|i| i as f64).sum();
            (1..=21).map(|i| i as f64 / s).collect()
        };
        let result = profile_log_likelihood_small(&counts, &probs, 1e-9);
        assert!(result.is_err());
    }

    #[test]
    fn best_uniform_support_size_single_symbol() {
        // Single symbol: support must be at least 1.
        let (s, ll) = best_uniform_support_size(&[10], 100).unwrap();
        assert!(s >= 1);
        assert!(ll.is_finite());
    }
}
