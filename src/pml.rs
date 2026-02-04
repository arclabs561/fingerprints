//! Profile-likelihood / PML-oriented utilities.
//!
//! This module intentionally starts with *small, exact* building blocks:
//! - profile log-likelihood for small observed supports (via a permanent computation)
//! - a “uniform family” profile-MLE baseline for support size
//!
//! Full PML (optimizing over arbitrary unlabeled distributions with unseen support) is an active
//! research area; this scaffolding is designed to support future solvers/heuristics cleanly.

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

/// Log-likelihood (natural log) of observing an **unlabeled** count profile under a uniform
/// distribution on `support_size` symbols.
///
/// We interpret `counts` as the multiplicities of the **observed** distinct symbols (all counts > 0),
/// and assume any remaining support symbols have 0 count.
///
/// This is not “full PML” (which optimizes over all distributions), but it is a meaningful baseline
/// family that is easy to optimize over support size.
pub fn uniform_profile_log_likelihood(counts: &[usize], support_size: usize) -> Result<f64> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    if counts.iter().any(|&c| c == 0) {
        return Err(PropEstError::Invalid("counts must be positive"));
    }

    let m = counts.len();
    if support_size < m {
        return Err(PropEstError::Invalid("support_size must be >= observed distinct count"));
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

/// Choose the `support_size` in `[m, s_max]` maximizing the uniform-family profile likelihood.
#[must_use]
pub fn best_uniform_support_size(counts: &[usize], s_max: usize) -> Result<(usize, f64)> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    let m = counts.len();
    if s_max < m {
        return Err(PropEstError::Invalid("s_max must be >= observed distinct count"));
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
    let m = xs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    if !m.is_finite() {
        return m;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// Exact log profile-likelihood for **small** observed support \(m \le 20\).
///
/// Computes:
/// \[
/// \log \left(\frac{n!}{\prod_i c_i!}\cdot\frac{1}{\prod_t \mathrm{mult}_t!}\cdot \mathrm{perm}(A)\right)
/// \]
/// where \(A_{ij} = p_j^{c_i}\), and `perm` is the permanent.
///
/// This captures the unlabeled assignment uncertainty (profile likelihood) exactly for small `m`.
pub fn profile_log_likelihood_small(counts: &[usize], probs: &[f64], tol: f64) -> Result<f64> {
    if counts.is_empty() {
        return Err(PropEstError::EmptySample);
    }
    if counts.len() != probs.len() {
        return Err(PropEstError::Invalid("counts and probs must have same length"));
    }
    if counts.len() > 20 {
        return Err(PropEstError::Invalid("observed support too large for exact profile likelihood"));
    }
    if counts.iter().any(|&c| c == 0) {
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
            log_a[i][j] = if p > 0.0 { c_f * p.ln() } else { f64::NEG_INFINITY };
        }
    }

    // Ryser formula in log domain for row sums; sum of subset terms in normal domain with sign.
    let mut terms: Vec<f64> = Vec::with_capacity((1usize << m) - 1);
    let full = 1usize << m;
    for mask in 1..full {
        let bits = mask.count_ones() as usize;
        let sign = if (m - bits) % 2 == 0 { 1.0 } else { -1.0 };

        let mut prod_ln = 0.0f64;
        for i in 0..m {
            // log(sum_{j in mask} exp(log_a[i][j]))
            let mut row_terms = Vec::new();
            row_terms.reserve(bits);
            for j in 0..m {
                if (mask >> j) & 1 == 1 {
                    row_terms.push(log_a[i][j]);
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

    if !(perm > 0.0) {
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

        fn heap(
            k: usize,
            idx: &mut [usize],
            probs: &[f64],
            counts: &[usize],
            sum: &mut f64,
        ) {
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
                if k % 2 == 0 {
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
}

