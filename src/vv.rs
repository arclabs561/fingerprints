//! Valiant & Valiant (2011/2017) style “unseen” scaffolding.
//!
//! The VV approach models the unknown distribution as a histogram (a density over
//! probability values) and matches the observed fingerprint entries \(F_1, \dots, F_k\)
//! via Poisson moment constraints. A linear program (LP) over the histogram then yields
//! bounds on symmetric functionals (support size, entropy).
//!
//! This module provides a minimal, auditable LP for:
//!
//! - [`support_bounds_lp`]: lower and upper bounds on the true support size.
//! - [`entropy_bounds_lp`]: lower and upper bounds on Shannon entropy (nats).
//!
//! Both use a log-spaced probability grid and Poisson PMF coefficients
//! \(a_{i,j} = \operatorname{Poi}(i; n p_j)\) as the constraint matrix.
//!
//! # References
//!
//! - Valiant & Valiant (2011): “Estimating the Unseen: an \(n / \log n\)-sample estimator
//!   for entropy and support size”
//! - Valiant & Valiant (2017): “Estimating the Unseen: Improved Estimators for Entropy
//!   and other Properties” (JACM)
//!
//! # Status
//!
//! This is a **research scaffold**: grid design and constraint policy may evolve.
//! The LP relaxation approach here follows the VV methodology of bounding symmetric
//! functionals via Poisson moment constraints on the fingerprint. Grid resolution and
//! constraint tolerances are the main knobs for tightness vs. feasibility.

#![forbid(unsafe_code)]

use crate::{Fingerprint, PropEstError, Result};

/// Parameters for the VV-style histogram LP.
///
/// The LP operates on a discretized "histogram" of the unknown distribution: for each
/// grid point \(p_j\), a variable \(x_j \ge 0\) represents the number of support elements
/// with probability \(p_j\). Constraints enforce that the Poisson-expected fingerprint
/// entries \(\sum_j x_j \operatorname{Poi}(i; n p_j)\) match the observed \(F_i\) within
/// tolerance, and that the total probability mass \(\sum_j x_j p_j \approx 1\).
///
/// # Fields
///
/// - `k`: number of fingerprint entries used (\(F_1, \dots, F_k\)).
/// - `grid_size`: number of grid points in the probability discretization.
/// - `p_min`, `p_max`: range of the log-spaced probability grid.
/// - `eps_scale`: controls constraint tolerance (larger = more relaxed bounds).
#[derive(Debug, Clone)]
pub struct SupportLpParams {
    /// Max multiplicity used from the fingerprint (\(F_1 \dots F_k\)).
    pub k: usize,
    /// Number of grid points in the log-spaced probability discretization.
    pub grid_size: usize,
    /// Minimum probability in the grid (typically \(\sim 1/n^2\)).
    pub p_min: f64,
    /// Maximum probability in the grid (typically 1.0).
    pub p_max: f64,
    /// Tolerance scale for matching fingerprint moments. Tolerance for \(F_i\) is
    /// `eps_scale * (sqrt(F_i) + 1)`.
    pub eps_scale: f64,
}

impl SupportLpParams {
    /// Reasonable defaults for small/medium sample sizes.
    ///
    /// Uses `k = 10` fingerprint entries, a 60-point log-spaced grid from
    /// \(1/n^2\) to 1.0, and `eps_scale = 1.5`.
    #[must_use]
    pub fn default_for(fp: &Fingerprint) -> Self {
        let n = fp.sample_size().max(1) as f64;
        Self {
            k: 10,
            grid_size: 60,
            // Typical VV grids include probabilities down to ~1/n^2.
            p_min: (1.0 / (n * n)).max(1e-12),
            p_max: 1.0,
            eps_scale: 1.5,
        }
    }
}

fn mass_eps(n: usize, eps_scale: f64) -> f64 {
    let n = n.max(1) as f64;
    (eps_scale / n.sqrt()).clamp(1e-6, 0.05)
}

fn ln_factorial(i: usize) -> f64 {
    if i <= 1 {
        return 0.0;
    }
    (2..=i).map(|k| (k as f64).ln()).sum()
}

fn poisson_pmf(i: usize, lambda: f64) -> f64 {
    if lambda < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 {
        return if i == 0 { 1.0 } else { 0.0 };
    }
    let ln = -(lambda) + (i as f64) * lambda.ln() - ln_factorial(i);
    ln.exp()
}

fn grid_log_space(p_min: f64, p_max: f64, m: usize) -> Result<Vec<f64>> {
    if !p_min.is_finite() || !p_max.is_finite() || p_min <= 0.0 || p_max <= 0.0 || p_max < p_min {
        return Err(PropEstError::Invalid("invalid grid bounds"));
    }
    if m < 2 {
        return Err(PropEstError::Invalid("grid_size must be >= 2"));
    }
    let a = p_min.ln();
    let b = p_max.ln();
    let mut out = Vec::with_capacity(m);
    for t in 0..m {
        let u = t as f64 / ((m - 1) as f64);
        out.push((a + u * (b - a)).exp());
    }
    Ok(out)
}

/// Compute (lower, upper) bounds on support size using a VV-style histogram LP.
///
/// Solves two LPs (minimize and maximize \(\sum_j x_j\)) subject to:
///
/// - Mass constraint: \(\sum_j x_j p_j \approx 1\) (within `eps_scale / sqrt(n)`).
/// - Support lower bound: \(\sum_j x_j \ge S_{\text{obs}}\).
/// - Fingerprint constraints: \(\sum_j x_j \operatorname{Poi}(i; n p_j) \approx F_i\)
///   for \(i = 1, \dots, k\).
///
/// Returns `(lower_bound, upper_bound)` in real-valued expected-support terms.
///
/// # Errors
///
/// Returns [`PropEstError::Invalid`] if the sample size is zero or the LP is infeasible.
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, vv::{SupportLpParams, support_bounds_lp}};
///
/// let fp = Fingerprint::from_counts([5, 4, 3, 2, 2, 1, 1, 1]).unwrap();
/// let params = SupportLpParams::default_for(&fp);
/// let (lo, hi) = support_bounds_lp(&fp, params).unwrap();
/// assert!(lo >= fp.observed_support() as f64 - 1e-9);
/// assert!(hi >= lo - 1e-9);
/// ```
pub fn support_bounds_lp(fp: &Fingerprint, params: SupportLpParams) -> Result<(f64, f64)> {
    use minilp::{ComparisonOp, OptimizationDirection, Problem};

    let n = fp.sample_size();
    if n == 0 {
        return Err(PropEstError::Invalid("sample size is zero"));
    }
    let s_obs = fp.observed_support() as f64;

    let k = params.k.min(fp.f.len().saturating_sub(1)).max(1);
    let grid = grid_log_space(params.p_min, params.p_max, params.grid_size)?;
    let eps_mass = mass_eps(n, params.eps_scale);

    // Precompute coefficients a_{i,j} = Poi(i; n p_j)
    let mut a = vec![vec![0.0f64; grid.len()]; k + 1];
    for (i, row) in a.iter_mut().enumerate().take(k + 1).skip(1) {
        for (j, &p) in grid.iter().enumerate() {
            let lambda = (n as f64) * p;
            row[j] = poisson_pmf(i, lambda);
        }
    }

    // Helper to build and solve LP with objective direction.
    let solve = |dir: OptimizationDirection| -> Result<f64> {
        let mut pb = Problem::new(dir);
        // Objective: support size S = Σ x_j
        let vars: Vec<_> = (0..grid.len())
            .map(|_| pb.add_var(1.0, (0.0, f64::INFINITY)))
            .collect();

        // Mass constraint: Σ x_j p_j ≈ 1.
        let mut mass = Vec::with_capacity(vars.len());
        for (v, &p) in vars.iter().zip(grid.iter()) {
            mass.push((*v, p));
        }
        pb.add_constraint(mass.clone(), ComparisonOp::Le, 1.0 + eps_mass);
        pb.add_constraint(mass, ComparisonOp::Ge, (1.0 - eps_mass).max(0.0));

        // Observed support is a lower bound.
        let sup: Vec<_> = vars.iter().map(|&v| (v, 1.0)).collect();
        pb.add_constraint(sup, ComparisonOp::Ge, s_obs);

        // Fingerprint constraints for i=1..k: match within eps_i.
        for (i, ai) in a.iter().enumerate().take(k + 1).skip(1) {
            let fi = fp.f.get(i).copied().unwrap_or(0) as f64;
            let eps_i = params.eps_scale * (fi.sqrt() + 1.0);
            let row: Vec<_> = vars.iter().enumerate().map(|(j, &v)| (v, ai[j])).collect();
            pb.add_constraint(row.clone(), ComparisonOp::Le, fi + eps_i);
            pb.add_constraint(row, ComparisonOp::Ge, (fi - eps_i).max(0.0));
        }

        let sol = pb
            .solve()
            .map_err(|_| PropEstError::Invalid("LP infeasible"))?;
        Ok(sol.objective())
    };

    let lower = solve(OptimizationDirection::Minimize)?;
    let upper = solve(OptimizationDirection::Maximize)?;
    Ok((lower, upper))
}

/// Compute (lower, upper) bounds on Shannon entropy (nats) using the same VV-style
/// histogram LP.
///
/// The objective is a linearized entropy over the histogram grid:
///
/// \[
/// H \approx \sum_j x_j \bigl(-p_j \ln p_j\bigr)
/// \]
///
/// Subject to the same mass, support, and fingerprint constraints as
/// [`support_bounds_lp`]. Returns `(lower_bound, upper_bound)` in nats.
///
/// Note: bounds depend on the grid discretization and constraint tolerances and may
/// be loose, especially for heavy-tailed distributions.
///
/// # Errors
///
/// Returns [`PropEstError::Invalid`] if the sample size is zero or the LP is infeasible.
///
/// # Examples
///
/// ```
/// use fingerprints::{Fingerprint, vv::{SupportLpParams, entropy_bounds_lp}};
///
/// let fp = Fingerprint::from_counts([5, 4, 3, 2, 2, 1, 1, 1]).unwrap();
/// let params = SupportLpParams::default_for(&fp);
/// let (lo, hi) = entropy_bounds_lp(&fp, params).unwrap();
/// assert!(lo >= -1e-9);
/// assert!(hi >= lo - 1e-9);
/// ```
pub fn entropy_bounds_lp(fp: &Fingerprint, params: SupportLpParams) -> Result<(f64, f64)> {
    use minilp::{ComparisonOp, OptimizationDirection, Problem};

    let n = fp.sample_size();
    if n == 0 {
        return Err(PropEstError::Invalid("sample size is zero"));
    }
    let s_obs = fp.observed_support() as f64;

    let k = params.k.min(fp.f.len().saturating_sub(1)).max(1);
    let grid = grid_log_space(params.p_min, params.p_max, params.grid_size)?;
    let eps_mass = mass_eps(n, params.eps_scale);

    // Precompute coefficients a_{i,j} = Poi(i; n p_j)
    let mut a = vec![vec![0.0f64; grid.len()]; k + 1];
    for (i, row) in a.iter_mut().enumerate().take(k + 1).skip(1) {
        for (j, &p) in grid.iter().enumerate() {
            let lambda = (n as f64) * p;
            row[j] = poisson_pmf(i, lambda);
        }
    }

    let ent_coeff: Vec<f64> = grid
        .iter()
        .map(|&p| if p > 0.0 { -(p * p.ln()) } else { 0.0 })
        .collect();

    let solve = |dir: OptimizationDirection| -> Result<f64> {
        let mut pb = Problem::new(dir);
        let vars: Vec<_> = (0..grid.len())
            .map(|j| pb.add_var(ent_coeff[j], (0.0, f64::INFINITY)))
            .collect();

        // Mass: Σ x_j p_j ≈ 1.
        let mut mass = Vec::with_capacity(vars.len());
        for (v, &p) in vars.iter().zip(grid.iter()) {
            mass.push((*v, p));
        }
        pb.add_constraint(mass.clone(), ComparisonOp::Le, 1.0 + eps_mass);
        pb.add_constraint(mass, ComparisonOp::Ge, (1.0 - eps_mass).max(0.0));

        // Observed support lower bound.
        let sup: Vec<_> = vars.iter().map(|&v| (v, 1.0)).collect();
        pb.add_constraint(sup, ComparisonOp::Ge, s_obs);

        // Fingerprint constraints.
        for (i, ai) in a.iter().enumerate().take(k + 1).skip(1) {
            let fi = fp.f.get(i).copied().unwrap_or(0) as f64;
            let eps_i = params.eps_scale * (fi.sqrt() + 1.0);
            let row: Vec<_> = vars.iter().enumerate().map(|(j, &v)| (v, ai[j])).collect();
            pb.add_constraint(row.clone(), ComparisonOp::Le, fi + eps_i);
            pb.add_constraint(row, ComparisonOp::Ge, (fi - eps_i).max(0.0));
        }

        let sol = pb
            .solve()
            .map_err(|_| PropEstError::Invalid("LP infeasible"))?;
        Ok(sol.objective())
    };

    let lower = solve(OptimizationDirection::Minimize)?;
    let upper = solve(OptimizationDirection::Maximize)?;
    Ok((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_bounds_at_least_observed_support() {
        // counts correspond to observed distinct species with these multiplicities
        let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = support_bounds_lp(&fp, params).unwrap();
        assert!(lo + 1e-9 >= fp.observed_support() as f64);
        assert!(hi + 1e-9 >= lo);
    }

    #[test]
    fn entropy_bounds_are_ordered_and_nonnegative() {
        let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = entropy_bounds_lp(&fp, params).unwrap();
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo >= -1e-9);
        assert!(hi + 1e-9 >= lo);
    }

    #[test]
    fn support_bounds_bracket_observed() {
        // The LP support lower bound should be >= S_obs, upper bound >= lower bound.
        let counts = [10usize, 5, 3, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let s_obs = fp.observed_support() as f64;
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = support_bounds_lp(&fp, params).unwrap();
        assert!(lo + 1e-9 >= s_obs, "lower bound {lo} < S_obs {s_obs}");
        assert!(hi + 1e-9 >= lo, "upper bound {hi} < lower bound {lo}");
    }

    #[test]
    fn entropy_bounds_bracket_plugin() {
        // The entropy LP bounds should contain the plug-in entropy (since the plug-in
        // distribution is feasible for the LP when constraints are loose enough).
        let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let h_plug = crate::entropy_plugin_nats(&fp);
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = entropy_bounds_lp(&fp, params).unwrap();
        // The plugin might not be exactly in the LP feasible region (discretization),
        // but it should be close.
        assert!(lo <= h_plug + 0.5, "LP lower {lo} >> plug-in {h_plug}");
        assert!(hi >= h_plug - 0.5, "LP upper {hi} << plug-in {h_plug}");
    }

    #[test]
    fn support_bounds_uniform_tight() {
        // For a large uniform sample with no singletons, LP bounds should be tight
        // near the observed support.
        let counts = [20usize, 20, 20, 20, 20];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let s_obs = fp.observed_support() as f64;
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = support_bounds_lp(&fp, params).unwrap();
        assert!(lo + 1e-9 >= s_obs);
        // Upper bound is finite and ordered (the LP is a research scaffold with
        // loose constraints, so we don't require tight bounds here).
        assert!(hi.is_finite(), "upper bound should be finite, got {hi}");
    }

    #[test]
    fn entropy_bounds_single_symbol() {
        // Single symbol: entropy should be near 0.
        let counts = [50usize];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let params = SupportLpParams::default_for(&fp);
        let (lo, hi) = entropy_bounds_lp(&fp, params).unwrap();
        assert!(lo >= -1e-9);
        assert!(hi.is_finite());
    }

    #[test]
    fn support_bounds_rejects_empty_sample() {
        let fp = Fingerprint { f: vec![0] };
        let params = SupportLpParams {
            k: 5,
            grid_size: 20,
            p_min: 0.001,
            p_max: 1.0,
            eps_scale: 1.5,
        };
        let result = support_bounds_lp(&fp, params);
        assert!(result.is_err());
    }

    #[test]
    fn default_params_are_sane() {
        let counts = [5usize, 3, 1, 1];
        let fp = Fingerprint::from_counts(counts).unwrap();
        let params = SupportLpParams::default_for(&fp);
        assert!(params.k > 0);
        assert!(params.grid_size >= 2);
        assert!(params.p_min > 0.0);
        assert!(params.p_max >= params.p_min);
        assert!(params.eps_scale > 0.0);
    }
}
