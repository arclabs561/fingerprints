//! Valiant–Valiant style “unseen” scaffolding.
//!
//! The classical VV approach (Poissonized fingerprints + histogram LP) is powerful but involves
//! design choices (grid, tolerances, constraints). This module provides a minimal, auditable LP
//! for **support size bounds** as a first step.
//!
//! Implementation notes:
//! - Behind the `vv-lp` feature flag to keep default deps lean.
//! - Uses a log-spaced probability grid and Poisson moment constraints for `F_1..F_k`.

#![forbid(unsafe_code)]

use crate::{Fingerprint, PropEstError, Result};

/// Parameters for the histogram LP.
#[derive(Debug, Clone)]
pub struct SupportLpParams {
    /// Max multiplicity used from the fingerprint (`F_1..F_k`).
    pub k: usize,
    /// Number of grid points.
    pub grid_size: usize,
    /// Minimum probability in the grid.
    pub p_min: f64,
    /// Maximum probability in the grid.
    pub p_max: f64,
    /// Tolerance scale for matching fingerprint moments.
    pub eps_scale: f64,
}

impl SupportLpParams {
    /// Heuristics that are reasonable for small/medium `n`.
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

#[cfg(feature = "vv-lp")]
fn ln_factorial(i: usize) -> f64 {
    if i <= 1 {
        return 0.0;
    }
    (2..=i).map(|k| (k as f64).ln()).sum()
}

#[cfg(feature = "vv-lp")]
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

#[cfg(feature = "vv-lp")]
fn grid_log_space(p_min: f64, p_max: f64, m: usize) -> Result<Vec<f64>> {
    if !(p_min > 0.0) || !(p_max > 0.0) || !(p_max >= p_min) {
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
/// Returns bounds in *expected* support terms; they are real-valued.
///
/// This is a **research scaffold**: expect future evolution in grid design and constraint policy.
pub fn support_bounds_lp(fp: &Fingerprint, params: SupportLpParams) -> Result<(f64, f64)> {
    support_bounds_lp_impl(fp, params)
}

#[cfg(not(feature = "vv-lp"))]
fn support_bounds_lp_impl(_fp: &Fingerprint, _params: SupportLpParams) -> Result<(f64, f64)> {
    Err(PropEstError::Invalid(
        "VV LP support bounds require feature `vv-lp`",
    ))
}

#[cfg(feature = "vv-lp")]
fn support_bounds_lp_impl(fp: &Fingerprint, params: SupportLpParams) -> Result<(f64, f64)> {
    use minilp::{ComparisonOp, OptimizationDirection, Problem};

    let n = fp.sample_size();
    if n == 0 {
        return Err(PropEstError::Invalid("sample size is zero"));
    }
    let s_obs = fp.observed_support() as f64;

    let k = params.k.min(fp.f.len().saturating_sub(1)).max(1);
    let grid = grid_log_space(params.p_min, params.p_max, params.grid_size)?;

    // Precompute coefficients a_{i,j} = Poi(i; n p_j)
    let mut a = vec![vec![0.0f64; grid.len()]; k + 1];
    for i in 1..=k {
        for (j, &p) in grid.iter().enumerate() {
            let lambda = (n as f64) * p;
            a[i][j] = poisson_pmf(i, lambda);
        }
    }

    // Helper to build and solve LP with objective direction.
    let solve = |dir: OptimizationDirection| -> Result<f64> {
        let mut pb = Problem::new(dir);
        // Objective: support size S = Σ x_j
        let vars: Vec<_> = (0..grid.len())
            .map(|_| pb.add_var(1.0, (0.0, f64::INFINITY)))
            .collect();

        // Mass constraint: Σ x_j p_j <= 1.
        let mut mass = Vec::with_capacity(vars.len());
        for (v, &p) in vars.iter().zip(grid.iter()) {
            mass.push((*v, p));
        }
        pb.add_constraint(mass, ComparisonOp::Le, 1.0);

        // Observed support is a lower bound.
        let sup: Vec<_> = vars.iter().map(|&v| (v, 1.0)).collect();
        pb.add_constraint(sup, ComparisonOp::Ge, s_obs);

        // Fingerprint constraints for i=1..k: match within eps_i.
        for i in 1..=k {
            let fi = fp.f.get(i).copied().unwrap_or(0) as f64;
            let eps_i = params.eps_scale * (fi.sqrt() + 1.0);
            let row: Vec<_> = vars
                .iter()
                .enumerate()
                .map(|(j, &v)| (v, a[i][j]))
                .collect();
            pb.add_constraint(row.clone(), ComparisonOp::Le, fi + eps_i);
            pb.add_constraint(row, ComparisonOp::Ge, (fi - eps_i).max(0.0));
        }

        let sol = pb.solve().map_err(|_| PropEstError::Invalid("LP infeasible"))?;
        Ok(sol.objective())
    };

    let lower = solve(OptimizationDirection::Minimize)?;
    let upper = solve(OptimizationDirection::Maximize)?;
    Ok((lower, upper))
}

#[cfg(all(test, feature = "vv-lp"))]
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
}

