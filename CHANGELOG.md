# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-11-03

### Added
- `coverage_chao_jost`, the Chao & Jost (2012) sample coverage estimator
  under its correct name.
- `rust-version = "1.87"` in the manifest. 0.2.1 already needed 1.87 for
  `usize::is_multiple_of`; the minimum is now declared, not raised.

### Changed
- `support_ichao1` now follows Chiu et al. (2014): the bias-corrected Chao1
  base with the `(n-1)/n`, `(n-3)/(4n)` and `(n-3)/(2(n-1))` finite-sample
  factors, and `F4 + 1` when `F4 = 0`, as SpadeR does. Results differ from
  0.2.1; the paper's worked example now gives 25.191 instead of 25.625.

### Deprecated
- `coverage_chao_shen`: it computes the Chao & Jost (2012) estimator, so it is
  now an alias for `coverage_chao_jost`.

### Fixed
- `Fingerprint::from_counts` returns `EstimationError::Invalid` instead of
  panicking when the largest count is too large to allocate the dense
  counts-of-counts vector (for example `usize::MAX`).
- `support_bounds_lp` and `entropy_bounds_lp` return
  `EstimationError::Invalid` for `k = 0`, a nonpositive or nonfinite
  `eps_scale`, or `p_max > 1`, and for nonfinite or inverted LP endpoints;
  tiny negative endpoints are clamped to zero.

## [0.2.1] - 2026-06-11

### Changed
- Expanded the pml and coverage module documentation.
- Simplified the crate description.

### Fixed
- Dropped a trailing zero in the `ln(40320)` f64 literal (clippy `excessive_precision`).

## [0.2.0] - 2026-04-06

### Added
- Pitman-Yor entropy estimator.
- `coverage_good_turing` and `from_frequency_counts`.
- `vocab_coverage` example.
- Convergence and edge-case tests.
- PML (profile maximum likelihood) and VV LP entropy bounds.
- Coverage baselines and code-length helpers.
- Paper citations across all estimator modules.
- Math formulas in the rustdocs.

### Changed
- API overhaul for 0.2.0.
- Enabled the `missing_docs` lint.
- Renamed the crate from `propest`/`unseen` to `fingerprints`.

### Fixed
- Broken doc comment in `pml.rs`.
