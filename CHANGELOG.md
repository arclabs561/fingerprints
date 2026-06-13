# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
