# fingerprints

[![crates.io](https://img.shields.io/crates/v/fingerprints.svg)](https://crates.io/crates/fingerprints)
[![Documentation](https://docs.rs/fingerprints/badge.svg)](https://docs.rs/fingerprints)

Property estimation from samples.

`fingerprints` estimates entropy, support size, unseen mass, and related
properties from counts or fingerprints.

It is aimed at finite samples from distributions whose support may be large or
unknown.

## Quickstart

Add to your `Cargo.toml`:

```toml
[dependencies]
fingerprints = "0.2"
```

Estimate a few basic quantities from per-symbol counts:

```rust
use fingerprints::{
    Fingerprint,
    entropy_plugin_nats,
    entropy_miller_madow_nats,
    entropy_jackknife_nats,
    entropy_pitman_yor_nats,
    pitman_yor_params_hat,
    unseen_mass_good_turing,
    support_chao1,
    to_bits,
};

let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
let fp = Fingerprint::from_counts(counts).unwrap();

let h_plugin = entropy_plugin_nats(&fp);
let h_mm = entropy_miller_madow_nats(&fp);
let h_jk = entropy_jackknife_nats(&fp);
let h_py = entropy_pitman_yor_nats(&fp);
let py = pitman_yor_params_hat(&fp);
let p_unseen = unseen_mass_good_turing(&fp);
let s_hat = support_chao1(&fp);

assert!(h_plugin >= 0.0);
assert!(h_mm >= h_plugin);
assert!(h_py.is_finite() && h_py >= 0.0);
assert!((0.0..1.0).contains(&py.d));
assert!((0.0..=1.0).contains(&p_unseen));
assert!(s_hat >= fp.observed_support() as f64);

// Convert nats to bits with to_bits().
let h_bits = to_bits(h_plugin);
assert!(h_bits >= 0.0);
```

## API tour

- **Fingerprint abstraction**:
  - `Fingerprint::from_counts`, `Fingerprint::from_frequency_counts`, `sample_size`, `observed_support`, `singletons`, `doubletons`, `count`, `max_freq`, `iter`, `as_slice`, `observed_species`
- **Entropy estimators (nats)**:
  - `entropy_default_nats` (opinionated default; currently aliases Pitman-Yor)
  - `entropy_plugin_nats`
  - `entropy_miller_madow_nats`
  - `entropy_jackknife_nats`
  - `entropy_pitman_yor_nats` (Pitman-Yor / DPYM; targets the unseen regime)
  - `pitman_yor_params_hat` (inspect selected hyperparameters)
- **Unit conversion**:
  - `to_bits(nats)` converts nats to bits
- **Coverage / support**:
  - `coverage_good_turing` (sample coverage `C_hat = 1 - F_1/n`)
  - `unseen_mass_good_turing` (unseen mass `p0_hat ~ F_1/n`)
  - `support_chao1`
- **Coverage baselines / toy problems** (`fingerprints::coverage`):
  - `german_tank_unbiased_*`: finite-support "max serial number" baseline (sampling without replacement). See [German tank problem](https://en.wikipedia.org/wiki/German_tank_problem)
  - `coupon_collector_expected_draws`, `expected_distinct_uniform`: uniform coverage baselines. See [Coupon collector's problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)
- **PML scaffolding** (`fingerprints::pml`):
  - `best_uniform_support_size` (baseline family)
  - `profile_log_likelihood_small` (exact profile likelihood for small observed support)
- **VV-style LP scaffold** (`fingerprints::vv`, requires `lp` feature):
  - `support_bounds_lp`, `entropy_bounds_lp`

## Invariants and conventions

- **Inputs**:
  - "counts" means per-symbol multiplicities for *observed* distinct symbols (all `> 0`).
  - `Fingerprint` stores `F[i]` = number of symbols seen exactly `i` times; `F[0]` is unused.
- **Units**:
  - Entropy is in **nats**. Use `to_bits()` for bits.
- **Codelength**:
  - `sample_codelen_plugin_nats` turns per-symbol entropy into **total sample code length** (`n * H(p_hat)`), a useful scalar for MDL-style comparisons.
- **Estimator semantics**:
  - `*_plugin_*` treats the empirical histogram as the true distribution.
  - Bias corrections (Miller-Madow, jackknife) can overshoot in some regimes; treat as estimators,
    not identities.

## Features

- `lp` (default): enables `fingerprints::vv` module with LP-backed bounds via `minilp`.

## Examples

See [`examples/README.md`](examples/README.md) for the full gallery: each
example states the question it answers, the run command, and real sample output.

Each targets the unseen regime above (estimating what a sample has not yet
revealed), the problem ecology calls species richness:

- `cargo run --example basic` estimates entropy, unseen mass, and support size from per-symbol counts, contrasting the plug-in estimator with the bias-corrected ones.
- `cargo run --example pml_uniform` runs profile maximum likelihood over the uniform family, recovering the size of a near-uniform alphabet from a sample.
- `cargo run --example vv_bounds` computes Valiant-Valiant LP bounds on support and entropy, the estimator with provable sample complexity (uses the default `lp` feature).
- `cargo run --example pitman_yor_zipf` samples from a heavy-tailed Pitman-Yor / Zipf distribution and estimates its properties, the realistic case for natural-language word frequencies.
- `cargo run --example unseen_report -- 5 4 3 2 2 1 1 1` takes a fingerprint (counts of counts) on the command line and prints the full unseen-regime report.
- `cargo run --example vocab_coverage` estimates how much of a text sample's vocabulary the sample has already revealed, the species-richness question on words.
- `cargo run --example mdl_codelength` compares two-part codelengths for exact and task-level representations, the MDL/Kolmogorov foothold.

## Tests

```bash
cargo test --all-features
```

102 tests (76 unit + 26 doc-tests).

## Roadmap (near-term)

- VV-style LP constraints that better track classical VV practice (grid policy, tighter moments).
- PML beyond the uniform family (solver/heuristics; keep behind clear, opt-in APIs).
- More properties beyond entropy/support (e.g. distance-to-uniformity proxies).

## References

Key papers motivating the estimator families in this crate:

- Good (1953), "The population frequencies of species and the estimation of population parameters": Good-Turing coverage
- Chao (1984), "Nonparametric estimation of the number of classes in a population": Chao1 support estimator
- Valiant & Valiant (2017), "Estimating the Unseen: Improved Estimators for Entropy and other Properties" (JACM): LP-based bounds
- Hao & Orlitsky (2019), "The Broad Optimality of Profile Maximum Likelihood": PML as unified sample-optimal estimator
- Hashino & Tsukuda (2026), "Estimating the Shannon Entropy Using the Pitman-Yor Process": PY entropy estimator
- Han, Jiao, Weissman (2025), "Besting Good-Turing: Optimality of NPMLE": theoretical motivation for PML direction

Ecology and biodiversity estimation is a primary motivating application domain for these methods. Species richness estimation, unseen species prediction, and diversity indices all reduce to the fingerprint-based estimation problems addressed here. See:

- Chen & Shen (2025), "Biogeographic Patterns of Estimation Bias of Biodiversity Indices": documents systematic estimation bias in biodiversity indices across geographic contexts, underscoring the need for bias-corrected estimators

## License

MIT OR Apache-2.0
