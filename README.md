# fingerprints

Property estimation from samples.

This crate provides estimators of information-theoretic functionals from *samples* (counts /
fingerprints / profiles), building on a definition layer for known distributions.

The center of gravity is the **unseen regime**: finite samples, large/unknown support, and
estimators with explicit bias/variance tradeoffs.

## Quickstart

Add to your `Cargo.toml`:

```toml
[dependencies]
fingerprints = "0.1"
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
```

## API tour

- **Fingerprint abstraction**:
  - `Fingerprint::from_counts`, `sample_size`, `observed_support`, `singletons`, `doubletons`
- **Entropy estimators (nats)**:
  - `entropy_default_nats` (opinionated default; currently aliases Pitman–Yor)
  - `entropy_plugin_nats`
  - `entropy_miller_madow_nats`
  - `entropy_jackknife_nats`
  - `entropy_pitman_yor_nats` (Pitman–Yor / DPYM; targets the unseen regime)
  - `pitman_yor_params_hat` (inspect selected hyperparameters)
- **Coverage / support**:
  - `unseen_mass_good_turing` (unseen mass \( \hat p_0 \approx F_1/n \))
  - `support_chao1`
- **Coverage baselines / toy problems** (`fingerprints::coverage`):
  - `german_tank_unbiased_*`: finite-support “max serial number” baseline (sampling without replacement) — see [German tank problem](https://en.wikipedia.org/wiki/German_tank_problem)
  - `coupon_collector_expected_draws`, `expected_distinct_uniform`: uniform coverage baselines — see [Coupon collector's problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)
- **PML scaffolding** (`fingerprints::pml`):
  - `best_uniform_support_size` (baseline family)
  - `profile_log_likelihood_small` (exact profile likelihood for small observed support)
- **VV-style LP scaffold** (`fingerprints::vv`, feature-gated):
  - `support_bounds_lp`, `entropy_bounds_lp`

## Invariants and conventions

- **Inputs**:
  - “counts” means per-symbol multiplicities for *observed* distinct symbols (all `> 0`).
  - `Fingerprint` stores `F[i]` = number of symbols seen exactly `i` times; `F[0]` is unused.
- **Units**:
  - entropy is in **nats** unless a `_bits` helper is used.
- **Codelength**:
  - `sample_codelen_plugin_*` turns per-symbol entropy into **total sample code length** (\(n\cdot H(\hat p)\)), which is a useful scalar for MDL-style comparisons.
- **Estimator semantics**:
  - `*_plugin_*` treats the empirical histogram as the true distribution.
  - Bias corrections (Miller–Madow, jackknife) can overshoot in some regimes; treat as estimators,
    not identities.

## Features

No feature flags. `fingerprints::vv`’s LP-backed bounds are available by default via `minilp`.

## Examples

- `cargo run --example basic`
- `cargo run --example pml_uniform`
- `cargo run --example vv_bounds`
- `cargo run --example pitman_yor_zipf`
- `cargo run --example unseen_report -- 5 4 3 2 2 1 1 1`

## Tests

```bash
cargo test -p fingerprints
```

72 tests (51 unit + 21 doc-tests) covering entropy estimator consistency, bias correction ordering, Good-Turing/Chao1 edge cases, profile likelihood monotonicity, VV LP bounds bracketing, Hellinger triangle inequality (proptest), and cross-crate property tests against logp (total Bregman normalization, rho-alpha identity).

## Roadmap (near-term)

- VV-style LP constraints that better track classical VV practice (grid policy, tighter moments).
- PML beyond the uniform family (solver/heuristics; keep behind clear, opt-in APIs).
- More properties beyond entropy/support (e.g. distance-to-uniformity proxies).

## References

Key papers motivating the estimator families in this crate:

- Good (1953), "The population frequencies of species and the estimation of population parameters" -- Good-Turing coverage
- Chao (1984), "Nonparametric estimation of the number of classes in a population" -- Chao1 support estimator
- Valiant & Valiant (2017), "Estimating the Unseen: Improved Estimators for Entropy and other Properties" (JACM) -- LP-based bounds
- Hao & Orlitsky (2019), "The Broad Optimality of Profile Maximum Likelihood" -- PML as unified sample-optimal estimator
- Hashino & Tsukuda (2026), "Estimating the Shannon Entropy Using the Pitman-Yor Process" -- PY entropy estimator
- Han, Jiao, Weissman (2025), "Besting Good-Turing: Optimality of NPMLE" -- theoretical motivation for PML direction

Ecology and biodiversity estimation is a primary motivating application domain for these methods. Species richness estimation, unseen species prediction, and diversity indices all reduce to the fingerprint-based estimation problems addressed here. See:

- Chen & Shen (2025), "Biogeographic Patterns of Estimation Bias of Biodiversity Indices" -- documents systematic estimation bias in biodiversity indices across geographic contexts, underscoring the need for bias-corrected estimators like those in this crate

## License

MIT OR Apache-2.0
