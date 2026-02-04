# propest

Property estimation from samples.

This crate lives “above” `logp`:

- `logp`: information-theoretic functionals on *known* distributions (simplex vectors).
- `propest`: estimators of those functionals from *samples* (counts / fingerprints / profiles).

The center of gravity is the **unseen regime**: finite samples, large/unknown support, and
estimators with explicit bias/variance tradeoffs.

## Quickstart

Add to your `Cargo.toml`:

```toml
[dependencies]
propest = "0.1"
```

Estimate a few basic quantities from per-symbol counts:

```rust
use propest::{
    Fingerprint,
    entropy_plugin_nats,
    entropy_miller_madow_nats,
    entropy_jackknife_nats,
    unseen_mass_good_turing,
    support_chao1,
};

let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
let fp = Fingerprint::from_counts(counts).unwrap();

let h_plugin = entropy_plugin_nats(&fp);
let h_mm = entropy_miller_madow_nats(&fp);
let h_jk = entropy_jackknife_nats(&fp);
let p_unseen = unseen_mass_good_turing(&fp);
let s_hat = support_chao1(&fp);

assert!(h_plugin >= 0.0);
assert!(h_mm >= h_plugin);
assert!((0.0..=1.0).contains(&p_unseen));
assert!(s_hat >= fp.observed_support() as f64);
```

## API tour

- **Fingerprint abstraction**:
  - `Fingerprint::from_counts`, `sample_size`, `observed_support`, `singletons`, `doubletons`
- **Entropy estimators (nats)**:
  - `entropy_plugin_nats`
  - `entropy_miller_madow_nats`
  - `entropy_jackknife_nats`
- **Coverage / support**:
  - `unseen_mass_good_turing` (unseen mass \( \hat p_0 \approx F_1/n \))
  - `support_chao1`
- **PML scaffolding** (`propest::pml`):
  - `best_uniform_support_size` (baseline family)
  - `profile_log_likelihood_small` (exact profile likelihood for small observed support)
- **VV-style LP scaffold** (`propest::vv`, feature-gated):
  - `support_bounds_lp`, `entropy_bounds_lp`

## Invariants and conventions

- **Inputs**:
  - “counts” means per-symbol multiplicities for *observed* distinct symbols (all `> 0`).
  - `Fingerprint` stores `F[i]` = number of symbols seen exactly `i` times; `F[0]` is unused.
- **Units**:
  - entropy is in **nats** unless a `_bits` helper is used.
- **Estimator semantics**:
  - `*_plugin_*` treats the empirical histogram as the true distribution.
  - Bias corrections (Miller–Madow, jackknife) can overshoot in some regimes; treat as estimators,
    not identities.

## Features

- `vv-lp`: enables a minimal VV-style histogram LP via `minilp` (bounds on support and entropy).

## Examples

- `cargo run --example basic`
- `cargo run --example pml_uniform`
- `cargo run --example vv_bounds --features vv-lp`

## Roadmap (near-term)

- VV-style LP constraints that better track classical VV practice (grid policy, tighter moments).
- PML beyond the uniform family (solver/heuristics; still behind clear feature gates).
- More properties beyond entropy/support (e.g. distance-to-uniformity proxies).

## License

MIT OR Apache-2.0

