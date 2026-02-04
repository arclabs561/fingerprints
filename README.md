# propest

Property estimation from samples.

This crate lives “above” `logp`: `logp` defines information-theoretic functionals on *known*
distributions; `propest` estimates those functionals from *samples* (fingerprints / profiles).

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

## Scope (initial)

- Fingerprint (counts-of-counts) utilities
- Classical entropy estimators (plug-in, Miller–Madow, jackknife)
- Coverage / unseen-mass helpers (Good–Turing)
- Support-size (“distinct elements”) helpers (Chao1)
- PML scaffolding for small supports (`pml` module)
- VV-style histogram LP bounds (`vv` module): support and entropy (feature-gated)

## Features

- `vv-lp`: enables a minimal VV-style histogram LP (support-size bounds) via `minilp`.
  Includes `support_bounds_lp` and `entropy_bounds_lp`.

## Examples

- `cargo run --example basic`
- `cargo run --example pml_uniform`
- `cargo run --example vv_bounds --features vv-lp`

