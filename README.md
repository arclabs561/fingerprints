# propest

Property estimation from samples.

This crate lives “above” `logp`: `logp` defines information-theoretic functionals on *known*
distributions; `propest` estimates those functionals from *samples* (fingerprints / profiles).

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

