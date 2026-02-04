# propest

Property estimation from samples.

This crate lives “above” `logp`: `logp` defines information-theoretic functionals on *known*
distributions; `propest` estimates those functionals from *samples* (fingerprints / profiles).

## Scope (initial)

- Fingerprint (counts-of-counts) utilities
- Classical entropy estimators (plug-in, Miller–Madow)
- Support-size / “distinct elements” helpers (basics)

Planned:
- Valiant–Valiant style unseen reconstruction (LP-based), behind an explicit feature gate.

