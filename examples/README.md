# fingerprints examples

Each example answers one question and is runnable as-is. All outputs below are
real, captured from a run.

## Getting started

### `basic`: what can I estimate from per-symbol counts?

Builds a `Fingerprint` from observed counts, then reports entropy estimates,
Good-Turing unseen mass, and Chao1 support.

```bash
cargo run --release --example basic
```
```text
n=19 S_obs=8 H_plugin=1.9097 H_MM=2.0939 H_JK=2.1923 H_PY=2.6595 (d=0.193, α=3.206) p_unseen≈0.1579 S_chao1≈10.25
```

### `unseen_report`: how do I inspect one sample from the command line?

Takes per-symbol counts as arguments and prints the full unseen-regime report in
nats and bits, including VV-style grid-feasible LP ranges when the `lp` feature
is enabled.

```bash
cargo run --release --example unseen_report -- 5 4 3 2 2 1 1 1
```
```text
n=19 S_obs=8 F1=3
unseen mass (Good-Turing) p0_hat~=0.1579
support (Chao1) S_hat~=10.25

entropy (bits):
  plugin         2.755058
  Miller-Madow   3.020817
  jackknife      3.162851
  Pitman-Yor     3.836913

VV-style LP support range: [8.000, 143.707]
VV-style LP entropy range (nats): [0.113688, 3.411798]
```

## Estimator families

### `pml_uniform`: which support size fits best within the uniform family?

Compares support sizes under a uniform-distribution assumption and returns the
one with the highest profile log-likelihood. It does not optimize over
non-uniform distributions.

```bash
cargo run --release --example pml_uniform
```
```text
best_uniform_support_size: Ŝ=8 (observed distinct m=8) ll=-3.1934
```

### `vv_bounds`: what range does the VV-style LP admit?

Computes the minimum and maximum support and entropy over the scaffold's
configured discretized feasible set. These are sensitivity ranges, not
confidence intervals for the true properties. This example needs the default
`lp` feature.

```bash
cargo run --release --example vv_bounds --features lp
```
```text
VV-style LP support range: [8.000, 143.707]
VV-style LP entropy range (nats): [0.113688, 3.411798]
```

### `pitman_yor_zipf`: how do estimators behave on a heavy-tailed distribution?

Samples from a Zipf distribution and compares plug-in, Miller-Madow, jackknife,
and Pitman-Yor entropy estimates as sample size grows.

```bash
cargo run --release --example pitman_yor_zipf
```
```text
Zipf demo: K=5000 s=1.10
true H(bits) = 7.9142

    N     S    F1  p0_hat err_plug   err_MM   err_JK   err_PY       d    alpha
  200   101    82  0.4100  -2.2451  -1.8845  -1.5798  -0.2188   0.000  125.849
  400   192   148  0.3700  -1.3858  -1.0413  -0.7629   0.5478   0.000  226.292
  800   299   228  0.2850  -1.4200  -1.1513  -0.9360   0.5788   0.000  369.345
 1600   510   382  0.2387  -1.0018  -0.7723  -0.5924   0.9947   0.000  654.186
```

## Applications

### `vocab_coverage`: how much vocabulary has this text sample covered?

Tokenizes sample text with `textprep`, builds word-frequency counts, and reports
the observed-vs-estimated vocabulary coverage.

```bash
cargo run --release --example vocab_coverage
```
```text
--- Vocabulary Coverage Report ---

Tokens (n):              186
Unique types (S_obs):    123
Singletons (F1):         97
Doubletons (F2):         15

Good-Turing unseen mass: 0.5215
Good-Turing coverage:    0.4785
Chao1 vocab estimate:    436.6
Entropy (bits, PY):      8.5168

Observed/estimated vocab: 28.2% (123 of 437 estimated types)
High unseen mass -- the sample covers a small fraction of the vocabulary.
```

### `mdl_codelength`: when is a coarser representation shorter?

Compares two-part codelengths for exact states and task labels. The task labels
are shorter for the task, but exact reconstruction must pay the residual bits.

```bash
cargo run --release --example mdl_codelength
```
```text
sample: 100 observations
representation  model bits  sample bits  task bits  exact bits
exact states          16.0        177.7      193.7       193.7
task labels           18.0         77.8       95.8       195.7
residual bits for exact reconstruction after merging: 99.9
```
