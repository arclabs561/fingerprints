use fingerprints::{
    entropy_jackknife_bits, entropy_miller_madow_bits, entropy_pitman_yor_bits,
    entropy_plugin_bits, pitman_yor_params_hat, support_chao1, unseen_mass_good_turing,
    Fingerprint,
};

// A tiny deterministic RNG for examples (SplitMix64).
// This avoids pulling in an RNG dependency while still letting us simulate sampling.
#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // Uniform in [0,1). Take the top 53 bits.
        let x = self.next_u64() >> 11;
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn zipf_probs(k: usize, s: f64) -> Vec<f64> {
    let mut w: Vec<f64> = (1..=k).map(|i| 1.0 / (i as f64).powf(s)).collect();
    let z: f64 = w.iter().sum();
    for wi in &mut w {
        *wi /= z;
    }
    w
}

fn shannon_entropy_bits(p: &[f64]) -> f64 {
    let ln2 = std::f64::consts::LN_2;
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * (pi.ln() / ln2);
        }
    }
    h
}

fn sample_counts(p: &[f64], n: usize, seed: u64) -> Vec<usize> {
    let mut cdf = Vec::with_capacity(p.len());
    let mut acc = 0.0;
    for &pi in p {
        acc += pi;
        cdf.push(acc);
    }
    // Force the last to be exactly 1.0 to avoid edge cases.
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut rng = SplitMix64::new(seed);
    let mut counts = vec![0usize; p.len()];
    for _ in 0..n {
        let u = rng.next_f64();
        let idx = cdf.partition_point(|&x| x < u).min(counts.len() - 1);
        counts[idx] += 1;
    }
    counts
}

fn main() {
    // Synthetic “unseen regime” demo:
    // - large support K
    // - Zipf-ish heavy tail
    // - sample size N << K
    let k = 5000usize;
    let s = 1.10;

    let p = zipf_probs(k, s);
    let h_true = shannon_entropy_bits(&p);

    println!("Zipf demo: K={} s={:.2}", k, s);
    println!("true H(bits) = {:.4}", h_true);
    println!();
    println!(
        "{:>5} {:>5} {:>5} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "N", "S", "F1", "p0_hat", "err_plug", "err_MM", "err_JK", "err_PY", "d", "alpha"
    );

    for &n in &[200usize, 400, 800, 1600] {
        let seed = 0xC0FFEE_u64 ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let counts_full = sample_counts(&p, n, seed);
        let counts_obs: Vec<usize> = counts_full.into_iter().filter(|&c| c > 0).collect();
        let fp = Fingerprint::from_counts(counts_obs).unwrap();

        let h_plugin = entropy_plugin_bits(&fp);
        let h_mm = entropy_miller_madow_bits(&fp);
        let h_jk = entropy_jackknife_bits(&fp);
        let h_py = entropy_pitman_yor_bits(&fp);
        let py = pitman_yor_params_hat(&fp);

        let p0_hat = unseen_mass_good_turing(&fp);

        let err_plugin = h_plugin - h_true;
        let err_mm = h_mm - h_true;
        let err_jk = h_jk - h_true;
        let err_py = h_py - h_true;

        println!(
            "{:>5} {:>5} {:>5} {:>7.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>7.3} {:>8.3}",
            n,
            fp.observed_support(),
            fp.singletons(),
            p0_hat,
            err_plugin,
            err_mm,
            err_jk,
            err_py,
            py.d,
            py.alpha
        );
    }

    println!();
    println!("(Also available: support_chao1 = {:.1})", {
        let counts_full = sample_counts(&p, 600, 0xC0FFEE);
        let counts_obs: Vec<usize> = counts_full.into_iter().filter(|&c| c > 0).collect();
        let fp = Fingerprint::from_counts(counts_obs).unwrap();
        support_chao1(&fp)
    });
}
