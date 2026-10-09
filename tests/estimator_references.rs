//! Estimator values checked against the formulas in the cited papers, plus a
//! seeded Monte Carlo check that the bias-corrected entropy estimators improve
//! on the plug-in estimator in the undersampled regime they target.

use fingerprints::{
    coverage_chao_jost, entropy_miller_madow_nats, entropy_pitman_yor_nats, entropy_plugin_nats,
    support_ichao1, Fingerprint,
};

/// Chiu, Wang, Walther & Chao (2014), with n = 36, S_obs = 12 and
/// (F1, F2, F3, F4) = (5, 1, 2, 2):
/// S = S_obs + (n-1)/n * F1^2 / (2 F2)
///       + (n-3)/(4n) * F3/F4 * max(F1 - (n-3)/(2(n-1)) * F2 F3 / F4, 0)
///   = 12 + 12.1528 + 1.0378 = 25.1906.
#[test]
fn ichao1_matches_chiu_2014_with_finite_sample_factors() {
    let fp = Fingerprint::from_counts([1usize, 1, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8]).unwrap();
    assert_eq!(fp.sample_size(), 36);
    assert_eq!(fp.observed_support(), 12);
    let n = 36.0_f64;
    let expected = 12.0
        + (n - 1.0) / n * 25.0 / 2.0
        + (n - 3.0) / (4.0 * n) * (2.0 / 2.0) * (5.0 - (n - 3.0) / (2.0 * (n - 1.0)) * 2.0 / 2.0);
    assert!((expected - 25.1906).abs() < 1e-4);
    let got = support_ichao1(&fp);
    assert!(
        (got - expected).abs() < 1e-12,
        "got {got}, expected {expected}"
    );
}

/// Chao & Jost (2012) coverage on counts [5, 3, 2, 2, 1, 1, 1, 1]:
/// n = 16, F1 = 4, F2 = 2, so C = 1 - (4/16) * (15*4) / (15*4 + 2*2) = 0.765625.
/// Chao & Shen (2003) would give 1 - 4/16 = 0.75.
#[test]
fn chao_jost_coverage_matches_closed_form() {
    let fp = Fingerprint::from_counts([5usize, 3, 2, 2, 1, 1, 1, 1]).unwrap();
    assert!((coverage_chao_jost(&fp) - 0.765625).abs() < 1e-12);
}

#[test]
#[allow(deprecated)]
fn deprecated_chao_shen_name_is_the_same_estimator() {
    let fp = Fingerprint::from_counts([5usize, 3, 2, 2, 1, 1, 1, 1]).unwrap();
    assert_eq!(
        fingerprints::coverage_chao_shen(&fp),
        coverage_chao_jost(&fp)
    );
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Zipf(s = 1) on K = 1000 symbols sampled n = 300 times: the plug-in
/// estimator is biased low by a large margin. Over 200 seeded replicates the
/// mean absolute error of Miller-Madow and Pitman-Yor must be smaller.
#[test]
fn bias_corrected_entropy_beats_plugin_on_zipf_samples() {
    let k = 1000;
    let n = 300;
    let weights: Vec<f64> = (1..=k).map(|i| 1.0 / i as f64).collect();
    let z: f64 = weights.iter().sum();
    let p: Vec<f64> = weights.iter().map(|w| w / z).collect();
    let truth: f64 = -p.iter().map(|&pi| pi * pi.ln()).sum::<f64>();
    let mut cdf = Vec::with_capacity(k);
    let mut acc = 0.0;
    for &pi in &p {
        acc += pi;
        cdf.push(acc);
    }
    *cdf.last_mut().unwrap() = 1.0;

    let mut rng = SplitMix64(20140601);
    let (mut err_plugin, mut err_mm, mut err_py) = (0.0, 0.0, 0.0);
    let reps = 200;
    for _ in 0..reps {
        let mut counts = vec![0usize; k];
        for _ in 0..n {
            let u = rng.next_f64();
            let idx = cdf.partition_point(|&c| c < u).min(k - 1);
            counts[idx] += 1;
        }
        let fp = Fingerprint::from_counts(counts.into_iter().filter(|&c| c > 0)).unwrap();
        err_plugin += (entropy_plugin_nats(&fp) - truth).abs();
        err_mm += (entropy_miller_madow_nats(&fp) - truth).abs();
        err_py += (entropy_pitman_yor_nats(&fp) - truth).abs();
    }
    let (err_plugin, err_mm, err_py) = (
        err_plugin / reps as f64,
        err_mm / reps as f64,
        err_py / reps as f64,
    );
    assert!(
        err_mm < err_plugin,
        "Miller-Madow MAE {err_mm} >= plug-in {err_plugin}"
    );
    assert!(
        err_py < err_plugin,
        "Pitman-Yor MAE {err_py} >= plug-in {err_plugin}"
    );
}
