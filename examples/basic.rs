use fingerprints::{
    entropy_jackknife_nats, entropy_miller_madow_nats, entropy_plugin_nats, support_chao1,
    unseen_mass_good_turing, Fingerprint,
};

fn main() {
    // Per-symbol counts for observed distinct symbols.
    let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
    let fp = Fingerprint::from_counts(counts).unwrap();

    let h_plugin = entropy_plugin_nats(&fp);
    let h_mm = entropy_miller_madow_nats(&fp);
    let h_jk = entropy_jackknife_nats(&fp);

    let p_unseen = unseen_mass_good_turing(&fp);
    let s_hat = support_chao1(&fp);

    assert!(h_plugin >= 0.0);
    assert!(h_mm >= h_plugin);
    assert!(h_jk.is_finite() && h_jk >= 0.0);
    assert!((0.0..=1.0).contains(&p_unseen));
    assert!(s_hat >= fp.observed_support() as f64);

    println!(
        "n={} S_obs={} H_plugin={:.4} H_MM={:.4} H_JK={:.4} p_unseen≈{:.4} S_chao1≈{:.2}",
        fp.sample_size(),
        fp.observed_support(),
        h_plugin,
        h_mm,
        h_jk,
        p_unseen,
        s_hat
    );
}

