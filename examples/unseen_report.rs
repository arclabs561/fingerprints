use fingerprints::{
    entropy_jackknife_bits, entropy_jackknife_nats, entropy_miller_madow_bits,
    entropy_miller_madow_nats, entropy_pitman_yor_bits, entropy_pitman_yor_nats,
    entropy_plugin_bits, entropy_plugin_nats, pitman_yor_params_hat, support_chao1,
    unseen_mass_good_turing, Fingerprint,
};

fn usage() -> ! {
    eprintln!(
        "Usage:\n  cargo run --example unseen_report -- <count1> <count2> ...\n\n\
If no counts are provided, a small default is used.\n\
Counts must be positive integers (per observed distinct symbol)."
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
    }

    let counts: Vec<usize> = if args.is_empty() {
        vec![5, 4, 3, 2, 2, 1, 1, 1]
    } else {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            let c: usize = a.parse().unwrap_or_else(|_| usage());
            if c == 0 {
                usage();
            }
            out.push(c);
        }
        out
    };

    let fp = Fingerprint::from_counts(counts).unwrap();

    let h_plugin_n = entropy_plugin_nats(&fp);
    let h_mm_n = entropy_miller_madow_nats(&fp);
    let h_jk_n = entropy_jackknife_nats(&fp);
    let h_py_n = entropy_pitman_yor_nats(&fp);

    let h_plugin_b = entropy_plugin_bits(&fp);
    let h_mm_b = entropy_miller_madow_bits(&fp);
    let h_jk_b = entropy_jackknife_bits(&fp);
    let h_py_b = entropy_pitman_yor_bits(&fp);

    let p0_hat = unseen_mass_good_turing(&fp);
    let s_chao1 = support_chao1(&fp);
    let py = pitman_yor_params_hat(&fp);

    println!(
        "n={} S_obs={} F1={}",
        fp.sample_size(),
        fp.observed_support(),
        fp.singletons()
    );
    println!("unseen mass (Good–Turing) p0_hat≈{:.4}", p0_hat);
    println!("support (Chao1) Ŝ≈{:.2}", s_chao1);
    println!();

    println!("entropy (nats):");
    println!("  plugin       {:>10.6}", h_plugin_n);
    println!("  Miller–Madow {:>10.6}", h_mm_n);
    println!("  jackknife    {:>10.6}", h_jk_n);
    println!(
        "  Pitman–Yor   {:>10.6}   (d={:.3}, α={:.3})",
        h_py_n, py.d, py.alpha
    );
    println!();

    println!("entropy (bits):");
    println!("  plugin       {:>10.6}", h_plugin_b);
    println!("  Miller–Madow {:>10.6}", h_mm_b);
    println!("  jackknife    {:>10.6}", h_jk_b);
    println!("  Pitman–Yor   {:>10.6}", h_py_b);
    println!();

    // LP-backed VV-style bounds (best-effort).
    let params = fingerprints::vv::SupportLpParams::default_for(&fp);
    match (
        fingerprints::vv::support_bounds_lp(&fp, params.clone()),
        fingerprints::vv::entropy_bounds_lp(&fp, params),
    ) {
        (Ok((s_lo, s_hi)), Ok((h_lo, h_hi))) => {
            println!("VV LP support bounds: [{:.3}, {:.3}]", s_lo, s_hi);
            println!("VV LP entropy bounds (nats): [{:.6}, {:.6}]", h_lo, h_hi);
        }
        (s, h) => {
            println!("VV LP bounds unavailable for this input.");
            println!("  support_bounds_lp: {:?}", s.err());
            println!("  entropy_bounds_lp: {:?}", h.err());
        }
    }
}
