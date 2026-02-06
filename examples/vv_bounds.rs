fn main() {
    use unseen::vv::{entropy_bounds_lp, support_bounds_lp, SupportLpParams};
    use unseen::Fingerprint;

    let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
    let fp = Fingerprint::from_counts(counts).unwrap();
    let params = SupportLpParams::default_for(&fp);

    let (s_lo, s_hi) = support_bounds_lp(&fp, params.clone()).unwrap();
    let (h_lo, h_hi) = entropy_bounds_lp(&fp, params).unwrap();

    assert!(s_hi >= s_lo);
    assert!(h_hi >= h_lo);

    println!("VV LP support bounds: [{:.3}, {:.3}]", s_lo, s_hi);
    println!("VV LP entropy bounds (nats): [{:.6}, {:.6}]", h_lo, h_hi);
}

