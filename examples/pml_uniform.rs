use fingerprints::pml::best_uniform_support_size;

fn main() {
    // Observed per-symbol counts (unlabeled).
    let counts = [5usize, 4, 3, 2, 2, 1, 1, 1];
    let m = counts.len();

    // Search support sizes from m..=m+200 and pick the best uniform-family profile likelihood.
    let (s_hat, ll) = best_uniform_support_size(&counts, m + 200).unwrap();
    assert!(s_hat >= m);
    assert!(ll.is_finite());

    println!(
        "best_uniform_support_size: Ŝ={} (observed distinct m={}) ll={:.4}",
        s_hat, m, ll
    );
}
