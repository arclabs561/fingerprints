//! Vocabulary coverage estimation: textprep tokenization -> fingerprints analysis.
//!
//! Tokenizes sample text with `textprep::tokenize::words`, counts word frequencies,
//! then uses fingerprint-based estimators to characterize vocabulary coverage:
//! how much of the underlying vocabulary the sample has likely captured.

use std::collections::HashMap;

use fingerprints::{
    coverage_good_turing, entropy_default_nats, support_chao1, to_bits, unseen_mass_good_turing,
    Fingerprint,
};

const SAMPLE_TEXT: &str = "\
Language models compress the statistical structure of text into weights. \
The compression is lossy: rare words and rare constructions are the first casualties. \
A model that has seen a billion tokens still has blind spots -- words it has never \
encountered, phrasings it cannot produce. Estimating how much vocabulary remains \
unseen after observing a corpus is a classical problem in ecology and linguistics alike. \
Good and Turing worked on this during the war, estimating the number of unseen German \
message types from intercepted traffic. Their estimator -- now called Good-Turing -- \
uses the count of singletons (words seen exactly once) as a proxy for the unseen mass. \
If many words appear only once, the sample likely covers a small fraction of the true \
vocabulary. If few words are singletons, coverage is high. \
The Chao1 estimator goes further, using both singletons and doubletons to produce a \
lower bound on the true vocabulary size. Together, these tools give a quick diagnostic: \
is our sample large enough, or is there a long tail of unseen types waiting beyond the \
horizon of our data?";

fn main() {
    // Tokenize into Unicode words (lowercased for frequency counting).
    let raw_words = textprep::tokenize::words(SAMPLE_TEXT);
    let lowered: Vec<String> = raw_words.iter().map(|w| w.to_lowercase()).collect();

    // Count per-word frequencies.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &lowered {
        *freq.entry(w.as_str()).or_default() += 1;
    }

    let counts: Vec<usize> = freq.values().copied().collect();
    let fp = Fingerprint::from_counts(counts).expect("non-empty text");

    let n = fp.sample_size();
    let s_obs = fp.observed_support();
    let f1 = fp.singletons();
    let f2 = fp.doubletons();

    let p_unseen = unseen_mass_good_turing(&fp);
    let coverage = coverage_good_turing(&fp);
    let s_chao1 = support_chao1(&fp);
    let h_bits = to_bits(entropy_default_nats(&fp));

    println!("--- Vocabulary Coverage Report ---");
    println!();
    println!("Tokens (n):              {n}");
    println!("Unique types (S_obs):    {s_obs}");
    println!("Singletons (F1):         {f1}");
    println!("Doubletons (F2):         {f2}");
    println!();
    println!("Good-Turing unseen mass: {p_unseen:.4}");
    println!("Good-Turing coverage:    {coverage:.4}");
    println!("Chao1 vocab estimate:    {s_chao1:.1}");
    println!("Entropy (bits, PY):      {h_bits:.4}");
    println!();

    // Interpret the results.
    let ratio = s_obs as f64 / s_chao1;
    println!(
        "Observed/estimated vocab: {:.1}% ({s_obs} of {s_chao1:.0} estimated types)",
        ratio * 100.0,
    );
    if p_unseen > 0.3 {
        println!("High unseen mass -- the sample covers a small fraction of the vocabulary.");
    } else if p_unseen > 0.1 {
        println!("Moderate unseen mass -- additional sampling would likely reveal new types.");
    } else {
        println!("Low unseen mass -- the sample covers most of the reachable vocabulary.");
    }
}
