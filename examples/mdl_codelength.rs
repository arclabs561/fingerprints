//! Two-part MDL codelength comparison.
//!
//! This is the executable "Kolmogorov/MDL" foothold: Kolmogorov complexity is
//! not computable, but concrete codelengths are. The example compares two
//! representations of the same sample and makes the distortion boundary
//! explicit: a compressed label representation is shorter for a coarse task,
//! but exact reconstruction must pay residual bits.
//!
//! Run: cargo run --example mdl_codelength

use fingerprints::{sample_codelen_plugin_nats, to_bits, Fingerprint};

#[derive(Clone, Copy)]
struct Representation<'a> {
    name: &'a str,
    counts: &'a [usize],
    model_bits: f64,
}

impl Representation<'_> {
    fn sample_bits(self) -> f64 {
        let fp = Fingerprint::from_counts(self.counts.iter().copied()).unwrap();
        to_bits(sample_codelen_plugin_nats(&fp))
    }

    fn total_bits(self) -> f64 {
        self.model_bits + self.sample_bits()
    }
}

fn main() {
    let exact_states = Representation {
        name: "exact states",
        counts: &[40, 37, 12, 11],
        model_bits: 16.0,
    };
    let task_labels = Representation {
        name: "task labels",
        counts: &[77, 23],
        model_bits: 18.0,
    };
    let residual_bits = residual_split_bits(&[40, 37]) + residual_split_bits(&[12, 11]);
    let task_exact_bits = task_labels.total_bits() + residual_bits;

    println!("sample: 100 observations");
    println!("representation  model bits  sample bits  task bits  exact bits");
    println!(
        "{:<14}  {:>10.1}  {:>11.1}  {:>9.1}  {:>10.1}",
        exact_states.name,
        exact_states.model_bits,
        exact_states.sample_bits(),
        exact_states.total_bits(),
        exact_states.total_bits()
    );
    println!(
        "{:<14}  {:>10.1}  {:>11.1}  {:>9.1}  {:>10.1}",
        task_labels.name,
        task_labels.model_bits,
        task_labels.sample_bits(),
        task_labels.total_bits(),
        task_exact_bits
    );
    println!("residual bits for exact reconstruction after merging: {residual_bits:.1}");

    assert!(task_labels.total_bits() < exact_states.total_bits());
    assert!(task_exact_bits > exact_states.total_bits());
}

fn residual_split_bits(counts: &[usize]) -> f64 {
    let fp = Fingerprint::from_counts(counts.iter().copied()).unwrap();
    to_bits(sample_codelen_plugin_nats(&fp))
}
