//! Input encodings: each character becomes a fixed set of input lines that
//! spike on that character's step. Three arms (frozen ladder, rung 3):
//! one-hot, sparse k-of-n, and articulatory features.

use crate::corpus::V;
use crate::reservoir::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// 27 lines, one active.
    OneHot,
    /// 32 lines, 4 active; codes drawn once from a fixed seed, all distinct.
    Sparse,
    /// 22 hand-written articulatory feature lines (see [`ARTIC_FEATURES`]).
    Artic,
}

pub const SPARSE_N: usize = 32;
pub const SPARSE_K: usize = 4;
pub const SPARSE_SEED: u64 = 0x5EED_C0DE;

pub const ARTIC_FEATURES: [&str; 22] = [
    "space", "letter", "vowel", "consonant", "voiced", "stop", "fricative", "nasal", "liquid",
    "glide", "affricate", "labial", "alveolar", "palatal", "velar", "glottal", "high", "low",
    "front", "back", "rounded", "sibilant",
];

/// Features of the most usual English sound of each letter. A letter is not
/// a phoneme; this is a deliberate, fixed approximation.
const ARTIC: [(char, &[&str]); 27] = [
    (' ', &["space"]),
    ('a', &["letter", "vowel", "voiced", "low", "back"]),
    ('b', &["letter", "consonant", "voiced", "stop", "labial"]),
    ('c', &["letter", "consonant", "stop", "velar", "alveolar"]),
    ('d', &["letter", "consonant", "voiced", "stop", "alveolar"]),
    ('e', &["letter", "vowel", "voiced", "front"]),
    ('f', &["letter", "consonant", "fricative", "labial"]),
    ('g', &["letter", "consonant", "voiced", "stop", "velar"]),
    ('h', &["letter", "consonant", "fricative", "glottal"]),
    ('i', &["letter", "vowel", "voiced", "high", "front"]),
    ('j', &["letter", "consonant", "voiced", "affricate", "palatal"]),
    ('k', &["letter", "consonant", "stop", "velar"]),
    ('l', &["letter", "consonant", "voiced", "liquid", "alveolar"]),
    ('m', &["letter", "consonant", "voiced", "nasal", "labial"]),
    ('n', &["letter", "consonant", "voiced", "nasal", "alveolar"]),
    ('o', &["letter", "vowel", "voiced", "back", "rounded"]),
    ('p', &["letter", "consonant", "stop", "labial"]),
    ('q', &["letter", "consonant", "stop", "velar", "rounded"]),
    ('r', &["letter", "consonant", "voiced", "liquid", "palatal"]),
    ('s', &["letter", "consonant", "fricative", "alveolar", "sibilant"]),
    ('t', &["letter", "consonant", "stop", "alveolar"]),
    ('u', &["letter", "vowel", "voiced", "high", "back", "rounded"]),
    ('v', &["letter", "consonant", "voiced", "fricative", "labial"]),
    ('w', &["letter", "consonant", "voiced", "glide", "labial", "rounded"]),
    ('x', &["letter", "consonant", "stop", "fricative", "velar", "alveolar", "sibilant"]),
    ('y', &["letter", "consonant", "voiced", "glide", "palatal", "high", "front"]),
    ('z', &["letter", "consonant", "voiced", "fricative", "alveolar", "sibilant"]),
];

impl Encoding {
    pub fn name(&self) -> &'static str {
        match self {
            Encoding::OneHot => "one-hot",
            Encoding::Sparse => "sparse 4-of-32",
            Encoding::Artic => "articulatory",
        }
    }
    pub fn d_in(&self) -> usize {
        match self {
            Encoding::OneHot => V,
            Encoding::Sparse => SPARSE_N,
            Encoding::Artic => ARTIC_FEATURES.len(),
        }
    }
    /// Active input lines per character, ascending.
    pub fn codes(&self) -> Vec<Vec<u32>> {
        match self {
            Encoding::OneHot => (0..V as u32).map(|c| vec![c]).collect(),
            Encoding::Sparse => {
                let mut rng = Rng::new(SPARSE_SEED);
                let mut out: Vec<Vec<u32>> = Vec::new();
                while out.len() < V {
                    let mut code: Vec<u32> = Vec::new();
                    while code.len() < SPARSE_K {
                        let j = (rng.next_u64() % SPARSE_N as u64) as u32;
                        if !code.contains(&j) {
                            code.push(j);
                        }
                    }
                    code.sort_unstable();
                    if !out.contains(&code) {
                        out.push(code);
                    }
                }
                out
            }
            Encoding::Artic => ARTIC
                .iter()
                .map(|(_, feats)| {
                    let mut code: Vec<u32> = feats
                        .iter()
                        .map(|f| ARTIC_FEATURES.iter().position(|g| g == f).expect("feature") as u32)
                        .collect();
                    code.sort_unstable();
                    code
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_in_range() {
        for enc in [Encoding::OneHot, Encoding::Sparse, Encoding::Artic] {
            let codes = enc.codes();
            assert_eq!(codes.len(), V);
            for (a, ca) in codes.iter().enumerate() {
                assert!(!ca.is_empty());
                assert!(ca.iter().all(|&j| (j as usize) < enc.d_in()));
                for cb in &codes[a + 1..] {
                    assert_ne!(ca, cb, "{} has a duplicate code", enc.name());
                }
            }
        }
    }

    #[test]
    fn artic_table_is_in_corpus_order() {
        assert_eq!(ARTIC[0].0, ' ');
        for (i, (ch, _)) in ARTIC.iter().enumerate().skip(1) {
            assert_eq!(*ch as u8, b'a' + i as u8 - 1);
        }
    }
}
