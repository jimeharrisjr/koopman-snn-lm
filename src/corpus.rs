//! The frozen text8 subset and split (docs/01-rung1-preregistration.md).

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Alphabet size: space and a–z.
pub const V: usize = 27;
/// The subset is the first 10,000,000 bytes of text8.
pub const SUBSET: usize = 10_000_000;
/// Train is `[0, TRAIN_END)`.
pub const TRAIN_END: usize = 9_000_000;
/// Validation is `[TRAIN_END, VALID_END)`; test is `[VALID_END, SUBSET)`.
pub const VALID_END: usize = 9_500_000;

/// Load the subset as symbols `0..27` (space = 0, `a`..`z` = 1..26).
pub fn load_subset(path: &Path) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; SUBSET];
    File::open(path)?.read_exact(&mut buf)?;
    for b in buf.iter_mut() {
        *b = match *b {
            b' ' => 0,
            b'a'..=b'z' => *b - b'a' + 1,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected byte {other} in text8 subset"),
                ))
            }
        };
    }
    Ok(buf)
}
