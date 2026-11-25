# koopman-snn-lm

Can a spiking language model built on the Koopman/DMD linear-plus-control
formulation run with much less compute and power than a conventional one?

This repository holds that investigation. It depends on the
[`kdmd-snn`](https://github.com/jimeharrisjr/koopman-based-SNN) crate for
neuron dynamics, identification and training.

## Status

**Scaffold only. No experiments have been run and none are authorized.**
The first commit exists to put the frozen decisions on record before any
result does.

## Goal

- **Goal:** beat a matched GRU on quality per operation — bits per character
  against energy-weighted operations per character.
- **Stretch goal:** match SpikeGPT's enwik8 result (1.262 bits per character
  at 46M parameters) at a smaller size.

The full set of frozen decisions, the experiment ladder, and what is still
to be fixed before a registered run are in
[docs/00-frozen-decisions.md](docs/00-frozen-decisions.md).

## Discipline

Inherited from `kdmd-snn`: a protocol is committed before the run it governs,
kill criteria are frozen in advance, and negative results are kept. Frozen
decisions are not tuned after results arrive.

## Building

Builds standalone: `kdmd-snn` is fetched from GitHub at the revision pinned
in `Cargo.toml`. Rust 1.85+.

```sh
cargo check
```
