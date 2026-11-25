//! Next-token language modelling with Koopman-DMD spiking neural networks.
//!
//! This crate is a scaffold. No model, training or evaluation code exists
//! yet, and no experiment is authorized. The plan, the success criteria and
//! the decisions frozen before any result are in `docs/00-frozen-decisions.md`.
//!
//! Neuron dynamics come from the [`kdmd_snn`] crate; anything that crate
//! itself needs is contributed upstream instead of being reimplemented here.

pub use kdmd_snn;
