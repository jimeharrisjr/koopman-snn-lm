//! Next-token language modelling with Koopman-DMD spiking neural networks.
//!
//! The plan, the success criteria and the decisions frozen before any result
//! are in `docs/00-frozen-decisions.md`. Each rung is a binary under
//! `src/bin/`, governed by its own pre-registration in `docs/`.
//!
//! Neuron dynamics come from the [`kdmd_snn`] crate; anything that crate
//! itself needs is contributed upstream instead of being reimplemented here.

pub use kdmd_snn;

pub mod corpus;
pub mod reservoir;
pub mod common;
