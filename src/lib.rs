//! vitni-tensor — no_std tensor framework for a verifiable runtime.
//!
//! See `../DESIGN.md` for the architectural rationale. Short version:
//! purpose-built no_std framework that inherits the runtime's invariants
//! (capabilities, event-sourcing, ExCert-per-op) end-to-end, rather
//! than retrofitting them onto a `std`-dependent framework like
//! Candle.
//!
//! # Status
//!
//! Milestone 2: CPU-only ops landed for binary (add/sub/mul/div),
//! matmul (rank-2 only), unary (silu/gelu/recip), softmax (last-dim),
//! rms_norm, embedding lookup, rope. Enough to walk a single
//! transformer layer manually.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod accel;
pub mod cert;
pub mod device;
pub mod dtype;
pub mod error;
pub mod model;
pub mod ops;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod tokenizer;

#[cfg(any(test, feature = "std-parallel"))]
extern crate std;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use shape::Shape;
pub use storage::Storage;
pub use tensor::Tensor;
