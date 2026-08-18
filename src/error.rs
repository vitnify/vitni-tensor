//! Error type. Stays simple — no thiserror (pulls std), no anyhow.

use core::fmt;

/// All errors vitni-tensor operations can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Op expected shape A, got B. Carries enough context for debugging.
    ShapeMismatch {
        /// What the op needed.
        expected: &'static str,
        /// What it got (rendered as text — Shape isn't fixed-size).
        got: &'static str,
    },
    /// Op got a dtype it doesn't support.
    DTypeMismatch {
        /// What's accepted.
        expected: &'static str,
        /// What was passed.
        got: &'static str,
    },
    /// Op got tensors on different devices and can't bridge.
    DeviceMismatch {
        /// Op name for the trace.
        op: &'static str,
    },
    /// GPU syscall failed. Wraps the the host runtime error.
    GpuError(&'static str),
    /// Out of CPU memory (Vec::with_capacity failure surface).
    OutOfMemory,
    /// Shape has too many dimensions (we cap at `shape::MAX_RANK`).
    TooManyDims {
        /// What was requested.
        got: usize,
        /// The configured cap.
        max: usize,
    },
    /// Generic shape problem (negative dim, zero stride where invalid, etc.).
    InvalidShape(&'static str),
    /// Operation not yet implemented for the (dtype, device) combo.
    NotImplemented {
        /// Op name.
        op: &'static str,
        /// What's missing.
        why: &'static str,
    },
    /// Catch-all for invariant violations that don't map to one of
    /// the structured variants above. Used by ops/quant.rs for
    /// block-alignment + layout sanity checks.
    Internal(&'static str),
    /// Untrusted input is structurally invalid — bad magic, an unknown
    /// type tag, a value that violates the format's invariants. Used by
    /// the GGUF loader (`model/gguf.rs`) on attacker-controlled files.
    Malformed(&'static str),
    /// Untrusted input ended before a field could be fully read — a
    /// truncated header, a length or offset that runs past the end of
    /// the blob.
    Truncated(&'static str),
    /// A size or offset computation over attacker-controlled dimensions
    /// or offsets overflowed. Guarded with checked arithmetic so it
    /// surfaces as an error instead of wrapping or panicking.
    Overflow(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Error::DTypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {expected}, got {got}")
            }
            Error::DeviceMismatch { op } => write!(f, "{op}: tensors on different devices"),
            Error::GpuError(msg) => write!(f, "gpu error: {msg}"),
            Error::OutOfMemory => write!(f, "out of memory"),
            Error::TooManyDims { got, max } => write!(f, "tensor rank {got} exceeds cap {max}"),
            Error::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
            Error::NotImplemented { op, why } => write!(f, "{op}: not implemented ({why})"),
            Error::Internal(msg) => write!(f, "internal: {msg}"),
            Error::Malformed(msg) => write!(f, "malformed input: {msg}"),
            Error::Truncated(msg) => write!(f, "truncated input: {msg}"),
            Error::Overflow(msg) => write!(f, "overflow: {msg}"),
        }
    }
}

/// Convenience `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
