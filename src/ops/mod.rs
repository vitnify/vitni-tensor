//! Op implementations. Each submodule owns one op family; `Tensor`
//! exposes them as methods (mirroring Candle's API so model
//! definitions port mechanically).
//!
//! CPU-only at M2. M3+ routes the matmul family to `SYS_GPU_*`
//! when a GPU device is selected.

pub mod argmax;
pub mod binary;
pub mod embedding;
pub mod matmul;
pub mod quant;
pub mod rms_norm;
pub mod rope;
pub mod softmax;
pub mod unary;

/// ExCert emission mode for an op (or chain of ops). Off by default;
/// turned on per-process or per-call for audit modes.
///
/// `PerOp` mode is the most expensive — every op produces a cert
/// binding `(op_id, input_hashes, params_hash, output_hash)`. Use for
/// legal/medical/defense deployments where layer-level proof is
/// required. `PerInference` covers most production use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExCertMode {
    /// No cert emitted. Default — for hot paths where overhead matters.
    Off,
    /// One cert covering an entire `forward()` pass.
    PerInference,
    /// One cert per op. Heaviest, finest audit granularity.
    PerOp,
}

impl Default for ExCertMode {
    fn default() -> Self {
        Self::Off
    }
}
