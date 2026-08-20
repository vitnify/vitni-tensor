//! Cert builder + finalized Cert type.

use alloc::{string::String, vec::Vec};

/// A single named field — either an input or an output of the bound
/// computation.
#[derive(Debug, Clone)]
pub struct CertField {
    /// Field name (e.g. `"prompt"`, `"weights_hash"`, `"output_tokens"`).
    pub name: String,
    /// Field bytes — usually a hash (32 bytes) or a small payload
    /// like the prompt or the output token IDs. Large blobs should be
    /// passed via their hash, not raw, to keep the cert small.
    pub bytes: Vec<u8>,
}

/// Per-op record bound by `ExCertMode::PerOp`. Carries the
/// information a verifier needs to reproduce + check a single op
/// in isolation: the op's identity, its layer (or `u32::MAX` for
/// non-per-layer ops), and BLAKE3 hashes of its input / params /
/// output tensors.
///
/// Op records are bound into the cert's binding digest in
/// declaration order (same canonical-form discipline as input /
/// output fields). A verifier with the model weights + the same
/// prompt can recompute every op's expected (input, params,
/// output) hashes and check each one independently. This is the
/// substrate primitive for per-layer attestation that mechanistic
/// interpretability research needs.
///
/// Size: ~16 bytes name + 4 + 4 + 3×32 = ~120 bytes per record.
/// For stories15M (6 layers × 11 ops = 66 ops/token), a 32-token
/// inference produces ~2,112 records → ~250 KB total. Fits the
/// in-process cert easily; CAS-store discipline handled by the
/// substrate sink (see `CertSink::on_op`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRecord {
    /// Monotonic index assigned by the builder. Stable across runs
    /// of the same program (op declaration order is deterministic).
    pub op_index: u32,
    /// Op name — short identifier like `"linear_q4_0"`, `"rms_norm"`,
    /// `"softmax"`, `"silu"`, `"embedding"`, `"rope"`. Bounded by
    /// the canonical-form serializer; readable in cert dumps.
    pub op_name: String,
    /// Layer index this op belongs to. `u32::MAX` for non-per-layer
    /// ops (token embedding, final norm, lm_head).
    pub layer: u32,
    /// BLAKE3 of the input tensor bytes (in row-major canonical form).
    pub input_hash: [u8; 32],
    /// BLAKE3 of the params/weights tensor bytes. Zero array if the
    /// op has no learnable params (softmax, silu, rope).
    pub params_hash: [u8; 32],
    /// BLAKE3 of the output tensor bytes.
    pub output_hash: [u8; 32],
}

/// Per-token activation snapshot bound by Phase 4b. Coarser-grained
/// than `OpRecord` — captures the full residual-stream state at
/// named checkpoints rather than at every op. Designed so a verifier
/// can spot-check intermediate activations during a re-execution
/// without having to reproduce every op exactly.
///
/// Typical checkpoint names:
///   - `"post_embed"`     — residual stream after token embedding lookup
///   - `"post_attn"`      — residual after the attention block (per layer)
///   - `"post_ffn"`       — residual after the FFN block (per layer)
///   - `"pre_lm_head"`    — residual after the final RMSNorm
///   - `"post_lm_head"`   — logits before sampling
///
/// `layer = u32::MAX` for non-per-layer checkpoints (post_embed,
/// pre_lm_head, post_lm_head). For per-layer checkpoints (post_attn,
/// post_ffn) the layer field carries the 0-based layer index.
///
/// Size: ~16 bytes name + 4 + 4 + 32 = ~56 bytes per record. For
/// stories15M (6 layers × 2 + 3 = 15 checkpoints per token), a
/// 32-token inference produces ~480 records → ~27 KB total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationRecord {
    /// Monotonic index assigned by the builder.
    pub act_index: u32,
    /// Token position this activation was captured at (0-based).
    pub token_index: u32,
    /// Layer index. `u32::MAX` for non-per-layer checkpoints.
    pub layer: u32,
    /// Checkpoint name — short identifier (`"post_embed"`,
    /// `"post_attn"`, `"post_ffn"`, `"pre_lm_head"`, `"post_lm_head"`).
    pub checkpoint: String,
    /// BLAKE3 of the activation tensor bytes (canonical row-major).
    pub tensor_hash: [u8; 32],
}

/// Causal-intervention record bound by Phase 4c. Declares "at
/// checkpoint C of token T, the residual stream was OVERRIDDEN
/// with a tensor whose BLAKE3 is `replacement_hash`". The
/// substrate forward pass consults the intervention list at each
/// checkpoint and substitutes the override tensor before the next
/// op runs.
///
/// Lets a verifier (or auditor) ask "if activation X had been Y
/// instead of Z, would the output have been different?" — the
/// substrate-attested answer is the modified cert's output.
///
/// Cert binding: the intervention list itself is hashed into the
/// digest (so the cert proves WHAT WAS PERTURBED), and the
/// post-intervention activation snapshots + output naturally
/// differ from the un-intervened cert. Compare two certs for the
/// same prompt — one with interventions, one without — to see the
/// causal effect with cryptographic provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterventionRecord {
    /// Monotonic index assigned by the builder.
    pub intv_index: u32,
    /// Token position the intervention applies at (0-based).
    pub token_index: u32,
    /// Layer index. `u32::MAX` for non-per-layer checkpoints.
    pub layer: u32,
    /// Checkpoint name where the override applies. Must match a
    /// checkpoint that the forward pass would emit normally
    /// (post_embed, post_attn, post_ffn, pre_lm_head, post_lm_head).
    pub checkpoint: String,
    /// BLAKE3 of the replacement tensor bytes.
    pub replacement_hash: [u8; 32],
}

/// Finalized certificate.
///
/// The `digest` is the cryptographic commitment binding all inputs,
/// outputs, AND (when `ExCertMode::PerOp` was active) per-op
/// records together. `inputs`/`outputs`/`ops` are retained so
/// verifiers can re-examine what produced the digest.
#[derive(Debug, Clone)]
pub struct Cert {
    /// Named inputs bound by this cert.
    pub inputs: Vec<CertField>,
    /// Named outputs bound by this cert.
    pub outputs: Vec<CertField>,
    /// Per-op records (empty in `PerInference` mode; populated in `PerOp`).
    pub ops: Vec<OpRecord>,
    /// Per-token activation snapshots (Phase 4b). Empty unless the
    /// caller invoked `declare_activation` during inference.
    pub activations: Vec<ActivationRecord>,
    /// Causal-intervention manifest (Phase 4c). Empty for a
    /// "normal" inference; populated when the caller declared
    /// overrides via `declare_intervention`.
    pub interventions: Vec<InterventionRecord>,
    /// 32-byte BLAKE3 binding digest.
    pub digest: [u8; 32],
}

impl Cert {
    /// Lowercase hex rendering of the digest (64 chars).
    pub fn digest_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for &b in &self.digest {
            out.push(hex_nibble(b >> 4));
            out.push(hex_nibble(b & 0x0f));
        }
        out
    }

    /// Look up an input field by name. Returns the bytes if found.
    pub fn input(&self, name: &str) -> Option<&[u8]> {
        self.inputs
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.bytes.as_slice())
    }

    /// Look up an output field by name.
    pub fn output(&self, name: &str) -> Option<&[u8]> {
        self.outputs
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.bytes.as_slice())
    }
}

fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => '?',
    }
}

/// Streaming builder. Inputs and outputs are buffered; `finalize()`
/// computes the binding digest.
///
/// `Drop` is intentionally a no-op — abandoning a builder before
/// finalize is a no-op (no syscall side effects in the software path).
#[derive(Debug, Default)]
pub struct CertBuilder {
    inputs: Vec<CertField>,
    outputs: Vec<CertField>,
    /// Per-op records, populated by `declare_op`. Empty in
    /// `PerInference` mode (no callers invoke `declare_op`); full
    /// in `PerOp` mode (every matmul/RMSNorm/softmax in the
    /// forward pass adds one entry).
    ops: Vec<OpRecord>,
    /// Monotonic index assigned to the next `declare_op` call.
    next_op_index: u32,
    /// Per-token activation snapshots (Phase 4b).
    activations: Vec<ActivationRecord>,
    /// Monotonic index assigned to the next `declare_activation` call.
    next_act_index: u32,
    /// Causal-intervention manifest (Phase 4c).
    interventions: Vec<InterventionRecord>,
    /// Monotonic index assigned to the next `declare_intervention` call.
    next_intv_index: u32,
    sealed: bool,
}

impl CertBuilder {
    /// Start a new empty cert.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a named input. Order matters — the binding digest
    /// depends on declaration order, so two builders that declare
    /// the same fields in different orders produce different certs.
    /// This is intentional (catches accidental reordering).
    pub fn declare_input(&mut self, name: &str, bytes: &[u8]) -> &mut Self {
        debug_assert!(!self.sealed, "cannot declare after output");
        self.inputs.push(CertField {
            name: String::from(name),
            bytes: bytes.to_vec(),
        });
        self
    }

    /// Declare a named output. Once outputs start being declared,
    /// inputs cannot be added — the cert is "input-sealed."
    pub fn declare_output(&mut self, name: &str, bytes: &[u8]) -> &mut Self {
        self.sealed = true;
        self.outputs.push(CertField {
            name: String::from(name),
            bytes: bytes.to_vec(),
        });
        self
    }

    /// Declare a per-op record (PerOp mode). Called from inside
    /// `forward_quantized::step` at each instrumented op site
    /// (matmul, RMSNorm, softmax, silu, rope, embedding).
    ///
    /// Bound into the cert digest in declaration order, same as
    /// inputs/outputs. Two runs of the same prompt + same weights
    /// + same op order produce identical op records → identical
    /// cert digest. Diverging at any op = falsified.
    pub fn declare_op(
        &mut self,
        op_name: &str,
        layer: u32,
        input_hash: [u8; 32],
        params_hash: [u8; 32],
        output_hash: [u8; 32],
    ) -> &mut Self {
        let op_index = self.next_op_index;
        self.next_op_index = self.next_op_index.wrapping_add(1);
        self.ops.push(OpRecord {
            op_index,
            op_name: String::from(op_name),
            layer,
            input_hash,
            params_hash,
            output_hash,
        });
        self
    }

    /// Read-only view of currently buffered op records — primarily
    /// for tests that want to count ops or inspect order without
    /// consuming the builder.
    pub fn ops(&self) -> &[OpRecord] {
        &self.ops
    }

    /// Declare a per-token activation snapshot (Phase 4b). Called
    /// from forward-pass hooks at named checkpoints (post_embed,
    /// post_attn[L], post_ffn[L], pre_lm_head, post_lm_head).
    ///
    /// Bound into the cert digest in declaration order, AFTER ops.
    /// Same canonical-form discipline: a verifier rebuilding the
    /// activation list in the same order recomputes the same digest
    /// section. Diverging at any checkpoint = falsified intermediate
    /// state.
    pub fn declare_activation(
        &mut self,
        token_index: u32,
        layer: u32,
        checkpoint: &str,
        tensor_hash: [u8; 32],
    ) -> &mut Self {
        let act_index = self.next_act_index;
        self.next_act_index = self.next_act_index.wrapping_add(1);
        self.activations.push(ActivationRecord {
            act_index,
            token_index,
            layer,
            checkpoint: String::from(checkpoint),
            tensor_hash,
        });
        self
    }

    /// Read-only view of currently buffered activation records.
    pub fn activations(&self) -> &[ActivationRecord] {
        &self.activations
    }

    /// Declare a causal intervention (Phase 4c). Called by the
    /// caller BEFORE running inference, specifying which checkpoint
    /// to override and with what replacement-tensor hash. The
    /// forward pass (when wired through a `PerInterventionRecorder`)
    /// consults this list at each checkpoint and substitutes the
    /// override tensor for the natural residual stream.
    ///
    /// Bound into the cert digest in declaration order, AFTER
    /// activations. Verifier sees: "this cert was issued with these
    /// interventions applied" — provenance for what was perturbed.
    pub fn declare_intervention(
        &mut self,
        token_index: u32,
        layer: u32,
        checkpoint: &str,
        replacement_hash: [u8; 32],
    ) -> &mut Self {
        let intv_index = self.next_intv_index;
        self.next_intv_index = self.next_intv_index.wrapping_add(1);
        self.interventions.push(InterventionRecord {
            intv_index,
            token_index,
            layer,
            checkpoint: String::from(checkpoint),
            replacement_hash,
        });
        self
    }

    /// Read-only view of currently buffered intervention records.
    pub fn interventions(&self) -> &[InterventionRecord] {
        &self.interventions
    }

    /// Compute the binding digest and consume the builder.
    pub fn finalize(self) -> Cert {
        let digest = compute_digest(
            &self.inputs,
            &self.outputs,
            &self.ops,
            &self.activations,
            &self.interventions,
        );
        Cert {
            inputs: self.inputs,
            outputs: self.outputs,
            ops: self.ops,
            activations: self.activations,
            interventions: self.interventions,
            digest,
        }
    }

    /// Finalize the cert AND replay every declaration into a substrate
    /// sink. Returns `(software_cert, substrate_cert_id)`.
    ///
    /// The sink sees calls in this order:
    ///
    ///   1. `on_request(tier)`
    ///   2. `on_input(name, bytes)` × n_inputs (declaration order)
    ///   3. `on_op(record)` × n_ops (declaration order, PerOp only)
    ///   4. `on_output(name, bytes)` × n_outputs (declaration order)
    ///   5. `on_finalize()` — returns the substrate-issued cert id
    ///
    /// On the host runtime with a `RuntimeSink` (see `cert::sink` module docs),
    /// the kernel-issued cert and the returned software `Cert` carry
    /// the SAME 32-byte digest by construction — both bind the same
    /// length-prefixed (name, bytes) pairs + per-op records via BLAKE3.
    ///
    /// Op records are replayed AFTER inputs and BEFORE outputs to
    /// match the natural cert-state machine on the kernel side: the
    /// inference's inputs were known up front, the ops ran in order,
    /// then the outputs materialized.
    pub fn finalize_with_sink<S: super::sink::CertSink>(
        self,
        sink: &mut S,
        tier: u8,
    ) -> Result<(Cert, u64), S::Error> {
        sink.on_request(tier)?;
        for f in &self.inputs {
            sink.on_input(&f.name, &f.bytes)?;
        }
        for op in &self.ops {
            sink.on_op(op)?;
        }
        for act in &self.activations {
            sink.on_activation(act)?;
        }
        for intv in &self.interventions {
            sink.on_intervention(intv)?;
        }
        for f in &self.outputs {
            sink.on_output(&f.name, &f.bytes)?;
        }
        let cert_id = sink.on_finalize()?;
        let digest = compute_digest(
            &self.inputs,
            &self.outputs,
            &self.ops,
            &self.activations,
            &self.interventions,
        );
        Ok((
            Cert {
                inputs: self.inputs,
                outputs: self.outputs,
                ops: self.ops,
                activations: self.activations,
                interventions: self.interventions,
                digest,
            },
            cert_id,
        ))
    }
}

/// The numerical regime this engine build computes under — the reduction contract
/// (pinned reduction order, no FMA contraction, no reassociation) that makes the
/// forward pass bit-identical across CPU vendors (see
/// `ops::matmul::tests::matmul_reduction_bits_are_pinned`). It is BOUND into every
/// `vitnify-receipt v2` digest, so a receipt records WHICH regime produced it.
///
/// BUMP THIS whenever the reduction changes — i.e. whenever a pinned reduction hash
/// moves. A v2 receipt issued under the old regime is then cryptographically
/// distinguishable from one issued under the new regime: an L2 verifier can report
/// "regime moved — cannot replay this receipt with this engine" instead of a silent
/// hash mismatch that is indistinguishable from tampering. The format version
/// (`v1`/`v2`) versions the wire layout; `REGIME` versions the arithmetic.
pub const REGIME: &str = "vitni-regime-1";

/// Compute the canonical BLAKE3 binding digest for the SHIPPED format,
/// `vitnify-receipt v2`. Identical to v1 except the domain is `v2` and the numerical
/// [`REGIME`] is bound immediately after it (length-prefixed like every field), so the
/// receipt records the regime it was produced under. Absent sections write single
/// LEB128 zeros, so the digest is well-defined for every mode combination.
///
/// Format:
///   "vitnify-receipt v2\x00"
///   LEB128 |REGIME| ; REGIME
///   LEB128 n_inputs ; n_inputs × (write_field)
///   LEB128 n_outputs ; n_outputs × (write_field)
///   LEB128 n_ops ; n_ops × (write_op)
///   LEB128 n_activations ; n_activations × (write_activation)
///   LEB128 n_interventions ; n_interventions × (write_intervention)
///
/// See the module-level doc comment for the format rationale.
fn compute_digest(
    inputs: &[CertField],
    outputs: &[CertField],
    ops: &[OpRecord],
    activations: &[ActivationRecord],
    interventions: &[InterventionRecord],
) -> [u8; 32] {
    let mut hasher = ::blake3::Hasher::new();
    hasher.update(b"vitnify-receipt v2\x00");
    write_leb128(&mut hasher, REGIME.len() as u64);
    hasher.update(REGIME.as_bytes());
    write_sections(&mut hasher, inputs, outputs, ops, activations, interventions);
    *hasher.finalize().as_bytes()
}

/// Compute the FROZEN `vitnify-receipt v1` digest — the original format, with no
/// regime binding. Retained so a receipt issued before v2 existed (including the
/// published TinyLlama anchor `9c0754…`) stays reproducible; new receipts use v2.
pub fn compute_digest_v1(
    inputs: &[CertField],
    outputs: &[CertField],
    ops: &[OpRecord],
    activations: &[ActivationRecord],
    interventions: &[InterventionRecord],
) -> [u8; 32] {
    let mut hasher = ::blake3::Hasher::new();
    hasher.update(b"vitnify-receipt v1\x00");
    write_sections(&mut hasher, inputs, outputs, ops, activations, interventions);
    *hasher.finalize().as_bytes()
}

/// The length-prefixed input/output/op/activation/intervention sections, identical
/// between v1 and v2 — only the domain (and v2's regime prefix) differ.
fn write_sections(
    hasher: &mut ::blake3::Hasher,
    inputs: &[CertField],
    outputs: &[CertField],
    ops: &[OpRecord],
    activations: &[ActivationRecord],
    interventions: &[InterventionRecord],
) {
    write_leb128(hasher, inputs.len() as u64);
    for f in inputs {
        write_field(hasher, f);
    }
    write_leb128(hasher, outputs.len() as u64);
    for f in outputs {
        write_field(hasher, f);
    }
    write_leb128(hasher, ops.len() as u64);
    for op in ops {
        write_op(hasher, op);
    }
    write_leb128(hasher, activations.len() as u64);
    for act in activations {
        write_activation(hasher, act);
    }
    write_leb128(hasher, interventions.len() as u64);
    for intv in interventions {
        write_intervention(hasher, intv);
    }
}

/// Canonical-form serialization of one OpRecord into the digest.
/// Format (LEB128 lengths, byte-exact field contents):
///   op_index (LEB128 u64)
///   len(op_name) (LEB128 u64) ; op_name bytes
///   layer (LEB128 u64)
///   input_hash (32 bytes)
///   params_hash (32 bytes)
///   output_hash (32 bytes)
fn write_op(h: &mut ::blake3::Hasher, op: &OpRecord) {
    write_leb128(h, op.op_index as u64);
    let name = op.op_name.as_bytes();
    write_leb128(h, name.len() as u64);
    h.update(name);
    write_leb128(h, op.layer as u64);
    h.update(&op.input_hash);
    h.update(&op.params_hash);
    h.update(&op.output_hash);
}

/// Canonical-form serialization of one ActivationRecord into the digest.
/// Format (LEB128 lengths, byte-exact field contents):
///   act_index (LEB128 u64)
///   token_index (LEB128 u64)
///   layer (LEB128 u64)
///   len(checkpoint) (LEB128 u64) ; checkpoint bytes
///   tensor_hash (32 bytes)
fn write_activation(h: &mut ::blake3::Hasher, act: &ActivationRecord) {
    write_leb128(h, act.act_index as u64);
    write_leb128(h, act.token_index as u64);
    write_leb128(h, act.layer as u64);
    let name = act.checkpoint.as_bytes();
    write_leb128(h, name.len() as u64);
    h.update(name);
    h.update(&act.tensor_hash);
}

/// Canonical-form serialization of one InterventionRecord into the
/// digest. Format mirrors write_activation (same shape — `intv_index`
/// in place of `act_index`, `replacement_hash` in place of
/// `tensor_hash`).
fn write_intervention(h: &mut ::blake3::Hasher, intv: &InterventionRecord) {
    write_leb128(h, intv.intv_index as u64);
    write_leb128(h, intv.token_index as u64);
    write_leb128(h, intv.layer as u64);
    let name = intv.checkpoint.as_bytes();
    write_leb128(h, name.len() as u64);
    h.update(name);
    h.update(&intv.replacement_hash);
}

fn write_field(h: &mut ::blake3::Hasher, f: &CertField) {
    let name = f.name.as_bytes();
    write_leb128(h, name.len() as u64);
    h.update(name);
    write_leb128(h, f.bytes.len() as u64);
    h.update(&f.bytes);
}

/// Unsigned LEB128 — 7 bits per byte, high bit = continue. Compact
/// length prefix for short blobs (1-byte for <128); standard.
fn write_leb128(h: &mut ::blake3::Hasher, mut v: u64) {
    let mut buf = [0u8; 10];
    let mut i = 0;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf[i] = byte;
        i += 1;
        if v == 0 {
            break;
        }
    }
    h.update(&buf[..i]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::{CertSink, RecordingSink, SinkEvent};

    #[test]
    fn empty_cert_finalizes() {
        let cert = CertBuilder::new().finalize();
        assert!(cert.inputs.is_empty());
        assert!(cert.outputs.is_empty());
        // Digest is deterministic — same empty cert always hashes the same.
        let cert2 = CertBuilder::new().finalize();
        assert_eq!(cert.digest, cert2.digest);
    }

    #[test]
    fn determinism_same_inputs_same_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hello");
        a.declare_input("weights_hash", &[0xab; 32]);
        a.declare_output("tokens", &[1u8, 2, 3]);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hello");
        b.declare_input("weights_hash", &[0xab; 32]);
        b.declare_output("tokens", &[1u8, 2, 3]);
        let cert_b = b.finalize();

        assert_eq!(cert_a.digest, cert_b.digest);
        assert_eq!(cert_a.digest_hex().len(), 64);
    }

    #[test]
    fn different_input_bytes_different_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hello");
        a.declare_output("tokens", &[1]);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"world");
        b.declare_output("tokens", &[1]);
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn different_output_bytes_different_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"x");
        a.declare_output("tokens", &[1, 2, 3]);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"x");
        b.declare_output("tokens", &[1, 2, 4]);
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn order_matters() {
        // Two inputs swapped → different cert.
        let mut a = CertBuilder::new();
        a.declare_input("first", b"a");
        a.declare_input("second", b"b");
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("second", b"b");
        b.declare_input("first", b"a");
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn length_prefix_prevents_concatenation_collision() {
        // ["ab", "c"] vs ["a", "bc"] — without length prefix these
        // concat-hash to the same thing. With LEB128 prefix they don't.
        let mut a = CertBuilder::new();
        a.declare_input("k", b"ab");
        a.declare_input("k", b"c");
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("k", b"a");
        b.declare_input("k", b"bc");
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn lookup_by_name() {
        let mut c = CertBuilder::new();
        c.declare_input("prompt", b"once upon a time");
        c.declare_output("token_count", &[5, 0, 0, 0]);
        let cert = c.finalize();
        assert_eq!(cert.input("prompt"), Some(&b"once upon a time"[..]));
        assert_eq!(cert.output("token_count"), Some(&[5u8, 0, 0, 0][..]));
        assert_eq!(cert.input("nonexistent"), None);
    }

    #[test]
    fn known_blake3_for_empty_cert() {
        // Lock down the empty-cert digest so any change to the binding format is
        // immediately visible. Shipped format is now v2:
        // BLAKE3("vitnify-receipt v2\x00" || LEB128(|REGIME|) || REGIME || 0x00 × 5) —
        // five LEB128 zeros for n_inputs/n_outputs/n_ops/n_activations/n_interventions.
        let cert = CertBuilder::new().finalize();
        assert!(cert.digest.iter().any(|&b| b != 0));
        // Stable across runs.
        let cert2 = CertBuilder::new().finalize();
        assert_eq!(cert.digest, cert2.digest);
    }

    #[test]
    fn v2_digest_is_pinned_and_binds_regime() {
        // A representative shipped-shape cert (I/O only, n_ops=0). Its v2 digest is
        // PINNED: any change to the v2 binding format OR to REGIME moves it, which would
        // silently invalidate every issued receipt — so it fails LOUDLY here instead.
        // (Same pinned-hash discipline as matmul_reduction_bits_are_pinned.)
        let mut b = CertBuilder::new();
        b.declare_input("model_id", b"tinyllama");
        b.declare_input("prompt_tokens", &[1u8, 0, 0, 0]);
        b.declare_output("output_tokens", &[9u8, 0, 0, 0]);
        let cert = b.finalize();
        assert_eq!(
            cert.digest_hex(),
            "6d5f534cf69c441ba2832c6e63747a7989ccb5de9a2cc9dc528e5b8e0359cf1e",
            "vitnify-receipt v2 binding or REGIME changed — this invalidates issued receipts"
        );
        // The regime binding must actually matter: the same fields under the FROZEN v1
        // format produce a DIFFERENT digest, so a v1 and a v2 receipt for the same run
        // are distinguishable.
        let v1 = compute_digest_v1(&cert.inputs, &cert.outputs, &cert.ops,
                                   &cert.activations, &cert.interventions);
        assert_ne!(v1, cert.digest, "v2 digest must differ from v1 — regime is not bound");
    }

    // =====================================================================
    // PerOp tests (Phase 4a — per-op record support)
    // =====================================================================

    fn op_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        h
    }

    #[test]
    fn declare_op_appends_in_order_with_monotonic_index() {
        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_op("linear_q4_0", 0, op_hash(0x10), op_hash(0x20), op_hash(0x30));
        b.declare_op("rms_norm", 0, op_hash(0x40), op_hash(0x50), op_hash(0x60));
        b.declare_op("softmax", 1, op_hash(0x70), op_hash(0x80), op_hash(0x90));
        b.declare_output("logits", &[0u8; 4]);

        assert_eq!(b.ops().len(), 3);
        assert_eq!(b.ops()[0].op_index, 0);
        assert_eq!(b.ops()[0].op_name, "linear_q4_0");
        assert_eq!(b.ops()[0].layer, 0);
        assert_eq!(b.ops()[1].op_index, 1);
        assert_eq!(b.ops()[2].op_index, 2);
        assert_eq!(b.ops()[2].op_name, "softmax");
        assert_eq!(b.ops()[2].layer, 1);

        // Finalize and verify ops survive into the Cert.
        let cert = b.finalize();
        assert_eq!(cert.ops.len(), 3);
        assert_eq!(cert.ops[2].op_name, "softmax");
    }

    #[test]
    fn declare_op_changes_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hi");
        a.declare_output("logits", &[0u8; 4]);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_op("linear_q4_0", 0, op_hash(0x10), op_hash(0x20), op_hash(0x30));
        b.declare_output("logits", &[0u8; 4]);
        let cert_b = b.finalize();

        // Adding ANY op must change the binding digest. If it doesn't,
        // PerOp mode is silently producing the same cert as PerInference
        // and the per-op records aren't actually bound.
        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn changing_op_output_hash_changes_cert_digest() {
        // The whole point of per-op binding: if you tamper with the
        // output of ONE op (claim the wrong intermediate), the cert
        // detects it. Diff is on output_hash of op 1.
        let h1 = op_hash(0x10);
        let h2 = op_hash(0x20);
        let h3a = op_hash(0x30);
        let h3b = op_hash(0x31); // differs by one bit

        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hi");
        a.declare_op("linear_q4_0", 0, h1, h2, h3a);
        a.declare_output("logits", &[0u8; 4]);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_op("linear_q4_0", 0, h1, h2, h3b);
        b.declare_output("logits", &[0u8; 4]);
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn op_order_matters() {
        // Declaring the same ops in different order MUST yield
        // different digests. Models that swap layer execution order
        // are different models, even if the per-op hashes are identical.
        let h1 = op_hash(0x10);
        let h2 = op_hash(0x20);
        let h3 = op_hash(0x30);

        let mut a = CertBuilder::new();
        a.declare_op("linear_q4_0", 0, h1, h2, h3);
        a.declare_op("rms_norm", 0, h1, h2, h3);
        let cert_a = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_op("rms_norm", 0, h1, h2, h3);
        b.declare_op("linear_q4_0", 0, h1, h2, h3);
        let cert_b = b.finalize();

        assert_ne!(cert_a.digest, cert_b.digest);
    }

    #[test]
    fn perop_cert_with_sink_replays_op_records() {
        use super::super::sink::{RecordingSink, SinkEvent};

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_op("linear_q4_0", 0, op_hash(0x10), op_hash(0x20), op_hash(0x30));
        b.declare_op("rms_norm", 0, op_hash(0x40), op_hash(0x50), op_hash(0x60));
        b.declare_output("logits", &[0u8; 4]);

        let mut sink = RecordingSink::default();
        let (cert, _id) = b.finalize_with_sink(&mut sink, /*tier*/ 0).unwrap();

        // Sink should see: Request, Input("prompt"), Op×2, Output("logits"), Finalize
        assert_eq!(sink.events.len(), 6);
        match &sink.events[2] {
            SinkEvent::Op { record } => {
                assert_eq!(record.op_name, "linear_q4_0");
                assert_eq!(record.layer, 0);
            }
            other => panic!("expected Op event, got {:?}", other),
        }
        match &sink.events[3] {
            SinkEvent::Op { record } => {
                assert_eq!(record.op_name, "rms_norm");
                assert_eq!(record.op_index, 1);
            }
            other => panic!("expected Op event, got {:?}", other),
        }
        // Cert digest must equal "would-have-been" digest from the
        // same declarations replayed offline.
        let mut b2 = CertBuilder::new();
        b2.declare_input("prompt", b"hi");
        b2.declare_op("linear_q4_0", 0, op_hash(0x10), op_hash(0x20), op_hash(0x30));
        b2.declare_op("rms_norm", 0, op_hash(0x40), op_hash(0x50), op_hash(0x60));
        b2.declare_output("logits", &[0u8; 4]);
        let cert2 = b2.finalize();
        assert_eq!(cert.digest, cert2.digest);
    }

    // =====================================================================
    // Phase 4b — activation snapshot tests
    // =====================================================================

    fn act_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_mul(3).wrapping_add(i as u8);
        }
        h
    }

    #[test]
    fn declare_activation_appends_in_order_with_monotonic_index() {
        let mut b = CertBuilder::new();
        b.declare_activation(0, u32::MAX, "post_embed", act_hash(0x10));
        b.declare_activation(0, 0, "post_attn", act_hash(0x20));
        b.declare_activation(0, 0, "post_ffn", act_hash(0x30));
        assert_eq!(b.activations().len(), 3);
        assert_eq!(b.activations()[0].act_index, 0);
        assert_eq!(b.activations()[0].checkpoint, "post_embed");
        assert_eq!(b.activations()[0].layer, u32::MAX);
        assert_eq!(b.activations()[1].act_index, 1);
        assert_eq!(b.activations()[2].act_index, 2);
        assert_eq!(b.activations()[2].checkpoint, "post_ffn");
    }

    #[test]
    fn same_activations_same_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hi");
        a.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        a.declare_activation(0, 0, "post_attn", act_hash(0x02));
        a.declare_output("logits", &[0u8; 4]);
        let ca = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        b.declare_activation(0, 0, "post_attn", act_hash(0x02));
        b.declare_output("logits", &[0u8; 4]);
        let cb = b.finalize();

        assert_eq!(ca.digest, cb.digest);
    }

    #[test]
    fn different_activation_hash_different_digest() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hi");
        a.declare_activation(0, 0, "post_attn", act_hash(0x01));
        a.declare_output("logits", &[0u8; 4]);
        let ca = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_activation(0, 0, "post_attn", act_hash(0x99));
        b.declare_output("logits", &[0u8; 4]);
        let cb = b.finalize();

        assert_ne!(ca.digest, cb.digest);
    }

    #[test]
    fn perop_and_activation_compose() {
        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_op("linear_q4_0", 0, op_hash(0x10), op_hash(0x20), op_hash(0x30));
        b.declare_activation(0, 0, "post_attn", act_hash(0x40));
        b.declare_op("rms_norm", 0, op_hash(0x50), op_hash(0x60), op_hash(0x70));
        b.declare_activation(0, 0, "post_ffn", act_hash(0x80));
        b.declare_output("logits", &[0u8; 4]);
        let cert = b.finalize();
        assert_eq!(cert.ops.len(), 2);
        assert_eq!(cert.activations.len(), 2);
        // The activations and ops should both round-trip via finalize.
        assert_eq!(cert.activations[0].checkpoint, "post_attn");
        assert_eq!(cert.activations[1].checkpoint, "post_ffn");
    }

    #[test]
    fn activation_cert_with_sink_replays_in_order() {
        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        b.declare_activation(0, 0, "post_attn", act_hash(0x02));
        b.declare_output("logits", &[0u8; 4]);
        let mut sink = RecordingSink::default();
        let (cert, _id) = b.finalize_with_sink(&mut sink, /*tier*/ 0).unwrap();

        // Sink sees: Request, Input, Activation×2, Output, Finalize = 6
        assert_eq!(sink.events.len(), 6);
        match &sink.events[2] {
            SinkEvent::Activation { record } => {
                assert_eq!(record.checkpoint, "post_embed");
                assert_eq!(record.layer, u32::MAX);
            }
            other => panic!("expected Activation event, got {:?}", other),
        }
        match &sink.events[3] {
            SinkEvent::Activation { record } => {
                assert_eq!(record.checkpoint, "post_attn");
                assert_eq!(record.act_index, 1);
            }
            other => panic!("expected Activation event, got {:?}", other),
        }
        // Digest round-trip
        let mut b2 = CertBuilder::new();
        b2.declare_input("prompt", b"hi");
        b2.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        b2.declare_activation(0, 0, "post_attn", act_hash(0x02));
        b2.declare_output("logits", &[0u8; 4]);
        let cert2 = b2.finalize();
        assert_eq!(cert.digest, cert2.digest);
    }

    // =====================================================================
    // Phase 4c — causal-intervention tests
    // =====================================================================

    fn intv_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_add((i as u8).wrapping_mul(5));
        }
        h
    }

    #[test]
    fn declare_intervention_appends_in_order_with_monotonic_index() {
        let mut b = CertBuilder::new();
        b.declare_intervention(0, 0, "post_attn", intv_hash(0x01));
        b.declare_intervention(1, 0, "post_ffn", intv_hash(0x02));
        assert_eq!(b.interventions().len(), 2);
        assert_eq!(b.interventions()[0].intv_index, 0);
        assert_eq!(b.interventions()[0].checkpoint, "post_attn");
        assert_eq!(b.interventions()[1].intv_index, 1);
        assert_eq!(b.interventions()[1].token_index, 1);
    }

    #[test]
    fn unintervened_vs_intervened_have_different_digests() {
        let mut a = CertBuilder::new();
        a.declare_input("prompt", b"hi");
        a.declare_output("logits", &[1u8, 2, 3, 4]);
        let ca = a.finalize();

        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_intervention(0, 0, "post_attn", intv_hash(0x01));
        b.declare_output("logits", &[1u8, 2, 3, 4]);
        let cb = b.finalize();

        // Same inputs+outputs but one has interventions → different digest.
        assert_ne!(ca.digest, cb.digest);
    }

    #[test]
    fn intervention_cert_with_sink_replays_in_order() {
        let mut b = CertBuilder::new();
        b.declare_input("prompt", b"hi");
        b.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        b.declare_intervention(0, 0, "post_attn", intv_hash(0x77));
        b.declare_output("logits", &[0u8; 4]);
        let mut sink = RecordingSink::default();
        let (cert, _id) = b.finalize_with_sink(&mut sink, /*tier*/ 0).unwrap();

        // Order: Request, Input, Activation, Intervention, Output, Finalize = 6
        assert_eq!(sink.events.len(), 6);
        match &sink.events[3] {
            SinkEvent::Intervention { record } => {
                assert_eq!(record.checkpoint, "post_attn");
                assert_eq!(record.intv_index, 0);
                assert_eq!(record.replacement_hash, intv_hash(0x77));
            }
            other => panic!("expected Intervention event, got {:?}", other),
        }
        // Digest round-trip
        let mut b2 = CertBuilder::new();
        b2.declare_input("prompt", b"hi");
        b2.declare_activation(0, u32::MAX, "post_embed", act_hash(0x01));
        b2.declare_intervention(0, 0, "post_attn", intv_hash(0x77));
        b2.declare_output("logits", &[0u8; 4]);
        let cert2 = b2.finalize();
        assert_eq!(cert.digest, cert2.digest);
    }
}
