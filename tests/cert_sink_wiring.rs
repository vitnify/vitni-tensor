//! Syscall-wiring proof: the substrate-side cert path (via the
//! `CertSink` trait) replays the EXACT same declarations as the
//! software-side `CertBuilder`, in the same order, with the same
//! bytes.
//!
//! That means: on the target runtime, a `RuntimeSink` implementing the trait by
//! calling `the runtime's cert API` will issue a kernel cert whose binding
//! digest equals the software `Cert.digest` returned by `run_with_sink`.
//! Verifiers can independently reconstruct either one.
//!
//! Host-side we use `RecordingSink` to capture the event sequence
//! and assert:
//!   1. on_request fires once with the right tier
//!   2. on_input fires N times in declaration order, with correct bytes
//!   3. on_output fires M times in declaration order, with correct bytes
//!   4. on_finalize fires last
//!   5. The cert id returned by the sink is surfaced to the caller
//!
//! Plus a digest-equivalence check: simulate substrate computing its
//! own digest from the recorded events (BLAKE3 over the canonical
//! length-prefixed format), confirm it matches the software digest.

extern crate alloc;

use vitni_tensor::cert::{CertBuilder, CertSink, RecordingSink, SinkEvent};
use vitni_tensor::model::{config::Config, inference, weights::Weights};

fn build_synthetic_blob() -> (Config, Vec<u8>) {
    let cfg = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 64,
        seq_len: 16,
        shared_weights: true,
    };
    let mut blob = Vec::new();
    blob.extend_from_slice(&(cfg.dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.hidden_dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_layers as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_kv_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.vocab_size as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.seq_len as i32).to_le_bytes());
    let hs = cfg.head_size();
    let sizes = [
        cfg.vocab_size * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.dim * (cfg.n_heads * hs),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * hs),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * hs),
        cfg.n_layers * (cfg.n_heads * hs) * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.dim,
    ];
    let mut seed = 1u32;
    for &n in &sizes {
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = (seed as i32 as f32) / (i32::MAX as f32) * 0.05;
            blob.extend_from_slice(&v.to_le_bytes());
        }
    }
    for _ in 0..(cfg.seq_len * hs) {
        blob.extend_from_slice(&0.0f32.to_le_bytes());
    }
    (cfg, blob)
}

/// Recompute the cert digest from a sequence of recorded sink events,
/// using the same canonical format `CertBuilder::finalize` uses.
/// This is what a the host runtime would compute from `the runtime's cert syscall` calls.
///
/// Bumped to v4 format (2026-06-25, Phase 4c): trailing interventions
/// section after activations. Same canonical-LEB128 shape; unintervened
/// inferences declare zero interventions so the section becomes a
/// single LEB128 zero — the version tag differentiates v4 from v3.
fn digest_from_events(events: &[SinkEvent]) -> [u8; 32] {
    let mut inputs: Vec<(&str, &[u8])> = Vec::new();
    let mut outputs: Vec<(&str, &[u8])> = Vec::new();
    let mut ops: Vec<&vitni_tensor::cert::builder::OpRecord> = Vec::new();
    let mut acts: Vec<&vitni_tensor::cert::builder::ActivationRecord> = Vec::new();
    let mut intvs: Vec<&vitni_tensor::cert::builder::InterventionRecord> = Vec::new();
    for e in events {
        match e {
            SinkEvent::Input { name, bytes } => inputs.push((name.as_str(), bytes.as_slice())),
            SinkEvent::Output { name, bytes } => outputs.push((name.as_str(), bytes.as_slice())),
            SinkEvent::Op { record } => ops.push(record),
            SinkEvent::Activation { record } => acts.push(record),
            SinkEvent::Intervention { record } => intvs.push(record),
            _ => {}
        }
    }

    let mut h = blake3::Hasher::new();
    h.update(b"vitnify-receipt v1\x00");
    write_leb128(&mut h, inputs.len() as u64);
    for (n, b) in &inputs {
        write_leb128(&mut h, n.len() as u64);
        h.update(n.as_bytes());
        write_leb128(&mut h, b.len() as u64);
        h.update(b);
    }
    write_leb128(&mut h, outputs.len() as u64);
    for (n, b) in &outputs {
        write_leb128(&mut h, n.len() as u64);
        h.update(n.as_bytes());
        write_leb128(&mut h, b.len() as u64);
        h.update(b);
    }
    write_leb128(&mut h, ops.len() as u64);
    for op in &ops {
        write_leb128(&mut h, op.op_index as u64);
        let name = op.op_name.as_bytes();
        write_leb128(&mut h, name.len() as u64);
        h.update(name);
        write_leb128(&mut h, op.layer as u64);
        h.update(&op.input_hash);
        h.update(&op.params_hash);
        h.update(&op.output_hash);
    }
    write_leb128(&mut h, acts.len() as u64);
    for act in &acts {
        write_leb128(&mut h, act.act_index as u64);
        write_leb128(&mut h, act.token_index as u64);
        write_leb128(&mut h, act.layer as u64);
        let name = act.checkpoint.as_bytes();
        write_leb128(&mut h, name.len() as u64);
        h.update(name);
        h.update(&act.tensor_hash);
    }
    write_leb128(&mut h, intvs.len() as u64);
    for intv in &intvs {
        write_leb128(&mut h, intv.intv_index as u64);
        write_leb128(&mut h, intv.token_index as u64);
        write_leb128(&mut h, intv.layer as u64);
        let name = intv.checkpoint.as_bytes();
        write_leb128(&mut h, name.len() as u64);
        h.update(name);
        h.update(&intv.replacement_hash);
    }
    *h.finalize().as_bytes()
}

fn write_leb128(h: &mut blake3::Hasher, mut v: u64) {
    let mut buf = [0u8; 10];
    let mut i = 0;
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        buf[i] = b;
        i += 1;
        if v == 0 {
            break;
        }
    }
    h.update(&buf[..i]);
}

#[test]
fn builder_finalize_with_sink_replays_declarations_in_order() {
    let mut b = CertBuilder::new();
    b.declare_input("model_id", b"test-model");
    b.declare_input("weights_hash", &[0xab; 32]);
    b.declare_output("output_tokens", &[1u8, 0, 0, 0, 2, 0, 0, 0]);

    let mut sink = RecordingSink::default();
    sink.finalize_id = 0x4242;
    let (cert, id) = b.finalize_with_sink(&mut sink, /* tier */ 7).unwrap();

    assert_eq!(id, 0x4242);
    assert_eq!(sink.events.len(), 5); // request + 2 inputs + 1 output + finalize
    assert_eq!(sink.events[0], SinkEvent::Request { tier: 7 });
    assert!(matches!(&sink.events[1], SinkEvent::Input { name, .. } if name == "model_id"));
    assert!(matches!(&sink.events[2], SinkEvent::Input { name, .. } if name == "weights_hash"));
    assert!(matches!(&sink.events[3], SinkEvent::Output { name, .. } if name == "output_tokens"));
    assert_eq!(sink.events[4], SinkEvent::Finalize);

    // The substrate-side digest (reconstructed from events) MUST equal
    // the software digest. This is what guarantees the kernel cert
    // and the returned Cert carry the same 32-byte binding.
    let kernel_digest = digest_from_events(&sink.events);
    assert_eq!(kernel_digest, cert.digest);
}

#[test]
fn inference_run_with_sink_matches_run() {
    // Two semantically-equivalent calls: run() and run_with_sink with
    // a NullSink should produce identical certs.
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = *blake3::hash(&blob).as_bytes();
    let req = inference::Request {
        model_id: "x",
        prompt_tokens: &[3, 7],
        n_new_tokens: 2,
    };

    let plain = inference::run(&cfg, &weights, &weights_hash, &req).unwrap();

    let mut sink = vitni_tensor::cert::NullSink;
    let (with_sink, cert_id) =
        inference::run_with_sink(&cfg, &weights, &weights_hash, &req, &mut sink, 0).unwrap();

    assert_eq!(cert_id, 0); // NullSink returns 0
    assert_eq!(plain.cert.digest, with_sink.cert.digest);
    assert_eq!(plain.generated_tokens, with_sink.generated_tokens);
}

#[test]
fn inference_run_with_recording_sink_kernel_digest_matches_software() {
    // The load-bearing property: when a the host runtime computes its cert
    // from the the runtime's cert syscall calls, the resulting 32-byte digest equals
    // the vitni-tensor software digest.
    //
    // We simulate this by replaying through RecordingSink, then
    // computing the digest from the captured events using the same
    // canonical hash format.
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = *blake3::hash(&blob).as_bytes();
    let req = inference::Request {
        model_id: "synthetic",
        prompt_tokens: &[1, 4, 9],
        n_new_tokens: 3,
    };

    let mut sink = RecordingSink::default();
    sink.finalize_id = 0xdeadbeef;
    let (outcome, kernel_cert_id) =
        inference::run_with_sink(&cfg, &weights, &weights_hash, &req, &mut sink, 0).unwrap();

    assert_eq!(kernel_cert_id, 0xdeadbeef);

    // Kernel-style digest reconstructed from the events the sink saw.
    let kernel_digest = digest_from_events(&sink.events);
    assert_eq!(
        kernel_digest, outcome.cert.digest,
        "kernel-replayed cert digest must equal software cert digest"
    );

    // Event sequence sanity: request, 5 inputs (model_id, weights_hash,
    // arch_hash, prompt_tokens, n_new_tokens), 2 outputs (output_tokens,
    // output_tokens_hash), finalize = 9 events total.
    assert_eq!(sink.events.len(), 9);
    let names: Vec<&str> = sink
        .events
        .iter()
        .filter_map(|e| match e {
            SinkEvent::Input { name, .. } | SinkEvent::Output { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "model_id",
            "weights_hash",
            "arch_hash",
            "prompt_tokens",
            "n_new_tokens",
            "output_tokens",
            "output_tokens_hash",
        ],
        "declaration order must match the reference implementation's field order"
    );
}

#[test]
fn sink_failure_propagates() {
    // A sink that fails on input declaration must abort the run.
    struct FailingSink;
    impl CertSink for FailingSink {
        type Error = &'static str;
        fn on_request(&mut self, _tier: u8) -> Result<(), Self::Error> {
            Ok(())
        }
        fn on_input(&mut self, _name: &str, _bytes: &[u8]) -> Result<(), Self::Error> {
            Err("simulated EXCERT_REQUEST_LIMIT")
        }
        fn on_output(&mut self, _name: &str, _bytes: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }
        fn on_finalize(&mut self) -> Result<u64, Self::Error> {
            Ok(0)
        }
    }

    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = [0u8; 32];
    let req = inference::Request {
        model_id: "x",
        prompt_tokens: &[1],
        n_new_tokens: 1,
    };

    let mut sink = FailingSink;
    let res = inference::run_with_sink(&cfg, &weights, &weights_hash, &req, &mut sink, 0);
    match res {
        Err(inference::RunError::Sink(msg)) => assert_eq!(msg, "simulated EXCERT_REQUEST_LIMIT"),
        _ => panic!("expected Sink error"),
    }
}
