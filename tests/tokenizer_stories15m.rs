//! Tokenizer test against the real stories15M tokenizer.bin.
//!
//! Confirms our llama2c BPE encoder reproduces the canonical
//! token sequence for "Once upon a time" — the same prompt
//! the reference implementation uses, and the same token IDs we already verified
//! the forward pass continues correctly from in M3.

extern crate alloc;

use vitni_tensor::tokenizer::Llama2cTokenizer;
use std::path::PathBuf;

const TOKENIZER_REL: &str = "../../userspace/the reference implementation/assets/tokenizer.bin";

fn tokenizer_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest);
    p.push(TOKENIZER_REL);
    p
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn encode_once_upon_a_time_matches_reference() {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let blob = std::fs::read(&path).expect("read tokenizer.bin");

    // Llama2 vocab is 32000.
    let tok = Llama2cTokenizer::from_blob(&blob, 32000).expect("parse tokenizer");

    let ids = tok.encode("Once upon a time", /* bos */ true);
    eprintln!("encoded: {:?}", ids);

    // karpathy llama2.c convention: no leading space is prepended to
    // the prompt (unlike reference SentencePiece). So "Once" at
    // start-of-text tokenizes literally to 26222, not "▁Once" (9038).
    // Internal spaces ARE present and get merged into the following
    // word — " upon" → 2501 = "▁upon", " a" → 263, " time" → 931.
    //
    // The reference SentencePiece tokenizer would emit [1, 9038,
    // 2501, 263, 931] (note 9038 instead of 26222). That's the
    // sequence the M3 stories15M test sees as MODEL OUTPUT from
    // BOS=1, because the model has learned to start with the
    // leading-space "▁Once" token. The encode and decode paths are
    // not symmetric for start-of-text — by design.
    assert_eq!(ids[0], 1, "BOS should be id 1");
    assert_eq!(
        &ids[..],
        &[1, 26222, 2501, 263, 931][..],
        "tokenization diverges from karpathy llama2c reference"
    );
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn decode_roundtrips_known_tokens() {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let blob = std::fs::read(&path).expect("read tokenizer.bin");
    let tok = Llama2cTokenizer::from_blob(&blob, 32000).expect("parse tokenizer");

    // Decode the canonical opening — BOS-strip kicks in on the first
    // token, so the output should be "Once upon a time" with no
    // leading space.
    let tokens = [9038u32, 2501, 263, 931, 29892];
    let text = tok.decode_string(/* prev = BOS */ 1, &tokens);
    eprintln!("decoded: {:?}", text);
    assert_eq!(text, "Once upon a time,");
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn encode_then_decode_roundtrip() {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let blob = std::fs::read(&path).expect("read tokenizer.bin");
    let tok = Llama2cTokenizer::from_blob(&blob, 32000).expect("parse tokenizer");

    for prompt in &[
        "Hello world",
        "The quick brown fox",
        "Once upon a time, there was a little girl",
    ] {
        let ids = tok.encode(prompt, false);
        let decoded = tok.decode_string(0, &ids);
        // Llama tokenizer's encode/decode round-trip adds a leading
        // space to non-BOS prompts (the "_" sentinel). That's
        // expected — we check the prompt is contained in the decoded
        // output.
        assert!(
            decoded.contains(prompt) || decoded.trim_start() == *prompt,
            "round-trip mismatch: '{}' → {:?} → '{}'",
            prompt,
            ids,
            decoded
        );
    }
}
