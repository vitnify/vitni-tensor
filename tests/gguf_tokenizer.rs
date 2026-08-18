//! Model-agnostic tokenizer: read each model's OWN vocab from its GGUF and encode
//! the SAME text — the IDs must match that model's tokenizer (llama.cpp reference),
//! which is exactly the mismatch that produced garbage on Mistral before.
extern crate alloc;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::tokenizer::GgufSpmTokenizer;

fn assets() -> String {
    std::env::var("VITNI_ASSETS").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../userspace/the reference implementation/assets").to_string()
    })
}

fn check(gguf_name: &str, expected: &[u32]) {
    let path = format!("{}/{}", assets(), gguf_name);
    let blob = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let tok = GgufSpmTokenizer::from_gguf(&gguf).expect("tokenizer from gguf");
    let ids = tok.encode("Once upon a time,", true);
    let decoded = tok.decode(&ids);
    eprintln!("{}  vocab={}  encode('Once upon a time,') = {:?}", gguf_name, tok.vocab_size(), ids);
    eprintln!("    round-trip decode: {:?}", decoded);
    assert_eq!(ids, expected, "tokenization does not match this model's own vocab (llama.cpp reference)");
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn tinyllama_gguf_tokenizer_matches_own_vocab() {
    // llama.cpp: "Once upon a time," -> [1, 9038, 2501, 263, 931, 29892]
    check("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", &[1, 9038, 2501, 263, 931, 29892]);
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn mistral_gguf_tokenizer_matches_own_vocab() {
    // llama.cpp: "Once upon a time," -> [1, 5713, 3714, 264, 727, 28725]  (DIFFERENT vocab!)
    check("mistral-7b-v0.1.Q4_K_M.gguf", &[1, 5713, 3714, 264, 727, 28725]);
}
