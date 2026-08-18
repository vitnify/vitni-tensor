//! Byte-level BPE (gpt2 family) tokenizer read from the GGUF must match llama.cpp for
//! a Qwen2.5 model — the other tokenizer family (Llama-3 / Qwen use "gpt2", not "llama").
extern crate alloc;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::tokenizer::GgufBpeTokenizer;

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn qwen_gguf_bpe_matches_llamacpp() {
    let path = std::env::var("QWEN_GGUF").expect("set QWEN_GGUF to the qwen2.5 gguf path");
    let blob = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let tok = GgufBpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf");
    let ids = tok.encode("Once upon a time,");
    eprintln!("vocab={}  encode('Once upon a time,') = {:?}", tok.vocab_size(), ids);
    eprintln!("round-trip decode: {:?}", tok.decode(&ids));
    // llama.cpp reference (Qwen2.5, no BOS):
    assert_eq!(ids, vec![12522u32, 5193, 264, 882, 11]);
}
