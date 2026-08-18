//! Decode the Qwen2.5 run's token IDs back to text (via the byte-BPE tokenizer)
//! so the engine output can be eyeballed against llama.cpp. No inference — fast.
extern crate alloc;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::tokenizer::GgufBpeTokenizer;

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn qwen_decode() {
    let path = std::env::var("QWEN_GGUF").expect("set QWEN_GGUF");
    let blob = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&blob).expect("parse");
    let tok = GgufBpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer");

    let prompt = [12522u32, 5193, 264, 882, 11];
    let gen = [572u32, 264, 3908, 883, 6941, 3757, 879, 12163, 304, 264, 2613, 14126, 13, 3757, 572, 1602];
    eprintln!("prompt : {:?}", tok.decode(&prompt));
    eprintln!("vitni-gen : {:?}", tok.decode(&gen));
    let mut all = prompt.to_vec();
    all.extend_from_slice(&gen);
    eprintln!("vitni-full: {:?}", tok.decode(&all));
}
