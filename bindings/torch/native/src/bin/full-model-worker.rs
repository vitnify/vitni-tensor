//! Resident standalone Rust control. Timers exclude stdin/stdout and file writes.
use std::{
    io::{self, BufRead, Write},
    time::Instant,
};
use vitni_torch_native::full_model::{hex, Session};
fn arg(flag: &str) -> String {
    let args: Vec<_> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .expect(flag)
        .clone()
}
fn main() {
    let start = Instant::now();
    let mut engine = Session::open(
        &arg("--gguf"),
        if arg("--backend") == "metal" { 1 } else { 0 },
        arg("--optimized") == "1",
    )
    .unwrap();
    let output = arg("--output");
    let model_id = arg("--model-id");
    println!(
        "{{\"ready\":true,\"load_seconds\":{},\"info\":{}}}",
        start.elapsed().as_secs_f64(),
        engine.info()
    );
    io::stdout().flush().unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "QUIT" {
            break;
        }
        if line == "INFO" {
            println!("{}", engine.info());
            io::stdout().flush().unwrap();
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let tokens: Vec<i64> = fields
            .last()
            .unwrap()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        match fields[0] {
            "F" => {
                let mut result = vec![0f32; tokens.len() * engine.cfg.vocab_size];
                let start = Instant::now();
                engine.forward(&tokens, &mut result).unwrap();
                let seconds = start.elapsed().as_secs_f64();
                let bytes: Vec<u8> = result
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                let digest = hex(blake3::hash(&bytes).as_bytes());
                std::fs::write(&output, &bytes).unwrap();
                println!(
                    "{{\"seconds\":{seconds},\"output_blake3\":\"{digest}\",\"words\":{}}}",
                    result.len()
                );
            }
            "G" => {
                let n = fields[1].parse().unwrap();
                let start = Instant::now();
                let receipt = engine.generate(&tokens, n, &model_id).unwrap();
                let seconds = start.elapsed().as_secs_f64();
                println!("{{\"seconds\":{seconds},\"receipt\":{receipt}}}");
            }
            _ => panic!("expected F or G"),
        }
        io::stdout().flush().unwrap();
    }
}
