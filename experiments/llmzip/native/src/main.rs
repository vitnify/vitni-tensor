//! Workload adapter only. Calls the unchanged, default-feature Vitni engine.
use std::{
    io::{self, BufRead, Write},
    time::Instant,
};
use vitni_tensor::{
    model::{
        config::Config, forward::RunState, forward_quantized, gguf::GgufFile,
        quant_weights::QuantizedWeights,
    },
    Storage,
};
use vitni_tensor_metal::{MetalForward, MetalOptions};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "MODEL cpu|metal CONTEXT");
    let blob = std::fs::read(&args[1]).unwrap();
    let gguf = GgufFile::parse(&blob).unwrap();
    assert_eq!(gguf.metadata_str("general.architecture").unwrap(), "llama");
    let mut cfg = Config::from_gguf(&gguf).unwrap();
    let capacity: usize = args[3].parse().unwrap();
    assert!(capacity > 0 && capacity <= cfg.seq_len && capacity <= 512);
    cfg.seq_len = capacity;
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
    let is_metal = match args[2].as_str() {
        "cpu" => false,
        "metal" => true,
        _ => panic!("invalid backend"),
    };
    let mut cpu = if is_metal {
        None
    } else {
        Some(RunState::new(&cfg))
    };
    let mut gpu = if is_metal {
        Some(objc::rc::autoreleasepool(|| {
            MetalForward::new_with_options(&cfg, &w, &MetalOptions::optimized()).unwrap()
        }))
    } else {
        None
    };
    let mut out = io::stdout().lock();
    writeln!(out,"{{\"ready\":true,\"backend\":\"{}\",\"vocab\":{},\"dim\":{},\"layers\":{},\"context\":{},\"weights_blake3\":\"{}\"}}",
        args[2],cfg.vocab_size,cfg.dim,cfg.n_layers,capacity,blake3::hash(&blob).to_hex()).unwrap();
    out.flush().unwrap();
    let mut pos = 0;
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "QUIT" {
            break;
        }
        let (op, values) = line.split_once(' ').unwrap();
        assert!(matches!(op, "P" | "S"));
        let ids: Vec<u32> = values.split(',').map(|v| v.parse().unwrap()).collect();
        assert!(!ids.is_empty() && ids.iter().all(|&t| (t as usize) < cfg.vocab_size));
        let start = Instant::now();
        if op == "P" {
            pos = 0;
            if let Some(s) = cpu.as_mut() {
                *s = RunState::new(&cfg);
            }
            if let Some(g) = gpu.as_mut() {
                g.reset();
            }
        }
        assert!(pos + ids.len() <= capacity);
        let mut row = Vec::new();
        for token in ids.iter() {
            row = if let Some(s) = cpu.as_mut() {
                let t = forward_quantized::step(&cfg, &w, s, *token, pos).unwrap();
                let Storage::Cpu(data) = t.storage() else {
                    panic!("expected CPU logits")
                };
                data.as_f32_slice().to_vec()
            } else {
                objc::rc::autoreleasepool(|| gpu.as_mut().unwrap().step(*token, pos))
            };
            pos += 1;
        }
        assert!(row.iter().all(|v| v.is_finite()), "nonfinite model output");
        let elapsed = start.elapsed().as_secs_f64();
        writeln!(
            out,
            "{{\"seconds\":{elapsed},\"positions\":{},\"pos\":{pos},\"words\":{}}}",
            ids.len(),
            row.len()
        )
        .unwrap();
        let raw: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.write_all(&raw).unwrap();
        out.flush().unwrap();
    }
}
