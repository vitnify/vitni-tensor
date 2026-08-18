//! Measures the cost of materialising the dequantized weight matrix.
//! Both paths produce bit-identical output (asserted here too).
use std::time::Instant;
use vitni_tensor::ops::quant::{linear_q4_k_cpu, linear_q4_k_fused};

fn main() {
    // Mistral-7B FFN shape, batch=1 — i.e. one autoregressive decode step.
    let in_feat = 4096usize;
    let out_feat = 14336usize;
    let row_bytes = (in_feat / 256) * 144;
    let mut w = vec![0u8; out_feat * row_bytes];
    let mut s: u64 = 0xC0FFEE;
    for v in w.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (s >> 40) as u8;
    }
    let x: Vec<f32> = (0..in_feat).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let mut y_ref = vec![0.0f32; out_feat];
    let mut y_fus = vec![0.0f32; out_feat];

    println!("Mistral-7B FFN linear, {}x{}, batch=1 (one decode step)", in_feat, out_feat);
    println!("  quantized weights : {:.1} MB", (out_feat*row_bytes) as f64/1e6);
    println!("  dequantized f32   : {:.1} MB  <- allocated per call by the current path",
             (out_feat*in_feat*4) as f64/1e6);
    println!();

    let mut d_ref = vec![]; let mut d_fus = vec![];
    for rep in 0..5 {
        let t = Instant::now();
        linear_q4_k_cpu(&x, &w, &mut y_ref, 1, in_feat, out_feat).unwrap();
        let a = t.elapsed().as_secs_f64();
        let t = Instant::now();
        linear_q4_k_fused(&x, &w, &mut y_fus, 1, in_feat, out_feat).unwrap();
        let b = t.elapsed().as_secs_f64();
        println!("  rep{}  materialise={:>8.1} ms   fused={:>8.1} ms   {:>5.1}x", rep, a*1e3, b*1e3, a/b);
        if rep >= 1 { d_ref.push(a); d_fus.push(b); }
    }
    d_ref.sort_by(|a,b| a.partial_cmp(b).unwrap());
    d_fus.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let m1 = d_ref[d_ref.len()/2]; let m2 = d_fus[d_fus.len()/2];
    let identical = y_ref.iter().zip(y_fus.iter()).all(|(p,q)| p.to_bits()==q.to_bits());
    println!();
    println!("  median materialise : {:>8.1} ms", m1*1e3);
    println!("  median fused       : {:>8.1} ms", m2*1e3);
    println!("  speedup            : {:>8.1}x", m1/m2);
    println!("  outputs            : {}", if identical {"BIT-IDENTICAL"} else {"*** DIFFER ***"});
}
