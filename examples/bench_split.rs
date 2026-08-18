//! Where does the fused kernel's time actually go: dequant or dot?
use std::time::Instant;
use vitni_tensor::ops::quant::{dequantize_q4_k, linear_q4_k_fused};

fn main() {
    let in_feat = 4096usize; let out_feat = 14336usize;
    let row_bytes = (in_feat / 256) * 144;
    let mut w = vec![0u8; out_feat * row_bytes];
    let mut s: u64 = 0xC0FFEE;
    for v in w.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *v = (s>>40) as u8; }
    // Random bytes make garbage f16 scales -> NaN/denormal arithmetic, which
    // does NOT run at the same speed as real weights. Overwrite each block's
    // d/dmin with plausible finite values so this measures the real workload.
    for blk in w.chunks_mut(144) {
        blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());  // d    ~= 0.1
        blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());  // dmin ~= 0.05
    }
    let x: Vec<f32> = (0..in_feat).map(|i| ((i*7919)%1000) as f32/500.0-1.0).collect();
    let mut y = vec![0.0f32; out_feat];

    // (a) full fused kernel
    linear_q4_k_fused(&x,&w,&mut y,1,in_feat,out_feat).unwrap();
    let t=Instant::now(); linear_q4_k_fused(&x,&w,&mut y,1,in_feat,out_feat).unwrap();
    let full=t.elapsed().as_secs_f64();

    // (b) dequant alone, same total weights, result consumed so it can't be elided
    let mut sink=0.0f32;
    let t=Instant::now();
    for o in 0..out_feat {
        let d = dequantize_q4_k(&w[o*row_bytes..(o+1)*row_bytes]).unwrap();
        sink += d[0] + d[d.len()-1];
    }
    let deq=t.elapsed().as_secs_f64();

    // (c) canonical dot alone on already-dequantized f32
    let row_f32 = dequantize_q4_k(&w[0..row_bytes]).unwrap();
    let mut part = vec![0.0f32; (in_feat+7)/8];
    let t=Instant::now();
    for _ in 0..out_feat {
        for b in 0..part.len() {
            let s0=b*8; let e=(s0+8).min(in_feat); let mut acc=0.0f32;
            for i in s0..e { let p = x[i]*row_f32[i]; acc+=p; }
            part[b]=acc;
        }
        let mut len=part.len();
        while len>1 { let h=(len+1)/2; for t2 in 0..h { let u=2*t2;
            part[t2]= if u+1<len {part[u]+part[u+1]} else {part[u]}; } len=h; }
        sink += part[0];
    }
    let dot=t.elapsed().as_secs_f64();

    println!("Mistral FFN 4096x14336, batch=1 — fused kernel time breakdown\n");
    println!("  (a) full fused kernel : {:>7.1} ms  100%", full*1e3);
    println!("  (b) dequant only      : {:>7.1} ms  {:>4.0}%   <- allocates a Vec per row", deq*1e3, deq/full*100.0);
    println!("  (c) canonical dot only: {:>7.1} ms  {:>4.0}%", dot*1e3, dot/full*100.0);
    println!("\n  (sink {})", sink);
}
