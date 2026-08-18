//! What does adopting regime 3 actually buy on the REAL kernel?
use std::time::Instant;
use vitni_tensor::ops::quant::{linear_q4_k_fused, linear_q4_k_fused_regime2};
fn main() {
    let in_feat = 4096usize; let out_feat = 14336usize;
    let row_bytes = (in_feat / 256) * 144;
    let mut w = vec![0u8; out_feat * row_bytes];
    let mut s: u64 = 0xC0FFEE;
    for v in w.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *v = (s>>40) as u8; }
    for blk in w.chunks_mut(144) {
        blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
    }
    let x: Vec<f32> = (0..in_feat).map(|i| ((i*7919)%1000) as f32/500.0-1.0).collect();
    let mut y1 = vec![0.0f32; out_feat];
    let mut y2 = vec![0.0f32; out_feat];
    let mut b1=Vec::new(); let mut b2=Vec::new();
    for rep in 0..6 {
        let t=Instant::now(); linear_q4_k_fused_regime2(&x,&w,&mut y1,1,in_feat,out_feat).unwrap();
        let a=t.elapsed().as_secs_f64();
        let t=Instant::now(); linear_q4_k_fused(&x,&w,&mut y2,1,in_feat,out_feat).unwrap();
        let b=t.elapsed().as_secs_f64();
        if rep>=1 { b1.push(a); b2.push(b); }
    }
    b1.sort_by(|a,b|a.partial_cmp(b).unwrap()); b2.sort_by(|a,b|a.partial_cmp(b).unwrap());
    let m1=b1[b1.len()/2]; let m2=b2[b2.len()/2];
    let ndiff = y1.iter().zip(y2.iter()).filter(|(p,q)| p.to_bits()!=q.to_bits()).count();
    let maxrel = y1.iter().zip(y2.iter())
        .map(|(p,q)| if *p!=0.0 {((p-q)/p).abs()} else {0.0})
        .fold(0.0f32,f32::max);
    println!("Mistral FFN 4096x14336, batch=1 — real kernel, regime 2 vs regime 3\n");
    println!("  regime 2 (v1 blocked, ships) : {:>7.2} ms   1.00x", m1*1e3);
    println!("  regime 3 (v2 lanes)          : {:>7.2} ms   {:.2}x", m2*1e3, m1/m2);
    println!("\n  rows differing : {} / {}", ndiff, out_feat);
    println!("  max rel delta  : {:e}  (reassociation only)", maxrel);
}
