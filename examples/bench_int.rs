use std::time::Instant;
use vitni_tensor::ops::quant::{linear_q4_k_fused, linear_q4_k_integer};
fn main(){
    let in_feat=4096usize; let out_feat=14336usize;
    let row_bytes=(in_feat/256)*144;
    let mut w=vec![0u8; out_feat*row_bytes];
    let mut s:u64=0xC0FFEE;
    for v in w.iter_mut(){ s=s.wrapping_mul(6364136223846793005).wrapping_add(1); *v=(s>>40) as u8; }
    for blk in w.chunks_mut(144){
        blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes()); }
    let x:Vec<f32>=(0..in_feat).map(|i| ((i*7919)%1000) as f32/500.0-1.0).collect();
    let mut yf=vec![0.0f32;out_feat]; let mut yi=vec![0.0f32;out_feat];
    let mut bf=Vec::new(); let mut bi=Vec::new();
    for rep in 0..5 {
        let t=Instant::now(); linear_q4_k_fused(&x,&w,&mut yf,1,in_feat,out_feat).unwrap();
        let a=t.elapsed().as_secs_f64();
        let t=Instant::now(); linear_q4_k_integer(&x,&w,&mut yi,1,in_feat,out_feat).unwrap();
        let b=t.elapsed().as_secs_f64();
        if rep>=1 { bf.push(a); bi.push(b); }
    }
    bf.sort_by(|a,b|a.partial_cmp(b).unwrap()); bi.sort_by(|a,b|a.partial_cmp(b).unwrap());
    let mf=bf[bf.len()/2]; let mi=bi[bi.len()/2];
    let flops=(out_feat*in_feat*2) as f64;
    println!("Mistral FFN 4096x14336, batch=1, single thread\n");
    println!("  float  (regime 3, fused) : {:>7.2} ms  {:>6.1} GFLOP/s  1.00x", mf*1e3, flops/mf/1e9);
    println!("  integer (q8 activations) : {:>7.2} ms  {:>6.1} GFLOP/s  {:.2}x", mi*1e3, flops/mi/1e9, mf/mi);
    let maxrel = yf.iter().zip(yi.iter())
        .map(|(p,q)| (p-q).abs()/p.abs().max(1.0)).fold(0.0f32,f32::max);
    println!("\n  max rel delta vs float : {:.2e}  (int8 activation loss)", maxrel);
}
