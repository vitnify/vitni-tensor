// Parallel scaling of the integer Q4_K linear at the shapes Mistral actually
// uses, so we can tell whether poor end-to-end scaling is the matmul or
// everything around it.
use vitni_tensor::ops::quant::{linear_q4_k_integer, linear_q4_k_integer_parallel};
use std::time::Instant;

fn make_w(in_feat: usize, out_feat: usize) -> Vec<u8> {
    let row_bytes = (in_feat / 256) * 144;
    let mut w = vec![0u8; out_feat * row_bytes];
    let mut s: u64 = 0xC0FFEE;
    for v in w.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (s >> 40) as u8;
    }
    for blk in w.chunks_mut(144) {
        blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
    }
    w
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    // (in_feat, out_feat, label, per-layer count in Mistral 7B)
    let shapes = [
        (4096usize, 4096usize, "attn q/o", 2usize),
        (4096, 1024, "attn k/v", 2),
        (4096, 14336, "ffn gate/up", 2),
        (14336, 4096, "ffn down", 1),
    ];
    let threads_list = [1usize, 2, 4, 8, 14];

    println!("Integer Q4_K linear, batch=1, median of 5 (M3 Max)\n");
    println!(
        "{:<14} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape", "1thr", "2thr", "4thr", "8thr", "14thr", "best-spd"
    );

    let mut serial_total = 0.0f64;
    let mut best_total = 0.0f64;

    for (in_feat, out_feat, label, count) in shapes {
        let w = make_w(in_feat, out_feat);
        let x: Vec<f32> = (0..in_feat)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut y = vec![0.0f32; out_feat];

        let mut row = String::new();
        let mut t1 = 0.0f64;
        let mut best = f64::MAX;
        for &t in &threads_list {
            let mut samples = Vec::new();
            for rep in 0..6 {
                let st = Instant::now();
                if t == 1 {
                    linear_q4_k_integer(&x, &w, &mut y, 1, in_feat, out_feat).unwrap();
                } else {
                    linear_q4_k_integer_parallel(&x, &w, &mut y, 1, in_feat, out_feat, t).unwrap();
                }
                let e = st.elapsed().as_secs_f64();
                if rep > 0 {
                    samples.push(e);
                }
            }
            let m = median(&mut samples);
            if t == 1 {
                t1 = m;
            }
            if m < best {
                best = m;
            }
            row.push_str(&format!("{:>9.2}", m * 1e3));
        }
        serial_total += t1 * count as f64;
        best_total += best * count as f64;
        println!("{:<14}{} {:>8.2}x", label, row, t1 / best);
    }

    println!(
        "\nPer-layer matmul total: serial {:.1} ms, best-threaded {:.1} ms  ({:.2}x)",
        serial_total * 1e3,
        best_total * 1e3,
        serial_total / best_total
    );
    println!(
        "x32 layers: serial {:.2} s/token, best-threaded {:.2} s/token",
        serial_total * 32.0,
        best_total * 32.0
    );
    println!("\nMeasured end-to-end marginal cost was ~0.265 s/token wall.");
    println!("If best-threaded matmul total is well under that, the gap is OUTSIDE the matmuls.");
}
