//! Standalone prototype checks plus a regression for the active SiLU math.
#[path = "support/q4_prepare_swiglu_gpu.rs"]
mod gpu;
#[path = "../src/q4_metadata.rs"]
mod metadata;
use vitni_tensor::ops::quant::linear_q4_k_fused;
fn fixture(k: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    let x = (0..k)
        .map(|i| {
            let r = ((next() >> 8) as f32 / 16777216.0) * 2.0 - 1.0;
            match i % 19 {
                0 => 0.0,
                1 => -0.0,
                2 => r * 1.0e4,
                3 => r * 1.0e-4,
                _ => r,
            }
        })
        .collect();
    let mut weights: Vec<u8> = (0..m * (k / 256) * 144).map(|_| next() as u8).collect();
    // Finite, ordinary Q4_K scale encodings; vary exponent and mantissa.
    for b in weights.chunks_exact_mut(144) {
        let d = 0x3000u16 + (next() as u16 & 0x0fff);
        let minimum = 0x2800u16 + (next() as u16 & 0x0fff);
        let cases = [0u16, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400];
        let d = if next() % 3 == 0 {
            cases[next() as usize % cases.len()]
        } else {
            d
        };
        b[..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&minimum.to_le_bytes());
    }
    let initial = (0..m + 3)
        .map(|i| {
            if i < m {
                (i as f32 - 17.0) * 0.375
            } else {
                123456.75
            }
        })
        .collect();
    (x, weights, initial)
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn q4_product_preparation_matches_cpu() {
    let gpu = gpu::Gpu::new();
    let mut words = 0;
    for (k, m) in [
        (256, 1),
        (256, 2),
        (256, 3),
        (768, 7),
        (2048, 9),
        (5632, 33),
        (8192, 8),
        (8448, 9),
        (14336, 33),
        (16640, 9),
        (27648, 65),
    ] {
        for seed in [7, 0x12345678, 0xfedcba9876543210] {
            let (x, w, init) = fixture(k, m, seed);
            let mut reference = vec![0.; m];
            linear_q4_k_fused(&x, &w, &mut reference, 1, k, m).unwrap();
            let xb = gpu.floats(&x);
            let wb = gpu.buffer(&metadata::pack(&w));
            for acc in [false, true] {
                let mut expected = init.clone();
                for i in 0..m {
                    expected[i] = if acc {
                        init[i] + reference[i]
                    } else {
                        reference[i]
                    };
                }
                for candidate in [false, true] {
                    let (y, _, _) = gpu.product(&xb, &wb, &init, k, m, candidate, acc, 1);
                    gpu::check(
                        &y,
                        &expected,
                        &format!("product K={k} M={m} seed={seed} acc={acc} candidate={candidate}"),
                    );
                    words += m;
                }
            }
        }
    }
    eprintln!("product: {words} scalar words including both modes, plus guards");
}
#[test]
#[ignore = "requires a real Metal GPU"]
fn fused_gate_up_swiglu_matches_cpu() {
    let gpu = gpu::Gpu::new();
    let mut intermediate = 0;
    let mut final_words = 0;
    for (k, m) in [
        (256, 1),
        (256, 2),
        (256, 3),
        (256, 4),
        (256, 5),
        (768, 7),
        (2048, 9),
        (5632, 17),
        (8192, 8),
        (8448, 9),
        (16640, 9),
        (27648, 17),
    ] {
        for seed in [7, 0x12345678, 0xfedcba9876543210] {
            let (mut x, w, init) = fixture(k, m, seed);
            let (_, w2, _) = fixture(k, m, seed ^ 0x99999999);
            let mut g = vec![0.; m];
            let mut u = g.clone();
            linear_q4_k_fused(&x, &w, &mut g, 1, k, m).unwrap();
            // Ordinary finite-exp coverage; overflow coverage is separate below.
            let mut bound = g.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            while bound > 16.0 {
                for v in &mut x {
                    *v *= 0.5;
                }
                bound *= 0.5;
            }
            linear_q4_k_fused(&x, &w, &mut g, 1, k, m).unwrap();
            linear_q4_k_fused(&x, &w2, &mut u, 1, k, m).unwrap();
            let mut eg = init.clone();
            let mut eu = init.clone();
            let mut ey = init.clone();
            for i in 0..m {
                eg[i] = g[i];
                eu[i] = u[i];
                ey[i] = (g[i] / (1.0 + libm::expf(-g[i]))) * u[i];
            }
            let xb = gpu.floats(&x);
            let wg = gpu.buffer(&metadata::pack(&w));
            let wu = gpu.buffer(&metadata::pack(&w2));
            for candidate in [false, true] {
                let (y, a, b, _, _) = gpu.fused(&xb, [&wg, &wu], &init, k, m, candidate, true, 1);
                gpu::check(&a, &eg, "gate including guards");
                gpu::check(&b, &eu, "up including guards");
                gpu::check(&y, &ey, &format!("inner K={k} M={m} candidate={candidate}"));
                intermediate += 2 * m;
                final_words += m;
                // The measured kernel elides debug stores; validate its distinct binary too.
                if candidate {
                    let (y, a, b, _, _) = gpu.fused(&xb, [&wg, &wu], &init, k, m, true, false, 1);
                    gpu::check(&y, &ey, "non-debug fused inner");
                    gpu::check(&a, &init, "no gate stores");
                    gpu::check(&b, &init, "no up stores");
                    final_words += m;
                }
            }
        }
    }
    eprintln!(
        "fusion: {intermediate} intermediate and {final_words} final scalar words, plus guards"
    );
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn swiglu_exp_overflow_matches_cpu() {
    let gpu = gpu::Gpu::new();
    let (k, m) = (256, 1);
    let mut overflow_cases = 0;
    for seed in [7, 0x12345678, 0xfedcba9876543210] {
        let (x, w, init) = fixture(k, m, seed);
        let (_, w2, _) = fixture(k, m, seed ^ 0x99999999);
        let mut g = vec![0.; m];
        let mut u = g.clone();
        linear_q4_k_fused(&x, &w, &mut g, 1, k, m).unwrap();
        linear_q4_k_fused(&x, &w2, &mut u, 1, k, m).unwrap();
        let cpu = (g[0] / (1.0 + libm::expf(-g[0]))) * u[0];
        let xb = gpu.floats(&x);
        let wg = gpu.buffer(&metadata::pack(&w));
        let wu = gpu.buffer(&metadata::pack(&w2));
        let (base, bg, bu, _, _) = gpu.fused(&xb, [&wg, &wu], &init, k, m, false, true, 1);
        let mut eg = init.clone();
        let mut eu = init.clone();
        eg[..m].copy_from_slice(&g);
        eu[..m].copy_from_slice(&u);
        gpu::check(&bg, &eg, "overflow gate versus CPU");
        gpu::check(&bu, &eu, "overflow up versus CPU");
        for debug in [false, true] {
            let (candidate, _, _, _, _) = gpu.fused(&xb, [&wg, &wu], &init, k, m, true, debug, 1);
            gpu::check(&candidate, &base, "fused and separate overflow behavior");
        }
        assert_eq!(
            base[0].to_bits(),
            cpu.to_bits(),
            "seed={seed} g={} up={}",
            g[0],
            u[0]
        );
        if libm::expf(-g[0]).is_infinite() {
            overflow_cases += 1;
            assert_eq!(base[0].abs().to_bits(), 0);
        }
    }
    assert!(
        overflow_cases > 0,
        "the original overflow regression must actually execute"
    );
}
