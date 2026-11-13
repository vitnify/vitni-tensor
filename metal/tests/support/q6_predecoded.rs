//! Lossless Q6_K integer layout; no quantization or float arithmetic occurs here.
pub const RAW: usize = 210;
pub const PACKED: usize = 276;

pub fn pack(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len() % RAW, 0);
    let mut out = vec![0u8; raw.len() / RAW * PACKED];
    for (b, p) in raw.chunks_exact(RAW).zip(out.chunks_exact_mut(PACKED)) {
        for hb in 0..2 {
            for l in 0..32 {
                let h = b[128 + hb * 32 + l];
                let b0 = b[hb * 64 + l];
                let b1 = b[hb * 64 + l + 32];
                let q = [
                    (b0 & 15) | ((h & 3) << 4),
                    (b1 & 15) | (((h >> 2) & 3) << 4),
                    (b0 >> 4) | (((h >> 4) & 3) << 4),
                    (b1 >> 4) | (((h >> 6) & 3) << 4),
                ];
                for j in 0..4 {
                    p[hb * 128 + l + 32 * j] = (q[j] as i16 - 32) as i8 as u8;
                }
            }
        }
        p[256..274].copy_from_slice(&b[192..210]);
    }
    out
}

pub fn unpack(packed: &[u8]) -> Vec<u8> {
    assert_eq!(packed.len() % PACKED, 0);
    let mut raw = vec![0u8; packed.len() / PACKED * RAW];
    for (p, b) in packed.chunks_exact(PACKED).zip(raw.chunks_exact_mut(RAW)) {
        for hb in 0..2 {
            for l in 0..32 {
                let q: [u8; 4] = std::array::from_fn(|j| {
                    let x = p[hb * 128 + l + 32 * j] as i8;
                    assert!((-32..=31).contains(&x));
                    (x as i16 + 32) as u8
                });
                b[hb * 64 + l] = (q[0] & 15) | ((q[2] & 15) << 4);
                b[hb * 64 + l + 32] = (q[1] & 15) | ((q[3] & 15) << 4);
                b[128 + hb * 32 + l] =
                    (q[0] >> 4) | ((q[1] >> 4) << 2) | ((q[2] >> 4) << 4) | ((q[3] >> 4) << 6);
            }
        }
        b[192..210].copy_from_slice(&p[256..274]);
    }
    raw
}

#[cfg(test)]
mod tests {
    #[test]
    fn q6_layout_roundtrips_every_half_header_and_arbitrary_quant_bits() {
        let mut raw: Vec<u8> = (0..65536 * super::RAW)
            .map(|i| ((i * 73 + 11) ^ (i >> 5)) as u8)
            .collect();
        for (h, b) in raw.chunks_exact_mut(super::RAW).enumerate() {
            b[208..].copy_from_slice(&(h as u16).to_le_bytes());
        }
        let packed = super::pack(&raw);
        assert_eq!(packed.len(), 65536 * super::PACKED);
        assert!(packed
            .chunks_exact(super::PACKED)
            .all(|b| b[274..] == [0, 0]));
        assert_eq!(super::unpack(&packed), raw);
    }
}
