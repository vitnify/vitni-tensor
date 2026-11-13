//! Research-only exact static Q4 metadata packing; nibbles remain quantized.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 0x1) as u32;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half → normal float.
            let mut m = mant;
            let mut e: i32 = 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (((127 - 15 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

pub fn pack(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len() % 144, 0);
    let mut packed = Vec::with_capacity(raw.len() / 144 * 192);
    for b in raw.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales: &[u8; 12] = b[4..16].try_into().unwrap();
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, scales);
            packed.extend_from_slice(&(d * sc as f32).to_le_bytes());
            packed.extend_from_slice(&(dmin * m as f32).to_le_bytes());
        }
        packed.extend_from_slice(&b[16..144]);
    }
    packed
}

pub fn headers(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(144)
        .flat_map(|b| b[..16].iter().copied())
        .collect()
}

pub fn unpack(packed: &[u8], headers: &[u8]) -> Vec<u8> {
    assert_eq!(packed.len() % 192, 0);
    assert_eq!(headers.len(), packed.len() / 192 * 16);
    let mut raw = Vec::with_capacity(packed.len() / 192 * 144);
    for (b, h) in packed.chunks_exact(192).zip(headers.chunks_exact(16)) {
        raw.extend_from_slice(h);
        raw.extend_from_slice(&b[64..192]);
    }
    raw
}

#[cfg(test)]
mod tests {
    #[test]
    fn metadata_layout_roundtrips_raw_bytes() {
        let mut raw: Vec<u8> = (0..144 * 19).map(|i| (i * 73 + 11) as u8).collect();
        for b in raw.chunks_exact_mut(144) {
            b[..4].copy_from_slice(&[0, 0x30, 1, 0]);
        }
        let packed = super::pack(&raw);
        assert_eq!(packed.len(), raw.len() / 144 * 192);
        assert_eq!(super::unpack(&packed, &super::headers(&raw)), raw);
    }
}
