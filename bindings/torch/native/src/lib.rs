//! Small C ABI over the actual Vitni CPU operations. No libtorch ABI dependency.
use std::{hint::black_box, panic::catch_unwind, slice};
use vitni_tensor::{ops::quant::linear_q4_k_fused, Shape, Storage, Tensor};

pub mod full_model;

#[no_mangle]
pub extern "C" fn vitni_torch_abi_version() -> u32 {
    1
}

fn environment_ok() -> bool {
    // Detect an ambient flush-to-zero or non-nearest rounding environment;
    // never change the caller's thread/global floating-point settings.
    let subnormal = black_box(f32::from_bits(1)) * black_box(1.0f32);
    let tie = black_box(f32::from_bits(0x3f800001)) + black_box(f32::from_bits(0x33800000));
    let even_tie = black_box(1.0f32) + black_box(f32::from_bits(0x33800000));
    subnormal.to_bits() == 1 && tie.to_bits() == 0x3f800002 && even_tie.to_bits() == 0x3f800000
}

fn lengths(m: usize, k: usize, n: usize) -> Option<(usize, usize, usize)> {
    let a = m.checked_mul(k)?;
    let b = k.checked_mul(n)?;
    let out = m.checked_mul(n)?;
    if [a, b, out].iter().any(|&x| x > isize::MAX as usize / 4) {
        return None;
    }
    Some((a, b, out))
}

unsafe fn input<'a, T>(pointer: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    if pointer.is_null() || pointer as usize % std::mem::align_of::<T>() != 0 {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(pointer, len) })
}

/// Callers own valid buffers of the stated lengths for the duration of the call.
/// Returns 0 success, 1 invalid metadata, 2 engine error, 3 panic, 4 nonfinite
/// result, or 5 unsupported ambient CPU floating-point environment.
#[no_mangle]
pub unsafe extern "C" fn vitni_torch_matmul(
    a: *const f32,
    a_len: usize,
    b: *const f32,
    b_len: usize,
    out: *mut f32,
    out_len: usize,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    catch_unwind(|| {
        if !environment_ok() {
            return 5;
        }
        if lengths(m, k, n) != Some((a_len, b_len, out_len)) || (out_len > 0 && out.is_null()) {
            return 1;
        }
        let Some(a) = (unsafe { input(a, a_len) }) else {
            return 1;
        };
        let Some(b) = (unsafe { input(b, b_len) }) else {
            return 1;
        };
        if a.iter().chain(b).any(|x| !x.is_finite()) {
            return 1;
        }
        let Ok(a) = Tensor::from_f32(
            a.to_vec(),
            match Shape::new(&[m, k]) {
                Ok(s) => s,
                Err(_) => return 1,
            },
        ) else {
            return 2;
        };
        let Ok(b) = Tensor::from_f32(
            b.to_vec(),
            match Shape::new(&[k, n]) {
                Ok(s) => s,
                Err(_) => return 1,
            },
        ) else {
            return 2;
        };
        let Ok(y) = a.matmul(&b) else {
            return 2;
        };
        let Storage::Cpu(values) = y.storage() else {
            return 2;
        };
        let values = values.as_f32_slice();
        if values.iter().any(|x| !x.is_finite()) {
            return 4;
        }
        if out_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), out, out_len);
            }
        }
        0
    })
    .unwrap_or(3)
}

#[no_mangle]
pub unsafe extern "C" fn vitni_torch_q4_k_linear(
    x: *const f32,
    x_len: usize,
    weights: *const u8,
    weights_len: usize,
    out: *mut f32,
    out_len: usize,
    m: usize,
    k: usize,
) -> i32 {
    catch_unwind(|| {
        if !environment_ok() {
            return 5;
        }
        if k == 0
            || k % 256 != 0
            || x_len != k
            || out_len != m
            || m.checked_mul(k / 256).and_then(|v| v.checked_mul(144)) != Some(weights_len)
            || (out_len > 0 && out.is_null())
        {
            return 1;
        }
        let Some(x) = (unsafe { input(x, x_len) }) else {
            return 1;
        };
        let Some(weights) = (unsafe { input(weights, weights_len) }) else {
            return 1;
        };
        if x.iter().any(|x| !x.is_finite()) {
            return 1;
        }
        let mut y = vec![0.0f32; m];
        if linear_q4_k_fused(x, weights, &mut y, 1, k, m).is_err() {
            return 2;
        }
        if y.iter().any(|x| !x.is_finite()) {
            return 4;
        }
        if m > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(y.as_ptr(), out, m);
            }
        }
        0
    })
    .unwrap_or(3)
}
