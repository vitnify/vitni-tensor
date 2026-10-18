// Compile the deterministic CUDA forward (vt_cuda.cu) into a static lib with
// nvcc, baking in the determinism flags, then link it + the CUDA runtime.
//
//   --fmad=false : no mul+add -> fma contraction (matches the CPU's separate ops)
//   -ftz=false   : IEEE denormals (correctly-rounded)
//   -arch=sm_XX  : real-SASS target (no PTX JIT that could re-contract). Set the
//                  arch for the box via VITNI_CUDA_ARCH (default sm_75 = T4);
//                  use sm_89 for L40S/Ada, sm_90 for H100.
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = "vt_cuda.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=VITNI_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let arch = env::var("VITNI_CUDA_ARCH").unwrap_or_else(|_| "sm_75".into());
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = format!("{cuda_home}/bin/nvcc");
    let nvcc = if PathBuf::from(&nvcc).exists() { nvcc } else { "nvcc".into() };

    let lib = out.join("libvtcuda.a");
    let status = Command::new(&nvcc)
        .args(["--fmad=false", "-ftz=false", "-O2", "-std=c++14"])
        .arg(format!("-arch={arch}"))
        .args(["-lib", src, "-o"])
        .arg(&lib)
        .status()
        .expect("failed to run nvcc — is the CUDA toolkit installed? set CUDA_HOME");
    assert!(status.success(), "nvcc failed to build {src}");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vtcuda");

    // CUDA runtime + C++ runtime nvcc links against.
    for p in [
        format!("{cuda_home}/lib64"),
        format!("{cuda_home}/targets/x86_64-linux/lib"),
    ] {
        if PathBuf::from(&p).exists() {
            println!("cargo:rustc-link-search=native={p}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
