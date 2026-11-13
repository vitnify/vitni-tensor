"""Registered CPU/MPS operators with an explicit, inference-only contract."""
from __future__ import annotations

import collections
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import struct
import sys

import torch
from torch import Tensor

MAX_K = 64 * 8192
MAX_NUMEL = 2**31 - 1
CONTRACT = "vitni-lane8-chunk8192-separate-f32-v1"
_COUNTS = collections.Counter()
_MPS_PROBES = {}


def _metadata(a: Tensor, b: Tensor, quantized: bool = False) -> None:
    if a.device != b.device or a.device.type not in ("cpu", "mps"):
        raise ValueError("Vitni operators require both inputs on the same CPU or MPS device")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("Vitni operators require strided tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Vitni operators require contiguous tensors; call contiguous() explicitly")
    if a.dtype != torch.float32 or b.dtype != (torch.uint8 if quantized else torch.float32):
        raise TypeError("Expected float32 activations and " + ("uint8 Q4_K blocks" if quantized else "float32 weights"))
    if a.ndim != (1 if quantized else 2) or b.ndim != 2:
        raise ValueError("Expected a vector and rank-2 packed weights" if quantized else "Expected rank-2 matrices")
    k = a.shape[-1]
    torch._check(k <= MAX_K, lambda: "K exceeds the supported 64 canonical chunks")
    torch._check(a.numel() <= MAX_NUMEL, lambda: "Activation extent exceeds the kernel index range")
    torch._check(b.numel() <= MAX_NUMEL, lambda: "Weight extent exceeds the kernel index range")
    if quantized:
        torch._check(k > 0, lambda: "Q4_K requires positive K")
        torch._check(k % 256 == 0, lambda: "Q4_K requires K divisible by 256")
        torch._check(b.shape[1] == (k // 256) * 144, lambda: "Expected raw 144-byte GGUF Q4_K blocks per row")
    else:
        torch._check(k == b.shape[0], lambda: "Matrix contraction dimensions differ")
        torch._check(a.shape[0] * b.shape[1] <= MAX_NUMEL, lambda: "Output extent exceeds the kernel index range")


def _finite(tensor: Tensor, label: str) -> None:
    if tensor.numel() and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} must be finite; nonfinite results and overflow are outside this operator contract")


def _domain(tensor: Tensor, label: str) -> None:
    # Integer classification is essential: MPS floating comparisons can already
    # treat a subnormal input as zero. Both devices enforce the same domain.
    if tensor.numel():
        word = tensor.view(torch.int32) & 0x7fffffff
        valid = (word == 0) | ((word >= 0x2f800000) & (word <= 0x4f800000))
        if not valid.all().item():
            raise ValueError(f"{label} must be finite and zero or have magnitude in [2**-32, 2**32]; subnormals and out-of-range values are unsupported")


def _training(*inputs: Tensor) -> None:
    if any(x.requires_grad for x in inputs):
        raise RuntimeError("Vitni operators are inference-only; detach inputs explicitly")


@lru_cache(maxsize=1)
def _cpu_lib():
    name = ("vitni_torch_native.dll" if sys.platform == "win32" else
            "libvitni_torch_native.dylib" if sys.platform == "darwin" else
            "libvitni_torch_native.so")
    path = Path(os.environ.get("VITNI_TORCH_LIBRARY", Path(__file__).with_name(name)))
    if not path.is_file():
        raise RuntimeError("Vitni native library is missing; install the built wheel or build this repository's Python package")
    lib = ctypes.CDLL(str(path))
    lib.vitni_torch_abi_version.restype = ctypes.c_uint32
    if lib.vitni_torch_abi_version() != 1:
        raise RuntimeError("Incompatible Vitni native ABI")
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    lib.vitni_torch_matmul.argtypes = [pointer, size, pointer, size, pointer, size, size, size, size]
    lib.vitni_torch_matmul.restype = ctypes.c_int
    lib.vitni_torch_q4_k_linear.argtypes = [pointer, size, pointer, size, pointer, size, size, size]
    lib.vitni_torch_q4_k_linear.restype = ctypes.c_int
    return lib


def _status(code: int) -> None:
    if code:
        meanings = {1: "invalid inputs", 2: "engine rejected the operation", 3: "native engine panic",
                    4: "nonfinite result or overflow", 5: "CPU floating-point environment changes rounding or flushes subnormals"}
        raise RuntimeError(f"Vitni: {meanings.get(code, 'unknown native error')}")


_PROBE = """
kernel void vitni_contract_probe(device const float* x [[buffer(0)]], device float* y [[buffer(1)]]) {
    float p = x[0] * x[1];
    y[0] = p + x[2];
    y[1] = fma(x[0], x[1], x[2]);
    y[2] = x[3] * x[4];
    y[3] = x[5] * x[4];
}
"""


@lru_cache(maxsize=2)
def _mps_lib(kind: str):
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Actual MPS execution and torch.mps.compile_shader are required")
    source = (Path(__file__).parent / "_shaders" / ("q4k.metal" if kind == "q4" else "canonical_matmul.metal")).read_text()
    prefix = "#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n#define VITNI_Q4K_NO_CONTRACT 1\n"
    library = torch.mps.compile_shader(prefix + source + "\n" + _PROBE)
    bits = [0x3f800001, 0x3f7ffffe, 0xbf800000, 1, 0x3f800000, 0x80000000]
    bits = [x if x < 2**31 else x - 2**32 for x in bits]
    values = torch.tensor(bits, dtype=torch.int32).view(torch.float32).to("mps")
    result = torch.empty(4, dtype=torch.float32, device="mps")
    library.vitni_contract_probe(values, result, threads=1, group_size=1)
    fused = struct.unpack("<i", struct.pack("<f", -(2.0**-46)))[0]
    actual = result.cpu().view(torch.int32).tolist()
    if actual[0] != 0 or actual[1] != fused or actual[3] != -(2**31) or actual[2] not in (0,1):
        raise RuntimeError("MPS compiler does not honor the required separate-rounding/signed-zero contract")
    _MPS_PROBES[kind] = {"separate_bits": actual[0], "fma_bits": actual[1],
                         "subnormal_bits": actual[2], "negative_zero_bits": actual[3]}
    return library


@torch.library.custom_op("vitni::canonical_matmul", mutates_args=(), device_types=("cpu", "mps"))
def _matmul_op(a: Tensor, b: Tensor) -> Tensor:
    _metadata(a, b)
    _training(a, b)
    _domain(a, "Left input")
    _domain(b, "Right input")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    result = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if result.numel() == 0:
        return result
    if a.device.type == "cpu":
        _status(_cpu_lib().vitni_torch_matmul(a.data_ptr(), a.numel(), b.data_ptr(), b.numel(),
                                            result.data_ptr(), result.numel(), m, k, n))
        _COUNTS["cpu_matmul"] += 1
    else:
        dims = torch.tensor([m, k, n], dtype=torch.int32, device="mps")
        _mps_lib("matmul").canonical_matmul(a, b, result, dims,
            threads=((m*n+63)//64)*64, group_size=64)
        _COUNTS["mps_matmul"] += 1
    _finite(result, "Output")
    return result


@_matmul_op.register_fake
def _matmul_fake(a: Tensor, b: Tensor) -> Tensor:
    _metadata(a, b)
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op("vitni::q4_k_linear", mutates_args=(), device_types=("cpu", "mps"))
def _q4_op(x: Tensor, weights: Tensor) -> Tensor:
    _metadata(x, weights, quantized=True)
    _training(x)
    _domain(x, "Input")
    m, k = weights.shape[0], x.shape[0]
    result = torch.empty((m,), dtype=torch.float32, device=x.device)
    if m == 0:
        return result
    if x.device.type == "cpu":
        _status(_cpu_lib().vitni_torch_q4_k_linear(x.data_ptr(), x.numel(), weights.data_ptr(), weights.numel(),
                                                result.data_ptr(), result.numel(), m, k))
        _COUNTS["cpu_q4_k"] += 1
    else:
        dims = torch.tensor([k, m, k//256], dtype=torch.int32, device="mps")
        _mps_lib("q4").q4k_linear_cooperative(x, weights, result, dims,
            threads=((m*8+63)//64)*64, group_size=64)
        _COUNTS["mps_q4_k"] += 1
    _finite(result, "Output")
    return result


@_q4_op.register_fake
def _q4_fake(x: Tensor, weights: Tensor) -> Tensor:
    _metadata(x, weights, quantized=True)
    return x.new_empty((weights.shape[0],))


def canonical_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Rank-2 contiguous float32 matmul using Vitni's pinned reduction order."""
    _training(a, b)
    return _matmul_op(a, b)


def q4_k_linear(x: Tensor, weights: Tensor) -> Tensor:
    """GGUF raw Q4_K rows [M, K/256*144] times float32 vector [K]."""
    _training(x)
    return _q4_op(x, weights)


def backend_info() -> dict:
    """Report the actual supported surface and completed arithmetic dispatches."""
    return {"contract": CONTRACT, "torch": torch.__version__, "inference_only": True,
            "supported_devices": ["cpu", "mps"], "native_abi": _cpu_lib().vitni_torch_abi_version(),
            "nonzero_input_magnitude": [2.0**-32, 2.0**32],
            "mps_available": torch.backends.mps.is_available(), "dispatches": dict(_COUNTS),
            "mps_compiler_probes": dict(_MPS_PROBES)}
