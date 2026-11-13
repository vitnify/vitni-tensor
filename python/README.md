# vitni-torch (experimental, opt-in)

This research checkout also includes a resident full-model `VitniModel`
wrapper. See [the full-model guide](FULL_MODEL.md) for its distinct API,
CPU/Metal execution, measurements and limits. This addition is not part of
the published 0.1.0a1 preview; the research package is versioned 0.1.0a2.

Explicit CPU and Apple MPS operators using Vitni's arithmetic contract:

- `canonical_matmul(a, b)`: contiguous rank-2 float32 matrices.
- `q4_k_linear(x, weights)`: float32 vector and raw GGUF Q4_K uint8 rows.

CPU computation calls the actual Rust engine through a small C ABI. MPS
computation runs the engine's Metal kernels directly on MPS tensors through
`torch.mps.compile_shader`; it does not copy the numerical operation to CPU.
Import registers the `vitni` namespace without replacing PyTorch operators or
changing PyTorch's global determinism settings.

Build/install from this repository with Rust/Cargo and Python 3.10+ available:

```sh
python -m pip install '.[torch]'
```

```python
import torch
import vitni_torch

a = torch.randn(4, 256, dtype=torch.float32)
b = torch.randn(256, 8, dtype=torch.float32)
cpu = vitni_torch.canonical_matmul(a, b)
mps = vitni_torch.canonical_matmul(a.to('mps'), b.to('mps')).cpu()
assert torch.equal(cpu.view(torch.int32), mps.view(torch.int32))
```

This preview requires PyTorch 2.14+ and supports inference only. Detach inputs
explicitly; there is no backward formula. Inputs must be contiguous, finite,
and on the same supported device. Each floating input must be zero or have
magnitude in `[2**-32, 2**32]`, on both devices. Integer bit classification
rejects subnormals and out-of-range inputs before arithmetic. Q4_K headers
use the ordinary finite GGUF half-scale encodings; nonfinite results are
rejected. The MPS kernel supports at most 64 chunks of 8192 input elements.
Non-contiguous inputs, other dtypes, CUDA, batched matmul, and automatic
replacement of an arbitrary PyTorch model are outside this preview's scope.

The package registers fake kernels for both operators. Local tests exercise
`torch.library.opcheck`, `torch.compile` with eager/AOT-eager/Inductor backends,
and export/reload on CPU and MPS. These registrations cover these operators;
they do not automatically convert an existing model.

The canonical output is defined by Vitni's numerical rules and need not match
`torch.matmul`'s native arithmetic. Exact CPU/MPS equality must be tested with
integer bit views, not `allclose`. Registration/FakeTensor/compilation checks
are separate from those numerical checks. MPS compilation probes separate
rounding, explicit FMA and signed zero and fails if the contract is not
honored. It also reports a subnormal probe: the tested PyTorch MPS shader path
flushes subnormals, which is why this preview enforces the restricted domain.
CPU calls reject ambient flush-to-zero/non-nearest settings.

With these input bounds, matrix products lie between `2**-64` and `2**64`.
Rounded products share a grid no finer than `2**-87`, so cancellation cannot
create a nonzero subnormal intermediate; at most `2**19` terms cannot overflow.
Finite Q4_K dequantization uses half values and bounded integer scales; its
nonzero weight grid is at least `2**-24`, and the same activation bounds keep
its products and sums in the normal float32 range. This is a deliberately
restricted operator contract, not support for arbitrary float32 bit patterns.

Finite-value validation can synchronize MPS work. Include this and transfer,
allocation and first-call compilation costs when measuring an application.
The project's approximately 2.47x TinyLlama/llama.cpp Metal forward-time ratio
does not describe these general PyTorch operators; benchmark them separately.
