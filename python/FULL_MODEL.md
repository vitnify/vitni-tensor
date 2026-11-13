# Opt-in resident full-model wrapper

Available in vitni-tensor v0.3.3 / vitni-torch 0.1.0a2 as an experimental,
explicitly opt-in API. It wraps the existing full inference engine. Importing
the package does not open a model, replace PyTorch operations or select an
engine backend. Construct `VitniModel` explicitly to use it.

Build/install this checkout with Cargo and Python/PyTorch available:

```sh
python -m pip install '.[torch,test]'
```

On macOS the package also builds the native Metal backend. CPU-only builds
do not require Metal. The optional backend is selected per model instance;
opening a model does not change process-wide optimization environment flags.

```python
import torch
from vitni_torch import VitniModel

ids = torch.tensor([1, 450, 11781, 338], dtype=torch.int64)
with VitniModel('model.gguf', backend='metal', optimized=True,
               model_id='my-model') as model:
    logits = model(ids)                 # CPU float32 [tokens, vocabulary]
    receipt = model.generate(ids, n_new=16)
    print(receipt['tokens'], receipt['model_digest'])
    print(model.info())                 # actual engine flags and call counts
```

`backend='cpu'` uses the core's quantized CPU forward. `backend='metal',
optimized=True` constructs the actual resident `MetalForward` with the retained
Q4 metadata/RMS/Q6/attention/concurrent-dispatch configuration. The numerical
regime and generation/certificate driver are the existing engine's.

The GPU engine owns its weights, cached metadata, scratch buffers and KV cache
for the model's lifetime. The CPU bridge retains the GGUF bytes and KV cache;
it reconstructs borrowed weight views per request without re-reading the file
or copying matrices. This avoids unsafe self-referential model storage.
`close()` or leaving the context manager releases the resident model.
Metal requests use explicit autorelease scopes so temporary command objects
are drained in a persistent Python process.

Each call starts a fresh sequence. Calls reuse resident weights and KV allocations;
ordinary engine temporaries and returned tensors can still allocate.
They do not append to a previous request's conversation. The native owner
serializes calls to each model, and closed/forked handles are rejected.
Returned tensors own their output storage.

## Exact replay

```python
with VitniModel('model.gguf', backend='cpu', model_id='same') as cpu, \
     VitniModel('model.gguf', backend='metal', optimized=True,
                model_id='same') as metal:
    limit = metal.info()['max_context']
    vocab = cpu.info()['vocab_size']
    history = torch.tensor([(137 + i * 11731) % vocab for i in range(limit)])
    a, b = cpu(history), metal(history)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert cpu.generate(ids, n_new=4) == metal.generate(ids, n_new=4)
```

The forward check compares every logit's bits. Receipt equality compares the
entire existing schema, including output tokens, both digest versions, regime
and weights hash. Receipt equality by itself does not compare all logits.

## Scope

- Eager inference through `torch.nn.Module`; no training, gradients, batched
  inputs, arbitrary PyTorch model conversion or Hugging Face compatibility.
- Input: contiguous rank-1 CPU int64 token IDs. Output: CPU float32 logits.
  `backend='metal'` describes the native engine's execution; outputs are not
  PyTorch MPS tensors. This bridge does not use PyTorch's MPS allocator/stream.
- At most `min(model_context,512)` positions per request, including generation.
  The original model configuration remains bound by the receipt.
- Current Metal projection types: Q4_K/Q6_K/Q5_0/Q8_0 with valid block widths;
  the existing engine's dimension limits also apply.
- Native handles cannot be exported, pickled or saved as a `state_dict`.
  Reopen the GGUF explicitly. `.to()`/dtype conversion is rejected; select
  the backend in the constructor. The forward is explicitly an eager boundary,
  without a full-graph `torch.compile`/export claim.
- The earlier individual operators' `[2**-32,2**32]` input restriction concerns
  their PyTorch shader path. This full-engine API accepts token IDs and uses
  the native engine instead. The finite-input Metal SiLU overflow, large-divisor
  and underflow cases are corrected in v0.3.3; see
  [the fix and validation](../docs/research/silu-overflow-2026-11-13.md).
  These tests do not establish universal IEEE-754 equality for every engine op
  or define cross-hardware NaN-payload behavior.
- Physical CUDA, other model families and other GPU generations are not
  established by this research run.

## Reproduce the comparison

On the tested Mac, build the resident Rust control and package:

```sh
export CARGO_TARGET_DIR="$PWD/build/native"
cargo build --release --manifest-path bindings/torch/native/Cargo.toml \
  --features metal --bin full-model-worker
python -m pip install '.[torch,test]'
VITNI_REQUIRE_MPS=1 VITNI_REQUIRE_METAL=1 VITNI_REQUIRE_FULL_MODEL=1 \
  VITNI_FULL_MODEL_GGUF=model.gguf python -m pytest -q python/tests
```

Save a JSON array of valid token IDs as `tokens.json`, then run:

```sh
python python/examples/full_model_benchmark.py --gguf model.gguf \
  --worker build/native/release/full-model-worker --backend metal \
  --tokens-json tokens.json --output-dir measured-metal --repetitions 9
```

Use `--backend cpu` for the CPU comparison. The driver alternates Rust and
PyTorch order, checks full-logit bytes or complete receipts on every repetition,
and reports first calls, startup, engine time and Python wall time separately.
Rust timers exclude process IPC, file writes and the preallocated output
buffer; PyTorch wall time includes module dispatch, validation, allocation,
FFI and conversion. Both execute synchronous completed engine work and reset
the sequence for every request. No Torch import or model load is included in
warm timings. Startup observations share the filesystem cache.

The control is the same resident engine/session invoked directly from Rust,
not a cold CLI process per repetition. The research report also checks the
unmodified v0.3.2 receipt CLIs independently. The older approximately 2.47x
comparison to ordinary llama.cpp is a separate experiment and is not remeasured
by this wrapper benchmark.

See [the research results and test plan](../docs/research/torch-full-model-2026-11-13.md)
for the measured costs, exact replay coverage and source/binary provenance.

Hosted CI sets `VITNI_FULL_MODEL_CPU_ONLY=1` for the real-model CPU contract.
The local hardware gate above requires native Metal as well as PyTorch MPS;
CPU-only results do not count as GPU validation.
