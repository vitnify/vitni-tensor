# vitni-tensor

**Deterministic, `no_std` tensor engine for verifiable LLM inference.** The same
inputs produce the **same output bits on different CPUs** — across vendors and
instruction sets — by fixing reduction order in floating point and doing k-quant
matmuls in exact integers. It runs stock GGUF models and emits a `vitnify-receipt v1`
model-computation digest for every run.

vitni-tensor is the deterministic engine under [vitnify](https://vitnify.com) —
execution receipts for AI agents.

## Why

Floating-point inference is non-deterministic across hardware (parallel reductions,
FMA fusion, vendor `libm`). vitni-tensor removes every source of divergence on the CPU
path, so a run can be **reproduced bit-for-bit** by an independent party and certified.

## What's inside

- `no_std` core; the only dependencies are a math library and a hash function.
- GGUF loader; grouped-query attention; interleaved (Llama) and NeoX (Qwen2) RoPE;
  optional attention biases; RMSNorm; SwiGLU; KV cache.
- Quantized matmul kernels: **F32 · Q4_0 · Q4_K · Q5_0 · Q6_K · Q8_0**.
- Pinned-order fp32 dot + exact-integer k-quant dots; deterministic `libm` transcendentals.
- A per-run **`vitnify-receipt v1`** model-computation digest (BLAKE3).

## The receipt binary

```
cargo build --release --bin vitni-receipt
./target/release/vitni-receipt --gguf model.gguf --prompt "1,2,3" --n 20 --model-id my-model
# -> {"model_digest":"…","weights_hash":"…","tokens":[…]}
```

The vitnify SDK calls this to bind the model's computation into a signed execution receipt.

## Tests

`cargo test` runs the unit and determinism tests. The model-driven integration tests
are `#[ignore]`d (they need a GGUF that isn't in the repo); run them with a model:

```
VITNI_GGUF=model.gguf cargo test --release --test tinyllama_gguf -- --ignored --nocapture
```

## Determinism, measured

Across TinyLlama-1.1B, Mistral-7B, and Qwen2.5 on Apple, Arm/AWS Graviton, Intel, and
AMD (two ISAs, two compiler builds), the same run produces a byte-identical certificate
digest. Details in the paper, *Reconstructable Execution Certificates for Tool-Using AI
Agents*.

## License

Apache-2.0. **"vitnify"** and **"vitnify-verified"** are trademarks — see
[TRADEMARKS.md](TRADEMARKS.md). A fork may use the code, not the name.
