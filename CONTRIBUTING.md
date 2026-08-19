# Contributing to vitni-tensor

Thanks for helping build a deterministic, verifiable inference engine. This is a small
`no_std` Rust crate with an outsized correctness bar: its entire reason to exist is that
the same inputs produce the **same output bits on every CPU**. Please read the
determinism rule below before you write a line of code.

## The one rule that is non-negotiable: determinism

> **A change must not alter the reference model-computation digest.**
>
> Reference (TinyLlama-1.1B-Chat, Q4_K_M GGUF, model_id `tinyllama-1.1b-chat-Q4_K_M`,
> prompt `[1, 9038, 2501, 263, 931, 29892]`, `n_new = 20`):
>
> ```
> 9c0754458633e863e0fb5bb2bd00df0d8b813934687b9a4097a1a9a4179f3b0f
> ```

This digest is a published conformance anchor (see the
[receipt spec](https://github.com/vitnify/vitnify-receipt-spec)). A run must reproduce
it byte-for-bit on any vendor or instruction set. **Any PR that changes this digest will
be rejected**, no matter how appealing the speedup or cleanup — unless the PR is
*explicitly* a versioned change to the computation itself, proposed and agreed up front
in an issue, with the new anchor updated in lockstep across the engine, SDK, and spec.

Things that quietly break determinism and therefore will not be accepted:

- reordering a floating-point reduction, or introducing a parallel reduction whose
  order depends on thread count or scheduling;
- relying on FMA contraction, fast-math, or a platform `libm` instead of the pinned
  deterministic `libm`;
- doing k-quant matmuls in floating point instead of exact integers;
- swapping the hash function or changing how digest inputs are length-prefixed.

Parallelism is fine *only* where it cannot touch reduction order — the `std-parallel`
feature splits independent output rows across threads and is verified to be
bit-identical to the serial path. Keep it that way.

## Build & test

Requires a Rust **nightly** toolchain (what the crate is developed against; edition 2021).

```bash
cargo build --release
cargo test --release --lib               # unit + determinism tests
cargo test --release --test gguf_fuzz    # adversarial GGUF parser tests
```

`gguf_fuzz` feeds the loader deliberately malformed byte streams (bad magic, truncated
headers, absurd counts, lengths past EOF) and asserts every one returns `Err` without
panicking, aborting, or hanging. If you touch `src/model/gguf.rs` or any untrusted-input
path, add a case there.

The model-driven integration tests (`tinyllama_gguf`, `mistral_gguf`, `qwen_gguf`, …) are
`#[ignore]`d because they need a GGUF checkpoint that isn't in the repo. Run them locally
against a model when you change the compute path — this is how you confirm the reference
digest still holds:

```bash
VITNI_GGUF=tinyllama.gguf cargo test --release --test tinyllama_gguf -- --ignored --nocapture
```

CI runs `cargo build --release`, `cargo test --release --lib`, and
`cargo test --release --test gguf_fuzz`. It does **not** run the `#[ignore]`d model tests
(no GGUF in CI), so verify the digest locally before you open the PR.

## Pull requests

- Keep changes focused; explain *why*, not just *what*.
- Add or update tests. New parser/robustness behavior belongs in `gguf_fuzz`; new compute
  behavior needs a determinism check.
- Confirm the reference digest is unchanged (run a model locally if your change is
  anywhere near the compute path).
- `cargo fmt` and a warning-clean `cargo build` before you push.

## Sign your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/). Sign off
every commit — it certifies you have the right to contribute the code:

```bash
git commit -s -m "your message"
```

This adds a `Signed-off-by: Your Name <you@example.com>` line. Commits without it will be
asked to amend.

## Reporting bugs vs. vulnerabilities

Ordinary bugs → open a GitHub issue. **Security issues** (a determinism/soundness escape,
or a crash on a malformed model) → **do not** open a public issue; email
[security@vitnify.com](mailto:security@vitnify.com). See [SECURITY.md](SECURITY.md).

By contributing you agree your contributions are licensed under Apache-2.0. Note that the
**vitnify** marks are trademarks — see [TRADEMARKS.md](TRADEMARKS.md).
