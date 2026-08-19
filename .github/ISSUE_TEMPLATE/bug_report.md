---
name: Bug report
about: A crash, an incorrect result, or a determinism divergence in vitni-tensor
title: "[bug] "
labels: bug
---

<!--
SECURITY: if this is a determinism/soundness escape (a way to make the same inputs
produce a different digest, or two computations collide on one digest) or a crash on a
malformed model file, do NOT file it here. Email security@vitnify.com — see SECURITY.md.
-->

## What happened

A clear description of the bug.

## Determinism impact

- [ ] This changes / could change the model-computation digest
- [ ] Crash, panic, abort, or hang
- [ ] Wrong numerical result
- [ ] Other

If a digest is involved, paste the expected and actual digests.

## Reproduction

Steps or a minimal snippet. For loader/parser bugs, the raw bytes or a builder like the
ones in `tests/gguf_fuzz.rs` is ideal.

```
# commands / code
```

## Environment

- `rustc -Vv`:
- Host architecture (e.g. aarch64-apple-darwin, x86_64 AVX2):
- Features enabled (e.g. `--features std-parallel`):
- Model / quant, if applicable (e.g. TinyLlama-1.1B Q4_K_M):

## Additional context

Backtrace (`RUST_BACKTRACE=1`), logs, or anything else that helps.
