<!-- Thanks for contributing to vitni-tensor. Please complete the checklist. -->

## What this changes

Briefly describe the change and the motivation.

## Determinism

The reference digest is a published conformance anchor and must not change:

```
9c0754458633e863e0fb5bb2bd00df0d8b813934687b9a4097a1a9a4179f3b0f
```

- [ ] **The reference model-computation digest is unchanged.** (If it changes, this PR
      must be an explicit, agreed-upon versioned migration — link the issue.)
- [ ] I did not reorder any floating-point reduction, rely on FMA/fast-math, or swap the
      pinned `libm`/hash.

## Checklist

- [ ] `cargo test --release --lib` passes
- [ ] `cargo test --release --test gguf_fuzz` passes
- [ ] For compute-path changes: I ran a model locally (`VITNI_GGUF=… --ignored`) and
      confirmed the reference digest still reproduces
- [ ] `cargo fmt` applied; build is warning-clean
- [ ] Tests added/updated for the change
- [ ] Commits are signed off (DCO): `git commit -s`

## Related issues

Closes #
