# Security Policy

vitni-tensor is the deterministic engine behind a verification product. A bug here
is not just a crash — it can be a way to make a run *look* reproducible when it isn't,
or to take down a process that is parsing an untrusted model file. We take reports
seriously and we want to hear from you.

## Reporting a vulnerability

**Email [security@vitnify.com](mailto:security@vitnify.com).** Please do **not** open
a public GitHub issue, pull request, or discussion for a suspected vulnerability —
that discloses it to everyone before there is a fix.

If you can, encrypt or otherwise protect any proof-of-concept; a private GitHub
Security Advisory (Security ▸ *Report a vulnerability*) is also fine.

A useful report includes:

- the affected component (e.g. the GGUF loader in `src/model/gguf.rs`, a quant kernel,
  the receipt/digest path);
- a minimal input that triggers it — for parser bugs, the raw bytes or a builder
  snippet like the ones in `tests/gguf_fuzz.rs`;
- what you observed (panic, abort, hang, out-of-bounds read, a digest that should not
  reproduce or should not have matched) and what you expected;
- the target/toolchain (`rustc -Vv`, host architecture, and any `--features`).

## What's in scope

This is a `no_std` inference core whose whole value is bit-exact reproducibility, so
we especially want to know about:

- **Memory safety and robustness on untrusted input** — the GGUF file is
  attacker-controllable (downloaded checkpoints, untrusted CAS blobs). Any malformed
  or hostile input that causes a panic, abort, hang, unbounded allocation, or
  out-of-bounds access is in scope.
- **Determinism / soundness escapes** — any way to make the same inputs produce a
  *different* `vitnify-receipt v1` model-computation digest on some CPU or instruction
  set, or to make two genuinely different computations collide on the same digest, or
  otherwise forge a digest without doing the computation. These are the crown jewels.
- **`unsafe` misuse** or undefined behavior anywhere in the crate.

## Not in scope

- The trust-boundary limits that are already documented (an embedded ed25519 key
  proves signer continuity, not runtime authority — see the receipt spec). That is a
  known, stated limit, not a vulnerability.
- Non-determinism you introduce by editing the reduction order, enabling non-default
  features, or building with a different math/hash library than the pinned ones.
- Denial of service from feeding an enormous but *well-formed* model (resource limits
  are the caller's responsibility).

## Response expectations

- We will **acknowledge your report within 3 business days**.
- We will confirm the issue and share an assessment, typically within 10 business days.
- We will keep you updated as we work on a fix and coordinate a disclosure timeline
  with you. We are glad to credit you in the release notes and advisory unless you
  prefer to remain anonymous.

## Safe harbor

We will not pursue or support legal action against anyone who, in good faith, follows
this policy while investigating or reporting a security issue — including accessing
only what is necessary to demonstrate the problem, avoiding privacy violations and
service disruption, and giving us reasonable time to respond before any public
disclosure. If in doubt, ask us at security@vitnify.com first. We consider good-faith
security research to be authorized conduct and will work with you.
