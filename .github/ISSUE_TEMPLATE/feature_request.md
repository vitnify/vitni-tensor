---
name: Feature request
about: Propose a new capability for vitni-tensor (a kernel, quant type, op, or target)
title: "[feature] "
labels: enhancement
---

## What do you want to be able to do

Describe the capability and the use case.

## Determinism plan

vitni-tensor's contract is bit-identical output across CPUs. Explain how the proposal
keeps that:

- Does it touch a floating-point reduction, quant matmul, or transcendental path?
- If so, how do you keep reduction order fixed and independent of thread count / ISA?
- Does it change the reference digest? (A yes needs an explicit, versioned migration
  agreed up front — see CONTRIBUTING.md.)

## Proposed approach

Sketch the design: new kernel/op, new quant type (e.g. Q3_K), a new target, etc.

## Alternatives considered

What else did you weigh, and why this?

## Additional context

Links, references, papers, or reference numbers.
