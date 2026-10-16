#!/usr/bin/env bash
#
# check-anchors.sh — guard regime-manifest.json against what the engine ACTUALLY emits.
#
# regime-manifest.json is the single source of truth for the published determinism
# anchors, but nothing re-derived it when the engine changed: a digest-affecting fix
# (e.g. the GGUF weight-load alignment fixes) once silently drifted the mistral/qwen
# anchors away from the engine's output with no test to catch it. This script closes
# that gap. It re-runs `gen-anchors` and, for every model whose GGUF is available,
# asserts the committed anchor still matches — model_digest, model_digest_v1, tokens,
# regime, and the pinned reduction hash.
#
# It is opt-in on MODEL AVAILABILITY, not on effort: a model whose checkpoint is not
# present is SKIPPED (reported, never a pass-by-omission). So hosted CI runs it with
# just the small TinyLlama checkpoint and a runner that also has Mistral/Qwen extends
# the same guard to them. Point the engine at checkpoints with the same env vars
# `gen-anchors` reads:
#
#   VITNI_GGUF_TINYLLAMA=/path/to/tinyllama.gguf \
#   VITNI_GGUF_MISTRAL=/path/to/mistral.gguf \
#   VITNI_GGUF_QWEN=/path/to/qwen.gguf \
#   tools/check-anchors.sh
#
# Exit 0 iff every AVAILABLE model's anchor matches; non-zero on any drift (then run
# `cargo run --release --bin gen-anchors > regime-manifest.json` and commit).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v jq >/dev/null || { echo "check-anchors: jq is required" >&2; exit 2; }

MANIFEST="regime-manifest.json"
[ -f "$MANIFEST" ] || { echo "check-anchors: $MANIFEST not found" >&2; exit 2; }

echo "check-anchors: building gen-anchors (release)…"
cargo build --release --bin gen-anchors >/dev/null 2>&1

NEW="$(mktemp)"
trap 'rm -f "$NEW"' EXIT
./target/release/gen-anchors > "$NEW"

fail=0
checked=0

# Top-level invariants (regime, prompt, n_new, pinned reduction hash) are not
# model-gated — gen-anchors always emits them, so compare unconditionally.
for key in regime prompt_tokens n_new_tokens pins; do
  want="$(jq -S -c ".$key" "$MANIFEST")"
  got="$(jq -S -c ".$key" "$NEW")"
  if [ "$want" != "$got" ]; then
    echo "DRIFT  .$key"
    echo "       committed: $want"
    echo "       engine:    $got"
    fail=1
  fi
done

# Per-model anchors. gen-anchors emits `null` for a model whose GGUF it could not
# read, so a null entry in its output means "not available here" -> skip.
for name in $(jq -r '.models | keys[]' "$MANIFEST"); do
  regen="$(jq -S -c --arg m "$name" '.models[$m]' "$NEW")"
  if [ "$regen" = "null" ] || [ -z "$regen" ]; then
    echo "SKIP   $name (checkpoint not available)"
    continue
  fi
  want="$(jq -S -c --arg m "$name" '.models[$m]' "$MANIFEST")"
  if [ "$want" = "$regen" ]; then
    echo "OK     $name"
    checked=$((checked + 1))
  else
    echo "DRIFT  $name"
    echo "       committed: $want"
    echo "       engine:    $regen"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "check-anchors: regime-manifest.json is stale vs the engine." >&2
  echo "               run: cargo run --release --bin gen-anchors > regime-manifest.json" >&2
  exit 1
fi

if [ "$checked" -eq 0 ]; then
  echo "check-anchors: no checkpoints available — nothing re-derived (set VITNI_GGUF_*)."
else
  echo "check-anchors: OK — $checked model anchor(s) reproduce the committed manifest."
fi
