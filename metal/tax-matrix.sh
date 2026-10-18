#!/usr/bin/env bash
# Consolidated determinism-tax matrix: native (llama.cpp) vs deterministic (vitni-tensor),
# across CPU and GPU, on the SAME model. Run from the metal/ crate dir.
#
#   VITNI_GGUF=/path/model.gguf  N_TOK=32  ./tax-matrix.sh
set -uo pipefail
MODEL="${VITNI_GGUF:?set VITNI_GGUF to a GGUF model path}"
N="${N_TOK:-32}"
echo "============================================================"
echo "  determinism-tax matrix   model: $(basename "$MODEL")   n=$N"
echo "============================================================"

echo
echo "### native baseline  (llama.cpp / llama-bench, token-generation t/s)"
if command -v llama-bench >/dev/null 2>&1; then
  echo "-- CPU (-ngl 0) --"
  llama-bench -m "$MODEL" -p 0 -n "$N" -ngl 0 2>/dev/null | grep -E "tg|t/s|model" | head -4 || echo "(llama-bench CPU run failed)"
  echo "-- Metal GPU (-ngl 99) --"
  llama-bench -m "$MODEL" -p 0 -n "$N" -ngl 99 2>/dev/null | grep -E "tg|t/s|model" | head -4 || echo "(llama-bench GPU run failed)"
else
  echo "(llama-bench not on PATH; skipping native baseline)"
fi

echo
echo "### deterministic (vitni-tensor): serial CPU + GPU"
VITNI_GGUF="$MODEL" N_TOK="$N" cargo run --release --quiet --bin taxmatrix

echo "### deterministic (vitni-tensor): std-parallel CPU + GPU"
VITNI_GGUF="$MODEL" N_TOK="$N" cargo run --release --quiet --features parallel --bin taxmatrix
