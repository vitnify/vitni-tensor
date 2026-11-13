"""Explicit inference-only operators. Import does not alter PyTorch operators."""
from ._ops import canonical_matmul, q4_k_linear, backend_info

__version__ = "0.1.0a1"
__all__ = ["canonical_matmul", "q4_k_linear", "backend_info"]
