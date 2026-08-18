//! Tokenizer module — text ↔ token ID conversion.
//!
//! Each tokenizer format gets its own submodule. They share a loose
//! convention (encode/decode methods) but don't require a trait
//! because the per-format APIs differ enough that a unified trait
//! would obscure rather than simplify.
//!
//! Status:
//!   - `llama2c` — karpathy's llama2.c binary tokenizer format
//!     (used by stories15M / stories110M / TinyLlama and most
//!     llama2.c-format checkpoints). BPE merge based on per-token
//!     scores baked into the blob.
//!   - Future: `huggingface` (JSON tokenizer.json), `sentencepiece`
//!     (proto), etc. — added as model ports demand.

pub mod gguf_bpe;
pub mod gguf_spm;
pub mod llama2c;

pub use gguf_bpe::GgufBpeTokenizer;
pub use gguf_spm::GgufSpmTokenizer;
pub use llama2c::Llama2cTokenizer;
