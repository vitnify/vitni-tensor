# Fuzzing the untrusted-input surface

`GgufFile::parse` (and `Config::from_gguf` downstream) consume **untrusted bytes** — a
model file a user downloaded, or one an attacker crafted. In a `no_std` crate, a bug in
that parser is the highest-value memory-safety / DoS target in the engine. These
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer) targets assert the
only thing a parser owes an attacker: any input returns `Err` — never a panic, an
out-of-bounds read, an overflow, or a runaway allocation.

## Targets
- **`gguf_parse`** — the pure parser: magic, version, tensor/metadata counts, metadata
  KV pairs, tensor descriptors, offsets.
- **`gguf_config`** — parse **+** `Config::from_gguf`: metadata *interpretation* (dims,
  head counts, RoPE base/eps — the divisions and casts on untrusted values, e.g.
  `head_size = dim / n_heads`, where a hostile `n_heads = 0` would divide by zero).

## Run
```sh
rustup toolchain install nightly     # cargo-fuzz needs nightly
cargo install cargo-fuzz
cargo fuzz run gguf_parse  -- -max_total_time=120 -max_len=262144
cargo fuzz run gguf_config -- -max_total_time=120 -max_len=262144
```
Seed the corpus from a real file for faster coverage (structural prefix is enough — the
parser reads descriptors, not the tensor payload):
```sh
head -c 262144 path/to/model.gguf > fuzz/corpus/gguf_parse/seed
```

## Status
Both targets run **clean** (no crash, no OOM) on the corpus seeded from the stock
TinyLlama-1.1B Q4_K_M GGUF — `gguf_parse` at ~26M runs / 2 min, corpus grown to ~450
coverage-increasing inputs. A crash writes a reproducer to `fuzz/artifacts/<target>/`;
replay it with `cargo fuzz run <target> fuzz/artifacts/<target>/<file>`.

## CI
A short smoke run gates PRs; a longer nightly run gives depth:
```sh
cargo fuzz run gguf_parse  -- -max_total_time=30 -max_len=262144
cargo fuzz run gguf_config -- -max_total_time=30 -max_len=262144
```
Cache `fuzz/corpus/` between runs so coverage compounds.
