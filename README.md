# tenferro-einsum-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [tenferro-einsum](https://github.com/tensor4all/tenferro-rs)

Only metadata is stored — tensors are generated at benchmark time (zero-filled), keeping the repo lightweight.

The structure mirrors [strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) for consistency and cross-comparison.

See [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) for the full library.

## Project Structure

```
tenferro-einsum-benchmark/
  src/
    main.rs                 # Rust benchmark runner (tenferro-einsum)
  scripts/
    run_all.sh              # Top-level orchestrator (delegates to run_all_rust.sh)
    run_all_rust.sh         # Build & run tenferro-einsum + strided-rs (faer)
    generate_dataset.py     # Filter & export benchmark instances as JSON
    format_results.py       # Parse logs and output markdown tables
  data/
    instances/              # Exported JSON metadata (one file per instance)
    results/                # Benchmark logs and markdown results
  Cargo.toml                # Rust project
  pyproject.toml            # Python project
```

## Setup

### Python (dataset export)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Rust

Requires a local clone of [tenferro-rs](https://github.com/tensor4all/tenferro-rs) at `../tenferro-rs`.

```bash
cargo build --release
```

## Usage

### 1. Export benchmark metadata

```bash
uv run python scripts/generate_dataset.py
```

This selects instances by category with laptop-scale criteria and saves JSON metadata to `data/instances/`. `rnd_mixed_` instances are excluded (not yet supported by tenferro-einsum).

**Selection criteria (per category):**

| Category | Prefix | log10[FLOPS] | log2[SIZE] | num_tensors | dtype |
|----------|--------|--------------|------------|-------------|-------|
| Language model | `lm_` | < 10 | < 25 | ≤ 100 | float64 or complex128 |
| Graphical model | `gm_` | < 10 | < 27 | ≤ 200 | float64 or complex128 |
| Structured | `str_` | < 11 | < 26 | ≤ 200 | float64 or complex128 |

### 2. Run all benchmarks

```bash
./scripts/run_all.sh          # 1 thread (default)
./scripts/run_all.sh 4        # 4 threads
```

- Delegates to `scripts/run_all_rust.sh` (tenferro-einsum + strided-rs faer)
- Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
- Formats results as a markdown table via `scripts/format_results.py`
- Saves all outputs to `data/results/` with timestamps

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error are reported as **SKIP** with the reason on stderr.

Requires `strided-rs-benchmark-suite` at `../strided-rs-benchmark-suite` for strided-rs comparison (optional; skipped with a note if not found).

### 3. Run a single instance

To run the benchmark for **one instance only**, set the environment variable `BENCH_INSTANCE` to the instance name:

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp cargo run --release
BENCH_INSTANCE=tensornetwork_permutation_light_415 cargo run --release
```

**With the full script:**

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 1
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 4
```

Instance name must match the `name` field in the JSON (i.e. the filename without `.json`). To list available names: `ls data/instances/` → e.g. `gm_queen5_5_3.wcsp.json` → use `gm_queen5_5_3.wcsp`.

### 4. Binary Einsum Diagnostic Instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`) — dense matrix multiplication baseline
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`) — batched GEMM with moderate batch size
- `bin_outer_product_4096` (`i,j->ij`) — outer-product path (sensitive to broadcast/pack overhead)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`) — pure elementwise multiplication throughput

Recommended invocation for consistent single-thread profiling:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 cargo run --release
```

These are intended for quick, reproducible profiling before running heavier LM/structured/network cases.

### 5. Run individually

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

### 6. Format existing logs

```bash
uv run python scripts/format_results.py data/results/tenferro_einsum_*.log
```

## Reproducing Benchmarks

```bash
./scripts/run_all.sh        # 1 thread
./scripts/run_all.sh 4      # 4 threads
```

This script:

1. Delegates to `scripts/run_all_rust.sh` which builds and runs tenferro-einsum and strided-rs (faer)
2. Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
3. Formats results as a markdown table via `scripts/format_results.py`
4. Saves all outputs to `data/results/` with timestamps

## Benchmark Instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite. Selection is per-category (see [Export benchmark metadata](#1-export-benchmark-metadata)); dtype is float64 or complex128; tensors are zero-filled at runtime.

| Instance | Category | Tensors | Steps | log10(FLOPS) | log2(SIZE) |
|----------|----------|--------:|------:|-------------:|------------:|
| `bin_matmul_256` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_batched_matmul_b32_m64_n64_k64` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_outer_product_4096` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_elementwise_mul_2048x2048` | Binary (diagnostic) | 2 | 1 | — | — |
| `gm_queen5_5_3.wcsp` | Graphical model | 160 | 159 | 9.75 | 26.94 |
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 99 | 8.48 | 17.26 |
| `str_mps_varying_inner_product_200` | Structured (MPS) | 200 | 199 | 8.31 | 15.48 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 25 | 10.49 | 25.36 |
| `tensornetwork_permutation_light_415` | Tensor network | 415 | 414 | 9.65 | 24.0 |
| `tensornetwork_permutation_focus_step409_316` | Tensor network (focused) | 316 | 315 | 9.65 | 24.0 |

- **Graphical model (gm_*)**: WCSP / constraint networks; many small 2D factors (e.g. 3×3), full contraction to scalar.
- **Language model (lm_*)**: many small multi-dimensional tensors (3D/4D) with large batch dimensions; many steps with small GEMM kernels.
- **Structured — matrix chain (str_matrix_chain_*)**: large 2D matrices; each step is one large GEMM.
- **Structured — MPS (str_mps_*)**: matrix product state–style networks; varying inner dimensions, many 2D contractions.
- **Structured — MERA (str_nw_mera_*)**: tensor networks from multi-scale entanglement renormalization; many small 3×3-like tensors, heavy contraction.
- **Tensor network (tensornetwork_permutation_light_415)**: 415 tensors extracted from the full TensorNetworkBenchmarks instance via BFS-connected subgraph.
- **Tensor network focused (tensornetwork_permutation_focus_step409_316)**: focused subtree for profiling the late bottleneck steps (408/409); 316 tensors with non-scalar output (rank-18).

## Benchmark Results

### Apple Silicon M4

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.023 ± 0.193 | **0.965 ± 0.260** |
| bin_elementwise_mul_2048x2048 | 1.504 ± 0.064 | **1.318 ± 0.062** |
| bin_matmul_256 | **0.595 ± 0.017** | 0.596 ± 0.011 |
| bin_outer_product_4096 | 3.272 ± 0.128 | **2.124 ± 0.035** |
| gm_queen5_5_3.wcsp | 3371.448 ± 210.094 | **1718.031 ± 6.705** |
| lm_batch_likelihood_brackets_4_4d | 21.869 ± 0.160 | **9.543 ± 0.137** |
| lm_batch_likelihood_sentence_3_12d | 62.664 ± 0.275 | **32.938 ± 0.114** |
| lm_batch_likelihood_sentence_4_4d | 24.084 ± 0.282 | **10.515 ± 0.200** |
| str_matrix_chain_multiplication_100 | 10.152 ± 0.098 | **8.647 ± 0.078** |
| str_mps_varying_inner_product_200 | 10.661 ± 0.062 | **8.808 ± 0.037** |
| str_nw_mera_closed_120 | 914.478 ± 28.520 | **842.302 ± 2.381** |
| str_nw_mera_open_26 | 714.434 ± 37.233 | **550.314 ± 1.396** |
| tensornetwork_permutation_focus_step409_316 | 262.624 ± 3.523 | **165.416 ± 0.703** |
| tensornetwork_permutation_light_415 | 265.324 ± 2.189 | **166.476 ± 0.549** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.658 ± 0.007 | **0.493 ± 0.002** |
| bin_elementwise_mul_2048x2048 | 1.744 ± 0.074 | **1.210 ± 0.059** |
| bin_matmul_256 | 0.598 ± 0.015 | **0.577 ± 0.014** |
| bin_outer_product_4096 | 2.999 ± 0.063 | **2.124 ± 0.021** |
| gm_queen5_5_3.wcsp | 1164.295 ± 12.885 | **666.903 ± 5.936** |
| lm_batch_likelihood_brackets_4_4d | 22.040 ± 0.305 | **10.002 ± 0.031** |
| lm_batch_likelihood_sentence_3_12d | 65.665 ± 1.158 | **34.563 ± 0.228** |
| lm_batch_likelihood_sentence_4_4d | 25.887 ± 0.160 | **11.317 ± 0.101** |
| str_matrix_chain_multiplication_100 | 10.110 ± 0.109 | **8.700 ± 0.063** |
| str_mps_varying_inner_product_200 | 9.743 ± 0.029 | **8.609 ± 0.086** |
| str_nw_mera_closed_120 | 885.906 ± 1.514 | **852.758 ± 0.905** |
| str_nw_mera_open_26 | 690.973 ± 2.728 | **553.280 ± 1.361** |
| tensornetwork_permutation_focus_step409_316 | 264.075 ± 1.485 | **165.384 ± 0.371** |
| tensornetwork_permutation_light_415 | 265.583 ± 1.893 | **166.402 ± 0.498** |

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.146 ± 0.255 | **1.128 ± 0.435** |
| bin_elementwise_mul_2048x2048 | 1.531 ± 0.265 | **1.268 ± 0.121** |
| bin_matmul_256 | 0.250 ± 0.013 | **0.196 ± 0.006** |
| bin_outer_product_4096 | 3.284 ± 0.121 | **1.941 ± 0.159** |
| gm_queen5_5_3.wcsp | 3235.469 ± 53.987 | **1854.348 ± 27.603** |
| lm_batch_likelihood_brackets_4_4d | 20.900 ± 0.211 | **8.327 ± 0.041** |
| lm_batch_likelihood_sentence_3_12d | 48.856 ± 0.984 | **17.915 ± 1.152** |
| lm_batch_likelihood_sentence_4_4d | 23.466 ± 0.307 | **8.683 ± 0.174** |
| str_matrix_chain_multiplication_100 | 9.309 ± 0.401 | **6.343 ± 0.497** |
| str_mps_varying_inner_product_200 | 11.337 ± 0.061 | **9.492 ± 0.202** |
| str_nw_mera_closed_120 | 391.572 ± 48.171 | **310.748 ± 3.119** |
| str_nw_mera_open_26 | 438.809 ± 44.976 | **211.457 ± 22.471** |
| tensornetwork_permutation_focus_step409_316 | 218.409 ± 11.147 | **120.869 ± 2.175** |
| tensornetwork_permutation_light_415 | 215.735 ± 2.202 | **122.797 ± 2.528** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.672 ± 0.013 | **0.515 ± 0.001** |
| bin_elementwise_mul_2048x2048 | 1.710 ± 0.131 | **1.267 ± 0.073** |
| bin_matmul_256 | **0.246 ± 0.005** | 0.253 ± 0.065 |
| bin_outer_product_4096 | 3.000 ± 0.172 | **1.836 ± 0.055** |
| gm_queen5_5_3.wcsp | 936.115 ± 21.103 | **509.918 ± 6.951** |
| lm_batch_likelihood_brackets_4_4d | 22.207 ± 0.214 | **9.220 ± 0.211** |
| lm_batch_likelihood_sentence_3_12d | 51.643 ± 1.182 | **17.862 ± 0.238** |
| lm_batch_likelihood_sentence_4_4d | 25.147 ± 0.135 | **9.476 ± 0.196** |
| str_matrix_chain_multiplication_100 | 9.289 ± 0.243 | **6.039 ± 0.159** |
| str_mps_varying_inner_product_200 | 10.545 ± 0.211 | **9.269 ± 0.251** |
| str_nw_mera_closed_120 | 356.761 ± 6.302 | **315.582 ± 5.638** |
| str_nw_mera_open_26 | 421.154 ± 5.651 | **199.370 ± 3.922** |
| tensornetwork_permutation_focus_step409_316 | 223.099 ± 22.663 | **121.368 ± 3.657** |
| tensornetwork_permutation_light_415 | 220.271 ± 14.192 | **124.385 ± 7.210** |


**Notes:**
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- Both backends use the same pre-computed contraction path for fair comparison.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) — Rust tensor library
- [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) — sibling benchmark suite for strided-rs

## License

MIT
