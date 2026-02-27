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
| bin_batched_matmul_b32_m64_n64_k64 | 1.257 ± 0.269 | **1.132 ± 0.343** |
| bin_elementwise_mul_2048x2048 | 1.540 ± 0.112 | **1.317 ± 0.084** |
| bin_matmul_256 | **0.583 ± 0.016** | 0.604 ± 0.021 |
| bin_outer_product_4096 | 7.679 ± 1.205 | **2.119 ± 0.038** |
| gm_queen5_5_3.wcsp | 4361.203 ± 32.993 | **1707.480 ± 4.783** |
| lm_batch_likelihood_brackets_4_4d | 11.543 ± 0.130 | **9.650 ± 0.125** |
| lm_batch_likelihood_sentence_3_12d | 39.640 ± 0.178 | **33.115 ± 0.284** |
| lm_batch_likelihood_sentence_4_4d | 12.224 ± 0.208 | **10.571 ± 0.200** |
| str_matrix_chain_multiplication_100 | 8.724 ± 0.145 | **8.710 ± 0.136** |
| str_mps_varying_inner_product_200 | **7.453 ± 0.032** | 8.761 ± 0.046 |
| str_nw_mera_closed_120 | 861.240 ± 5.015 | **843.357 ± 1.288** |
| str_nw_mera_open_26 | 674.893 ± 1.701 | **550.076 ± 1.476** |
| tensornetwork_permutation_focus_step409_316 | 222.450 ± 14.265 | **165.368 ± 0.724** |
| tensornetwork_permutation_light_415 | 222.037 ± 2.086 | **166.477 ± 0.443** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.603 ± 0.008 | **0.496 ± 0.001** |
| bin_elementwise_mul_2048x2048 | 1.553 ± 0.090 | **1.193 ± 0.033** |
| bin_matmul_256 | **0.583 ± 0.014** | 0.586 ± 0.014 |
| bin_outer_product_4096 | 6.619 ± 0.026 | **2.122 ± 0.029** |
| gm_queen5_5_3.wcsp | 1153.491 ± 17.509 | **664.450 ± 1.667** |
| lm_batch_likelihood_brackets_4_4d | 11.100 ± 0.077 | **10.084 ± 0.130** |
| lm_batch_likelihood_sentence_3_12d | 38.722 ± 0.526 | **34.319 ± 0.071** |
| lm_batch_likelihood_sentence_4_4d | 12.895 ± 0.099 | **11.183 ± 0.088** |
| str_matrix_chain_multiplication_100 | **8.646 ± 0.108** | 8.792 ± 0.061 |
| str_mps_varying_inner_product_200 | **7.192 ± 0.040** | 8.617 ± 0.041 |
| str_nw_mera_closed_120 | 862.404 ± 13.353 | **852.064 ± 2.696** |
| str_nw_mera_open_26 | 670.021 ± 2.820 | **553.358 ± 1.817** |
| tensornetwork_permutation_focus_step409_316 | 220.674 ± 1.220 | **165.382 ± 0.393** |
| tensornetwork_permutation_light_415 | 221.702 ± 0.989 | **166.651 ± 0.508** |

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.910 ± 0.169** | 1.120 ± 0.357 |
| bin_elementwise_mul_2048x2048 | 1.604 ± 0.288 | **1.289 ± 0.174** |
| bin_matmul_256 | 0.568 ± 0.015 | **0.197 ± 0.002** |
| bin_outer_product_4096 | 6.624 ± 0.032 | **1.815 ± 0.071** |
| gm_queen5_5_3.wcsp | 4393.381 ± 42.625 | **1818.873 ± 18.936** |
| lm_batch_likelihood_brackets_4_4d | 11.464 ± 0.111 | **7.959 ± 0.163** |
| lm_batch_likelihood_sentence_3_12d | 39.608 ± 0.161 | **17.217 ± 1.027** |
| lm_batch_likelihood_sentence_4_4d | 12.173 ± 0.173 | **8.611 ± 0.231** |
| str_matrix_chain_multiplication_100 | 8.785 ± 0.190 | **6.247 ± 0.540** |
| str_mps_varying_inner_product_200 | **7.407 ± 0.051** | 9.472 ± 0.203 |
| str_nw_mera_closed_120 | 864.155 ± 6.530 | **313.307 ± 2.753** |
| str_nw_mera_open_26 | 676.042 ± 1.255 | **194.276 ± 2.322** |
| tensornetwork_permutation_focus_step409_316 | 220.118 ± 1.661 | **119.400 ± 1.423** |
| tensornetwork_permutation_light_415 | 222.065 ± 1.549 | **120.717 ± 1.773** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.591 ± 0.005 | **0.509 ± 0.001** |
| bin_elementwise_mul_2048x2048 | 1.769 ± 0.099 | **1.211 ± 0.036** |
| bin_matmul_256 | 0.581 ± 0.007 | **0.229 ± 0.065** |
| bin_outer_product_4096 | 6.623 ± 0.042 | **1.832 ± 0.134** |
| gm_queen5_5_3.wcsp | 1162.212 ± 12.981 | **513.113 ± 10.355** |
| lm_batch_likelihood_brackets_4_4d | 10.998 ± 0.179 | **9.145 ± 0.174** |
| lm_batch_likelihood_sentence_3_12d | 38.126 ± 0.285 | **17.868 ± 0.665** |
| lm_batch_likelihood_sentence_4_4d | 13.024 ± 0.213 | **9.513 ± 0.208** |
| str_matrix_chain_multiplication_100 | 8.628 ± 0.104 | **6.029 ± 0.227** |
| str_mps_varying_inner_product_200 | **7.252 ± 0.057** | 9.329 ± 0.112 |
| str_nw_mera_closed_120 | 858.527 ± 3.044 | **316.367 ± 3.520** |
| str_nw_mera_open_26 | 666.410 ± 1.704 | **196.557 ± 1.914** |
| tensornetwork_permutation_focus_step409_316 | 220.263 ± 1.149 | **119.459 ± 0.712** |
| tensornetwork_permutation_light_415 | 221.319 ± 0.981 | **120.869 ± 2.163** |

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
