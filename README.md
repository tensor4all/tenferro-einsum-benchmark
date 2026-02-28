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
| bin_batched_matmul_b32_m64_n64_k64 | 1.121 ± 0.378 | **1.120 ± 0.342** |
| bin_elementwise_mul_2048x2048 | 1.679 ± 0.313 | **1.305 ± 0.038** |
| bin_matmul_256 | **0.589 ± 0.016** | 0.602 ± 0.020 |
| bin_outer_product_4096 | 3.461 ± 0.152 | **2.068 ± 0.028** |
| gm_queen5_5_3.wcsp | **1222.655 ± 12.807** | 1684.874 ± 5.130 |
| lm_batch_likelihood_brackets_4_4d | **9.324 ± 0.068** | 9.630 ± 0.125 |
| lm_batch_likelihood_sentence_3_12d | 33.150 ± 0.124 | **32.837 ± 0.065** |
| lm_batch_likelihood_sentence_4_4d | 10.608 ± 0.040 | **10.441 ± 0.094** |
| str_matrix_chain_multiplication_100 | **8.130 ± 0.190** | 8.520 ± 0.074 |
| str_mps_varying_inner_product_200 | **7.048 ± 0.028** | 8.689 ± 0.016 |
| str_nw_mera_closed_120 | **842.711 ± 2.537** | 889.166 ± 7.552 |
| str_nw_mera_open_26 | 560.668 ± 5.843 | **552.898 ± 2.716** |
| tensornetwork_permutation_focus_step409_316 | 180.818 ± 1.965 | **164.933 ± 0.767** |
| tensornetwork_permutation_light_415 | 181.752 ± 1.538 | **167.211 ± 2.191** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.486 ± 0.008** | 0.514 ± 0.015 |
| bin_elementwise_mul_2048x2048 | 1.542 ± 0.077 | **1.277 ± 0.067** |
| bin_matmul_256 | **0.574 ± 0.021** | 0.585 ± 0.008 |
| bin_outer_product_4096 | 3.207 ± 0.047 | **2.147 ± 0.014** |
| gm_queen5_5_3.wcsp | 810.777 ± 4.571 | **685.465 ± 16.417** |
| lm_batch_likelihood_brackets_4_4d | 10.281 ± 0.047 | **10.013 ± 0.050** |
| lm_batch_likelihood_sentence_3_12d | **33.765 ± 0.115** | 34.415 ± 0.081 |
| lm_batch_likelihood_sentence_4_4d | 11.352 ± 0.061 | **11.317 ± 0.087** |
| str_matrix_chain_multiplication_100 | **8.455 ± 0.080** | 8.764 ± 0.060 |
| str_mps_varying_inner_product_200 | **6.792 ± 0.019** | 8.589 ± 0.104 |
| str_nw_mera_closed_120 | **848.573 ± 2.793** | 851.552 ± 2.400 |
| str_nw_mera_open_26 | 563.575 ± 1.885 | **556.740 ± 1.373** |
| tensornetwork_permutation_focus_step409_316 | 182.376 ± 0.875 | **165.133 ± 0.455** |
| tensornetwork_permutation_light_415 | 183.752 ± 1.257 | **166.649 ± 0.679** |

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.967 ± 0.258** | 0.987 ± 0.189 |
| bin_elementwise_mul_2048x2048 | 1.511 ± 0.204 | **1.300 ± 0.061** |
| bin_matmul_256 | 0.584 ± 0.013 | **0.207 ± 0.009** |
| bin_outer_product_4096 | 3.479 ± 0.122 | **1.843 ± 0.123** |
| gm_queen5_5_3.wcsp | **1234.635 ± 16.463** | 1784.081 ± 20.114 |
| lm_batch_likelihood_brackets_4_4d | 9.693 ± 0.333 | **8.001 ± 0.105** |
| lm_batch_likelihood_sentence_3_12d | 34.110 ± 0.178 | **16.888 ± 1.234** |
| lm_batch_likelihood_sentence_4_4d | 10.848 ± 0.131 | **8.555 ± 0.235** |
| str_matrix_chain_multiplication_100 | 8.571 ± 0.109 | **5.942 ± 0.138** |
| str_mps_varying_inner_product_200 | **7.300 ± 0.105** | 9.415 ± 0.227 |
| str_nw_mera_closed_120 | 856.015 ± 3.790 | **308.202 ± 3.519** |
| str_nw_mera_open_26 | 565.111 ± 2.777 | **196.093 ± 2.024** |
| tensornetwork_permutation_focus_step409_316 | 183.230 ± 0.622 | **118.905 ± 1.930** |
| tensornetwork_permutation_light_415 | 184.342 ± 0.695 | **119.480 ± 1.600** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.499 ± 0.001** | 0.502 ± 0.001 |
| bin_elementwise_mul_2048x2048 | 1.553 ± 0.066 | **1.210 ± 0.034** |
| bin_matmul_256 | 0.583 ± 0.017 | **0.225 ± 0.003** |
| bin_outer_product_4096 | 3.278 ± 0.070 | **1.834 ± 0.030** |
| gm_queen5_5_3.wcsp | 820.282 ± 8.630 | **507.016 ± 7.450** |
| lm_batch_likelihood_brackets_4_4d | 10.566 ± 0.115 | **8.933 ± 0.176** |
| lm_batch_likelihood_sentence_3_12d | 34.177 ± 0.431 | **17.602 ± 0.381** |
| lm_batch_likelihood_sentence_4_4d | 11.570 ± 0.164 | **9.425 ± 0.083** |
| str_matrix_chain_multiplication_100 | 8.768 ± 0.185 | **6.021 ± 0.186** |
| str_mps_varying_inner_product_200 | **7.024 ± 0.113** | 9.278 ± 0.129 |
| str_nw_mera_closed_120 | 855.142 ± 1.572 | **312.284 ± 3.343** |
| str_nw_mera_open_26 | 566.609 ± 3.293 | **198.526 ± 2.653** |
| tensornetwork_permutation_focus_step409_316 | 183.050 ± 0.451 | **119.161 ± 1.950** |
| tensornetwork_permutation_light_415 | 184.278 ± 0.532 | **119.705 ± 2.118** |

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
