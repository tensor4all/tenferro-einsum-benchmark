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
    run_all.sh              # Run all benchmarks (configurable thread count)
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

- Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
- Builds and runs the Rust benchmark
- Formats results as a markdown table via `scripts/format_results.py`
- Saves all outputs to `data/results/` with timestamps

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error are reported as **SKIP** with the reason on stderr.

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

1. Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
2. Builds and runs the Rust benchmark (release)
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

Environment: Apple Silicon M4. Median ± IQR (ms) of 15 runs (3 warmup). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1. Run date: 2026-02-25.

#### Strategy: opt_flops

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.693 ± 0.006** | 0.729 ± 0.552 |
| bin_elementwise_mul_2048x2048 | 1.589 ± 0.073 | **1.462 ± 0.083** |
| bin_matmul_256 | 0.626 ± 0.007 | **0.596 ± 0.035** |
| bin_outer_product_4096 | 3.279 ± 0.095 | **2.162 ± 0.046** |
| gm_queen5_5_3.wcsp | **0.467 ± 0.022** | 1787.717 ± 42.108 |
| lm_batch_likelihood_brackets_4_4d | 60.782 ± 0.534 | **9.938 ± 0.294** |
| lm_batch_likelihood_sentence_3_12d | 954.167 ± 47.744 | **33.102 ± 0.209** |
| lm_batch_likelihood_sentence_4_4d | 189.056 ± 1.357 | **10.929 ± 0.188** |
| str_matrix_chain_multiplication_100 | - | **8.815 ± 0.121** |
| str_mps_varying_inner_product_200 | - | **8.835 ± 0.052** |
| str_nw_mera_closed_120 | - | **848.963 ± 1.435** |
| str_nw_mera_open_26 | - | **552.778 ± 1.911** |
| tensornetwork_permutation_focus_step409_316 | **9.932 ± 2.803** | 166.298 ± 0.498 |
| tensornetwork_permutation_light_415 | **0.892 ± 0.088** | 167.526 ± 0.347 |

#### Strategy: opt_size

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.685 ± 0.007 | **0.498 ± 0.002** |
| bin_elementwise_mul_2048x2048 | 1.606 ± 0.069 | **1.238 ± 0.122** |
| bin_matmul_256 | 0.624 ± 0.009 | **0.577 ± 0.003** |
| bin_outer_product_4096 | 3.213 ± 0.067 | **2.116 ± 0.031** |
| gm_queen5_5_3.wcsp | **0.446 ± 0.006** | 675.833 ± 6.907 |
| lm_batch_likelihood_brackets_4_4d | 46.479 ± 0.555 | **10.266 ± 0.148** |
| lm_batch_likelihood_sentence_3_12d | 915.454 ± 25.753 | **34.695 ± 0.658** |
| lm_batch_likelihood_sentence_4_4d | 94.324 ± 0.857 | **11.346 ± 0.251** |
| str_matrix_chain_multiplication_100 | - | **8.993 ± 0.307** |
| str_mps_varying_inner_product_200 | - | **8.742 ± 0.283** |
| str_nw_mera_closed_120 | - | **884.260 ± 17.634** |
| str_nw_mera_open_26 | - | **555.995 ± 1.449** |
| tensornetwork_permutation_focus_step409_316 | **8.229 ± 1.203** | 166.382 ± 0.557 |
| tensornetwork_permutation_light_415 | **0.787 ± 0.075** | 167.383 ± 0.300 |

**Notes:**
- `-` indicates the instance was skipped (OOM or unsupported). Skipped instances are reported as **SKIP** with the reason on stderr.
- `str_*` instances are currently not supported by tenferro-einsum (OOM kill on M4).
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- tenferro-einsum and strided-rs use the same pre-computed contraction path for fair comparison.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) — Rust tensor library
- [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) — sibling benchmark suite for strided-rs

## License

MIT
