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

Environment: Apple Silicon M4. Median ± IQR (ms) of 15 runs (3 warmup). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1. Run date: 2026-02-26.

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.676 ± 0.012** | 0.981 ± 0.164 |
| bin_elementwise_mul_2048x2048 | 1.750 ± 0.258 | **1.349 ± 0.028** |
| bin_matmul_256 | 0.622 ± 0.023 | **0.599 ± 0.028** |
| bin_outer_product_4096 | 3.224 ± 0.155 | **2.157 ± 0.028** |
| gm_queen5_5_3.wcsp | 3404.841 ± 279.844 | **1767.186 ± 4.682** |
| lm_batch_likelihood_brackets_4_4d | 24.493 ± 1.396 | **9.918 ± 0.179** |
| lm_batch_likelihood_sentence_3_12d | 67.311 ± 1.591 | **33.260 ± 0.153** |
| lm_batch_likelihood_sentence_4_4d | 27.195 ± 1.300 | **10.915 ± 0.812** |
| str_matrix_chain_multiplication_100 | 10.931 ± 0.329 | **8.875 ± 0.137** |
| str_mps_varying_inner_product_200 | 11.195 ± 0.151 | **8.829 ± 0.064** |
| str_nw_mera_closed_120 | 969.616 ± 10.422 | **849.542 ± 1.492** |
| str_nw_mera_open_26 | 716.492 ± 37.860 | **553.762 ± 1.894** |
| tensornetwork_permutation_focus_step409_316 | 267.890 ± 3.083 | **170.660 ± 1.366** |
| tensornetwork_permutation_light_415 | 267.738 ± 2.239 | **168.012 ± 0.661** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.668 ± 0.026 | **0.498 ± 0.007** |
| bin_elementwise_mul_2048x2048 | 1.743 ± 0.333 | **1.184 ± 0.052** |
| bin_matmul_256 | 0.600 ± 0.054 | **0.576 ± 0.004** |
| bin_outer_product_4096 | 3.238 ± 0.298 | **2.139 ± 0.054** |
| gm_queen5_5_3.wcsp | 1260.923 ± 21.310 | **681.685 ± 7.665** |
| lm_batch_likelihood_brackets_4_4d | 24.458 ± 1.085 | **10.452 ± 0.248** |
| lm_batch_likelihood_sentence_3_12d | 70.142 ± 1.705 | **35.210 ± 0.177** |
| lm_batch_likelihood_sentence_4_4d | 28.827 ± 1.291 | **11.868 ± 0.354** |
| str_matrix_chain_multiplication_100 | 10.858 ± 0.158 | **8.981 ± 0.172** |
| str_mps_varying_inner_product_200 | 9.915 ± 0.435 | **8.787 ± 0.128** |
| str_nw_mera_closed_120 | **892.920 ± 3.443** | 897.435 ± 13.205 |
| str_nw_mera_open_26 | 699.450 ± 2.671 | **611.122 ± 24.217** |
| tensornetwork_permutation_focus_step409_316 | 268.065 ± 1.746 | **184.465 ± 8.098** |
| tensornetwork_permutation_light_415 | 269.012 ± 3.166 | **178.819 ± 10.542** |

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
