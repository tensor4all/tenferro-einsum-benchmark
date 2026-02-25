# tenferro-einsum-benchmark

Benchmark suite for [tenferro-einsum](https://github.com/tensor4all/tenferro-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [tenferro-einsum](https://github.com/tensor4all/tenferro-rs)

Only metadata is stored — tensors are generated at benchmark time (zero-filled), keeping the repo lightweight.

The structure mirrors [strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) for consistency and cross-comparison.

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

### 2. Run benchmarks

```bash
./scripts/run_all.sh [NUM_THREADS]
```

Default: single-threaded. Pass `4` for 4 threads.

### 3. Run a single instance

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp cargo run --release
```

### 4. Binary Einsum Diagnostic Instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`)
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`)
- `bin_outer_product_4096` (`i,j->ij`)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`)

Recommended invocation for consistent single-thread profiling:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 cargo run --release
```

## Benchmark Result M4 

#### Strategy: opt_flops

Median time (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) |
|---|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **1.129** |
| bin_elementwise_mul_2048x2048 | **1.887** |
| bin_matmul_256 | **0.625** |
| bin_outer_product_4096 | **2.969** |
| gm_queen5_5_3.wcsp | **0.432** |
| lm_batch_likelihood_brackets_4_4d | **58.821** |
| lm_batch_likelihood_sentence_3_12d | **939.580** |
| lm_batch_likelihood_sentence_4_4d | **186.254** |
| str_matrix_chain_multiplication_100 | - |
| str_mps_varying_inner_product_200 | - |
| str_nw_mera_closed_120 | - |
| str_nw_mera_open_26 | - |
| tensornetwork_permutation_focus_step409_316 | **14.588** |
| tensornetwork_permutation_light_415 | **1.141** |

#### Strategy: opt_size

Median time (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) |
|---|---:|
| bin_batched_matmul_b32_m64_n64_k64 | **0.714** |
| bin_elementwise_mul_2048x2048 | **1.536** |
| bin_matmul_256 | **0.606** |
| bin_outer_product_4096 | **3.056** |
| gm_queen5_5_3.wcsp | **0.463** |
| lm_batch_likelihood_brackets_4_4d | **43.508** |
| lm_batch_likelihood_sentence_3_12d | **914.487** |
| lm_batch_likelihood_sentence_4_4d | **90.349** |
| str_matrix_chain_multiplication_100 | - |
| str_mps_varying_inner_product_200 | - |
| str_nw_mera_closed_120 | - |
| str_nw_mera_open_26 | - |
| tensornetwork_permutation_focus_step409_316 | **12.276** |
| tensornetwork_permutation_light_415 | **0.843** |

## License

MIT
