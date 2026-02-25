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

## Limitations

- **float64 only**: `complex128` instances are skipped (not yet supported).
- **Index labels**: tenferro-einsum's `Subscripts::parse` supports only `a-z` and `A-Z`. Instances using Unicode index labels (e.g. `Á`, `Â`) will be skipped with an error.

## Comparison with strided-rs-benchmark-suite

| Feature | strided-rs-benchmark-suite | tenferro-einsum-benchmark |
|---------|---------------------------|---------------------------|
| Rust backend | strided-opteinsum (faer/blas) | tenferro-einsum |
| Julia backend | OMEinsum.jl | — |
| Data format | Same JSON | Same JSON |
| Path strategies | opt_flops, opt_size | opt_flops, opt_size |

## License

MIT
