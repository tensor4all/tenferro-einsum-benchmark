//! Benchmark runner for tenferro-einsum using einsum benchmark instances.
//!
//! Loads JSON metadata from data/instances/, builds ContractionTree from
//! pre-computed paths (opt_flops / opt_size), and times tenferro-einsum execution.

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{einsum_with_plan, ContractionTree, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

// ---------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BenchmarkInstance {
    name: String,
    format_string_colmajor: String,
    shapes_colmajor: Vec<Vec<usize>>,
    dtype: String,
    num_tensors: usize,
    paths: PathInfo,
}

#[derive(Deserialize)]
struct PathInfo {
    opt_size: PathMeta,
    opt_flops: PathMeta,
}

#[derive(Deserialize)]
struct PathMeta {
    path: Vec<[usize; 2]>,
    log2_size: f64,
    log10_flops: f64,
}

// ---------------------------------------------------------------------------
// Contraction path conversion
// ---------------------------------------------------------------------------

/// Convert opt_einsum/cotengra path format to tenferro ContractionTree pairs.
///
/// Path convention: each step [i, j] contracts tensors at indices i and j
/// in the current list. Same as tenferro's from_pairs (left, right).
fn path_to_pairs(path: &[[usize; 2]]) -> Vec<(usize, usize)> {
    path.iter().map(|p| (p[0], p[1])).collect()
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn create_operands(shapes: &[Vec<usize>], dtype: &str) -> Vec<Tensor<f64>> {
    let col = MemoryOrder::ColumnMajor;
    let mem = LogicalMemorySpace::MainMemory;

    match dtype {
        "float64" => shapes
            .iter()
            .map(|shape| Tensor::<f64>::zeros(shape, mem, col))
            .collect(),
        "complex128" => panic!("complex128 not yet supported in tenferro-einsum-benchmark"),
        other => panic!("unsupported dtype: {other}"),
    }
}

fn run_instance(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
    ctx: &mut CpuContext,
) -> Result<(Duration, Duration), tenferro_device::Error> {
    if instance.dtype == "complex128" {
        return Err(tenferro_device::Error::InvalidArgument(
            "complex128 not supported".into(),
        ));
    }

    let subs = Subscripts::parse(&instance.format_string_colmajor)?;
    let shapes: Vec<&[usize]> = instance
        .shapes_colmajor
        .iter()
        .map(|s| s.as_slice())
        .collect();
    let pairs = path_to_pairs(&path_meta.path);
    let tree = ContractionTree::from_pairs(&subs, &shapes, &pairs)?;

    let operands: Vec<Tensor<f64>> = create_operands(&instance.shapes_colmajor, &instance.dtype);
    let operands_refs: Vec<&Tensor<f64>> = operands.iter().collect();

    // Warmup
    for _ in 0..3 {
        let _ = einsum_with_plan::<_, CpuBackend>(ctx, &tree, &operands_refs, None)?;
    }

    // Timed runs
    let num_runs = 15;
    let mut durations = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let operands: Vec<Tensor<f64>> =
            create_operands(&instance.shapes_colmajor, &instance.dtype);
        let operands_refs: Vec<&Tensor<f64>> = operands.iter().collect();
        let t0 = Instant::now();
        let result = einsum_with_plan::<_, CpuBackend>(ctx, &tree, &operands_refs, None)?;
        let elapsed = t0.elapsed();
        black_box(&result);
        durations.push(elapsed);
    }

    durations.sort();
    let median = durations[num_runs / 2];
    let q1 = durations[num_runs / 4];
    let q3 = durations[3 * num_runs / 4];
    let iqr = q3.saturating_sub(q1);
    Ok((median, iqr))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const BACKEND_NAME: &str = "tenferro-einsum";

fn load_instances() -> Vec<BenchmarkInstance> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut paths: Vec<_> = std::fs::read_dir(&data_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", data_dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    paths
        .iter()
        .filter_map(|path| {
            let json_str = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Warning: skip {} (read failed: {e})", path.display());
                    return None;
                }
            };
            match serde_json::from_str(&json_str) {
                Ok(instance) => Some(instance),
                Err(e) => {
                    eprintln!("Warning: skip {} (parse failed: {e})", path.display());
                    None
                }
            }
        })
        .collect()
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut instances = load_instances();
    if let Ok(filter) = std::env::var("BENCH_INSTANCE") {
        instances.retain(|i| i.name == filter);
        if instances.is_empty() {
            eprintln!("BENCH_INSTANCE={filter:?}: no matching instance found");
            std::process::exit(1);
        }
    }

    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    let omp_threads = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into());

    let mut ctx = CpuContext::new(4);

    println!("{BACKEND_NAME} benchmark suite");
    println!("==================================");
    println!(
        "Loaded {} instances from {}",
        instances.len(),
        data_dir.display()
    );
    println!("Backend: {BACKEND_NAME}");
    println!("RAYON_NUM_THREADS={rayon_threads}, OMP_NUM_THREADS={omp_threads}");
    println!("Timing: median ± IQR of 15 runs (3 warmup)");

    let strategies: &[(&str, fn(&PathInfo) -> &PathMeta)] = &[
        ("opt_flops", |p| &p.opt_flops),
        ("opt_size", |p| &p.opt_size),
    ];

    for &(strategy_name, get_path) in strategies {
        println!();
        println!("Strategy: {strategy_name}");
        println!(
            "{:<50} {:>8} {:>10} {:>12} {:>12} {:>10}",
            "Instance", "Tensors", "log10FLOPS", "log2SIZE", "Median (ms)", "IQR (ms)"
        );
        println!("{}", "-".repeat(108));

        for (i, instance) in instances.iter().enumerate() {
            eprintln!("  [{}/{}] {}...", i + 1, instances.len(), instance.name);
            let path_meta = get_path(&instance.paths);
            match run_instance(instance, path_meta, &mut ctx) {
                Ok((median, iqr)) => {
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12.3} {:>10.3}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        median.as_secs_f64() * 1e3,
                        iqr.as_secs_f64() * 1e3,
                    );
                }
                Err(e) => {
                    eprintln!("  -> {} (error: {e})", instance.name);
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12} {:>10}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        "SKIP",
                        "-",
                    );
                }
            }
        }
    }
}
