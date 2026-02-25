#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Benchmark runner for tenferro-einsum-benchmark
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading, if used)
#   - RAYON_NUM_THREADS (Rust rayon parallelism)
#
# Requires tenferro-rs at ../tenferro-rs (sibling directory).
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export RAYON_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " tenferro-einsum benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Timestamp:    $TIMESTAMP"
echo ""
echo "Threading policy (threads=${NUM_THREADS}):"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo ""

# ---------------------------------------------------------------------------
# Rust benchmark (tenferro-einsum)
# ---------------------------------------------------------------------------
echo "============================================"
echo " [1/1] Rust: tenferro-einsum"
echo "============================================"

RUST_LOG="$RESULTS_DIR/tenferro_einsum_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release)..."
cargo build --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

echo "Running Rust benchmark..."
cargo run --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_LOG"

echo ""
echo "Rust results saved to: $RUST_LOG"
echo ""

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${TIMESTAMP}.md"

echo "Formatting results as markdown..."
if command -v uv &>/dev/null; then
    uv run python "$PROJECT_DIR/scripts/format_results.py" "$RUST_LOG" | tee "$MARKDOWN_OUT"
else
    python3 "$PROJECT_DIR/scripts/format_results.py" "$RUST_LOG" | tee "$MARKDOWN_OUT"
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
echo "  Rust log:     $RUST_LOG"
echo "  Markdown:     $MARKDOWN_OUT"
