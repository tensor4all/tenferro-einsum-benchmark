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

BENCH_LOG="$RESULTS_DIR/tenferro_einsum_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release)..."
cargo build --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

BIN="$PROJECT_DIR/target/release/tenferro-einsum-benchmark"
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: benchmark binary not found: $BIN"
    exit 1
fi

INSTANCES=()
while IFS= read -r inst; do
    INSTANCES+=("$inst")
done < <(
    find "$PROJECT_DIR/data/instances" -maxdepth 1 -type f -name '*.json' -print \
        | sed -E 's#.*/##; s#\.json$##' \
        | sort
)

if [[ "${#INSTANCES[@]}" -eq 0 ]]; then
    echo "ERROR: no benchmark instances found in $PROJECT_DIR/data/instances"
    exit 1
fi

if [[ -n "${BENCH_INSTANCE:-}" ]]; then
    INSTANCES=("$BENCH_INSTANCE")
fi

echo "Running Rust benchmark (instance-by-instance)..."
echo "Instances: ${#INSTANCES[@]}"
echo ""
: > "$BENCH_LOG"

FAILED=()
for i in "${!INSTANCES[@]}"; do
    inst="${INSTANCES[$i]}"
    n=$((i + 1))
    total="${#INSTANCES[@]}"
    echo "  [$n/$total] $inst"
    if BENCH_INSTANCE="$inst" "$BIN" 2>&1 | tee -a "$BENCH_LOG"; then
        :
    else
        rc=${PIPESTATUS[0]}
        FAILED+=("$inst(rc=$rc)")
        echo "WARNING: instance failed: $inst (exit=$rc). Continue." | tee -a "$BENCH_LOG"
        # Ensure formatter can render this instance as missing for both strategies.
        {
            echo "Strategy: opt_flops"
            printf "%-50s %8s %10s %12s %12s\n" "$inst" "0" "0.00" "0.00" "SKIP"
            echo "Strategy: opt_size"
            printf "%-50s %8s %10s %12s %12s\n" "$inst" "0" "0.00" "0.00" "SKIP"
        } >> "$BENCH_LOG"
    fi
done

echo ""
echo "Rust results saved to: $BENCH_LOG"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "Failed instances (${#FAILED[@]}): ${FAILED[*]}"
fi
echo ""

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${TIMESTAMP}.md"

echo "Formatting results as markdown..."
if command -v uv &>/dev/null; then
    uv run python "$PROJECT_DIR/scripts/format_results.py" "$BENCH_LOG" | tee "$MARKDOWN_OUT"
else
    python3 "$PROJECT_DIR/scripts/format_results.py" "$BENCH_LOG" | tee "$MARKDOWN_OUT"
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
echo "  Rust log:     $BENCH_LOG"
echo "  Markdown:     $MARKDOWN_OUT"
