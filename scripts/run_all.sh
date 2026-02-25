#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Benchmark runner: tenferro-einsum vs strided-rs (faer + optional OpenBLAS)
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading, if used)
#   - RAYON_NUM_THREADS (Rust rayon parallelism)
#
# Requires:
#   - tenferro-rs at ../tenferro-rs
#   - strided-rs-benchmark-suite at ../strided-rs-benchmark-suite (for comparison)
#     └─ strided-rs at ../strided-rs (dependency of strided-rs-benchmark-suite)
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export RAYON_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STRIDED_DIR="$(cd "$PROJECT_DIR/../strided-rs-benchmark-suite" 2>/dev/null && pwd || true)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " tenferro vs strided-rs benchmark"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Timestamp:    $TIMESTAMP"
echo ""
echo "Threading (threads=${NUM_THREADS}):"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo ""

LOGS=()
STEP=1

# ---------------------------------------------------------------------------
# [1] tenferro-einsum
# ---------------------------------------------------------------------------
echo "============================================"
echo " [$STEP] tenferro-einsum"
echo "============================================"
STEP=$((STEP + 1))

TENFERRO_LOG="$RESULTS_DIR/tenferro_einsum_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building tenferro-einsum (release)..."
cargo build --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

BIN="$PROJECT_DIR/target/release/tenferro-einsum-benchmark"
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: tenferro-einsum binary not found: $BIN"
    exit 1
fi

echo "Running tenferro-einsum benchmark..."
"$BIN" 2>/dev/null | tee "$TENFERRO_LOG"
LOGS+=("$TENFERRO_LOG")
echo ""
echo "Saved: $TENFERRO_LOG"
echo ""

# ---------------------------------------------------------------------------
# [2/3] strided-rs (faer + optional OpenBLAS)
# ---------------------------------------------------------------------------
if [[ -z "$STRIDED_DIR" ]] || [[ ! -d "$STRIDED_DIR" ]]; then
    echo "NOTE: strided-rs-benchmark-suite not found at ../strided-rs-benchmark-suite"
    echo "  Skipping strided-rs comparison."
    echo "  To enable: git clone https://github.com/tensor4all/strided-rs-benchmark-suite ../strided-rs-benchmark-suite"
    echo ""
else
    STRIDED_BIN="$STRIDED_DIR/target/release/strided-rs-benchmark-suite"

    # -------------------------------------------------------------------------
    # [2] strided-rs faer
    # -------------------------------------------------------------------------
    echo "============================================"
    echo " [$STEP] strided-rs (faer)"
    echo "============================================"
    STEP=$((STEP + 1))

    STRIDED_FAER_LOG="$RESULTS_DIR/strided_faer_t${NUM_THREADS}_${TIMESTAMP}.log"

    echo "Building strided-rs (faer, release)..."
    if cargo build --release --no-default-features --features faer,parallel \
            --manifest-path="$STRIDED_DIR/Cargo.toml" 2>&1; then
        echo "Running strided-rs (faer) benchmark..."
        "$STRIDED_BIN" 2>/dev/null | tee "$STRIDED_FAER_LOG"
        LOGS+=("$STRIDED_FAER_LOG")
        echo ""
        echo "Saved: $STRIDED_FAER_LOG"
    else
        echo "WARNING: strided-rs (faer) build failed. Skipping."
    fi
    echo ""

    # -------------------------------------------------------------------------
    # [3] strided-rs OpenBLAS (optional)
    # -------------------------------------------------------------------------
    echo "============================================"
    echo " [$STEP] strided-rs (OpenBLAS)"
    echo "============================================"
    STEP=$((STEP + 1))

    # Auto-detect OpenBLAS 0.3.29 built locally
    CUSTOM_OPENBLAS="$HOME/opt/openblas-0.3.29/lib"
    if [[ -z "${OPENBLAS_LIB_DIR:-}" ]] && [[ -d "$CUSTOM_OPENBLAS" ]]; then
        export OPENBLAS_LIB_DIR="$CUSTOM_OPENBLAS"
        export LD_LIBRARY_PATH="${CUSTOM_OPENBLAS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "Auto-detected OpenBLAS 0.3.29 at $CUSTOM_OPENBLAS"
    fi

    STRIDED_BLAS_LOG="$RESULTS_DIR/strided_blas_t${NUM_THREADS}_${TIMESTAMP}.log"

    echo "Building strided-rs (blas, release)..."
    if cargo build --release --no-default-features --features blas,parallel \
            --manifest-path="$STRIDED_DIR/Cargo.toml" 2>&1; then
        echo "Running strided-rs (OpenBLAS) benchmark..."
        "$STRIDED_BIN" 2>/dev/null | tee "$STRIDED_BLAS_LOG"
        LOGS+=("$STRIDED_BLAS_LOG")
        echo ""
        echo "Saved: $STRIDED_BLAS_LOG"
    else
        echo "WARNING: strided-rs (blas) build failed (OpenBLAS not found?). Skipping."
        echo "  macOS:  brew install openblas"
        echo "  Linux:  build OpenBLAS >= 0.3.29 from source and set OPENBLAS_LIB_DIR"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
if [[ ${#LOGS[@]} -eq 0 ]]; then
    echo "No benchmark logs produced. Exiting."
    exit 1
fi

MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${TIMESTAMP}.md"

echo "Formatting results as markdown..."
if command -v uv &>/dev/null; then
    uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
else
    python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Logs:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
echo "Markdown: $MARKDOWN_OUT"
