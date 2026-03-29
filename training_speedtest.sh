#!/usr/bin/env bash
set -euo pipefail

# Training speed benchmark: before vs after optimizations.
#
# Runs training_speedtest.py at the pre-optimization commit and at the
# current (optimized) commit, rebuilding the Metal extension each time,
# then prints a per-phase comparison table.
#
# Usage:  ./training_speedtest.sh [dataset]

DATASET="${1:-test79-2022-05-may-12tb7p.min-v2.binpack}"
SCRIPT="training_speedtest.py"
BEFORE_COMMIT="7439eec"   # last commit before optimization work
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
CURRENT_SHA="$(git rev-parse HEAD)"
RESULTS_DIR="$(mktemp -d)"
BEFORE_LOG="${RESULTS_DIR}/before.txt"
AFTER_LOG="${RESULTS_DIR}/after.txt"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "============================================================"
echo "  Training Speed Benchmark: Before vs After Optimizations"
echo "============================================================"
echo ""
echo "  Dataset:   ${DATASET}"
echo "  Before:    ${BEFORE_COMMIT} (pre-optimization)"
echo "  After:     ${CURRENT_SHA:0:7} (current HEAD)"
echo ""

# ---- Phase 1: Benchmark the CURRENT (optimized) code ----

echo "============================================================"
echo "  Phase 1/2: Benchmarking AFTER (optimized) code"
echo "============================================================"
echo ""

python3 setup_metal.py build_ext -i 2>&1 | tail -1
echo ""

python3 "${SCRIPT}" "${DATASET}" 2>&1 | tee "${AFTER_LOG}"

# ---- Phase 2: Benchmark the BEFORE (pre-optimization) code ----

echo ""
echo "============================================================"
echo "  Phase 2/2: Benchmarking BEFORE (pre-optimization) code"
echo "============================================================"
echo ""

cp "${SCRIPT}" "${RESULTS_DIR}/${SCRIPT}"
git checkout "${BEFORE_COMMIT}" -- . 2>/dev/null
cp "${RESULTS_DIR}/${SCRIPT}" "${SCRIPT}"

python3 setup_metal.py build_ext -i 2>&1 | tail -1
echo ""

python3 "${SCRIPT}" "${DATASET}" 2>&1 | tee "${BEFORE_LOG}"

# ---- Restore optimized code ----

git checkout "${CURRENT_BRANCH}" -- . 2>/dev/null
python3 setup_metal.py build_ext -i 2>&1 | tail -1

# ---- Print comparison table ----

extract() {
    local file="$1" label="$2"
    grep "$label" "$file" | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1
}

B_CLIP=$(extract "${BEFORE_LOG}" "clip.*median=")
B_FWD=$(extract "${BEFORE_LOG}" "forward.*median=")
B_LOSS=$(extract "${BEFORE_LOG}" "loss.*median=")
B_BWD=$(extract "${BEFORE_LOG}" "backward.*median=")
B_OPT=$(extract "${BEFORE_LOG}" "optimizer.*median=")
B_TOTAL=$(extract "${BEFORE_LOG}" "TOTAL.*median=")
B_TPUT=$(grep "Median throughput" "${BEFORE_LOG}" | grep -oE '[0-9,]+' | head -1 | tr -d ',')

A_CLIP=$(extract "${AFTER_LOG}" "clip.*median=")
A_FWD=$(extract "${AFTER_LOG}" "forward.*median=")
A_LOSS=$(extract "${AFTER_LOG}" "loss.*median=")
A_BWD=$(extract "${AFTER_LOG}" "backward.*median=")
A_OPT=$(extract "${AFTER_LOG}" "optimizer.*median=")
A_TOTAL=$(extract "${AFTER_LOG}" "TOTAL.*median=")
A_TPUT=$(grep "Median throughput" "${AFTER_LOG}" | grep -oE '[0-9,]+' | head -1 | tr -d ',')

B_OPTNAME=$(grep "Optimizer:" "${BEFORE_LOG}" | sed 's/.*Optimizer: *//')
A_OPTNAME=$(grep "Optimizer:" "${AFTER_LOG}" | sed 's/.*Optimizer: *//')
B_LOSSNAME=$(grep "Loss:" "${BEFORE_LOG}" | sed 's/.*Loss: *//')
A_LOSSNAME=$(grep "Loss:" "${AFTER_LOG}" | sed 's/.*Loss: *//')

echo ""
echo ""
echo "============================================================"
echo "  COMPARISON: BEFORE vs AFTER"
echo "============================================================"
echo ""
printf "  %-14s  %12s  %12s  %10s\n" "Phase" "BEFORE (ms)" "AFTER (ms)" "Change"
printf "  %-14s  %12s  %12s  %10s\n" "--------------" "------------" "------------" "----------"

print_row() {
    local label="$1" bval="$2" aval="$3"
    if [[ -n "$bval" && -n "$aval" ]]; then
        local diff
        diff=$(python3 -c "b=$bval; a=$aval; d=a-b; print(f'{d:+.2f}')")
        printf "  %-14s  %9s ms  %9s ms  %7s ms\n" "$label" "$bval" "$aval" "$diff"
    fi
}

print_row "clip" "$B_CLIP" "$A_CLIP"
print_row "forward" "$B_FWD" "$A_FWD"
print_row "loss" "$B_LOSS" "$A_LOSS"
print_row "backward" "$B_BWD" "$A_BWD"
print_row "optimizer" "$B_OPT" "$A_OPT"
echo "  --------------  ------------  ------------  ----------"
print_row "TOTAL" "$B_TOTAL" "$A_TOTAL"

echo ""
printf "  %-14s  %12s  %12s\n" "Optimizer" "${B_OPTNAME}" "${A_OPTNAME}"
printf "  %-14s  %12s  %12s\n" "Loss" "${B_LOSSNAME}" "${A_LOSSNAME}"

if [[ -n "${B_TPUT}" && -n "${A_TPUT}" ]]; then
    SPEEDUP=$(python3 -c "print(f'{int(\"${A_TPUT}\") / int(\"${B_TPUT}\"):.3f}')")
    echo ""
    printf "  Throughput BEFORE:  %10s pos/s\n" "$(printf "%'d" "${B_TPUT}")"
    printf "  Throughput AFTER:   %10s pos/s\n" "$(printf "%'d" "${A_TPUT}")"
    echo "  Speedup:            ${SPEEDUP}x"
fi

echo ""
echo "  Raw logs:  ${RESULTS_DIR}/"
echo ""
