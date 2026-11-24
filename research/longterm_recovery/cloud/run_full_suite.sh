#!/usr/bin/env bash
# Streamlined Cloud Benchmark Matrix on 1x RTX 4090 (24GB) — Mechanistic Diagnosis + Calibration-Preserving Control
# Aligned with 2026-11-22 Go/No-Go Positioning Switch:
#   - REMOVED structurally zero-effect `+CARE` arms on SAMURAI & DAM4SAM (~50% cost reduction!)
#   - ADDED 5-line absent-skip gating baselines (`gate_skip_absent`, `gate_skip_absent_keept1`) on Tiny & Base+
#   - ADDED full 50-sequence VOT-LT2022 9-arm counterfactual memory-inertia decomposition
#   - RETAINED full CARE component & verifier ablations (`none`, `sam2_native`, `dinov2`, `no_dual`, `no_certify`, `no_scale`, `no_rollback`)
#
# Usage (inside `tmux new -s care` after switching AutoDL to GPU mode):
#   bash run_full_suite.sh --vot-only      # Recommended first pass: complete 50-seq VOT-LT2022 matrix (~12-16h, ~¥35-55)
#   bash run_full_suite.sh --with-lasot    # Complete VOT-LT2022 (50) + LaSOT-ext Protocol III (150) matrix
set -euo pipefail

RUN_LASOT=0
INCLUDE_DINOV2=1
for arg in "$@"; do
    case "${arg}" in
        --vot-only|--skip-lasot) RUN_LASOT=0 ;;
        --with-lasot)            RUN_LASOT=1 ;;
        --no-dinov2)             INCLUDE_DINOV2=0 ;;
        *) echo "Unknown argument: ${arg} (use --vot-only or --with-lasot [--no-dinov2])"; exit 2 ;;
    esac
done

CODE_DIR="${CODE_DIR:-/root/care_code/research/longterm_recovery}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp/care_eval}"
WEIGHT_T="${DATA_DISK}/weights/sam2.1_hiera_tiny.pt"
WEIGHT_B="${DATA_DISK}/weights/sam2.1_hiera_base_plus.pt"
# Strict host separation: vanilla sam21 MUST use Meta's official facebookresearch/sam2 tree!
META_SAM2_SRC="${DATA_DISK}/repos/sam2_meta"
SAMURAI_SRC="${DATA_DISK}/repos/samurai/sam2"
DAM4SAM_SRC="${DATA_DISK}/repos/DAM4SAM"
VOT_DIR="${DATA_DISK}/datasets/vot_lt2022"
LASOT_DIR="${DATA_DISK}/datasets/lasot_ext"
LASOT_LIST="${DATA_DISK}/datasets/extension_testing_set.txt"
RUNS_DIR="${DATA_DISK}/runs"
NUM_SHARDS="${NUM_SHARDS:-4}"

# Pre-flight verification (fail fast, never silently skip!)
for req in "${META_SAM2_SRC}" "${SAMURAI_SRC}" "${DAM4SAM_SRC}" "${WEIGHT_T}" "${WEIGHT_B}" "${VOT_DIR}/list.txt"; do
    if [ ! -e "${req}" ]; then
        echo "FATAL: Required path missing: ${req}" >&2
        exit 1
    fi
done

VOT_COUNT=$(grep -cve '^\s*$' "${VOT_DIR}/list.txt")
if [ "${VOT_COUNT}" -ne 50 ]; then
    echo "FATAL: Expected 50 VOT-LT2022 sequences in ${VOT_DIR}/list.txt, found ${VOT_COUNT}" >&2
    exit 1
fi

if [ "${RUN_LASOT}" -eq 1 ]; then
    python - "${LASOT_DIR}" "${LASOT_LIST}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
list_file = Path(sys.argv[2])
if not list_file.exists():
    sys.exit(f"FATAL: Official LaSOT-ext list missing: {list_file}")
expected = {ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
present = {p.parent.name for p in root.rglob("groundtruth.txt") if (p.parent / "img").is_dir()}
missing = sorted(expected - present)
if len(expected) != 150 or missing:
    sys.exit(f"FATAL: Incomplete LaSOT-ext ({len(expected) - len(missing)}/150 matched; missing e.g. {missing[:5]}).")
print("Pre-flight OK: 150/150 official LaSOT-ext sequences verified.")
PY
fi

COMPLETED_TAGS=()

run_care_sharded () {
    local TAG="$1"
    local DATASET="$2"
    local DATA_PATH="$3"
    local HOST="$4"
    local BACKBONE="$5"
    local VERIFIER="$6"
    local WEIGHTS="$7"
    local HOST_SRC="$8"
    local ARMS_STR="$9"
    shift 9
    local EXTRA_ARGS=("$@")
    local OUT_DIR="${RUNS_DIR}/${TAG}"
    mkdir -p "${OUT_DIR}"

    # shellcheck disable=SC2206
    local ARMS_ARR=(${ARMS_STR})

    echo ">>> [CARE] Launching ${TAG} (host=${HOST}, bb=${BACKBONE}, arms=${ARMS_STR}, ${NUM_SHARDS} shards)..."
    local PIDS=()
    for ((k=0; k<NUM_SHARDS; k++)); do
        python -u "${CODE_DIR}/care_sam2_plugin.py" \
            --dataset "${DATASET}" --data "${DATA_PATH}" \
            --host "${HOST}" --backbone "${BACKBONE}" --verifier "${VERIFIER}" \
            --weights "${WEIGHTS}" --host-source "${HOST_SRC}" \
            --output "${OUT_DIR}" --arms "${ARMS_ARR[@]}" \
            --sdpa-kernel-mode allow_all \
            --shard-idx "${k}" --num-shards "${NUM_SHARDS}" --resume \
            "${EXTRA_ARGS[@]}" > "${OUT_DIR}/shard_${k}.log" 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo ">>> [CARE] Merging & independently auditing ${TAG}..."
    python -u "${CODE_DIR}/care_sam2_plugin.py" \
        --dataset "${DATASET}" --data "${DATA_PATH}" \
        --host "${HOST}" --backbone "${BACKBONE}" --verifier "${VERIFIER}" \
        --weights "${WEIGHTS}" --host-source "${HOST_SRC}" \
        --output "${OUT_DIR}" --arms "${ARMS_ARR[@]}" \
        --sdpa-kernel-mode allow_all --merge-shards "${EXTRA_ARGS[@]}"

    python -u "${CODE_DIR}/audit_care_sam2_plugin.py" \
        --dataset "${DATASET}" --data "${DATA_PATH}" --run "${OUT_DIR}"

    COMPLETED_TAGS+=("${TAG}")
}

run_gating_sharded () {
    local TAG="$1"
    local DATASET="$2"
    local DATA_PATH="$3"
    local BACKBONE="$4"
    local WEIGHTS="$5"
    local ARMS_STR="$6"
    local OUT_DIR="${RUNS_DIR}/${TAG}"
    mkdir -p "${OUT_DIR}"

    # shellcheck disable=SC2206
    local ARMS_ARR=(${ARMS_STR})

    echo ">>> [GATING] Launching ${TAG} (bb=${BACKBONE}, arms=${ARMS_STR}, ${NUM_SHARDS} shards)..."
    local PIDS=()
    for ((k=0; k<NUM_SHARDS; k++)); do
        python -u "${CODE_DIR}/run_memory_gating_benchmark.py" \
            --dataset "${DATASET}" --data "${DATA_PATH}" \
            --backbone "${BACKBONE}" --weights "${WEIGHTS}" --host-source "${META_SAM2_SRC}" \
            --output "${OUT_DIR}" --arms "${ARMS_ARR[@]}" \
            --sdpa-kernel-mode allow_all \
            --shard-idx "${k}" --num-shards "${NUM_SHARDS}" --resume \
            > "${OUT_DIR}/shard_${k}.log" 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo ">>> [GATING] Merging & auditing ${TAG}..."
    python -u "${CODE_DIR}/run_memory_gating_benchmark.py" \
        --dataset "${DATASET}" --data "${DATA_PATH}" \
        --backbone "${BACKBONE}" --weights "${WEIGHTS}" --host-source "${META_SAM2_SRC}" \
        --output "${OUT_DIR}" --arms "${ARMS_ARR[@]}" \
        --sdpa-kernel-mode allow_all --merge-shards

    COMPLETED_TAGS+=("${TAG}")
}

run_probe_sharded () {
    local TAG="$1"
    local DATASET="$2"
    local DATA_PATH="$3"
    local BACKBONE="$4"
    local WEIGHTS="$5"
    local OUT_DIR="${RUNS_DIR}/${TAG}"
    mkdir -p "${OUT_DIR}"

    echo ">>> [A6-9ARM] Launching ${TAG} (bb=${BACKBONE}, ${NUM_SHARDS} shards)..."
    local PIDS=()
    for ((k=0; k<NUM_SHARDS; k++)); do
        python -u "${CODE_DIR}/run_counterfactual_inertia_benchmark.py" \
            --dataset "${DATASET}" --data "${DATA_PATH}" \
            --backbone "${BACKBONE}" --weights "${WEIGHTS}" --host-source "${META_SAM2_SRC}" \
            --output "${OUT_DIR}" --sdpa-kernel-mode allow_all \
            --shard-idx "${k}" --num-shards "${NUM_SHARDS}" --resume \
            > "${OUT_DIR}/shard_${k}.log" 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo ">>> [A6-9ARM] Merging & scoring ${TAG}..."
    python -u "${CODE_DIR}/run_counterfactual_inertia_benchmark.py" \
        --dataset "${DATASET}" --data "${DATA_PATH}" \
        --backbone "${BACKBONE}" --weights "${WEIGHTS}" --host-source "${META_SAM2_SRC}" \
        --output "${OUT_DIR}" --sdpa-kernel-mode allow_all --merge-shards

    COMPLETED_TAGS+=("${TAG}")
}

# =============================================================================
# PART 1: Full VOT-LT2022 (50 sequences, 168,282 frames)
# =============================================================================
# 1A. Core Pareto & Multi-Scale Comparisons (SAM 2.1 Tiny & Base+: vanilla, 5-line gating, CARE)
run_care_sharded   "vot22_sam21_tiny_native"       vot_lt_dir "${VOT_DIR}" sam21   tiny      sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "baseline care"
run_gating_sharded "vot22_sam21_tiny_gating"       vot_lt_dir "${VOT_DIR}"         tiny                  "${WEIGHT_T}"                    "gate_skip_absent gate_skip_absent_keept1"

run_care_sharded   "vot22_sam21_baseplus_native"   vot_lt_dir "${VOT_DIR}" sam21   base_plus sam2_native "${WEIGHT_B}" "${META_SAM2_SRC}" "baseline care"
run_gating_sharded "vot22_sam21_baseplus_gating"   vot_lt_dir "${VOT_DIR}"         base_plus             "${WEIGHT_B}"                    "gate_skip_absent gate_skip_absent_keept1"

# 1B. External Published Baselines (single baseline arm ONLY — no wasted +CARE arms!)
run_care_sharded   "vot22_samurai_tiny_m1"         vot_lt_dir "${VOT_DIR}" samurai tiny      sam2_native "${WEIGHT_T}" "${SAMURAI_SRC}"   "baseline" --samurai-baseline-min-obj -1.0
run_care_sharded   "vot22_samurai_tiny_m0"         vot_lt_dir "${VOT_DIR}" samurai tiny      sam2_native "${WEIGHT_T}" "${SAMURAI_SRC}"   "baseline" --samurai-baseline-min-obj  0.0
run_care_sharded   "vot22_dam4sam_tiny"            vot_lt_dir "${VOT_DIR}" dam4sam tiny      sam2_native "${WEIGHT_T}" "${DAM4SAM_SRC}"   "baseline"

# 1C. Full-Benchmark 9-Arm Counterfactual Memory-Inertia Decomposition (50 sequences!)
run_probe_sharded  "vot22_sam21_tiny_a6_decomposed" vot_lt_dir "${VOT_DIR}"        tiny                  "${WEIGHT_T}"

# 1D. Complete CARE Component & Verifier Ablations on Full VOT-LT2022 (single care arm each)
run_care_sharded   "vot22_abl_verifier_none"       vot_lt_dir "${VOT_DIR}" sam21   tiny      none        "${WEIGHT_T}" "${META_SAM2_SRC}" "care"
run_care_sharded   "vot22_abl_no_rollback"         vot_lt_dir "${VOT_DIR}" sam21   tiny      sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "care" --no-rollback
run_care_sharded   "vot22_abl_no_dual_anchor"      vot_lt_dir "${VOT_DIR}" sam21   tiny      sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "care" --no-dual-anchor
run_care_sharded   "vot22_abl_no_certify"          vot_lt_dir "${VOT_DIR}" sam21   tiny      sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "care" --no-certify-anchor
run_care_sharded   "vot22_abl_no_scale_norm"       vot_lt_dir "${VOT_DIR}" sam21   tiny      sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "care" --no-scale-norm
if [ "${INCLUDE_DINOV2}" -eq 1 ]; then
    run_care_sharded "vot22_abl_verifier_dinov2"   vot_lt_dir "${VOT_DIR}" sam21   tiny      dinov2      "${WEIGHT_T}" "${META_SAM2_SRC}" "care"
fi

# =============================================================================
# PART 2 (Optional): Full LaSOT-ext Protocol III (150 sequences)
# =============================================================================
if [ "${RUN_LASOT}" -eq 1 ]; then
    run_care_sharded   "lasot_ext_sam21_tiny_native" lasot_ext "${LASOT_DIR}" sam21   tiny sam2_native "${WEIGHT_T}" "${META_SAM2_SRC}" "baseline care"
    run_gating_sharded "lasot_ext_sam21_tiny_gating" lasot_ext "${LASOT_DIR}"         tiny             "${WEIGHT_T}"                    "gate_skip_absent gate_skip_absent_keept1"
    run_care_sharded   "lasot_ext_samurai_tiny_m1"   lasot_ext "${LASOT_DIR}" samurai tiny sam2_native "${WEIGHT_T}" "${SAMURAI_SRC}"   "baseline" --samurai-baseline-min-obj -1.0
    run_care_sharded   "lasot_ext_samurai_tiny_m0"   lasot_ext "${LASOT_DIR}" samurai tiny sam2_native "${WEIGHT_T}" "${SAMURAI_SRC}"   "baseline" --samurai-baseline-min-obj  0.0
    run_care_sharded   "lasot_ext_dam4sam_tiny"      lasot_ext "${LASOT_DIR}" dam4sam tiny sam2_native "${WEIGHT_T}" "${DAM4SAM_SRC}"   "baseline"
fi

# =============================================================================
# PART 3: Programmatic Summary Table & Paired Bootstrap CI Generation
# =============================================================================
python -u "${CODE_DIR}/summarize_cloud_results.py" \
    --runs-dir "${RUNS_DIR}" \
    --output "${RUNS_DIR}/CLOUD_SUMMARY_REPORT.md"

if [ "${RUN_LASOT}" -eq 1 ]; then
    echo "CLOUD_MATRIX_COMPLETE (VOT-LT2022 + LaSOT-ext): ${#COMPLETED_TAGS[@]} runs audited."
else
    echo "VOT_LT2022_MATRIX_COMPLETE (50 sequences; pass --with-lasot for LaSOT-ext): ${#COMPLETED_TAGS[@]} runs audited."
fi
printf '  - %s\n' "${COMPLETED_TAGS[@]}"
