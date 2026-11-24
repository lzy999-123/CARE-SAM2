#!/usr/bin/env bash
# Download & Verify Full Official Benchmarks (VOT-LT2022 50 seqs / 168,282 frames + optional LaSOT-ext 150 seqs)
# Includes 50GB free-disk overflow guard: routes temp archives to a self-cleaning dir, checks free GB,
# and removes residual archives/pip caches automatically.
# Pass `--vot-only` to fetch and verify VOT-LT2022 (50 seqs) only; without it, strictly verifies LaSOT-ext 150/150.
set -euo pipefail

VOT_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --vot-only) VOT_ONLY=1 ;;
        *) echo "Unknown argument: ${arg}"; exit 2 ;;
    esac
done

DATA_DISK="${DATA_DISK:-/root/autodl-tmp/care_eval}"
DS_ROOT="${DATA_DISK}/datasets"
TMP_DL="${DATA_DISK}/.tmp_dl"
mkdir -p "${DS_ROOT}/vot_lt2022" "${DS_ROOT}/lasot_ext" "${TMP_DL}"
export TMPDIR="${TMP_DL}"
trap 'rm -rf "${TMP_DL}"' EXIT

# 50GB free-disk guard: fail fast with exact free-space report before filling the disk
AVAIL_GB=$(df -BG --output=avail "${DATA_DISK}" | tail -n 1 | tr -dc '0-9')
MIN_GB=32
if [ "${VOT_ONLY}" -eq 0 ]; then
    MIN_GB=100
fi
echo "[Disk Guard] ${DATA_DISK} free space: ${AVAIL_GB} GB (minimum required: ${MIN_GB} GB)"
if [ "${AVAIL_GB}" -lt "${MIN_GB}" ]; then
    echo "FATAL: Insufficient free disk on ${DATA_DISK} (${AVAIL_GB} GB < ${MIN_GB} GB). Expand data disk or free space first." >&2
    exit 1
fi

if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

echo "[1/2] Fetching full VOT-LT2022 (50 sequences, 168,282 frames) via official vot-toolkit..."
pip install --no-cache-dir vot-toolkit
python - <<'PY'
from pathlib import Path
import os
import vot
from vot.dataset import download_dataset

print(f"Using official vot-toolkit {vot.__version__} (TMPDIR={os.environ.get('TMPDIR')})")
out = Path(os.environ.get("DATA_DISK", "/root/autodl-tmp/care_eval")) / "datasets" / "vot_lt2022"
# Official VOT-LT2022 manifest (verified in vot/stack/vot2022/longterm.yaml)
download_dataset("https://data.votchallenge.net/vot2022/lt/description.json", str(out))
# Sweep any residual zip archives immediately to protect 50GB free data disks
for residual in out.rglob("*.zip"):
    residual.unlink(missing_ok=True)
seqs = [l.strip() for l in (out / "list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
total_frames = 0
for s in seqs:
    gt = out / s / "groundtruth.txt"
    assert gt.exists(), f"Missing groundtruth.txt for {s}"
    total_frames += sum(1 for _ in gt.open(encoding="utf-8"))
print(f"VOT-LT2022 verified: {len(seqs)} sequences, {total_frames} frames")
assert len(seqs) == 50, f"Expected 50 VOT-LT2022 sequences, got {len(seqs)}"
assert total_frames == 168282, f"Expected 168282 VOT-LT2022 frames, got {total_frames}"
PY

rm -rf "${TMP_DL}"/* || true
pip cache purge >/dev/null 2>&1 || true
df -h / "${DATA_DISK}"

if [ "${VOT_ONLY}" -eq 1 ]; then
    echo "VOT_LT2022_ONLY_VERIFIED: 50/50 sequences (168,282 frames). Skipped LaSOT-ext per --vot-only."
    exit 0
fi

echo "[2/2] Verifying LaSOT-ext Protocol III against official 150-sequence test list..."
LASOT_LIST="${DS_ROOT}/extension_testing_set.txt"
if [ ! -f "${LASOT_LIST}" ]; then
    wget -qO "${LASOT_LIST}" https://hengfan2010.github.io/projects/LaSOT/extension_testing_set.txt
fi
python - <<'PY'
from pathlib import Path
import os
import sys

root = Path(os.environ.get("DATA_DISK", "/root/autodl-tmp/care_eval")) / "datasets" / "lasot_ext"
list_path = Path(os.environ.get("DATA_DISK", "/root/autodl-tmp/care_eval")) / "datasets" / "extension_testing_set.txt"
expected = {ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
assert len(expected) == 150, f"Official LaSOT-ext list should have 150 sequences, got {len(expected)}"
present = {
    p.parent.name: p.parent
    for p in root.rglob("groundtruth.txt")
    if (p.parent / "img").is_dir()
}
missing = sorted(expected - set(present.keys()))
print(f"LaSOT-ext sequences matched: {len(expected) - len(missing)}/150")
if missing:
    print(f"ERROR: Missing {len(missing)} official LaSOT-ext sequences (first 10: {missing[:10]}).")
    print("Place/extract full LaSOT-ext under datasets/lasot_ext (or pass --vot-only).")
    sys.exit(2)
print("LASOT_EXT_VERIFIED: 150/150 official Protocol III sequences present.")
PY
