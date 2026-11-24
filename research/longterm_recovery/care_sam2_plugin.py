"""CARE: Counterfactual Absence-Inertia Recovery with 1-Step Rollback.

Host-agnostic, zero-external-model plug-and-play recovery plugin for SAM 2.1,
SAMURAI, and DAM4SAM across Hiera Tiny / Base+ / Large backbones.

Features:
- Zero-external-model SAM 2 native feature verifier (`--verifier sam2_native`),
  reading cached FPN features (`backbone_fpn[-1]`) directly with zero extra
  models, zero extra weights, and zero extra image encoder forwards.
- Retains `--verifier dinov2` and `--verifier none` for controlled ablation.
- Full component ablation & sensitivity flags (`--no-dual-anchor`,
  `--no-certify-anchor`, `--no-scale-norm`, `--no-rollback`, `--k-sil`,
  `--tau-rb`, `--rho-max`, `--keep-window`).
- Sharded multi-process batching (`--shard-idx`, `--num-shards`), crash-safe
  resume (`--resume`), and shard merging (`--merge-shards`).
- Strictly causal inference (only frame 0 + init_box read before tracking;
  all evaluation metrics computed strictly post-hoc).
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
from statistics import median, quantiles
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vision_f

from audit_journal_vot_results import overlap
from evaluation import box_iou
from hoot_diagnostic import labels_from
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_hoot_mask_identity import descriptors as dino_descriptors
from tip_hoot_mask_identity import silhouette as dino_silhouette
from tip_hoot_mask_identity import similarity as dino_similarity
from tip_local_first_frame_calibration import global_boxes, offset_boxes
from tip_paired_query_continuous import combine
from tip_scale_consistent_recovery import BORDER_MARGIN, GAMMA_MIN, LINK, QUALITY, border_gamma
from tip_token_identity import SURROUND, TokenEncoder, crop


# Single frozen default universal parameter set (identical to DAIR-R baseline)
DEFAULT_RHO_MIN = 0.10
DEFAULT_RHO_MAX = 2.50
DEFAULT_TAU_OBJ = 2.0
DEFAULT_TAU_OBJ_BORDER = 3.5
DEFAULT_CLEAN_ANCHOR_OBJ = 4.0
DEFAULT_CLEAN_ANCHOR_STRIDE = 5
DEFAULT_SILENCE_MIN_STREAK = 6
DEFAULT_SILENCE_PROBE_STRIDE = 2
DEFAULT_ROLLBACK_OBJ_MIN = 0.80
DEFAULT_ROLLBACK_COOLDOWN = 12
DEFAULT_MEMORY_PURGE_WINDOW = 15
DEFAULT_MEMORY_KEEP_WINDOW = 32
DEFAULT_SAMURAI_BASELINE_MIN_OBJ: float | None = None  # None -> stock (-1.0)
DEFAULT_SAMURAI_CARE_MIN_OBJ: float | None = None      # None -> match stock (-1.0); never silently mismatch baseline!
DEFAULT_SDPA_KERNEL_MODE = "allow_all"

# Native feature verifier sampling grid (16x16 over 2x surround box)
NATIVE_GRID_SIDE = 16
NATIVE_MIN_FG_POINTS = 4
NATIVE_MIN_BG_POINTS = 8
_NY, _NX = np.meshgrid(np.arange(NATIVE_GRID_SIDE), np.arange(NATIVE_GRID_SIDE), indexing="ij")
NATIVE_CENTERS = (np.column_stack([_NX.ravel(), _NY.ravel()]) + 0.5) / float(NATIVE_GRID_SIDE)
NATIVE_INSIDE = ((NATIVE_CENTERS >= 0.25) & (NATIVE_CENTERS < 0.75)).all(axis=1)

BACKBONE_CONFIGS = {
    "sam21": {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_t.yaml"),
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_b+.yaml"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_l.yaml"),
    },
    "samurai": {
        "tiny": ("configs/samurai/sam2.1_hiera_t.yaml",),
        "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml",),
        "large": ("configs/samurai/sam2.1_hiera_l.yaml",),
    },
    "dam4sam": {
        "tiny": ("sam21pp_hiera_t.yaml", "sam21pp-T"),
        "base_plus": ("sam21pp_hiera_b+.yaml", "sam21pp-B"),
        "large": ("sam21pp_hiera_l.yaml", "sam21pp-L"),
    },
}


def mask_to_bbox(mask) -> list[float] | None:
    if mask is None:
        return None
    arr = np.asarray(mask) > 0
    ys = np.flatnonzero(arr.any(axis=1))
    xs = np.flatnonzero(arr.any(axis=0))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs[0]), float(ys[0]), float(xs[-1] - xs[0] + 1), float(ys[-1] - ys[0] + 1)]


def _scalar(value) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    fin = arr[np.isfinite(arr)]
    return float(fin.reshape(-1)[0]) if fin.size else None


def _sigmoid(x: float | None) -> float:
    if x is None or not math.isfinite(x):
        return 0.5
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clone_mem_entry(entry: dict) -> dict:
    cloned = {}
    for k, v in entry.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().clone()
        elif isinstance(v, list):
            cloned[k] = [x.detach().clone() if torch.is_tensor(x) else x for x in v]
        else:
            cloned[k] = v
    return cloned


def _resolve_sam2_config(sam2_pkg_dir: Path, candidates: tuple[str, ...]) -> tuple[str, Path | None]:
    for cfg in candidates:
        p1 = sam2_pkg_dir / cfg
        if p1.exists():
            return cfg, p1
        p2 = sam2_pkg_dir / "configs" / cfg
        if p2.exists():
            return cfg, p2
    return candidates[0], None


def _git_head_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    cur = Path(path).resolve()
    for cand in (cur, *cur.parents[:3]):
        if (cand / ".git").exists():
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(cand), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                return out.decode("utf-8", errors="ignore").strip() or None
            except Exception:
                return None
    return None


def _read_first_line_only(path: Path) -> str:
    """Strictly causal helper: read ONLY line 0 of a groundtruth file without reading subsequent lines."""
    with path.open("r", encoding="utf-8") as fh:
        return fh.readline().strip()


def _dir_frames_manifest_sha(name: str, frames: list[Path]) -> str:
    """Hash frame count, names, and byte sizes without reading post-initialization annotations."""
    payload = f"{name}:{len(frames)}:" + ";".join(f"{p.name}:{p.stat().st_size}" for p in frames)
    return sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Evaluation Metrics (Strictly Post-Hoc: Custom + Official VOT-LT & LaSOT)
# =============================================================================

def macro_prf(videos_data: dict) -> tuple[dict, dict]:
    """Fixed-point Macro Precision, Recall, F1 over frames 1..T-1."""
    ps, rs = [], []
    by_video = {}
    for name, (labels, boxes) in videos_data.items():
        vis = 0
        rep = 0
        ov_sum = 0.0
        for gt, pred in zip(labels[1:], boxes[1:], strict=True):
            if gt is not None:
                vis += 1
            if pred is not None:
                rep += 1
            if gt is not None and pred is not None:
                ov_sum += box_iou(pred, gt)
        p = (ov_sum / rep) if rep > 0 else 1.0
        r = (ov_sum / vis) if vis > 0 else 0.0
        ps.append(p)
        rs.append(r)
        by_video[name] = {
            "visible": vis,
            "reported": rep,
            "overlap_sum": ov_sum,
            "precision": p,
            "recall": r,
        }
    mp = sum(ps) / len(ps) if ps else 0.0
    mr = sum(rs) / len(rs) if rs else 0.0
    mf1 = (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
    return {"precision": mp, "recall": mr, "f1": mf1}, by_video


def vot_lt_curve_metrics(videos_conf_data: dict, num_thresholds: int = 100) -> dict:
    """Custom local VOT-LT-style confidence-sweep Pr / Re / Max F-score (non-official diagnostic).

    Use `vot_lt_toolkit_metrics()` backed by official `vot-toolkit` for leaderboard reporting.
    """
    if not videos_conf_data:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0, "best_threshold": 0.0}

    # Collect per-video arrays of (gt_vis, overlap, conf) on frames 1..T-1
    per_video = []
    all_confs = []
    for _name, (labels, boxes, confs) in videos_conf_data.items():
        vis = 0
        ovs = []
        cs = []
        for gt, pred, conf in zip(labels[1:], boxes[1:], confs[1:], strict=True):
            if gt is not None:
                vis += 1
            if pred is not None:
                c = float(conf) if conf is not None and math.isfinite(conf) else 0.5
                ov = box_iou(pred, gt) if gt is not None else 0.0
                ovs.append(ov)
                cs.append(c)
                all_confs.append(c)
        per_video.append((vis, np.asarray(ovs, dtype=np.float64), np.asarray(cs, dtype=np.float64)))

    if not all_confs:
        return {"f_score": 0.0, "precision": 1.0, "recall": 0.0, "best_threshold": 0.0}

    # Quantile-spaced thresholds + uniform grid over [0, 1]
    c_arr = np.asarray(all_confs, dtype=np.float64)
    q_grid = np.quantile(c_arr, np.linspace(0.0, 1.0, num_thresholds))
    u_grid = np.linspace(0.0, 1.0, 51)
    thresholds = np.unique(np.concatenate(([0.0], q_grid, u_grid)))

    best_f = -1.0
    best_p = 0.0
    best_r = 0.0
    best_t = 0.0
    for tau in thresholds:
        ps, rs = [], []
        for vis, ovs, cs in per_video:
            if len(cs) == 0:
                ps.append(1.0)
                rs.append(0.0)
                continue
            keep = cs >= tau
            n_rep = int(keep.sum())
            ov_sum = float(ovs[keep].sum()) if n_rep > 0 else 0.0
            ps.append(ov_sum / n_rep if n_rep > 0 else 1.0)
            rs.append(ov_sum / vis if vis > 0 else 0.0)
        mp = float(np.mean(ps))
        mr = float(np.mean(rs))
        f1 = (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
        if f1 > best_f:
            best_f, best_p, best_r, best_t = f1, mp, mr, float(tau)
    return {
        "f_score": best_f,
        "precision": best_p,
        "recall": best_r,
        "best_threshold": best_t,
    }


def vot_lt_toolkit_metrics(
    videos_toolkit_data: dict,
    toolkit_path: Path | None = None,
    resolution: int = 100,
) -> dict:
    """Official `vot-toolkit` long-term Precision/Recall/F-score (`vot.analysis.longterm`).

    IMPORTANT: Frame 0 and abstained frames (`bbox is None`) receive `confidence = float('nan')`
    so `confidence >= threshold` is always False on abstentions, preventing `vot-toolkit`'s
    `calculate_overlap(Special(0), Special(0)) == 1.0` from inflating Recall at low thresholds.
    """
    if not videos_toolkit_data:
        return {"available": False, "reason": "empty input"}

    if toolkit_path is not None and Path(toolkit_path).exists():
        tp = str(Path(toolkit_path).resolve())
        existing_vot = sys.modules.get("vot")
        if existing_vot is not None:
            ex_file = getattr(existing_vot, "__file__", "") or ""
            if tp.lower() not in str(Path(ex_file).resolve()).lower():
                for mod_name in [k for k in list(sys.modules.keys()) if k == "vot" or k.startswith("vot.")]:
                    sys.modules.pop(mod_name, None)
        if tp in sys.path:
            sys.path.remove(tp)
        sys.path.insert(0, tp)

    try:
        import vot
        import vot.analysis.longterm as vot_lt_mod
        from vot.analysis.longterm import compute_tpr_curves, determine_thresholds
        from vot.region import Rectangle, Special
    except Exception as exc:
        return {"available": False, "reason": f"vot-toolkit import failed: {exc}"}

    class _Seq:
        def __init__(self, labels, size):
            self._gt = [Rectangle(*b) if b is not None else Special(0) for b in labels]
            self.size = size

        def groundtruth(self):
            return self._gt

    data = []
    for _name, (labels, boxes, confs, size) in videos_toolkit_data.items():
        traj = [Special(1)] + [
            Rectangle(*b) if b is not None else Special(0)
            for b in boxes[1:]
        ]
        scores = [float("nan")] + [
            (float(c) if (c is not None and math.isfinite(float(c))) else 0.5) if b is not None else float("nan")
            for b, c in zip(boxes[1:], confs[1:], strict=True)
        ]
        data.append((traj, scores, _Seq(labels, size)))

    thresholds = determine_thresholds(
        [s for _, sc, _ in data for s in sc],
        resolution,
    )
    reports = [
        compute_tpr_curves(tr, sc, sq, thresholds, bounded=True)
        for tr, sc, sq in data
    ]
    arr = np.asarray(reports, dtype=np.float64)  # (N_videos, 2, N_thresholds)
    pr = arr[:, 0, :].mean(axis=0)
    re = arr[:, 1, :].mean(axis=0)
    fsc = np.divide(2.0 * pr * re, pr + re, out=np.zeros_like(pr), where=(pr + re) > 0)
    bi = int(np.argmax(fsc))
    best_th = thresholds[bi]
    return {
        "available": True,
        "toolkit_version": getattr(vot, "__version__", "unknown"),
        "longterm_py_path": str(Path(vot_lt_mod.__file__).resolve()),
        "longterm_py_sha256": digest(Path(vot_lt_mod.__file__)),
        "resolution": int(resolution),
        "abstention_encoding": "Special(0) with confidence=NaN (prevents empty-empty overlap=1.0 inflation)",
        "f_score": float(fsc[bi]),
        "precision": float(pr[bi]),
        "recall": float(re[bi]),
        "best_threshold": float(best_th) if math.isfinite(best_th) else str(best_th),
    }


def lasot_metrics(videos_data: dict) -> dict:
    """Compute LaSOT / LaSOT-ext Success (AUC), Normalized Precision, and Precision@20 under BOTH protocols:

    - `matlab_conf`: exact conference-version `HengLan/LaSOT_Evaluation_Toolkit/utils/eval_tracker.m`
      (removes absent frames in `calc_seq_err_robust.m`, uses strict `>` on overlap thresholds `0:0.05:1`,
      and divides by total sequence frames `len_all = size(anno, 1)`).
    - `vis_strict`: visible-frames-only denominator with strict `>` on overlap thresholds.
    - Top-level `auc`, `p_norm`, `p20`: legacy visible-frames `>=` local metric (kept for backward compatibility
      and explicitly documented as non-MATLAB-conference protocol).
    """
    iou_thresholds = np.linspace(0.0, 1.0, 21)
    norm_thresholds = np.linspace(0.0, 0.5, 51)

    seq_vis_ge_auc, seq_vis_gt_auc, seq_mat_auc = [], [], []
    seq_vis_p20, seq_mat_p20 = [], []
    seq_vis_pnorm, seq_mat_pnorm = [], []
    by_video = {}

    for name, (labels, boxes) in videos_data.items():
        n_total = len(labels)
        ious = []
        cle = []
        ncle = []
        for gt, pred in zip(labels, boxes, strict=True):
            if gt is None:
                continue
            gx, gy, gw, gh = gt
            if pred is None:
                ious.append(0.0)
                cle.append(float("inf"))
                ncle.append(float("inf"))
            else:
                ious.append(box_iou(pred, gt))
                px, py, pw, ph = pred
                dx = (px + 0.5 * pw) - (gx + 0.5 * gw)
                dy = (py + 0.5 * ph) - (gy + 0.5 * gh)
                cle.append(math.hypot(dx, dy))
                ncle.append(math.hypot(dx / max(1.0, gw), dy / max(1.0, gh)))
        if not ious or n_total == 0:
            continue
        iou_arr = np.asarray(ious, dtype=np.float64)
        cle_arr = np.asarray(cle, dtype=np.float64)
        ncle_arr = np.asarray(ncle, dtype=np.float64)

        # 1) Legacy visible-only with >= (for t > 0)
        sc_vis_ge = np.array(
            [(iou_arr > 0.0).mean() if t == 0.0 else (iou_arr >= t).mean() for t in iou_thresholds],
            dtype=np.float64,
        )
        auc_vis_ge = float(sc_vis_ge.mean())
        p20_vis = float((cle_arr <= 20.0).mean())
        pnorm_vis = float(np.mean([(ncle_arr <= t).mean() for t in norm_thresholds]))

        # 2) Visible-only with strict >
        sc_vis_gt = np.array([(iou_arr > t).mean() for t in iou_thresholds], dtype=np.float64)
        auc_vis_gt = float(sc_vis_gt.mean())

        # 3) Official HengLan/LaSOT_Evaluation_Toolkit eval_tracker.m: strict >, denominator = total sequence frames
        denom = float(n_total)
        sc_mat = np.array([float((iou_arr > t).sum()) / denom for t in iou_thresholds], dtype=np.float64)
        auc_mat = float(sc_mat.mean())
        p20_mat = float((cle_arr <= 20.0).sum()) / denom
        pnorm_mat = float(np.mean([float((ncle_arr <= t).sum()) / denom for t in norm_thresholds]))

        seq_vis_ge_auc.append(auc_vis_ge)
        seq_vis_gt_auc.append(auc_vis_gt)
        seq_mat_auc.append(auc_mat)
        seq_vis_p20.append(p20_vis)
        seq_mat_p20.append(p20_mat)
        seq_vis_pnorm.append(pnorm_vis)
        seq_mat_pnorm.append(pnorm_mat)
        by_video[name] = {
            "total_frames": n_total,
            "visible_frames": len(ious),
            "auc": auc_vis_ge,
            "auc_vis_strict": auc_vis_gt,
            "auc_matlab_conf": auc_mat,
            "mean_iou": float(iou_arr.mean()),
            "p20": p20_vis,
            "p20_matlab_conf": p20_mat,
            "p_norm": pnorm_vis,
            "p_norm_matlab_conf": pnorm_mat,
        }

    return {
        "auc": float(np.mean(seq_vis_ge_auc)) if seq_vis_ge_auc else 0.0,
        "p_norm": float(np.mean(seq_vis_pnorm)) if seq_vis_pnorm else 0.0,
        "p20": float(np.mean(seq_vis_p20)) if seq_vis_p20 else 0.0,
        "vis_strict": {
            "auc": float(np.mean(seq_vis_gt_auc)) if seq_vis_gt_auc else 0.0,
            "p_norm": float(np.mean(seq_vis_pnorm)) if seq_vis_pnorm else 0.0,
            "p20": float(np.mean(seq_vis_p20)) if seq_vis_p20 else 0.0,
        },
        "matlab_conf": {
            "protocol": "HengLan/LaSOT_Evaluation_Toolkit eval_tracker.m (strict >, total-frame denominator)",
            "auc": float(np.mean(seq_mat_auc)) if seq_mat_auc else 0.0,
            "p_norm": float(np.mean(seq_mat_pnorm)) if seq_mat_pnorm else 0.0,
            "p20": float(np.mean(seq_mat_p20)) if seq_mat_p20 else 0.0,
        },
        "by_video": by_video,
    }


# =============================================================================
# Verifiers: Zero-External-Model SAM 2 Native vs. DINOv2 vs. None
# =============================================================================

class BaseVerifier:
    name = "base"

    def initialize(self, rgb_np, init_box, init_mask, feat_map, width, height) -> dict:
        raise NotImplementedError

    def score_candidate(self, rgb_np, box, binary_mask, feat_map, width, height) -> dict | None:
        raise NotImplementedError

    def verify_anchor(self, rgb_np, box, binary_mask, feat_map, width, height) -> bool:
        raise NotImplementedError

    def verify_probe(self, rgb_np, box, binary_mask, feat_map, width, height, obj_score, gamma, tau_obj_border) -> tuple[bool, dict]:
        raise NotImplementedError


class NoneVerifier(BaseVerifier):
    """Ablation verifier: disables appearance contrast checking completely."""
    name = "none"

    def initialize(self, rgb_np, init_box, init_mask, feat_map, width, height) -> dict:
        return {"verifier": self.name, "calibrated": True, "fg_floor": None, "fgc_floor": None, "neg_count": 0}

    def score_candidate(self, rgb_np, box, binary_mask, feat_map, width, height) -> dict | None:
        return {"foreground": None, "foreground_contrast": None}

    def verify_anchor(self, rgb_np, box, binary_mask, feat_map, width, height) -> bool:
        return True

    def verify_probe(self, rgb_np, box, binary_mask, feat_map, width, height, obj_score, gamma, tau_obj_border) -> tuple[bool, dict]:
        return True, {"fg": None, "fgc": None}


def _sample_native_descriptors(
    feat_norm: torch.Tensor,
    box: list[float],
    binary_mask: np.ndarray | None,
    width: int,
    height: int,
    precomputed_sil: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample L2-normalized (fg, bg) prototypes directly from cached SAM 2 FPN feature map.

    Args:
        feat_norm: [1, C, Hf, Wf] spatially centered & channel-L2-normalized feature map.
        box: [x, y, w, h] in original image pixel coordinates.
        binary_mask: [H, W] bool/uint8 binary mask (optional if precomputed_sil is provided).
        width, height: original image dimensions.
        precomputed_sil: optional [G*G] bool silhouette array (used for first-frame negative boxes).
    """
    x, y, w, h = [float(v) for v in box]
    if w <= 1e-3 or h <= 1e-3:
        return None, None

    # 2x surround box centered on [x, y, w, h]
    px = NATIVE_CENTERS[:, 0] * (2.0 * w) + (x - 0.5 * w)
    py = NATIVE_CENTERS[:, 1] * (2.0 * h) + (y - 0.5 * h)
    valid = (px >= 0.0) & (px < float(width)) & (py >= 0.0) & (py < float(height))

    if precomputed_sil is not None:
        fg_sel = valid & precomputed_sil
    else:
        if binary_mask is None:
            return None, None
        ix = np.clip(np.floor(px).astype(np.int64), 0, width - 1)
        iy = np.clip(np.floor(py).astype(np.int64), 0, height - 1)
        fg_sel = valid & NATIVE_INSIDE & (binary_mask[iy, ix] > 0)

    bg_sel = valid & (~fg_sel)
    if int(fg_sel.sum()) < NATIVE_MIN_FG_POINTS or int(bg_sel.sum()) < NATIVE_MIN_BG_POINTS:
        # Soft fallback for ultra-thin masks: use valid INSIDE points if binary mask has >=1 pixel
        if precomputed_sil is None and binary_mask is not None and int(fg_sel.sum()) >= 1 and int(bg_sel.sum()) >= NATIVE_MIN_BG_POINTS:
            pass
        else:
            return None, None

    # Convert pixel coords to grid_sample normalized [-1, 1] coordinates (align_corners=False)
    gx = ((px + 0.5) / float(width)) * 2.0 - 1.0
    gy = ((py + 0.5) / float(height)) * 2.0 - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32)).to(feat_norm.device)
    grid = grid.view(1, NATIVE_GRID_SIDE, NATIVE_GRID_SIDE, 2)

    sampled = F.grid_sample(
        feat_norm, grid, mode="bilinear", padding_mode="border", align_corners=False
    )  # [1, C, G, G]
    tokens = F.normalize(sampled[0].flatten(1).transpose(0, 1), dim=-1)  # [G*G, C]

    fg_idx = torch.from_numpy(fg_sel).to(tokens.device)
    bg_idx = torch.from_numpy(bg_sel).to(tokens.device)
    fg_vec = F.normalize(tokens[fg_idx].mean(dim=0), dim=0)
    bg_vec = F.normalize(tokens[bg_idx].mean(dim=0), dim=0)
    return fg_vec.detach().cpu().numpy(), bg_vec.detach().cpu().numpy()


def _prepare_native_feat_map(feat_map: torch.Tensor | None) -> torch.Tensor | None:
    if feat_map is None:
        return None
    f = feat_map.detach().float()
    if f.dim() == 3:
        f = f.unsqueeze(0)
    # Channel-wise L2 normalization (exact analogue to DINOv2 patch normalization,
    # without subtracting global spatial mean which would turn uniform background into -fg).
    return F.normalize(f, dim=1)


def native_silhouette(binary_mask: np.ndarray, box: list[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    h_img, w_img = binary_mask.shape[:2]
    px = NATIVE_CENTERS[:, 0] * (2.0 * w) + (x - 0.5 * w)
    py = NATIVE_CENTERS[:, 1] * (2.0 * h) + (y - 0.5 * h)
    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)
    valid = (ix >= 0) & (ix < w_img) & (iy >= 0) & (iy < h_img)
    res = np.zeros(len(NATIVE_CENTERS), dtype=bool)
    res[valid] = binary_mask[iy[valid], ix[valid]] > 0
    return res & NATIVE_INSIDE


class SAM2NativeVerifier(BaseVerifier):
    """Zero-external-model foreground-contrast verifier using SAM 2's cached FPN features."""

    name = "sam2_native"

    def __init__(self):
        self.ref_fg: np.ndarray | None = None
        self.ref_bg: np.ndarray | None = None
        self.fg_floor: float | None = None
        self.fgc_floor: float | None = None

    def initialize(self, rgb_np, init_box, init_mask, feat_map, width, height) -> dict:
        self.ref_fg = None
        self.ref_bg = None
        self.fg_floor = None
        self.fgc_floor = None

        feat_norm = _prepare_native_feat_map(feat_map)
        if feat_norm is None or init_mask is None:
            return {"verifier": self.name, "calibrated": False, "fg_floor": None, "fgc_floor": None, "neg_count": 0}

        mask_bool = np.asarray(init_mask) > 0
        sil0 = native_silhouette(mask_bool, init_box)
        if int(sil0.sum()) < NATIVE_MIN_FG_POINTS:
            sil0 = NATIVE_INSIDE.copy()

        ref_fg, ref_bg = _sample_native_descriptors(feat_norm, init_box, mask_bool, width, height, precomputed_sil=sil0)
        if ref_fg is None or ref_bg is None:
            return {"verifier": self.name, "calibrated": False, "fg_floor": None, "fgc_floor": None, "neg_count": 0}

        self.ref_fg = ref_fg
        self.ref_bg = ref_bg

        neg_boxes = global_boxes(init_box, width, height)
        if len(neg_boxes) < 4:
            neg_boxes = offset_boxes(init_box, SURROUND)

        fg_vals, fgc_vals = [], []
        for nb in neg_boxes:
            n_fg, n_bg = _sample_native_descriptors(feat_norm, nb, None, width, height, precomputed_sil=sil0)
            if n_fg is None or n_bg is None:
                continue
            fg_sim = float(ref_fg @ n_fg)
            fgc_sim = fg_sim - max(float(ref_fg @ n_bg), float(ref_bg @ n_fg))
            fg_vals.append(fg_sim)
            fgc_vals.append(fgc_sim)

        self.fg_floor = max(fg_vals) if fg_vals else None
        # A valid target must always exhibit strictly positive foreground-vs-background contrast.
        self.fgc_floor = max(0.0, max(fgc_vals)) if fgc_vals else None
        return {
            "verifier": self.name,
            "calibrated": self.fgc_floor is not None,
            "fg_floor": self.fg_floor,
            "fgc_floor": self.fgc_floor,
            "neg_count": len(fgc_vals),
        }

    def score_candidate(self, rgb_np, box, binary_mask, feat_map, width, height) -> dict | None:
        if self.ref_fg is None or self.ref_bg is None:
            return None
        feat_norm = _prepare_native_feat_map(feat_map)
        if feat_norm is None:
            return None
        fg, bg = _sample_native_descriptors(feat_norm, box, np.asarray(binary_mask) > 0, width, height)
        if fg is None or bg is None:
            return None
        fg_sim = float(self.ref_fg @ fg)
        fgc_sim = fg_sim - max(float(self.ref_fg @ bg), float(self.ref_bg @ fg))
        return {"foreground": fg_sim, "foreground_contrast": fgc_sim}

    def verify_anchor(self, rgb_np, box, binary_mask, feat_map, width, height) -> bool:
        if self.ref_fg is None or self.fgc_floor is None:
            return True
        sim = self.score_candidate(rgb_np, box, binary_mask, feat_map, width, height)
        if sim is None or sim["foreground_contrast"] is None:
            return False
        return sim["foreground_contrast"] > self.fgc_floor

    def verify_probe(self, rgb_np, box, binary_mask, feat_map, width, height, obj_score, gamma, tau_obj_border) -> tuple[bool, dict]:
        if self.ref_fg is None or self.fgc_floor is None:
            return True, {"fg": None, "fgc": None}
        sim = self.score_candidate(rgb_np, box, binary_mask, feat_map, width, height)
        if sim is None or sim["foreground"] is None or sim["foreground_contrast"] is None:
            return False, {}
        fg = sim["foreground"]
        fgc = sim["foreground_contrast"]
        pass_fgc = fgc > self.fgc_floor
        pass_border = (
            gamma < 1.0
            and obj_score >= tau_obj_border
            and self.fg_floor is not None
            and (fg / math.sqrt(gamma)) > self.fg_floor
        )
        return (pass_fgc or pass_border), {"fg": round(fg, 4), "fgc": round(fgc, 4)}


class _CloudHFTokenEncoder:
    """Drop-in Linux/Cloud fallback for TokenEncoder using facebook/dinov2-small when local F:/ cache is absent."""

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(model_id).eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        self.images_encoded = 0
        self.batch_calls = 0

    @torch.inference_mode()
    def tokens(self, crops):
        output = [None] * len(crops)
        valid = [i for i, item in enumerate(crops) if item is not None]
        for start in range(0, len(valid), 16):
            indices = valid[start:start + 16]
            value = torch.from_numpy(np.stack([crops[i][0] for i in indices])).to("cuda")
            value = (value.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            features = F.normalize(self.model(pixel_values=value).last_hidden_state, dim=-1).cpu().numpy()
            for i, item in zip(indices, features, strict=True):
                output[i] = item, crops[i][1]
            self.images_encoded += len(indices)
            self.batch_calls += 1
        return output


class DINOv2Verifier(BaseVerifier):
    """Frozen external DINOv2 ViT-S/14 verifier (retained strictly for fair ablation)."""

    name = "dinov2"

    def __init__(self, encoder=None):
        if encoder is not None:
            self.encoder = encoder
        else:
            from appearance_features import WEIGHTS as LOCAL_DINO_WEIGHTS
            if Path(LOCAL_DINO_WEIGHTS).exists():
                self.encoder = TokenEncoder()
            else:
                self.encoder = _CloudHFTokenEncoder(os.environ.get("CARE_DINOV2_MODEL", "facebook/dinov2-small"))
        self.target_desc = None
        self.fg_floor = None
        self.fgc_floor = None

    def initialize(self, rgb_np, init_box, init_mask, feat_map, width, height) -> dict:
        self.target_desc = None
        self.fg_floor = None
        self.fgc_floor = None
        sil0 = dino_silhouette(np.asarray(init_mask) > 0, init_box)
        neg_boxes = global_boxes(init_box, width, height)
        if len(neg_boxes) < 4:
            neg_boxes = offset_boxes(init_box, SURROUND)
        ref_tok, *neg_toks = self.encoder.tokens([crop(rgb_np, b) for b in (init_box, *neg_boxes)])
        if ref_tok is None:
            return {"verifier": self.name, "calibrated": False, "fg_floor": None, "fgc_floor": None, "neg_count": 0}
        target = dino_descriptors(ref_tok, sil0)
        if target[1] is None or target[2] is None:
            return {"verifier": self.name, "calibrated": False, "fg_floor": None, "fgc_floor": None, "neg_count": 0}
        self.target_desc = target
        fg_vals, fgc_vals = [], []
        for tok in neg_toks:
            if tok is None:
                continue
            sim = dino_similarity(target, dino_descriptors(tok, sil0))
            if sim["foreground"] is not None:
                fg_vals.append(float(sim["foreground"]))
            if sim["foreground_contrast"] is not None:
                fgc_vals.append(float(sim["foreground_contrast"]))
        self.fg_floor = max(fg_vals) if fg_vals else None
        # A valid target must always exhibit strictly positive foreground-vs-background contrast.
        self.fgc_floor = max(0.0, max(fgc_vals)) if fgc_vals else None
        return {
            "verifier": self.name,
            "calibrated": self.fgc_floor is not None,
            "fg_floor": self.fg_floor,
            "fgc_floor": self.fgc_floor,
            "neg_count": len(fgc_vals),
        }

    def score_candidate(self, rgb_np, box, binary_mask, feat_map, width, height) -> dict | None:
        if self.target_desc is None:
            return None
        tok = self.encoder.tokens([crop(rgb_np, box)])[0]
        if tok is None:
            return None
        return dino_similarity(self.target_desc, dino_descriptors(tok, dino_silhouette(np.asarray(binary_mask) > 0, box)))

    def verify_anchor(self, rgb_np, box, binary_mask, feat_map, width, height) -> bool:
        if self.target_desc is None or self.fgc_floor is None:
            return True
        sim = self.score_candidate(rgb_np, box, binary_mask, feat_map, width, height)
        if sim is None or sim["foreground_contrast"] is None:
            return False
        return sim["foreground_contrast"] > self.fgc_floor

    def verify_probe(self, rgb_np, box, binary_mask, feat_map, width, height, obj_score, gamma, tau_obj_border) -> tuple[bool, dict]:
        if self.target_desc is None or self.fgc_floor is None:
            return True, {"fg": None, "fgc": None}
        sim = self.score_candidate(rgb_np, box, binary_mask, feat_map, width, height)
        if sim is None or sim["foreground"] is None or sim["foreground_contrast"] is None:
            return False, {}
        fg = sim["foreground"]
        fgc = sim["foreground_contrast"]
        pass_fgc = fgc > self.fgc_floor
        pass_border = (
            gamma < 1.0
            and obj_score >= tau_obj_border
            and self.fg_floor is not None
            and (fg / math.sqrt(gamma)) > self.fg_floor
        )
        return (pass_fgc or pass_border), {"fg": round(fg, 4), "fgc": round(fgc, 4)}


def build_verifier(kind: str) -> BaseVerifier:
    if kind == "sam2_native":
        return SAM2NativeVerifier()
    if kind == "dinov2":
        return DINOv2Verifier()
    if kind == "none":
        return NoneVerifier()
    raise ValueError(f"Unknown verifier: {kind}")


# =============================================================================
# Host-Agnostic Runtime Adapters: SAM 2.1, SAMURAI, DAM4SAM
# =============================================================================

class BaseHostAdapter:
    host_name = "base"

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.frame_idx = -1
        self.state = None
        self.predictor = None
        self.provenance: dict = {}

    def configure_arm(self, arm: str, care_params: dict):
        pass

    def initialize(self, pil_rgb: Image.Image, init_box: list[float]) -> dict:
        raise NotImplementedError

    def track(self, pil_rgb: Image.Image) -> dict:
        raise NotImplementedError

    def get_cached_feature_map(self) -> torch.Tensor | None:
        if self.state is None:
            return None
        cached = self.state.get("cached_features", {}).get(self.frame_idx)
        if cached is None:
            return None
        _, backbone_out = cached
        fpn = backbone_out.get("backbone_fpn")
        if not fpn:
            return None
        return fpn[-1]

    def get_non_cond_slot(self, frame_idx: int) -> dict | None:
        raise NotImplementedError

    def dual_anchor_probe(self, last_clean_out: dict | None, use_dual_anchor: bool = True) -> dict:
        raise NotImplementedError

    def commit_reentry(self, probe: dict, purge_window: int) -> dict:
        raise NotImplementedError

    def rollback_reentry(self, inj_frame: int, snapshot: dict) -> tuple[list[float] | None, float | None]:
        raise NotImplementedError

    def prune_old_memory(self, keep_window: int):
        raise NotImplementedError

    def release_clip(self):
        self.state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SAM21HostAdapter(BaseHostAdapter):
    """Adapter for vanilla SAM 2.1 (Tiny / Base+ / Large) via SENTRY SAM2Runtime or native sam2."""

    host_name = "sam21"

    def __init__(
        self,
        weights: Path,
        backbone: str = "tiny",
        sam2_root: Path | None = None,
        deps_dirs: list[Path] | None = None,
        use_sentry_runtime: bool = False,
        sentry_source: Path | None = None,
        device: str = "cuda:0",
    ):
        super().__init__(device=device)
        self.backbone = backbone
        self.weights = Path(weights)
        self.use_sentry_runtime = bool(use_sentry_runtime)
        self.runtime = None
        self.image_size = 1024
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(self.device)

        extra = []
        if use_sentry_runtime and sentry_source is not None:
            extra.append(str(sentry_source / "src"))
        if sam2_root is not None:
            extra.append(str(sam2_root))
        if deps_dirs:
            extra.extend(str(d) for d in deps_dirs if d is not None and Path(d).exists())
        for p in reversed(extra):
            if p not in sys.path:
                sys.path.insert(0, p)

        if self.use_sentry_runtime:
            import sentry_tracking.backends.sam2_runtime as s_rt_mod
            from sentry_tracking.backends.sam2_runtime import SAM2Runtime
            tmap = {"tiny": "sam21-T", "base_plus": "sam21-B", "large": "sam21-L"}
            self.runtime = SAM2Runtime(tracker_name=tmap[backbone], checkpoint=self.weights, device=device)
            self.predictor = self.runtime.predictor
            import sam2 as sam2_pkg
            self.provenance = {
                "adapter": "SAM21HostAdapter(sentry_runtime)",
                "host_source_root": str(sentry_source.resolve()) if sentry_source is not None else None,
                "host_git_commit": _git_head_sha(sentry_source),
                "host_module_path": str(Path(s_rt_mod.__file__).resolve()),
                "sam2_module_path": str(Path(sam2_pkg.__file__).resolve()),
                "config_name": tmap[backbone],
                "config_path": None,
                "config_sha256": None,
            }
        else:
            import sam2 as sam2_pkg
            from sam2.build_sam import build_sam2_video_predictor
            pkg_dir = Path(sam2_pkg.__file__).resolve().parent
            cfg, cfg_path = _resolve_sam2_config(pkg_dir, BACKBONE_CONFIGS["sam21"][backbone])
            pred_mod = importlib.import_module("sam2.sam2_video_predictor")
            pred_mod.tqdm = lambda it, **_: it
            self.predictor = build_sam2_video_predictor(cfg, str(self.weights), device=device)
            if hasattr(self.predictor, "samurai_mode"):
                self.predictor.samurai_mode = False
            if hasattr(self.predictor, "min_obj_score_logits"):
                self.predictor.min_obj_score_logits = 0.0
            sam2_mod_path = str(Path(sam2_pkg.__file__).resolve())
            if "samurai" in sam2_mod_path.lower():
                print(
                    f"WARNING: --host sam21 imported sam2 from a SAMURAI tree ({sam2_mod_path}). "
                    "Use Meta's official facebookresearch/sam2 tree for strict vanilla SAM 2.1.",
                    flush=True,
                )
            self.provenance = {
                "adapter": "SAM21HostAdapter(native_sam2)",
                "host_source_root": str(sam2_root.resolve()) if sam2_root is not None else None,
                "host_git_commit": _git_head_sha(sam2_root),
                "host_module_path": sam2_mod_path,
                "sam2_module_path": sam2_mod_path,
                "config_name": cfg,
                "config_path": str(cfg_path.resolve()) if cfg_path is not None else None,
                "config_sha256": digest(cfg_path) if cfg_path is not None else None,
                "has_samurai_mode_attr": hasattr(self.predictor, "samurai_mode"),
            }

    def _new_state(self, width: int, height: int) -> dict:
        return {
            "images": {},
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": self.device,
            "storage_device": self.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "frames_tracked_per_obj": {},
        }

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        tensor = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).to(self.device)
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        tensor = vision_f.resize(tensor, (self.image_size, self.image_size))
        return (tensor - self.mean) / self.std

    def _sync_both_dicts(self, frame_idx: int, entry: dict):
        obj_idx = self.state["obj_id_to_idx"][0]
        self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][frame_idx] = entry
        if "output_dict" in self.state and "non_cond_frame_outputs" in self.state["output_dict"]:
            self.state["output_dict"]["non_cond_frame_outputs"][frame_idx] = entry

    def _pop_both_dicts(self, frame_idx: int):
        obj_idx = self.state["obj_id_to_idx"][0]
        self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
        if "output_dict" in self.state and "non_cond_frame_outputs" in self.state["output_dict"]:
            self.state["output_dict"]["non_cond_frame_outputs"].pop(frame_idx, None)

    def initialize(self, pil_rgb: Image.Image, init_box: list[float]) -> dict:
        if self.use_sentry_runtime:
            res = self.runtime.initialize(pil_rgb, init_box)
            self.state = self.runtime.state
            self.frame_idx = 0
            return {"pred_mask": res["pred_mask"], "pred_bbox": res["pred_bbox"], "object_score": None}

        self.frame_idx = 0
        width, height = pil_rgb.size
        self.state = self._new_state(width, height)
        self.state["images"][0] = self._prepare_image(pil_rgb)
        self.predictor.reset_state(self.state)
        self.predictor._get_image_feature(self.state, frame_idx=0, batch_size=1)
        x, y, w, h = init_box
        _, _, logits = self.predictor.add_new_points_or_box(
            self.state, frame_idx=0, obj_id=0, box=[x, y, x + w, y + h]
        )
        self.predictor.propagate_in_video_preflight(self.state)
        pred_mask = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
        self.state["images"].pop(0, None)
        return {"pred_mask": pred_mask, "pred_bbox": mask_to_bbox(pred_mask), "object_score": None}

    def track(self, pil_rgb: Image.Image) -> dict:
        if self.use_sentry_runtime:
            res = self.runtime.track(pil_rgb)
            self.state = self.runtime.state
            self.frame_idx = self.runtime.frame_idx
            return {
                "pred_mask": res["pred_mask"],
                "pred_bbox": res["pred_bbox"],
                "object_score": _scalar(res.get("object_score_logits")),
            }

        self.frame_idx += 1
        self.state["num_frames"] = self.frame_idx + 1
        self.state["images"][self.frame_idx] = self._prepare_image(pil_rgb)
        for _, _, mask_logits in self.predictor.propagate_in_video(
            self.state, start_frame_idx=self.frame_idx, max_frame_num_to_track=0
        ):
            mask = (mask_logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
            slot = self.get_non_cond_slot(self.frame_idx)
            obj_score = _scalar(slot.get("object_score_logits")) if slot is not None else None
            self.state["images"].pop(self.frame_idx, None)
            return {
                "pred_mask": mask,
                "pred_bbox": mask_to_bbox(mask),
                "object_score": obj_score,
            }
        raise RuntimeError(f"SAM 2.1 propagation yielded no output at frame {self.frame_idx}")

    def get_non_cond_slot(self, frame_idx: int) -> dict | None:
        obj_idx = self.state["obj_id_to_idx"][0]
        slot = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx)
        if slot is None and "output_dict" in self.state:
            slot = self.state["output_dict"]["non_cond_frame_outputs"].get(frame_idx)
        return slot

    def dual_anchor_probe(self, last_clean_out: dict | None, use_dual_anchor: bool = True) -> dict:
        obj_idx = self.state["obj_id_to_idx"][0]
        cond_outputs = self.state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]
        cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        dual_dict = {
            "cond_frame_outputs": cond_0,
            "non_cond_frame_outputs": {self.frame_idx - 1: last_clean_out} if (use_dual_anchor and last_clean_out is not None) else {},
        }
        cur_out = self.predictor.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=dual_dict,
            num_frames=self.frame_idx + 1,
            run_mem_encoder=False,
        )
        _, video_mask = self.predictor._get_orig_video_res_output(self.state, cur_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        obj_score = _scalar(cur_out.get("object_score_logits"))
        ious = cur_out.get("multimask_ious")
        if ious is not None:
            quality = float(ious.detach().float().max().item())
        elif cur_out.get("best_iou_score") is not None:
            quality = float(_scalar(cur_out.get("best_iou_score")) or 0.0)
        else:
            quality = 1.0 if obj_score is not None and obj_score > 0 else 0.0
        return {
            "cur_out": cur_out,
            "mask": binary,
            "box": mask_to_bbox(binary),
            "obj_score": obj_score,
            "quality": quality,
        }

    def commit_reentry(self, probe: dict, purge_window: int) -> dict:
        cur_out = probe["cur_out"]
        obj_idx = self.state["obj_id_to_idx"][0]
        non_cond = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        current = self.get_non_cond_slot(self.frame_idx)
        if current is None:
            raise RuntimeError(f"Missing non_cond output at frame {self.frame_idx}")

        high_res_masks = cur_out["pred_masks_high_res"].to(self.device)
        obj_logits = cur_out["object_score_logits"].to(self.device)
        maskmem_features, maskmem_pos_enc = self.predictor._run_memory_encoder(
            self.state, self.frame_idx, 1, high_res_masks, obj_logits, is_mask_from_pts=False
        )

        saved_orig = _clone_mem_entry(current)
        saved_evicted = {}
        evicted = 0
        if purge_window > 0:
            lo = max(1, self.frame_idx - purge_window)
            for k in range(lo, self.frame_idx):
                slot = non_cond.get(k)
                if slot is not None:
                    slot_obj = _scalar(slot.get("object_score_logits"))
                    if slot_obj is None or slot_obj <= 0.0:
                        saved_evicted[k] = slot
                        self._pop_both_dicts(k)
                        evicted += 1

        storage = self.state["storage_device"]
        updated = dict(current)
        updated["pred_masks"] = cur_out["pred_masks"].to(storage, dtype=current["pred_masks"].dtype)
        updated["obj_ptr"] = cur_out["obj_ptr"].to(storage, dtype=current["obj_ptr"].dtype)
        updated["maskmem_features"] = maskmem_features.to(storage)
        updated["maskmem_pos_enc"] = [p.to(storage) for p in maskmem_pos_enc]
        updated["object_score_logits"] = obj_logits.to(storage)
        self._sync_both_dicts(self.frame_idx, updated)
        return {
            "applied": True,
            "frame_idx": self.frame_idx,
            "evicted_no_obj": evicted,
            "saved_orig": saved_orig,
            "saved_evicted": saved_evicted,
        }

    def rollback_reentry(self, inj_frame: int, snapshot: dict) -> tuple[list[float] | None, float | None]:
        obj_idx = self.state["obj_id_to_idx"][0]
        obj_dict = self.state["output_dict_per_obj"][obj_idx]
        self._sync_both_dicts(inj_frame, snapshot["saved_orig"])
        for k, v in snapshot["saved_evicted"].items():
            self._sync_both_dicts(k, v)

        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        restored_out = self.predictor.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=obj_dict,
            num_frames=self.frame_idx + 1,
            run_mem_encoder=True,
        )
        self._sync_both_dicts(self.frame_idx, restored_out)
        _, video_mask = self.predictor._get_orig_video_res_output(self.state, restored_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        return mask_to_bbox(binary), _scalar(restored_out.get("object_score_logits"))

    def prune_old_memory(self, keep_window: int):
        if keep_window <= 0 or self.state is None:
            return
        obj_idx = self.state["obj_id_to_idx"].get(0)
        if obj_idx is None:
            return
        cutoff = self.frame_idx - keep_window
        non_cond = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        for k in [idx for idx in non_cond if idx < cutoff]:
            self._pop_both_dicts(k)

    def release_clip(self):
        if self.runtime is not None:
            self.runtime.state = None
        super().release_clip()


class SAMURAIHostAdapter(BaseHostAdapter):
    """Adapter for SAMURAI (IEEE TIP 2026) with Kalman Filter + Motion-Aware Memory."""

    host_name = "samurai"

    def __init__(
        self,
        weights: Path,
        backbone: str = "tiny",
        samurai_root: Path | None = None,
        deps_dirs: list[Path] | None = None,
        device: str = "cuda:0",
    ):
        super().__init__(device=device)
        self.backbone = backbone
        self.weights = Path(weights)
        self.image_size = 1024
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(self.device)

        extra = []
        if samurai_root is not None:
            extra.append(str(samurai_root))
        if deps_dirs:
            extra.extend(str(d) for d in deps_dirs if d is not None and Path(d).exists())
        for p in reversed(extra):
            if p not in sys.path:
                sys.path.insert(0, p)

        import sam2 as sam2_pkg
        from sam2.build_sam import build_sam2_video_predictor
        pred_mod = importlib.import_module("sam2.sam2_video_predictor")
        pred_mod.tqdm = lambda it, **_: it

        pkg_dir = Path(sam2_pkg.__file__).resolve().parent
        cfg, cfg_path = _resolve_sam2_config(pkg_dir, BACKBONE_CONFIGS["samurai"][backbone])
        self.predictor = build_sam2_video_predictor(cfg, str(self.weights), device=device)
        self._stock_min_obj = float(self.predictor.min_obj_score_logits)
        self.provenance = {
            "adapter": "SAMURAIHostAdapter",
            "host_source_root": str(samurai_root.resolve()) if samurai_root is not None else None,
            "host_git_commit": _git_head_sha(samurai_root),
            "host_module_path": str(Path(sam2_pkg.__file__).resolve()),
            "sam2_module_path": str(Path(sam2_pkg.__file__).resolve()),
            "config_name": cfg,
            "config_path": str(cfg_path.resolve()) if cfg_path is not None else None,
            "config_sha256": digest(cfg_path) if cfg_path is not None else None,
            "stock_min_obj_score_logits": self._stock_min_obj,
        }

    def configure_arm(self, arm: str, care_params: dict):
        if arm == "baseline":
            val = care_params.get("samurai_baseline_min_obj")
            self.predictor.min_obj_score_logits = self._stock_min_obj if val is None else float(val)
        else:
            val = care_params.get("samurai_care_min_obj")
            # Default matches stock threshold (-1.0) so baseline and care are strictly threshold-matched
            # unless the user explicitly specifies a 2x2 factorial threshold override.
            self.predictor.min_obj_score_logits = self._stock_min_obj if val is None else float(val)

    def _reset_kf(self):
        self.predictor.kf_mean = None
        self.predictor.kf_covariance = None
        self.predictor.stable_frames = 0
        self.predictor.frame_cnt = 0

    def _snapshot_kf(self) -> dict:
        return {
            "kf_mean": None if self.predictor.kf_mean is None else self.predictor.kf_mean.copy(),
            "kf_covariance": None if self.predictor.kf_covariance is None else self.predictor.kf_covariance.copy(),
            "stable_frames": int(self.predictor.stable_frames),
            "frame_cnt": int(self.predictor.frame_cnt),
        }

    def _restore_kf(self, snap: dict):
        self.predictor.kf_mean = None if snap["kf_mean"] is None else snap["kf_mean"].copy()
        self.predictor.kf_covariance = None if snap["kf_covariance"] is None else snap["kf_covariance"].copy()
        self.predictor.stable_frames = int(snap["stable_frames"])
        self.predictor.frame_cnt = int(snap["frame_cnt"])

    def _new_state(self, width: int, height: int) -> dict:
        return {
            "images": {},
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": self.device,
            "storage_device": self.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB").resize((self.image_size, self.image_size)))
        tensor = torch.from_numpy(array / 255.0).float().permute(2, 0, 1).to(self.device)
        return (tensor - self.mean) / self.std

    def _set_both_non_cond(self, frame_idx: int, entry: dict):
        obj_idx = self.state["obj_id_to_idx"][0]
        self.state["output_dict"]["non_cond_frame_outputs"][frame_idx] = entry
        self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][frame_idx] = entry

    def initialize(self, pil_rgb: Image.Image, init_box: list[float]) -> dict:
        self._reset_kf()
        self.frame_idx = 0
        width, height = pil_rgb.size
        self.state = self._new_state(width, height)
        self.state["images"][0] = self._prepare_image(pil_rgb)
        self.predictor.reset_state(self.state)
        self.predictor._get_image_feature(self.state, frame_idx=0, batch_size=1)
        x, y, w, h = init_box
        _, _, logits = self.predictor.add_new_points_or_box(
            self.state, frame_idx=0, obj_id=0, box=[x, y, x + w, y + h]
        )
        self.predictor.propagate_in_video_preflight(self.state)
        pred_mask = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
        self.state["images"].pop(0, None)
        return {"pred_mask": pred_mask, "pred_bbox": mask_to_bbox(pred_mask), "object_score": None}

    def track(self, pil_rgb: Image.Image) -> dict:
        self.frame_idx += 1
        self.state["num_frames"] = self.frame_idx + 1
        self.state["images"][self.frame_idx] = self._prepare_image(pil_rgb)
        for _, _, mask_logits in self.predictor.propagate_in_video(
            self.state, start_frame_idx=self.frame_idx, max_frame_num_to_track=0
        ):
            mask = (mask_logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
            slot = self.get_non_cond_slot(self.frame_idx)
            obj_score = _scalar(slot.get("object_score_logits")) if slot is not None else None
            self.state["images"].pop(self.frame_idx, None)
            return {
                "pred_mask": mask,
                "pred_bbox": mask_to_bbox(mask),
                "object_score": obj_score,
            }
        raise RuntimeError(f"SAMURAI propagation yielded no output at frame {self.frame_idx}")

    def get_non_cond_slot(self, frame_idx: int) -> dict | None:
        return self.state["output_dict"]["non_cond_frame_outputs"].get(frame_idx)

    def dual_anchor_probe(self, last_clean_out: dict | None, use_dual_anchor: bool = True) -> dict:
        cond_outputs = self.state["output_dict"]["cond_frame_outputs"]
        cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        dual_dict = {
            "cond_frame_outputs": cond_0,
            "non_cond_frame_outputs": {self.frame_idx - 1: last_clean_out} if (use_dual_anchor and last_clean_out is not None) else {},
        }
        # Counterfactual probe MUST NOT mutate SAMURAI's Kalman filter or trigger full-history scan
        saved_mode = self.predictor.samurai_mode
        saved_kf = self._snapshot_kf()
        try:
            self.predictor.samurai_mode = False
            cur_out = self.predictor.track_step(
                frame_idx=self.frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=dual_dict,
                num_frames=self.frame_idx + 1,
                run_mem_encoder=False,
            )
        finally:
            self.predictor.samurai_mode = saved_mode
            self._restore_kf(saved_kf)

        _, video_mask = self.predictor._get_orig_video_res_output(self.state, cur_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        obj_score = _scalar(cur_out.get("object_score_logits"))
        quality = float(_scalar(cur_out.get("best_iou_score")) or 0.0)
        return {
            "cur_out": cur_out,
            "mask": binary,
            "box": mask_to_bbox(binary),
            "obj_score": obj_score,
            "quality": quality,
        }

    def commit_reentry(self, probe: dict, purge_window: int) -> dict:
        cur_out = probe["cur_out"]
        non_cond = self.state["output_dict"]["non_cond_frame_outputs"]
        current = non_cond.get(self.frame_idx)
        if current is None:
            raise RuntimeError(f"Missing SAMURAI non_cond output at frame {self.frame_idx}")

        high_res_masks = cur_out["pred_masks_high_res"].to(self.device)
        obj_logits = cur_out["object_score_logits"].to(self.device)
        maskmem_features, maskmem_pos_enc = self.predictor._run_memory_encoder(
            self.state, self.frame_idx, 1, high_res_masks, obj_logits, is_mask_from_pts=False
        )

        saved_orig = _clone_mem_entry(current)
        saved_kf = self._snapshot_kf()
        saved_evicted = {}
        evicted = 0

        # For SAMURAI, we MUST preserve integer keys 2..frame_idx-1 (SAMURAI indexes non_cond[i] directly).
        # To purge no_obj pointer inertia without KeyError, replace silent slot obj_ptr with cur_out["obj_ptr"]
        # while keeping best_iou_score=0 / object_score_logits=-10 so SAMURAI's spatial selector skips them.
        if purge_window > 0:
            lo = max(1, self.frame_idx - purge_window)
            zero_iou = torch.tensor(0.0, device=self.device)
            neg_obj = torch.tensor([[-10.0]], device=self.device)
            for k in range(lo, self.frame_idx):
                slot = non_cond.get(k)
                if slot is not None:
                    slot_obj = _scalar(slot.get("object_score_logits"))
                    if slot_obj is None or slot_obj <= 0.0:
                        saved_evicted[k] = _clone_mem_entry(slot)
                        purged = dict(slot)
                        purged["obj_ptr"] = cur_out["obj_ptr"].to(slot["obj_ptr"].device, dtype=slot["obj_ptr"].dtype)
                        purged["best_iou_score"] = zero_iou
                        purged["object_score_logits"] = neg_obj
                        purged["kf_score"] = zero_iou
                        self._set_both_non_cond(k, purged)
                        evicted += 1

        storage = self.state["storage_device"]
        updated = dict(current)
        updated["pred_masks"] = cur_out["pred_masks"].to(storage, dtype=current["pred_masks"].dtype)
        updated["obj_ptr"] = cur_out["obj_ptr"].to(storage, dtype=current["obj_ptr"].dtype)
        updated["maskmem_features"] = maskmem_features.to(storage, dtype=torch.bfloat16) if maskmem_features is not None else None
        updated["maskmem_pos_enc"] = [p.to(storage) for p in maskmem_pos_enc]
        updated["object_score_logits"] = obj_logits.to(storage)
        updated["best_iou_score"] = torch.tensor(float(probe["quality"]), device=self.device)
        updated["kf_score"] = torch.tensor(1.0, device=self.device)
        self._set_both_non_cond(self.frame_idx, updated)

        # Cleanly re-lock SAMURAI's Kalman Filter onto the verified re-entry mask in 1024x1024 space
        nz = torch.argwhere(high_res_masks[0, 0] > 0.0)
        if len(nz) > 0:
            y_min, x_min = nz.min(dim=0).values
            y_max, x_max = nz.max(dim=0).values
            kf_box = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
            self.predictor.kf_mean, self.predictor.kf_covariance = self.predictor.kf.initiate(
                self.predictor.kf.xyxy_to_xyah(kf_box)
            )
            self.predictor.stable_frames = 1

        return {
            "applied": True,
            "frame_idx": self.frame_idx,
            "evicted_no_obj": evicted,
            "saved_orig": saved_orig,
            "saved_evicted": saved_evicted,
            "saved_kf": saved_kf,
        }

    def rollback_reentry(self, inj_frame: int, snapshot: dict) -> tuple[list[float] | None, float | None]:
        self._set_both_non_cond(inj_frame, snapshot["saved_orig"])
        for k, v in snapshot["saved_evicted"].items():
            self._set_both_non_cond(k, v)
        self._restore_kf(snapshot["saved_kf"])

        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        restored_out = self.predictor.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=self.state["output_dict"],
            num_frames=self.frame_idx + 1,
            run_mem_encoder=True,
        )
        storage = self.state["storage_device"]
        compact = {
            "maskmem_features": restored_out["maskmem_features"].to(torch.bfloat16).to(storage) if restored_out["maskmem_features"] is not None else None,
            "maskmem_pos_enc": self.predictor._get_maskmem_pos_enc(self.state, restored_out),
            "pred_masks": restored_out["pred_masks"].to(storage),
            "obj_ptr": restored_out["obj_ptr"],
            "object_score_logits": restored_out["object_score_logits"],
            "best_iou_score": restored_out["best_iou_score"],
            "kf_score": restored_out["kf_ious"],
        }
        self._set_both_non_cond(self.frame_idx, compact)
        _, video_mask = self.predictor._get_orig_video_res_output(self.state, restored_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        return mask_to_bbox(binary), _scalar(restored_out.get("object_score_logits"))

    def prune_old_memory(self, keep_window: int):
        """Cap GPU VRAM on long videos while preserving SAMURAI's backwards-scan invariant."""
        if keep_window <= 0 or self.state is None:
            return
        non_cond = self.state["output_dict"]["non_cond_frame_outputs"]
        cutoff = self.frame_idx - keep_window
        if cutoff <= 2:
            return

        # Keep up to max_obj_ptrs_in_encoder valid past memory frames intact for SAMURAI's valid_indices
        protected = set()
        valid_found = 0
        for i in range(self.frame_idx - 1, 1, -1):
            slot = non_cond.get(i)
            if slot is None:
                continue
            iou_s = _scalar(slot.get("best_iou_score")) or 0.0
            obj_s = _scalar(slot.get("object_score_logits")) or -10.0
            kf_s = _scalar(slot.get("kf_score"))
            if (
                iou_s > self.predictor.memory_bank_iou_threshold
                and obj_s > self.predictor.memory_bank_obj_score_threshold
                and (kf_s is None or kf_s > self.predictor.memory_bank_kf_score_threshold)
            ):
                protected.add(i)
                valid_found += 1
                if valid_found >= self.predictor.max_obj_ptrs_in_encoder:
                    break

        zero_t = torch.tensor(0.0, device=self.device)
        neg_t = torch.tensor([[-10.0]], device=self.device)
        for k in list(non_cond.keys()):
            if k < cutoff and k not in protected:
                slot = non_cond[k]
                if slot.get("maskmem_features") is not None:
                    stub = {
                        "maskmem_features": None,
                        "maskmem_pos_enc": None,
                        "pred_masks": None,
                        "obj_ptr": None,
                        "object_score_logits": neg_t,
                        "best_iou_score": zero_t,
                        "kf_score": zero_t,
                    }
                    self._set_both_non_cond(k, stub)


class DAM4SAMHostAdapter(BaseHostAdapter):
    """Adapter for DAM4SAM (IJCV 2026) with RAM + Distractor-Resolving Memory (DRM)."""

    host_name = "dam4sam"

    def __init__(
        self,
        weights: Path,
        backbone: str = "tiny",
        dam4sam_root: Path | None = None,
        deps_dirs: list[Path] | None = None,
        device: str = "cuda:0",
    ):
        super().__init__(device=device)
        self.backbone = backbone
        self.weights = Path(weights)
        extra = []
        if dam4sam_root is not None:
            extra.append(str(dam4sam_root))
        if deps_dirs:
            extra.extend(str(d) for d in deps_dirs if d is not None and Path(d).exists())
        for p in reversed(extra):
            if p not in sys.path:
                sys.path.insert(0, p)

        import dam4sam_tracker as implementation
        cfg_name, tracker_name = BACKBONE_CONFIGS["dam4sam"][backbone]
        orig_det = implementation.determine_tracker
        implementation.determine_tracker = lambda _: (str(self.weights), cfg_name)
        try:
            self.model = implementation.DAM4SAMTracker(tracker_name)
        finally:
            implementation.determine_tracker = orig_det
        self.predictor = self.model.predictor
        pred_mod = importlib.import_module("sam2.sam2_video_predictor")
        pred_mod.tqdm = lambda it, **_: it
        import sam2 as sam2_pkg
        pkg_dir = Path(sam2_pkg.__file__).resolve().parent
        _, cfg_path = _resolve_sam2_config(pkg_dir, (cfg_name,))
        self.provenance = {
            "adapter": "DAM4SAMHostAdapter",
            "host_source_root": str(dam4sam_root.resolve()) if dam4sam_root is not None else None,
            "host_git_commit": _git_head_sha(dam4sam_root),
            "host_module_path": str(Path(implementation.__file__).resolve()),
            "sam2_module_path": str(Path(sam2_pkg.__file__).resolve()),
            "config_name": cfg_name,
            "config_path": str(cfg_path.resolve()) if cfg_path is not None else None,
            "config_sha256": digest(cfg_path) if cfg_path is not None else None,
        }

    def initialize(self, pil_rgb: Image.Image, init_box: list[float]) -> dict:
        if hasattr(self.model, "inference_state"):
            del self.model.inference_state
        self.predictor.curr_out = None
        x, y, w, h = init_box
        # DAM4SAMTracker.estimate_mask_from_box expects [x, y, w, h] (computes x+w, y+h internally)
        res = self.model.initialize(pil_rgb, None, bbox=[x, y, w, h])
        self.state = self.model.inference_state
        self.frame_idx = 0
        mask = (np.asarray(res["pred_mask"]) > 0).astype(np.uint8)
        return {"pred_mask": mask, "pred_bbox": mask_to_bbox(mask), "object_score": None}

    def track(self, pil_rgb: Image.Image) -> dict:
        res = self.model.track(pil_rgb)
        self.state = self.model.inference_state
        self.frame_idx = self.model.frame_index
        mask = (np.asarray(res["pred_mask"]) > 0).astype(np.uint8)
        slot = self.get_non_cond_slot(self.frame_idx)
        obj_score = _scalar(slot.get("object_score_logits")) if slot is not None else None
        return {"pred_mask": mask, "pred_bbox": mask_to_bbox(mask), "object_score": obj_score}

    def get_non_cond_slot(self, frame_idx: int) -> dict | None:
        obj_idx = self.state["obj_id_to_idx"][0]
        slot = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx)
        if slot is None:
            slot = self.state["output_dict"]["non_cond_frame_outputs"].get(frame_idx)
        return slot

    def _set_both_non_cond(self, frame_idx: int, entry: dict):
        obj_idx = self.state["obj_id_to_idx"][0]
        self.state["output_dict"]["non_cond_frame_outputs"][frame_idx] = entry
        self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][frame_idx] = entry

    def dual_anchor_probe(self, last_clean_out: dict | None, use_dual_anchor: bool = True) -> dict:
        obj_idx = self.state["obj_id_to_idx"][0]
        cond_outputs = self.state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]
        # Use pure frame-0 conditioning anchor (immune to any drifted DRM entries!)
        cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        clean_slot = None
        if use_dual_anchor and last_clean_out is not None:
            clean_slot = dict(last_clean_out)
            clean_slot["n_pixels_pos"] = max(1, int(clean_slot.get("n_pixels_pos", 1)))
        dual_dict = {
            "cond_frame_outputs": cond_0,
            "non_cond_frame_outputs": {self.frame_idx - 1: clean_slot} if clean_slot is not None else {},
        }
        cur_out = self.predictor.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=dual_dict,
            num_frames=self.frame_idx + 1,
            run_mem_encoder=False,
        )
        _, video_mask = self.predictor._get_orig_video_res_output(self.state, cur_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        obj_score = _scalar(cur_out.get("object_score_logits"))
        # DAM4SAM track_step does not return multimask_ious directly; use object_score sigmoid or iou
        quality = _sigmoid(obj_score) if obj_score is not None else 0.0
        return {
            "cur_out": cur_out,
            "mask": binary,
            "box": mask_to_bbox(binary),
            "obj_score": obj_score,
            "quality": quality,
        }

    def commit_reentry(self, probe: dict, purge_window: int) -> dict:
        cur_out = probe["cur_out"]
        obj_idx = self.state["obj_id_to_idx"][0]
        non_cond = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        current = self.get_non_cond_slot(self.frame_idx)
        if current is None:
            raise RuntimeError(f"Missing DAM4SAM non_cond output at frame {self.frame_idx}")

        high_res_masks = cur_out["pred_masks_high_res"].to(self.device)
        obj_logits = cur_out["object_score_logits"].to(self.device)
        maskmem_features, maskmem_pos_enc = self.predictor._run_memory_encoder(
            self.state, self.frame_idx, 1, high_res_masks, obj_logits, is_mask_from_pts=False
        )

        saved_orig = _clone_mem_entry(current)
        saved_evicted = {}
        evicted = 0
        if purge_window > 0:
            lo = max(1, self.frame_idx - purge_window)
            for k in range(lo, self.frame_idx):
                slot = non_cond.get(k)
                if slot is not None:
                    slot_obj = _scalar(slot.get("object_score_logits"))
                    if slot_obj is None or slot_obj <= 0.0 or int(slot.get("n_pixels_pos", 0)) < 1:
                        saved_evicted[k] = _clone_mem_entry(slot)
                        purged = dict(slot)
                        purged["n_pixels_pos"] = 0
                        self._set_both_non_cond(k, purged)
                        evicted += 1

        storage = self.state["storage_device"]
        updated = dict(current)
        updated["pred_masks"] = cur_out["pred_masks"].to(storage, dtype=current["pred_masks"].dtype)
        updated["obj_ptr"] = cur_out["obj_ptr"].to(storage, dtype=current["obj_ptr"].dtype)
        updated["maskmem_features"] = maskmem_features.to(storage)
        updated["maskmem_pos_enc"] = [p.to(storage) for p in maskmem_pos_enc]
        updated["object_score_logits"] = obj_logits.to(storage)
        updated["n_pixels_pos"] = int((cur_out["pred_masks"][0, 0] > 0.0).sum().item())
        updated["iou"] = float(probe["quality"])
        if "all_pred_masks" in cur_out:
            updated["all_pred_masks"] = cur_out["all_pred_masks"]
        self._set_both_non_cond(self.frame_idx, updated)
        return {
            "applied": True,
            "frame_idx": self.frame_idx,
            "evicted_no_obj": evicted,
            "saved_orig": saved_orig,
            "saved_evicted": saved_evicted,
        }

    def rollback_reentry(self, inj_frame: int, snapshot: dict) -> tuple[list[float] | None, float | None]:
        obj_idx = self.state["obj_id_to_idx"][0]
        self._set_both_non_cond(inj_frame, snapshot["saved_orig"])
        for k, v in snapshot["saved_evicted"].items():
            self._set_both_non_cond(k, v)
        obj_dict = self.state["output_dict_per_obj"][obj_idx]
        _, _, vision_feats, vision_pos, feat_sizes = self.predictor._get_image_feature(self.state, self.frame_idx, 1)
        restored_out = self.predictor.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=obj_dict,
            num_frames=self.frame_idx + 1,
            run_mem_encoder=True,
        )
        restored_out["n_pixels_pos"] = int((restored_out["pred_masks"][0, 0] > 0.0).sum().item())
        restored_out["iou"] = 1.0
        self._set_both_non_cond(self.frame_idx, restored_out)
        _, video_mask = self.predictor._get_orig_video_res_output(self.state, restored_out["pred_masks"])
        binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
        return mask_to_bbox(binary), _scalar(restored_out.get("object_score_logits"))

    def prune_old_memory(self, keep_window: int):
        if keep_window <= 0 or self.state is None:
            return
        obj_idx = self.state["obj_id_to_idx"].get(0)
        if obj_idx is None:
            return
        non_cond = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        cutoff = self.frame_idx - keep_window
        # Keep up to 16 most recent visible RAM frames for DAM4SAM's RAM lookback
        visible_recent = sorted(
            [k for k, v in non_cond.items() if v is not None and int(v.get("n_pixels_pos", 0)) >= 1],
            reverse=True,
        )[:16]
        protected = set(visible_recent)
        for k in [idx for idx in non_cond if idx < cutoff and idx not in protected]:
            self.state["output_dict"]["non_cond_frame_outputs"].pop(k, None)
            non_cond.pop(k, None)

    def release_clip(self):
        if hasattr(self.model, "inference_state"):
            del self.model.inference_state
        super().release_clip()


# =============================================================================
# CARE Controller (Host-Agnostic, Verifier-Agnostic)
# =============================================================================

class CAREController:
    """Counterfactual Absence-Inertia Recovery with 1-Step Rollback."""

    def __init__(
        self,
        verifier: BaseVerifier,
        enable_dual_anchor: bool = True,
        certify_clean_anchor: bool = True,
        enable_scale_norm: bool = True,
        enable_rollback: bool = True,
        rho_min: float = DEFAULT_RHO_MIN,
        rho_max: float = DEFAULT_RHO_MAX,
        tau_obj: float = DEFAULT_TAU_OBJ,
        tau_obj_border: float = DEFAULT_TAU_OBJ_BORDER,
        clean_anchor_obj: float = DEFAULT_CLEAN_ANCHOR_OBJ,
        clean_anchor_stride: int = DEFAULT_CLEAN_ANCHOR_STRIDE,
        k_sil: int = DEFAULT_SILENCE_MIN_STREAK,
        probe_stride: int = DEFAULT_SILENCE_PROBE_STRIDE,
        tau_rb: float = DEFAULT_ROLLBACK_OBJ_MIN,
        rollback_cooldown: int = DEFAULT_ROLLBACK_COOLDOWN,
        purge_window: int = DEFAULT_MEMORY_PURGE_WINDOW,
    ):
        self.verifier = verifier
        self.enable_dual_anchor = bool(enable_dual_anchor)
        self.certify_clean_anchor = bool(certify_clean_anchor)
        self.enable_scale_norm = bool(enable_scale_norm)
        self.enable_rollback = bool(enable_rollback)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.tau_obj = float(tau_obj)
        self.tau_obj_border = float(tau_obj_border)
        self.clean_anchor_obj = float(clean_anchor_obj)
        self.clean_anchor_stride = int(clean_anchor_stride)
        self.k_sil = int(k_sil)
        self.probe_stride = int(probe_stride)
        self.tau_rb = float(tau_rb)
        self.rollback_cooldown = int(rollback_cooldown)
        self.purge_window = int(purge_window)

        self.init_box: list[float] | None = None
        self.a0: float = 1.0
        self.width: int = 0
        self.height: int = 0
        self.last_clean_out: dict | None = None
        self.last_clean_frame: int = 0
        self.pending: tuple[int, list[float]] | None = None
        self.just_injected: tuple[int, list[float], dict] | None = None
        self.rollback_until: int = -1
        self.silent_streak: int = 0

    def initialize(self, init_rgb: np.ndarray, init_box: list[float], init_mask: np.ndarray, adapter: BaseHostAdapter) -> dict:
        self.height, self.width = init_rgb.shape[:2]
        self.init_box = [float(v) for v in init_box]
        self.a0 = max(1.0, self.init_box[2] * self.init_box[3])
        self.last_clean_out = None
        self.last_clean_frame = 0
        self.pending = None
        self.just_injected = None
        self.rollback_until = -1
        self.silent_streak = 0
        feat_map = adapter.get_cached_feature_map()
        return self.verifier.initialize(init_rgb, self.init_box, init_mask, feat_map, self.width, self.height)

    def _update_clean_anchor(self, frame_idx, rgb_np, adapter, native_bbox, native_mask, obj_score, stats):
        if not self.enable_dual_anchor:
            return
        if native_bbox is None or native_mask is None or obj_score is None or obj_score < self.clean_anchor_obj:
            return
        if self.last_clean_out is not None and (frame_idx - self.last_clean_frame) < self.clean_anchor_stride:
            return
        if self.enable_scale_norm:
            gamma = border_gamma(native_bbox, self.init_box, self.width, self.height, margin=BORDER_MARGIN)
            rho_tilde = ((native_bbox[2] * native_bbox[3]) / self.a0) / gamma
            if not (self.rho_min <= rho_tilde <= self.rho_max):
                return
        if self.certify_clean_anchor:
            stats["anchor_checks"] += 1
            feat_map = adapter.get_cached_feature_map()
            if not self.verifier.verify_anchor(rgb_np, native_bbox, native_mask, feat_map, self.width, self.height):
                return
        slot = adapter.get_non_cond_slot(frame_idx)
        if slot is not None:
            self.last_clean_out = _clone_mem_entry(slot)
            self.last_clean_frame = frame_idx
            stats["anchor_updates"] += 1

    def _verify_candidate(self, rgb_np, adapter, probe) -> tuple[bool, dict]:
        box = probe["box"]
        obj = probe["obj_score"]
        qual = probe["quality"]
        if box is None or obj is None or obj < self.tau_obj or qual < QUALITY:
            return False, {}
        if self.enable_scale_norm:
            gamma = border_gamma(box, self.init_box, self.width, self.height, margin=BORDER_MARGIN)
            rho_tilde = ((box[2] * box[3]) / self.a0) / gamma
            if not (self.rho_min <= rho_tilde <= self.rho_max):
                return False, {}
        else:
            gamma = 1.0
            rho_tilde = (box[2] * box[3]) / self.a0

        feat_map = adapter.get_cached_feature_map()
        ok, vdiag = self.verifier.verify_probe(
            rgb_np, box, probe["mask"], feat_map, self.width, self.height, obj, gamma, self.tau_obj_border
        )
        if not ok:
            return False, {}
        return True, {
            "obj_probe": round(obj, 3),
            "quality": round(qual, 3),
            "gamma": round(gamma, 3),
            "rho_tilde": round(rho_tilde, 3),
            **vdiag,
        }

    def step(self, frame_idx, rgb_np, adapter: BaseHostAdapter, native_bbox, native_mask, obj_score, stats):
        """Strictly passive during occupied tracking; runs Counterfactual Probe ONLY when silent."""
        # 1-Step Post-Injection Self-Verification Rollback
        if self.just_injected is not None:
            inj_frame, inj_box, snapshot = self.just_injected
            self.just_injected = None
            if self.enable_rollback and frame_idx == inj_frame + 1:
                link_iou = overlap(native_bbox, inj_box) if native_bbox is not None else 0.0
                if link_iou < LINK or obj_score is None or obj_score < self.tau_rb:
                    restored_box, restored_obj = adapter.rollback_reentry(inj_frame, snapshot)
                    stats["rollbacks"] += 1
                    self.pending = None
                    self.rollback_until = frame_idx + self.rollback_cooldown
                    self.silent_streak = self.k_sil if restored_box is None else 0
                    return restored_box, "rollback", False, {
                        "rolled_back_from": inj_frame,
                        "link_iou": round(link_iou, 3),
                        "obj_t1": round(obj_score, 3) if obj_score is not None else None,
                        "restored_obj": round(restored_obj, 3) if restored_obj is not None else None,
                    }

        if native_bbox is not None:
            self.silent_streak = 0
            self.pending = None
            self._update_clean_anchor(frame_idx, rgb_np, adapter, native_bbox, native_mask, obj_score, stats)
            return native_bbox, "baseline", False, {}

        # Host tracker is SILENT (native_bbox is None)
        self.silent_streak += 1
        if self.silent_streak < self.k_sil or frame_idx <= self.rollback_until:
            self.pending = None
            return None, "baseline", False, {}

        has_pending = (self.pending is not None and 1 <= (frame_idx - self.pending[0]) <= 2)
        if not has_pending:
            self.pending = None

        if not (has_pending or ((self.silent_streak - self.k_sil) % self.probe_stride == 0)):
            return None, "baseline", False, {}

        stats["probe_calls"] += 1
        probe = adapter.dual_anchor_probe(self.last_clean_out, use_dual_anchor=self.enable_dual_anchor)
        ok, diag = self._verify_candidate(rgb_np, adapter, probe)
        if not ok:
            if self.pending is not None and frame_idx - self.pending[0] >= 2:
                self.pending = None
            return None, "baseline", False, {}

        stats["verified_probes"] += 1
        confirmed = (
            self.pending is not None
            and 1 <= (frame_idx - self.pending[0]) <= 2
            and overlap(probe["box"], self.pending[1]) >= LINK
        )
        if not confirmed:
            self.pending = (frame_idx, probe["box"])
            return None, "baseline", False, {"pending": True, **diag}

        # Confirmed over 2 consecutive frames -> Commit Closed-Loop Inertia-Free Re-Entry
        snapshot = adapter.commit_reentry(probe, purge_window=self.purge_window)
        stats["memory_updates"] += 1
        self.pending = None
        self.silent_streak = 0
        if self.enable_rollback:
            self.just_injected = (frame_idx, probe["box"], snapshot)
        diag["evicted_no_obj"] = snapshot["evicted_no_obj"]
        diag["last_clean_frame"] = self.last_clean_frame
        return probe["box"], "recovered_care", True, diag


# =============================================================================
# Dataset Loaders (Local Verified ZIPs + Standard Cloud Extracted Directories)
# =============================================================================

def _parse_vot_gt_line(line: str) -> list[float] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4 or any(p.lower() == "nan" for p in parts[:4]):
        return None
    vals = [float(p) for p in parts]
    if len(vals) == 8:
        xs = vals[0::2]
        ys = vals[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        w, h = x1 - x0, y1 - y0
        return [x0, y0, w, h] if w > 0 and h > 0 else None
    return vals[:4] if vals[2] > 0 and vals[3] > 0 else None


def load_dataset_spec(dataset: str, data_root: Path, clips_filter: set[str] | None = None) -> list[dict]:
    """Load sequence specifications supporting both local ZIP archives and cloud extracted directories.

    Strictly causal: only frame-0 initialization boxes (`B0`) are read before tracking.
    Full groundtruth annotations are read lazily inside `load_labels()` strictly post-hoc.
    """
    data_root = Path(data_root)
    specs = []

    if dataset in ("hoot_dev", "hoot_transfer"):
        plan = json.loads((data_root / "media-plan.json").read_text(encoding="utf-8"))
        for name, meta in plan["videos"].items():
            if clips_filter and name not in clips_filter:
                continue
            zip_path = data_root / f"{name}-rgb.zip"
            dl = json.loads((data_root / f"{name}-download.json").read_text(encoding="utf-8"))
            reads = {"count": 0}

            def _make_labels(zp=zip_path, nm=name, ln=meta["length"], ann_sha=meta["annotation_sha256"], rc=reads):
                rc["count"] += 1
                with ZipFile(zp) as z:
                    raw = z.read("anno.json")
                if sha256(raw).hexdigest() != ann_sha:
                    raise ValueError(f"Annotation SHA mismatch: {nm}")
                lbls, _ = labels_from(raw, ln, nm)
                return lbls

            specs.append({
                "name": name,
                "storage": "zip",
                "archive_path": zip_path,
                "archive_sha256": dl["sha256"],
                "length": int(meta["length"]),
                "init_box": [float(v) for v in meta["initial_box"]],
                "suffix": ".png",
                "load_labels": _make_labels,
                "label_reads": reads,
            })

    elif dataset == "lasot_ext":
        plan_path = data_root / "media-plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            for name, meta in plan["videos"].items():
                if clips_filter and name not in clips_filter:
                    continue
                zip_path = data_root / f"{name}.zip"
                reads = {"count": 0}

                def _make_labels(zp=zip_path, nm=name, ln=meta["length"], shas=meta["labels_sha256"], rc=reads):
                    rc["count"] += 1
                    with ZipFile(zp) as z:
                        gt_raw = z.read("groundtruth.txt")
                        occ_raw = z.read("full_occlusion.txt")
                        oov_raw = z.read("out_of_view.txt")
                    if (
                        sha256(gt_raw).hexdigest() != shas["groundtruth.txt"]
                        or sha256(occ_raw).hexdigest() != shas["full_occlusion.txt"]
                        or sha256(oov_raw).hexdigest() != shas["out_of_view.txt"]
                    ):
                        raise ValueError(f"LaSOT label SHA mismatch: {nm}")
                    gt_lines = gt_raw.decode("utf-8").splitlines()
                    occ = [int(v) for v in occ_raw.decode("utf-8").replace(",", " ").split()]
                    oov = [int(v) for v in oov_raw.decode("utf-8").replace(",", " ").split()]
                    if not (len(gt_lines) == len(occ) == len(oov) == ln):
                        raise ValueError(f"LaSOT label length mismatch: {nm}")
                    lbls = []
                    for line, fo, ov in zip(gt_lines, occ, oov, strict=True):
                        vals = [float(x) for x in line.split(",")]
                        lbls.append(None if (fo or ov or vals[2] <= 0 or vals[3] <= 0) else vals)
                    return lbls

                specs.append({
                    "name": name,
                    "storage": "zip",
                    "archive_path": zip_path,
                    "archive_sha256": meta["archive_sha256"],
                    "length": int(meta["length"]),
                    "init_box": [float(v) for v in meta["first_box"]],
                    "suffix": ".jpg",
                    "load_labels": _make_labels,
                    "label_reads": reads,
                })
        else:
            # Standard extracted LaSOT-ext directory layout: <root>/<class>/<seq>/ or <root>/<seq>/
            # Strictly causal: read ONLY line 0 of groundtruth.txt and hash image frame manifest only.
            seq_dirs = sorted({p.parent for p in data_root.rglob("groundtruth.txt") if (p.parent / "img").is_dir()})
            for sdir in seq_dirs:
                name = sdir.name
                if clips_filter and name not in clips_filter:
                    continue
                img_dir = sdir / "img"
                frames = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                gt_path = sdir / "groundtruth.txt"
                occ_path = sdir / "full_occlusion.txt"
                oov_path = sdir / "out_of_view.txt"
                first_line = _read_first_line_only(gt_path)
                init_box = [float(x) for x in first_line.split(",")[:4]]
                manifest_hash = _dir_frames_manifest_sha(name, frames)
                reads = {"count": 0}

                def _make_dir_labels(gp=gt_path, op=occ_path, vp=oov_path, ln=len(frames), rc=reads):
                    rc["count"] += 1
                    gt_lines = gp.read_text(encoding="utf-8").splitlines()
                    occ = [int(v) for v in op.read_text(encoding="utf-8").replace(",", " ").split()] if op.exists() else [0] * ln
                    oov = [int(v) for v in vp.read_text(encoding="utf-8").replace(",", " ").split()] if vp.exists() else [0] * ln
                    if not (len(gt_lines) == len(occ) == len(oov) == ln):
                        raise ValueError(f"LaSOT-ext dir length mismatch: {gp}")
                    lbls = []
                    for line, fo, ov in zip(gt_lines, occ, oov, strict=True):
                        vals = [float(x) for x in line.split(",")]
                        lbls.append(None if (fo or ov or vals[2] <= 0 or vals[3] <= 0) else vals)
                    return lbls

                specs.append({
                    "name": name,
                    "storage": "dir",
                    "frame_paths": frames,
                    "archive_path": sdir,
                    "archive_sha256": manifest_hash,
                    "length": len(frames),
                    "init_box": init_box,
                    "suffix": frames[0].suffix.lower() if frames else ".jpg",
                    "load_labels": _make_dir_labels,
                    "label_reads": reads,
                })

    elif dataset == "vot_reserve":
        manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
        raw_dir = data_root / "raw"
        for seq in manifest["sequences"]:
            name = seq["name"]
            if clips_filter and name not in clips_filter:
                continue
            color_zip = raw_dir / f"{name}-color.zip"
            ann_zip = raw_dir / f"{name}-annotations.zip"
            length = int(seq["length"])
            reads = {"count": 0}

            def _make_labels(az=ann_zip, ln=length, rc=reads):
                rc["count"] += 1
                with ZipFile(az) as z:
                    gt_name = [n for n in z.namelist() if n.endswith("groundtruth.txt")][0]
                    lines = z.read(gt_name).decode("utf-8").splitlines()
                if len(lines) != ln:
                    raise ValueError(f"VOT reserve length mismatch: {az}")
                return [_parse_vot_gt_line(line) for line in lines]

            with ZipFile(ann_zip) as z:
                gt_name = [n for n in z.namelist() if n.endswith("groundtruth.txt")][0]
                with z.open(gt_name) as fh:
                    first_line = fh.readline().decode("utf-8").strip()
                init_box = _parse_vot_gt_line(first_line)
            specs.append({
                "name": name,
                "storage": "zip",
                "archive_path": color_zip,
                "archive_sha256": digest(color_zip),
                "length": length,
                "init_box": init_box,
                "suffix": ".jpg",
                "load_labels": _make_labels,
                "label_reads": reads,
            })

    elif dataset == "vot_lt_dir":
        list_file = data_root / "list.txt"
        if list_file.exists():
            seq_names = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            seq_dirs = [data_root / s for s in seq_names]
        else:
            seq_dirs = sorted(p.parent for p in data_root.rglob("groundtruth.txt"))
        for sdir in seq_dirs:
            name = sdir.name
            if clips_filter and name not in clips_filter:
                continue
            color_dir = sdir / "color" if (sdir / "color").is_dir() else sdir
            frames = sorted(p for p in color_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            gt_path = sdir / "groundtruth.txt"
            # Strictly causal: read ONLY line 0 of groundtruth.txt and hash image frame manifest only.
            first_line = _read_first_line_only(gt_path)
            init_box = _parse_vot_gt_line(first_line)
            manifest_hash = _dir_frames_manifest_sha(name, frames)
            reads = {"count": 0}

            def _make_vot_dir_labels(gp=gt_path, ln=len(frames), rc=reads):
                rc["count"] += 1
                raw_lines = gp.read_text(encoding="utf-8").splitlines()
                if len(raw_lines) != ln:
                    raise ValueError(f"VOT dir length mismatch: {gp} ({len(raw_lines)} != {ln})")
                return [_parse_vot_gt_line(line) for line in raw_lines]

            specs.append({
                "name": name,
                "storage": "dir",
                "frame_paths": frames,
                "archive_path": sdir,
                "archive_sha256": manifest_hash,
                "length": len(frames),
                "init_box": init_box,
                "suffix": frames[0].suffix.lower() if frames else ".jpg",
                "load_labels": _make_vot_dir_labels,
                "label_reads": reads,
            })

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    specs.sort(key=lambda s: s["name"])
    return specs


def _iter_spec_frames(spec: dict, max_frames: int | None):
    """Yield (frame_idx, raw_bytes) strictly sequentially for ZIP or directory storage."""
    if spec["storage"] == "zip":
        with ZipFile(spec["archive_path"]) as archive:
            members = sorted(m for m in archive.namelist() if m.lower().endswith(spec["suffix"]))
            if len(members) != spec["length"]:
                raise ValueError(f"Member count mismatch for {spec['name']}: {len(members)} != {spec['length']}")
            limit = min(len(members), max_frames) if max_frames is not None else len(members)
            for i in range(limit):
                yield i, limit, archive.read(members[i])
    else:
        frames = spec["frame_paths"]
        if len(frames) != spec["length"]:
            raise ValueError(f"Frame count mismatch for {spec['name']}: {len(frames)} != {spec['length']}")
        limit = min(len(frames), max_frames) if max_frames is not None else len(frames)
        for i in range(limit):
            yield i, limit, frames[i].read_bytes()


# =============================================================================
# Causal Clip Runner, Sharding, Resume, and Post-Hoc Scoring
# =============================================================================

def run_clip_arm(
    spec: dict,
    arm: str,
    adapter: BaseHostAdapter,
    controller: CAREController | None,
    out_dir: Path,
    care_params: dict,
    keep_window: int = DEFAULT_MEMORY_KEEP_WINDOW,
    max_frames: int | None = None,
    resume: bool = False,
) -> tuple[dict, Path]:
    """Run one strictly causal arm ('baseline' or 'care') on one clip with optional resume."""
    name = spec["name"]
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    final_path = clip_dir / f"{arm}.jsonl"
    status_path = clip_dir / f"{arm}.status.json"

    if resume and final_path.exists() and status_path.exists():
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        if saved.get("trace_sha256") == digest(final_path):
            print(f"CARE {name} [{arm}] RESUMED (verified SHA-256)", flush=True)
            return saved, final_path

    adapter.configure_arm(arm, care_params)
    # Enforce identical SDPA kernel selection & RNG seeds across every arm and clip
    # (prevents Windows/Ampere FlashAttention fallback drift between arm 1 and arm 2).
    sdpa_mode = care_params.get("sdpa_kernel_mode", DEFAULT_SDPA_KERNEL_MODE)
    tmod = sys.modules.get("sam2.modeling.sam.transformer")
    if tmod is not None and hasattr(tmod, "ALLOW_ALL_KERNELS"):
        tmod.ALLOW_ALL_KERNELS = (sdpa_mode == "allow_all")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    partial_path = clip_dir / f"{arm}.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()

    frame_ms = []
    sources = Counter()
    stats = Counter()
    cal_info = None
    total_frames = 0
    img_width = 0
    img_height = 0

    with partial_path.open("w", encoding="utf-8") as writer, \
            torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i, limit, encoded in _iter_spec_frames(spec, max_frames):
            total_frames = limit
            sha = sha256(encoded).hexdigest()
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
                if i == 0:
                    img_width, img_height = pil_rgb.size
                rgb_np = np.asarray(pil_rgb)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = perf_counter()

            if i == 0:
                res = adapter.initialize(pil_rgb, spec["init_box"])
                final_bbox = [float(v) for v in spec["init_box"]]
                native_bbox = res["pred_bbox"]
                source = "init"
                mem_updated = False
                obj_score = None
                confidence = 1.0
                diag = {}
                if arm == "care" and controller is not None:
                    cal_info = controller.initialize(rgb_np, spec["init_box"], res["pred_mask"], adapter)
            else:
                res = adapter.track(pil_rgb)
                native_bbox = res["pred_bbox"]
                obj_score = res.get("object_score")
                if arm == "care" and controller is not None:
                    final_bbox, source, mem_updated, diag = controller.step(
                        i, rgb_np, adapter, native_bbox, res.get("pred_mask"), obj_score, stats
                    )
                    if source == "recovered_care" and "obj_probe" in diag:
                        confidence = _sigmoid(diag["obj_probe"])
                    elif source == "rollback":
                        confidence = _sigmoid(diag.get("restored_obj")) if final_bbox is not None else 0.0
                    else:
                        confidence = _sigmoid(obj_score) if final_bbox is not None else 0.0
                else:
                    final_bbox = native_bbox
                    source = "baseline"
                    mem_updated = False
                    confidence = _sigmoid(obj_score) if final_bbox is not None else 0.0
                    diag = {}
                adapter.prune_old_memory(keep_window=keep_window)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = 1000.0 * (perf_counter() - t_start)

            if i > 0:
                frame_ms.append(elapsed_ms)
            sources[source] += 1

            writer.write(json.dumps({
                "frame": i,
                "sha256": sha,
                "bbox": final_bbox,
                "native_bbox": native_bbox,
                "object_score": obj_score,
                "confidence": round(confidence, 6),
                "source": source,
                "memory_update": mem_updated,
                "synchronized_model_ms": elapsed_ms,
                "diag": diag,
            }, separators=(",", ":"), allow_nan=False) + "\n")

            if (i + 1) % 250 == 0 or (i + 1) == limit:
                writer.flush()
                print(
                    f"CARE {adapter.host_name}/{name} [{arm}] {i + 1}/{limit} "
                    f"updates={stats['memory_updates']} rb={stats['rollbacks']} "
                    f"probes={stats['probe_calls']} med_ms={median(frame_ms) if frame_ms else 0.0:.1f}",
                    flush=True,
                )

    partial_path.replace(final_path)
    p95 = quantiles(frame_ms, n=100)[94] if len(frame_ms) >= 20 else (max(frame_ms) if frame_ms else 0.0)
    effective_min_obj = (
        float(adapter.predictor.min_obj_score_logits)
        if adapter.predictor is not None and hasattr(adapter.predictor, "min_obj_score_logits")
        else None
    )
    summary = {
        "frames": total_frames,
        "width": img_width,
        "height": img_height,
        "effective_min_obj_score_logits": effective_min_obj,
        "trace_sha256": digest(final_path),
        "sources": dict(sources),
        "stats": dict(stats),
        "calibration": cal_info,
        "median_synchronized_step_ms": median(frame_ms) if frame_ms else 0.0,
        "p95_synchronized_step_ms": p95,
        "mean_synchronized_step_ms": (sum(frame_ms) / len(frame_ms)) if frame_ms else 0.0,
        "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    save(status_path, summary)

    adapter.release_clip()
    if controller is not None:
        controller.last_clean_out = None
        controller.just_injected = None
    return summary, final_path


def _spec_image_size(spec: dict, operations: dict | None = None, arm: str | None = None) -> tuple[int, int] | None:
    if operations and spec["name"] in operations:
        op = operations[spec["name"]].get(arm) if arm else next(iter(operations[spec["name"]].values()), None)
        if op and op.get("width") and op.get("height"):
            return int(op["width"]), int(op["height"])
    try:
        for _, _, encoded in _iter_spec_frames(spec, max_frames=1):
            with Image.open(BytesIO(encoded)) as img:
                return img.size
    except Exception:
        return None
    return None


def score_traces(
    specs: list[dict],
    trace_paths: dict,
    arms: list[str],
    operations: dict | None = None,
    vot_toolkit_path: Path | None = None,
) -> dict:
    """Strictly post-hoc scoring: event stats, Fixed Macro PRF, official vot-toolkit, custom VOT-LT, and dual-protocol LaSOT."""
    per_arm_videos = {a: {} for a in arms}
    per_arm_prf_inputs = {a: {} for a in arms}
    per_arm_conf_inputs = {a: {} for a in arms}
    per_arm_toolkit_inputs = {a: {} for a in arms}
    all_ms = {a: [] for a in arms}

    for spec in specs:
        name = spec["name"]
        labels = spec["load_labels"]()
        for arm in arms:
            with trace_paths[name][arm].open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            limit = len(rows)
            lbl_slice = labels[:limit]
            boxes = [r["bbox"] for r in rows]
            confs = [r.get("confidence", 1.0 if r["bbox"] is not None else 0.0) for r in rows]
            per_arm_videos[arm][name] = assess(lbl_slice, boxes)
            per_arm_prf_inputs[arm][name] = (lbl_slice, boxes)
            per_arm_conf_inputs[arm][name] = (lbl_slice, boxes, confs)
            size = _spec_image_size(spec, operations=operations, arm=arm)
            if size is not None:
                per_arm_toolkit_inputs[arm][name] = (lbl_slice, boxes, confs, size)
            all_ms[arm].extend(r["synchronized_model_ms"] for r in rows[1:])

    out = {"arms": {}}
    for arm in arms:
        totals = combine(per_arm_videos[arm])
        prf, prf_by_video = macro_prf(per_arm_prf_inputs[arm])
        vot_lt_custom = vot_lt_curve_metrics(per_arm_conf_inputs[arm])
        vot_lt_tk = (
            vot_lt_toolkit_metrics(per_arm_toolkit_inputs[arm], toolkit_path=vot_toolkit_path)
            if len(per_arm_toolkit_inputs[arm]) == len(per_arm_conf_inputs[arm])
            else {"available": False, "reason": "missing image dimensions"}
        )
        lasot = lasot_metrics(per_arm_prf_inputs[arm])
        ms_list = all_ms[arm]
        out["arms"][arm] = {
            "totals": totals,
            "fixed_point_macro_prf": prf,
            "fixed_point_by_video": prf_by_video,
            "vot_lt_toolkit": vot_lt_tk,
            "vot_lt_custom": vot_lt_custom,
            "lasot_metrics": lasot,
            "ms": {
                "median": median(ms_list) if ms_list else 0.0,
                "p95": quantiles(ms_list, n=100)[94] if len(ms_list) >= 20 else (max(ms_list) if ms_list else 0.0),
                "mean": (sum(ms_list) / len(ms_list)) if ms_list else 0.0,
            },
            "videos": per_arm_videos[arm],
        }
    return out


def build_host_adapter(args) -> BaseHostAdapter:
    deps = [Path(d) for d in (args.deps or [])]
    if args.host == "sam21":
        return SAM21HostAdapter(
            weights=args.weights,
            backbone=args.backbone,
            sam2_root=args.host_source,
            deps_dirs=deps,
            use_sentry_runtime=args.use_sentry_runtime,
            sentry_source=args.sentry_source,
            device=args.device,
        )
    if args.host == "samurai":
        return SAMURAIHostAdapter(
            weights=args.weights,
            backbone=args.backbone,
            samurai_root=args.host_source,
            deps_dirs=deps,
            device=args.device,
        )
    if args.host == "dam4sam":
        return DAM4SAMHostAdapter(
            weights=args.weights,
            backbone=args.backbone,
            dam4sam_root=args.host_source,
            deps_dirs=deps,
            device=args.device,
        )
    raise ValueError(f"Unknown host: {args.host}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hoot_dev", "hoot_transfer", "lasot_ext", "vot_reserve", "vot_lt_dir"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--host", choices=("sam21", "samurai", "dam4sam"), default="sam21")
    parser.add_argument("--backbone", choices=("tiny", "base_plus", "large"), default="tiny")
    parser.add_argument("--verifier", choices=("sam2_native", "dinov2", "none"), default="sam2_native")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host-source", type=Path, default=None, help="Root of host source tree (sam2 / samurai/sam2 / DAM4SAM).")
    parser.add_argument("--sentry-source", type=Path, default=None, help="Optional SENTRY source root if --use-sentry-runtime.")
    parser.add_argument("--use-sentry-runtime", action="store_true", help="Use bundled SENTRY SAM2Runtime for exact legacy sam21 parity.")
    parser.add_argument("--deps", type=Path, nargs="*", default=None, help="Optional extra Python dependency paths.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("baseline", "care"), default=["baseline", "care"])
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Component ablations & hyperparameter sensitivity switches
    parser.add_argument("--no-dual-anchor", action="store_true")
    parser.add_argument("--no-certify-anchor", action="store_true")
    parser.add_argument("--no-scale-norm", action="store_true")
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--k-sil", type=int, default=DEFAULT_SILENCE_MIN_STREAK)
    parser.add_argument("--tau-rb", type=float, default=DEFAULT_ROLLBACK_OBJ_MIN)
    parser.add_argument("--rho-max", type=float, default=DEFAULT_RHO_MAX)
    parser.add_argument("--keep-window", type=int, default=DEFAULT_MEMORY_KEEP_WINDOW)
    parser.add_argument("--tau-obj", type=float, default=DEFAULT_TAU_OBJ)
    parser.add_argument(
        "--samurai-baseline-min-obj",
        type=float,
        default=DEFAULT_SAMURAI_BASELINE_MIN_OBJ,
        help="Optional min_obj_score_logits for SAMURAI baseline arm (None -> stock -1.0).",
    )
    parser.add_argument(
        "--samurai-care-min-obj",
        type=float,
        default=DEFAULT_SAMURAI_CARE_MIN_OBJ,
        help="Optional min_obj_score_logits for SAMURAI CARE arm (None -> match stock -1.0).",
    )
    parser.add_argument(
        "--sdpa-kernel-mode",
        choices=("allow_all", "stock"),
        default=DEFAULT_SDPA_KERNEL_MODE,
        help="Consistent SDPA kernel mode across arms (`allow_all` prevents first-arm fallback drift).",
    )
    parser.add_argument(
        "--vot-toolkit-path",
        type=Path,
        default=None,
        help="Optional path to official vot-toolkit Python root for official VOT-LT scoring.",
    )

    # Sharding & resume
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true", help="Merge all completed shard clip statuses and score.")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        parser.error("Invalid --shard-idx / --num-shards")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False

    all_specs = load_dataset_spec(args.dataset, args.data, set(args.clips) if args.clips else None)
    if not all_specs:
        raise ValueError("No clips selected")

    args.output.mkdir(parents=True, exist_ok=True)
    care_params = {
        "host": args.host,
        "backbone": args.backbone,
        "verifier": args.verifier,
        "quality": QUALITY,
        "link": LINK,
        "border_margin": BORDER_MARGIN,
        "gamma_min": GAMMA_MIN,
        "rho_min": DEFAULT_RHO_MIN,
        "rho_max": args.rho_max,
        "tau_obj": args.tau_obj,
        "tau_obj_border": DEFAULT_TAU_OBJ_BORDER,
        "clean_anchor_obj": DEFAULT_CLEAN_ANCHOR_OBJ,
        "clean_anchor_stride": DEFAULT_CLEAN_ANCHOR_STRIDE,
        "k_sil": args.k_sil,
        "probe_stride": DEFAULT_SILENCE_PROBE_STRIDE,
        "enable_dual_anchor": not args.no_dual_anchor,
        "certify_clean_anchor": not args.no_certify_anchor,
        "enable_scale_norm": not args.no_scale_norm,
        "enable_rollback": not args.no_rollback,
        "tau_rb": args.tau_rb,
        "rollback_cooldown": DEFAULT_ROLLBACK_COOLDOWN,
        "purge_window": DEFAULT_MEMORY_PURGE_WINDOW,
        "keep_window": args.keep_window,
        "samurai_baseline_min_obj": args.samurai_baseline_min_obj,
        "samurai_care_min_obj": args.samurai_care_min_obj,
        "sdpa_kernel_mode": args.sdpa_kernel_mode,
    }

    freeze_path = args.output / "freeze.json"
    pre_label_reads = sum(s["label_reads"]["count"] for s in all_specs)
    if pre_label_reads != 0:
        raise RuntimeError(f"Causality violation: {pre_label_reads} label reads occurred before inference")

    def _build_freeze_payload(host_prov: dict | None = None) -> dict:
        return {
            "utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "host": args.host,
            "backbone": args.backbone,
            "verifier": args.verifier,
            "clips": {
                s["name"]: {
                    "length": s["length"],
                    "archive_sha256": s["archive_sha256"],
                    "init_box": s["init_box"],
                }
                for s in all_specs
            },
            "arms": list(args.arms),
            "max_frames": args.max_frames,
            "checkpoint_sha256": digest(args.weights),
            "script_sha256": digest(Path(__file__)),
            "host_provenance": host_prov or {},
            "precision": "matched CUDA FP16 autocast",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "parameters": care_params,
            "post_initialization_label_reads": pre_label_reads,
        }

    if args.merge_shards:
        if not freeze_path.exists():
            save(freeze_path, _build_freeze_payload())
        operations = {}
        trace_paths = {}
        for spec in all_specs:
            name = spec["name"]
            operations[name] = {}
            trace_paths[name] = {}
            for arm in args.arms:
                tpath = args.output / name / f"{arm}.jsonl"
                spath = args.output / name / f"{arm}.status.json"
                if not (tpath.exists() and spath.exists()):
                    raise FileNotFoundError(f"Missing shard output for {name}/{arm}")
                op = json.loads(spath.read_text(encoding="utf-8"))
                if op["trace_sha256"] != digest(tpath):
                    raise ValueError(f"Trace SHA mismatch during merge for {name}/{arm}")
                operations[name][arm] = op
                trace_paths[name][arm] = tpath
        pre_score_reads = sum(s["label_reads"]["count"] for s in all_specs)
        if pre_score_reads != 0:
            raise RuntimeError(f"Causality violation before scoring: {pre_score_reads}")
        complete_path = args.output / "complete.json"
        save(complete_path, {
            "freeze_sha256": digest(freeze_path),
            "post_initialization_label_reads_before_scoring": pre_score_reads,
            "operations": operations,
        })
        scored = score_traces(all_specs, trace_paths, args.arms, operations=operations, vot_toolkit_path=args.vot_toolkit_path)
        save(args.output / "results.json", {
            "freeze_sha256": digest(freeze_path),
            "complete_sha256": digest(complete_path),
            **scored,
        })
        _print_summary(args, scored)
        return

    shard_specs = [s for idx, s in enumerate(all_specs) if idx % args.num_shards == args.shard_idx]
    if not shard_specs:
        print(f"Shard {args.shard_idx}/{args.num_shards} has 0 clips; exiting cleanly.", flush=True)
        return

    adapter = build_host_adapter(args)
    # Write freeze.json AFTER adapter initialization so actual imported module paths,
    # config SHA-256, and host repo git commit are frozen before inference starts.
    if args.num_shards == 1 or (args.shard_idx == 0 and not freeze_path.exists()):
        save(freeze_path, _build_freeze_payload(adapter.provenance))

    controller = None
    if "care" in args.arms:
        verifier = build_verifier(args.verifier)
        controller = CAREController(
            verifier=verifier,
            enable_dual_anchor=not args.no_dual_anchor,
            certify_clean_anchor=not args.no_certify_anchor,
            enable_scale_norm=not args.no_scale_norm,
            enable_rollback=not args.no_rollback,
            rho_min=DEFAULT_RHO_MIN,
            rho_max=args.rho_max,
            tau_obj=args.tau_obj,
            tau_obj_border=DEFAULT_TAU_OBJ_BORDER,
            clean_anchor_obj=DEFAULT_CLEAN_ANCHOR_OBJ,
            clean_anchor_stride=DEFAULT_CLEAN_ANCHOR_STRIDE,
            k_sil=args.k_sil,
            probe_stride=DEFAULT_SILENCE_PROBE_STRIDE,
            tau_rb=args.tau_rb,
            rollback_cooldown=DEFAULT_ROLLBACK_COOLDOWN,
            purge_window=DEFAULT_MEMORY_PURGE_WINDOW,
        )

    operations = {}
    trace_paths = {}
    for spec in shard_specs:
        name = spec["name"]
        operations[name] = {}
        trace_paths[name] = {}
        for arm in args.arms:
            summary, tpath = run_clip_arm(
                spec=spec,
                arm=arm,
                adapter=adapter,
                controller=controller,
                out_dir=args.output,
                care_params=care_params,
                keep_window=args.keep_window,
                max_frames=args.max_frames,
                resume=args.resume,
            )
            operations[name][arm] = summary
            trace_paths[name][arm] = tpath

    if args.num_shards > 1:
        save(args.output / f"shard_{args.shard_idx}_of_{args.num_shards}.json", {
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "operations": operations,
        })
        print(f"SHARD_DONE {args.shard_idx}/{args.num_shards} ({len(shard_specs)} clips)", flush=True)
        return

    pre_score_reads = sum(s["label_reads"]["count"] for s in shard_specs)
    if pre_score_reads != 0:
        raise RuntimeError(f"Causality violation during inference: {pre_score_reads} label reads before scoring")
    complete_path = args.output / "complete.json"
    save(complete_path, {
        "freeze_sha256": digest(freeze_path),
        "post_initialization_label_reads_before_scoring": pre_score_reads,
        "operations": operations,
    })
    scored = score_traces(shard_specs, trace_paths, args.arms, operations=operations, vot_toolkit_path=args.vot_toolkit_path)
    save(args.output / "results.json", {
        "freeze_sha256": digest(freeze_path),
        "complete_sha256": digest(complete_path),
        **scored,
    })
    _print_summary(args, scored)


def _print_summary(args, scored: dict):
    print(f"=== CARE PLUGIN SUMMARY ({args.host}/{args.backbone}/{args.verifier} on {args.dataset}) ===", flush=True)
    for arm in args.arms:
        a = scored["arms"][arm]
        t = a["totals"]
        prf = a["fixed_point_macro_prf"]
        vtk = a["vot_lt_toolkit"]
        vcu = a["vot_lt_custom"]
        ls = a["lasot_metrics"]
        ms = a["ms"]
        vtk_str = f"{vtk['f_score']:.4f} (v{vtk['toolkit_version']})" if vtk.get("available") else "N/A"
        print(
            f"  [{arm:8s}] timely={t['timely']} first={t['first']} fp={t['fp']} "
            f"vis_iou={t['visible_mean_iou']:.4f} | "
            f"MacroP={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f} | "
            f"VOT-tk={vtk_str} custom={vcu['f_score']:.4f} | "
            f"LaSOT(vis)={ls['auc']:.4f} (matlab_conf)={ls['matlab_conf']['auc']:.4f} | "
            f"med={ms['median']:.1f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
