"""Programmatic Cloud Benchmark Summary & Paired Video-Level Bootstrap CI Generator.

Reads strictly from independently audited `independent-audit.json` (`status == "AUDIT_OK"`)
and `results.json` files produced by `care_sam2_plugin.py`, `run_memory_gating_benchmark.py`,
and `run_counterfactual_inertia_benchmark.py`. Zero manual number transcription.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _per_video_f1(arm_obj: dict) -> dict[str, float]:
    out = {}
    for vid, d in arm_obj.get("fixed_point_by_video", {}).items():
        p = float(d["precision"])
        r = float(d["recall"])
        out[vid] = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return out


def _per_video_vis_iou(arm_obj: dict) -> dict[str, float]:
    out = {}
    for vid, d in arm_obj.get("videos", {}).items():
        if "vis_iou" in d:
            out[vid] = float(d["vis_iou"])
        else:
            vis = max(1, int(d.get("visible", 0)))
            out[vid] = float(d.get("visible_iou_sum", 0.0)) / vis
    return out


def _per_video_fp(arm_obj: dict) -> dict[str, int]:
    out = {}
    for vid, d in arm_obj.get("videos", {}).items():
        out[vid] = int(d["fp"] if "fp" in d else d.get("false_reports", 0))
    return out


def paired_video_bootstrap_ci(
    base_map: dict[str, float],
    trial_map: dict[str, float],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    vids = sorted(set(base_map.keys()) & set(trial_map.keys()))
    if not vids:
        return {"n_videos": 0, "mean_delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "wins": 0, "ties": 0, "losses": 0}
    deltas = np.array([float(trial_map[v]) - float(base_map[v]) for v in vids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    boot_means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_videos": len(vids),
        "mean_delta": float(deltas.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "wins": int((deltas > 1e-9).sum()),
        "ties": int((np.abs(deltas) <= 1e-9).sum()),
        "losses": int((deltas < -1e-9).sum()),
    }


def _row_from_arm(label: str, audit: dict, arm_name: str) -> str:
    a = audit["arms"][arm_name]
    t = a["totals"]
    prf = a["fixed_point_macro_prf"]
    vtk = a.get("vot_lt_toolkit", {})
    ls = a.get("lasot_metrics", {})
    ms = a.get("ms", {})
    iv = a.get("interventions", {})
    upd = sum(int(v.get("memory_updates", 0)) for v in iv.values())
    rb = sum(int(v.get("rollbacks", 0)) for v in iv.values())
    vtk_f = f"{vtk['f_score']:.4f}" if vtk.get("available") else "N/A"
    best_th = vtk.get("best_threshold", "N/A")
    if isinstance(best_th, float):
        th_s = f"{best_th:.4f}"
    else:
        th_s = str(best_th)
    auc_v = f"{ls['auc']:.4f}" if "auc" in ls else "N/A"
    auc_m = f"{ls['matlab_conf']['auc']:.4f}" if "matlab_conf" in ls else "N/A"
    return (
        f"| {label} | `{audit.get('status', 'OK')}` | {t['timely']} | {t['first']} | {t['fp']} | {upd} / {rb} | "
        f"{t['visible_mean_iou']:.4f} | {prf['precision']:.4f} | {prf['recall']:.4f} | **{prf['f1']:.4f}** | "
        f"**{vtk_f}** | `{th_s}` | {auc_v} | {auc_m} | {ms.get('median', 0.0):.1f} |"
    )


def build_report(runs_dir: Path) -> str:
    lines = [
        "# Cloud Full-Benchmark Programmatic Summary Report",
        "",
        f"Source directory: `{runs_dir}` (parsed strictly from `independent-audit.json` / `results.json`).",
        "",
    ]

    def _get_audit(tag: str) -> dict | None:
        p = runs_dir / tag / "independent-audit.json"
        if p.exists():
            obj = _load_json(p)
            if obj.get("status") != "AUDIT_OK":
                raise ValueError(f"Non-AUDIT_OK status in {p}: {obj.get('status')}")
            return obj
        return None

    table_header = [
        "| Configuration | Status | Timely | First | Absent FP | Upd / RB | Vis.IoU | Online P | Online R | Online F1 | VOT-tk F | `best_th` | LaSOT AUC (vis) | LaSOT AUC (MATLAB) | Med ms |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # 1. VOT-LT2022 Main Table
    vot_specs = [
        ("SAM 2.1 Tiny — `vanilla`", "vot22_sam21_tiny_native", "baseline"),
        ("SAM 2.1 Tiny — `gate_skip_absent` (5-line)", "vot22_sam21_tiny_gating", "gate_skip_absent"),
        ("SAM 2.1 Tiny — `gate_skip_absent_keept1` (SAMURAI rule, no KF)", "vot22_sam21_tiny_gating", "gate_skip_absent_keept1"),
        ("SAM 2.1 Tiny — `CARE (sam2_native)`", "vot22_sam21_tiny_native", "care"),
        ("SAM 2.1 Base+ — `vanilla`", "vot22_sam21_baseplus_native", "baseline"),
        ("SAM 2.1 Base+ — `gate_skip_absent` (5-line)", "vot22_sam21_baseplus_gating", "gate_skip_absent"),
        ("SAM 2.1 Base+ — `gate_skip_absent_keept1`", "vot22_sam21_baseplus_gating", "gate_skip_absent_keept1"),
        ("SAM 2.1 Base+ — `CARE (sam2_native)`", "vot22_sam21_baseplus_native", "care"),
        ("SAMURAI Tiny — stock (`min_obj=-1.0`)", "vot22_samurai_tiny_m1", "baseline"),
        ("SAMURAI Tiny — calibrated (`min_obj=0.0`)", "vot22_samurai_tiny_m0", "baseline"),
        ("DAM4SAM Tiny — stock", "vot22_dam4sam_tiny", "baseline"),
        ("SAM2Long Tiny — official (`K=3`, online top-1)", "vot22_sam2long_tiny", "baseline"),
        ("SAM2Long Tiny — official (`K=3`, full-video backtrack)", "vot22_sam2long_tiny", "backtrack"),
    ]
    vot_rows = []
    cached = {}
    for label, tag, arm in vot_specs:
        au = cached.setdefault(tag, _get_audit(tag))
        if au is not None and arm in au.get("arms", {}):
            vot_rows.append(_row_from_arm(label, au, arm))
    if vot_rows:
        lines.extend(["## 1. Full VOT-LT2022 (50 Sequences) Main Pareto Table", "", *table_header, *vot_rows, ""])

    # 2. VOT-LT2022 Ablations
    abl_specs = [
        ("SAM 2.1 Tiny — `vanilla`", "vot22_sam21_tiny_native", "baseline"),
        ("CARE — full (`sam2_native`, dual-anchor + scale-norm + 1-step rollback)", "vot22_sam21_tiny_native", "care"),
        ("CARE — verifier `none` (no appearance verifier)", "vot22_abl_verifier_none", "care"),
        ("CARE — verifier `dinov2` (external ViT-S/14)", "vot22_abl_verifier_dinov2", "care"),
        ("CARE — `--no-dual-anchor` (`m0` only)", "vot22_abl_no_dual_anchor", "care"),
        ("CARE — `--no-certify-anchor`", "vot22_abl_no_certify", "care"),
        ("CARE — `--no-scale-norm`", "vot22_abl_no_scale_norm", "care"),
        ("CARE — `--no-rollback` (no 1-step self-verification rollback)", "vot22_abl_no_rollback", "care"),
    ]
    abl_rows = []
    for label, tag, arm in abl_specs:
        au = cached.setdefault(tag, _get_audit(tag))
        if au is not None and arm in au.get("arms", {}):
            abl_rows.append(_row_from_arm(label, au, arm))
    if abl_rows:
        lines.extend(["## 2. Full VOT-LT2022 Component & Verifier Ablation Table", "", *table_header, *abl_rows, ""])

    # 3. Paired Video-Level Bootstrap 95% CI on VOT-LT2022
    au_t = cached.get("vot22_sam21_tiny_native")
    au_g = cached.get("vot22_sam21_tiny_gating")
    if au_t is not None and "baseline" in au_t["arms"] and "care" in au_t["arms"]:
        lines.extend(["## 3. Paired Video-Level Bootstrap 95% CIs (10,000 Resamples)", ""])
        pairs = [("CARE vs Vanilla (Tiny)", au_t["arms"]["baseline"], au_t["arms"]["care"])]
        if au_g is not None and "gate_skip_absent" in au_g["arms"]:
            pairs.append(("gate_skip_absent vs Vanilla (Tiny)", au_t["arms"]["baseline"], au_g["arms"]["gate_skip_absent"]))
            pairs.append(("CARE vs gate_skip_absent (Tiny)", au_g["arms"]["gate_skip_absent"], au_t["arms"]["care"]))
        lines.append("| Pair | Videos | Metric | Mean Delta | 95% Bootstrap CI | Win / Tie / Loss |")
        lines.append("| :--- | :---: | :--- | :---: | :---: | :---: |")
        for pname, b_arm, t_arm in pairs:
            for mname, fn in [("Online Video F1", _per_video_f1), ("Visible IoU", _per_video_vis_iou), ("Absent FP (lower better)", _per_video_fp)]:
                ci = paired_video_bootstrap_ci(fn(b_arm), fn(t_arm))
                fmt = "{:+.2f}" if "FP" in mname else "{:+.4f}"
                lines.append(
                    f"| {pname} | {ci['n_videos']} | {mname} | {fmt.format(ci['mean_delta'])} | "
                    f"[{fmt.format(ci['ci_lo'])}, {fmt.format(ci['ci_hi'])}] | {ci['wins']} / {ci['ties']} / {ci['losses']} |"
                )
        lines.append("")

    # 4. Full-Benchmark 9-Arm Counterfactual Decomposition
    probe_p = runs_dir / "vot22_sam21_tiny_a6_decomposed" / "results.json"
    if probe_p.exists():
        pr = _load_json(probe_p)
        lines.extend([
            f"## 4. Full VOT-LT2022 9-Arm Counterfactual Memory Decomposition ({pr['affected_silent_visible_videos']}/{pr['total_videos']} affected videos, {pr['silent_qualifying_events_count']} silent qualifying events)",
            "",
            "| Counterfactual Memory Condition | Pooled Vis `IoU>=0.5` | Pooled Vis Mean `obj` | Pooled Absent FP (`obj>0`) | Video-Macro Vis `IoU>=0.5` Rate | Video-Macro Absent FP Rate | Silent Events `IoU>=0.5` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for a, pv in pr["pooled_silent_visible"].items():
            pa = pr["pooled_silent_absent"][a]
            mv = pr["video_macro_silent_visible"][a]
            ma = pr["video_macro_silent_absent"][a]
            ev = pr["event_recovered_ge_05"][a]
            lines.append(
                f"| `{a}` | {pv['iou_ge_05']}/{pv['n']} ({100*pv['iou_ge_05']/max(1,pv['n']):.1f}%) | {pv['mean_obj']:+.4f} | "
                f"{pa['pos_obj']}/{pa['n']} ({100*pa['pos_obj']/max(1,pa['n']):.1f}%) | "
                f"{100*mv['mean_iou_ge_05_rate']:.2f}% | {100*ma['mean_fp_rate']:.2f}% | "
                f"{ev}/{pr['silent_qualifying_events_count']} |"
            )
        lines.append("")

    # 5. LaSOT-ext Main Table (if present)
    lasot_specs = [
        ("SAM 2.1 Tiny — `vanilla`", "lasot_ext_sam21_tiny_native", "baseline"),
        ("SAM 2.1 Tiny — `gate_skip_absent` (5-line)", "lasot_ext_sam21_tiny_gating", "gate_skip_absent"),
        ("SAM 2.1 Tiny — `gate_skip_absent_keept1`", "lasot_ext_sam21_tiny_gating", "gate_skip_absent_keept1"),
        ("SAM 2.1 Tiny — `CARE (sam2_native)`", "lasot_ext_sam21_tiny_native", "care"),
        ("SAMURAI Tiny — stock (`min_obj=-1.0`)", "lasot_ext_samurai_tiny_m1", "baseline"),
        ("SAMURAI Tiny — calibrated (`min_obj=0.0`)", "lasot_ext_samurai_tiny_m0", "baseline"),
        ("DAM4SAM Tiny — stock", "lasot_ext_dam4sam_tiny", "baseline"),
    ]
    lasot_rows = []
    for label, tag, arm in lasot_specs:
        au = cached.setdefault(tag, _get_audit(tag))
        if au is not None and arm in au.get("arms", {}):
            lasot_rows.append(_row_from_arm(label, au, arm))
    if lasot_rows:
        lines.extend(["## 5. Full LaSOT-ext Protocol III (150 Sequences) Main Table", "", *table_header, *lasot_rows, ""])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    md = build_report(args.runs_dir)
    if args.output is not None:
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote summary report: {args.output}", flush=True)
    else:
        print(md)


if __name__ == "__main__":
    main()
