"""Independent auditor for CARE plug-and-play SAM 2.1 / SAMURAI / DAM4SAM runs.

Independently verifies:
- SHA-256 provenance across freeze.json, complete.json, results.json, and all trace JSONL files;
- Identical per-frame pixel SHA-256 streams across arms;
- Zero post-initialization label reads before scoring;
- Independent recomputation of event stats, fixed-point Macro P/R/F1, official `vot-toolkit`
  VOT-LT F-score (`vot.analysis.longterm`), custom VOT-LT diagnostic curve, and dual-protocol
  LaSOT metrics (`matlab_conf` strict `>` with total-frame denominator AND visible-frame metrics)
  WITHOUT importing the scoring functions from `care_sam2_plugin.py`;
- Transparent per-video CARE intervention counts (`probe_calls`, `memory_updates`, `rollbacks`)
  and effective host thresholds so causal attribution is directly auditable.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import median, quantiles
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

from care_sam2_plugin import load_dataset_spec
from evaluation import box_iou
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_paired_query_continuous import combine


def _independent_macro_prf(videos_data: dict) -> tuple[dict, dict]:
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


def _independent_lasot_metrics(videos_data: dict) -> dict:
    """Independent LaSOT recomputation under both MATLAB conference (`>`, total frames) and visible protocols."""
    iou_ths = np.linspace(0.0, 1.0, 21)
    norm_ths = np.linspace(0.0, 0.5, 51)
    vis_ge_aucs, vis_gt_aucs, mat_aucs = [], [], []
    vis_p20s, mat_p20s = [], []
    vis_pnorms, mat_pnorms = [], []
    by_video = {}
    for name, (labels, boxes) in videos_data.items():
        n_total = len(labels)
        ious, cles, ncles = [], [], []
        for gt, pred in zip(labels, boxes, strict=True):
            if gt is None:
                continue
            gx, gy, gw, gh = gt
            if pred is None:
                ious.append(0.0)
                cles.append(float("inf"))
                ncles.append(float("inf"))
            else:
                ious.append(box_iou(pred, gt))
                px, py, pw, ph = pred
                dx = (px + 0.5 * pw) - (gx + 0.5 * gw)
                dy = (py + 0.5 * ph) - (gy + 0.5 * gh)
                cles.append(math.hypot(dx, dy))
                ncles.append(math.hypot(dx / max(1.0, gw), dy / max(1.0, gh)))
        if not ious or n_total == 0:
            continue
        i_arr = np.asarray(ious, dtype=np.float64)
        c_arr = np.asarray(cles, dtype=np.float64)
        n_arr = np.asarray(ncles, dtype=np.float64)

        auc_ge = float(np.mean([(i_arr > 0.0).mean() if t == 0.0 else (i_arr >= t).mean() for t in iou_ths]))
        auc_gt = float(np.mean([(i_arr > t).mean() for t in iou_ths]))
        p20_vis = float((c_arr <= 20.0).mean())
        pnorm_vis = float(np.mean([(n_arr <= t).mean() for t in norm_ths]))

        denom = float(n_total)
        auc_mat = float(np.mean([float((i_arr > t).sum()) / denom for t in iou_ths]))
        p20_mat = float((c_arr <= 20.0).sum()) / denom
        pnorm_mat = float(np.mean([float((n_arr <= t).sum()) / denom for t in norm_ths]))

        vis_ge_aucs.append(auc_ge)
        vis_gt_aucs.append(auc_gt)
        mat_aucs.append(auc_mat)
        vis_p20s.append(p20_vis)
        mat_p20s.append(p20_mat)
        vis_pnorms.append(pnorm_vis)
        mat_pnorms.append(pnorm_mat)
        by_video[name] = {
            "total_frames": n_total,
            "visible_frames": len(ious),
            "auc": auc_ge,
            "auc_vis_strict": auc_gt,
            "auc_matlab_conf": auc_mat,
            "p20": p20_vis,
            "p20_matlab_conf": p20_mat,
            "p_norm": pnorm_vis,
            "p_norm_matlab_conf": pnorm_mat,
        }
    return {
        "auc": float(np.mean(vis_ge_aucs)) if vis_ge_aucs else 0.0,
        "p_norm": float(np.mean(vis_pnorms)) if vis_pnorms else 0.0,
        "p20": float(np.mean(vis_p20s)) if vis_p20s else 0.0,
        "vis_strict": {
            "auc": float(np.mean(vis_gt_aucs)) if vis_gt_aucs else 0.0,
            "p_norm": float(np.mean(vis_pnorms)) if vis_pnorms else 0.0,
            "p20": float(np.mean(vis_p20s)) if vis_p20s else 0.0,
        },
        "matlab_conf": {
            "protocol": "HengLan/LaSOT_Evaluation_Toolkit eval_tracker.m (strict >, total-frame denominator)",
            "auc": float(np.mean(mat_aucs)) if mat_aucs else 0.0,
            "p_norm": float(np.mean(mat_pnorms)) if mat_pnorms else 0.0,
            "p20": float(np.mean(mat_p20s)) if mat_p20s else 0.0,
        },
        "by_video": by_video,
    }


def _independent_vot_toolkit(videos_toolkit_data: dict, toolkit_path: Path | None = None, resolution: int = 100) -> dict:
    if not videos_toolkit_data:
        return {"available": False, "reason": "empty input"}
    if toolkit_path is not None and Path(toolkit_path).exists():
        tp = str(Path(toolkit_path).resolve())
        if tp not in sys.path:
            sys.path.insert(0, tp)
    try:
        import vot
        import vot.analysis.longterm as vot_lt_mod
        from vot.analysis.longterm import compute_tpr_curves, determine_thresholds
        from vot.region import Rectangle, Special
    except Exception as exc:
        return {"available": False, "reason": f"vot-toolkit unavailable: {exc}"}

    class _Seq:
        def __init__(self, labels, size):
            self._gt = [Rectangle(*b) if b is not None else Special(0) for b in labels]
            self.size = size

        def groundtruth(self):
            return self._gt

    data = []
    for _name, (labels, boxes, confs, size) in videos_toolkit_data.items():
        traj = [Special(1)] + [Rectangle(*b) if b is not None else Special(0) for b in boxes[1:]]
        scores = [float("nan")] + [
            (float(c) if (c is not None and math.isfinite(float(c))) else 0.5) if b is not None else float("nan")
            for b, c in zip(boxes[1:], confs[1:], strict=True)
        ]
        data.append((traj, scores, _Seq(labels, size)))

    ths = determine_thresholds([s for _, sc, _ in data for s in sc], resolution)
    reps = [compute_tpr_curves(tr, sc, sq, ths, bounded=True) for tr, sc, sq in data]
    arr = np.asarray(reps, dtype=np.float64)
    pr = arr[:, 0, :].mean(axis=0)
    re = arr[:, 1, :].mean(axis=0)
    fsc = np.divide(2.0 * pr * re, pr + re, out=np.zeros_like(pr), where=(pr + re) > 0)
    bi = int(np.argmax(fsc))
    best_th = ths[bi]
    return {
        "available": True,
        "toolkit_version": getattr(vot, "__version__", "unknown"),
        "longterm_py_path": str(Path(vot_lt_mod.__file__).resolve()),
        "longterm_py_sha256": digest(Path(vot_lt_mod.__file__)),
        "resolution": int(resolution),
        "f_score": float(fsc[bi]),
        "precision": float(pr[bi]),
        "recall": float(re[bi]),
        "best_threshold": float(best_th) if math.isfinite(best_th) else str(best_th),
    }


def _spec_image_size(spec: dict, op: dict | None = None) -> tuple[int, int] | None:
    if op and op.get("width") and op.get("height"):
        return int(op["width"]), int(op["height"])
    try:
        if spec["storage"] == "zip":
            with ZipFile(spec["archive_path"]) as z:
                members = sorted(m for m in z.namelist() if m.lower().endswith(spec["suffix"]))
                with Image.open(BytesIO(z.read(members[0]))) as img:
                    return img.size
        else:
            with Image.open(spec["frame_paths"][0]) as img:
                return img.size
    except Exception:
        return None


def paired_deltas(base_videos: dict, trial_videos: dict) -> dict:
    gained = []
    lost = []
    for name in base_videos:
        b_ev = base_videos[name]["windows"]["0.5"]
        t_ev = trial_videos[name]["windows"]["0.5"]
        for be, te in zip(b_ev, t_ev, strict=True):
            if be["t0"] != te["t0"]:
                raise ValueError(f"Event alignment mismatch in {name}")
            if te["correct_recovered"] and not be["correct_recovered"]:
                gained.append([name, te["t0"]])
            elif be["correct_recovered"] and not te["correct_recovered"]:
                lost.append([name, te["t0"]])
    return {"gained_events": gained, "lost_events": lost}


def audit_run(
    dataset: str,
    data_root: Path,
    run_dir: Path,
    vot_toolkit_path: Path | None = None,
) -> dict:
    freeze_path = run_dir / "freeze.json"
    complete_path = run_dir / "complete.json"
    results_path = run_dir / "results.json"

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    if complete["freeze_sha256"] != digest(freeze_path):
        raise ValueError("complete.json freeze_sha256 mismatch")
    if results["freeze_sha256"] != digest(freeze_path):
        raise ValueError("results.json freeze_sha256 mismatch")
    if results["complete_sha256"] != digest(complete_path):
        raise ValueError("results.json complete_sha256 mismatch")
    if freeze.get("post_initialization_label_reads") != 0:
        raise ValueError("Non-zero post_initialization_label_reads in freeze.json")
    if complete.get("post_initialization_label_reads_before_scoring", 0) != 0:
        raise ValueError("Non-zero post_initialization_label_reads_before_scoring in complete.json")

    clip_names = set(freeze["clips"].keys())
    specs = load_dataset_spec(dataset, data_root, clip_names)
    if {s["name"] for s in specs} != clip_names:
        raise ValueError("Clip set mismatch between freeze and dataset")

    arms = freeze["arms"]
    recomputed_videos = {a: {} for a in arms}
    recomputed_prf_inputs = {a: {} for a in arms}
    recomputed_toolkit_inputs = {a: {} for a in arms}
    recomputed_ms = {a: [] for a in arms}
    interventions = {a: {} for a in arms}
    verified_hashes = {
        str(freeze_path): digest(freeze_path),
        str(complete_path): digest(complete_path),
        str(results_path): digest(results_path),
    }

    for spec in specs:
        name = spec["name"]
        if spec["archive_sha256"] != freeze["clips"][name]["archive_sha256"]:
            raise ValueError(f"Archive SHA mismatch for {name}")
        labels = spec["load_labels"]()
        arm_shas = {}
        for arm in arms:
            trace_path = run_dir / name / f"{arm}.jsonl"
            op = complete["operations"][name][arm]
            actual_sha = digest(trace_path)
            if actual_sha != op["trace_sha256"]:
                raise ValueError(f"Trace SHA mismatch for {name}/{arm}")
            verified_hashes[str(trace_path)] = actual_sha
            with trace_path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            if len(rows) != op["frames"]:
                raise ValueError(f"Frame count mismatch for {name}/{arm}")
            if [r["frame"] for r in rows] != list(range(len(rows))):
                raise ValueError(f"Non-contiguous frame index for {name}/{arm}")
            arm_shas[arm] = [r["sha256"] for r in rows]
            lbl_slice = labels[:len(rows)]
            boxes = [r["bbox"] for r in rows]
            confs = [r.get("confidence", 1.0 if r["bbox"] is not None else 0.0) for r in rows]
            recomputed_videos[arm][name] = assess(lbl_slice, boxes)
            recomputed_prf_inputs[arm][name] = (lbl_slice, boxes)
            size = _spec_image_size(spec, op)
            if size is not None:
                recomputed_toolkit_inputs[arm][name] = (lbl_slice, boxes, confs, size)
            recomputed_ms[arm].extend(r["synchronized_model_ms"] for r in rows[1:])
            interventions[arm][name] = {
                "effective_min_obj_score_logits": op.get("effective_min_obj_score_logits"),
                "probe_calls": int(op.get("stats", {}).get("probe_calls", 0)),
                "verified_probes": int(op.get("stats", {}).get("verified_probes", 0)),
                "memory_updates": int(sum(bool(r.get("memory_update")) for r in rows)),
                "rollbacks": int(sum(r.get("source") == "rollback" for r in rows)),
            }
        if len(arms) > 1:
            base_shas = arm_shas[arms[0]]
            for arm in arms[1:]:
                if arm_shas[arm] != base_shas:
                    raise ValueError(f"Pixel SHA stream differs between {arms[0]} and {arm} on {name}")

    audit_arms = {}
    for arm in arms:
        totals = combine(recomputed_videos[arm])
        prf, prf_by_video = _independent_macro_prf(recomputed_prf_inputs[arm])
        vot_tk = (
            _independent_vot_toolkit(recomputed_toolkit_inputs[arm], toolkit_path=vot_toolkit_path)
            if len(recomputed_toolkit_inputs[arm]) == len(recomputed_prf_inputs[arm])
            else {"available": False, "reason": "missing image dimensions"}
        )
        lasot = _independent_lasot_metrics(recomputed_prf_inputs[arm])
        ms_vals = recomputed_ms[arm]
        ms_summary = {
            "median": median(ms_vals) if ms_vals else 0.0,
            "p95": quantiles(ms_vals, n=100)[94] if len(ms_vals) >= 20 else (max(ms_vals) if ms_vals else 0.0),
            "mean": (sum(ms_vals) / len(ms_vals)) if ms_vals else 0.0,
        }
        reported = results["arms"][arm]
        for key in ("timely", "first", "fp"):
            if totals[key] != reported["totals"][key]:
                raise ValueError(f"Audit mismatch on {arm}.{key}: {totals[key]} != {reported['totals'][key]}")
        if abs(totals["visible_mean_iou"] - reported["totals"]["visible_mean_iou"]) > 1e-9:
            raise ValueError(f"Audit mismatch on {arm}.visible_mean_iou")
        if abs(prf["f1"] - reported["fixed_point_macro_prf"]["f1"]) > 1e-9:
            raise ValueError(f"Audit mismatch on {arm}.macro_f1")

        # Verify official vot-toolkit score if both computed it
        rep_vtk = reported.get("vot_lt_toolkit")
        if rep_vtk and rep_vtk.get("available") and vot_tk.get("available"):
            if abs(vot_tk["f_score"] - rep_vtk["f_score"]) > 1e-9:
                raise ValueError(f"Audit mismatch on {arm}.vot_lt_toolkit.f_score")

        # Verify LaSOT metrics (supports new `lasot_metrics` and legacy `lasot_official`)
        rep_ls = reported.get("lasot_metrics") or reported.get("lasot_official")
        if rep_ls is not None:
            if abs(lasot["auc"] - rep_ls["auc"]) > 1e-9:
                raise ValueError(f"Audit mismatch on {arm}.lasot_auc")
            if "matlab_conf" in rep_ls:
                if abs(lasot["matlab_conf"]["auc"] - rep_ls["matlab_conf"]["auc"]) > 1e-9:
                    raise ValueError(f"Audit mismatch on {arm}.lasot_matlab_conf_auc")

        audit_arms[arm] = {
            "totals": totals,
            "fixed_point_macro_prf": prf,
            "fixed_point_by_video": prf_by_video,
            "vot_lt_toolkit": vot_tk,
            "lasot_metrics": lasot,
            "interventions": interventions[arm],
            "ms": ms_summary,
            "videos": recomputed_videos[arm],
        }

    paired = None
    if "baseline" in arms and "care" in arms:
        paired = paired_deltas(recomputed_videos["baseline"], recomputed_videos["care"])
        # Also compute frame-level bbox diff count per video for transparent drift detection
        bbox_diffs = {}
        for spec in specs:
            name = spec["name"]
            with (run_dir / name / "baseline.jsonl").open(encoding="utf-8") as fb, \
                    (run_dir / name / "care.jsonl").open(encoding="utf-8") as fc:
                b_rows = [json.loads(line) for line in fb]
                c_rows = [json.loads(line) for line in fc]
            bbox_diffs[name] = sum(a["bbox"] != b["bbox"] for a, b in zip(b_rows, c_rows, strict=True))
        paired["bbox_diff_frames"] = bbox_diffs

    audit_report = {
        "status": "AUDIT_OK",
        "dataset": dataset,
        "host": freeze.get("host"),
        "backbone": freeze.get("backbone"),
        "verifier": freeze.get("verifier"),
        "script_sha256": freeze.get("script_sha256"),
        "host_provenance": freeze.get("host_provenance", {}),
        "audit_script_sha256": digest(Path(__file__)),
        "verified_sha256": verified_hashes,
        "arms": audit_arms,
        "paired": paired,
    }
    audit_path = run_dir / "independent-audit.json"
    save(audit_path, audit_report)
    return audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hoot_dev", "hoot_transfer", "lasot_ext", "vot_reserve", "vot_lt_dir"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--vot-toolkit-path", type=Path, default=None)
    args = parser.parse_args()

    report = audit_run(args.dataset, args.data, args.run, vot_toolkit_path=args.vot_toolkit_path)
    print(f"AUDIT_OK {args.run / 'independent-audit.json'} ({report['host']}/{report['backbone']}/{report['verifier']})", flush=True)
    for arm, a in report["arms"].items():
        t = a["totals"]
        prf = a["fixed_point_macro_prf"]
        vtk = a["vot_lt_toolkit"]
        ls = a["lasot_metrics"]
        ms = a["ms"]
        vtk_str = f"{vtk['f_score']:.4f}(v{vtk['toolkit_version']})" if vtk.get("available") else "N/A"
        tot_upd = sum(v["memory_updates"] for v in a["interventions"].values())
        tot_rb = sum(v["rollbacks"] for v in a["interventions"].values())
        print(
            f"  [{arm:8s}] timely={t['timely']} first={t['first']} fp={t['fp']} "
            f"vis_iou={t['visible_mean_iou']:.4f} | "
            f"MacroP={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f} | "
            f"VOT-tk={vtk_str} | LaSOT(vis)={ls['auc']:.4f} (mat_conf)={ls['matlab_conf']['auc']:.4f} | "
            f"upd={tot_upd} rb={tot_rb} | med={ms['median']:.1f}ms",
            flush=True,
        )
    if report["paired"]:
        print(f"  paired={json.dumps(report['paired'])}", flush=True)


if __name__ == "__main__":
    main()
