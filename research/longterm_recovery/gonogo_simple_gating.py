"""Go/No-Go Experiment 1: Closed-Loop 5-Line Absent-Frame Memory Gating vs Vanilla SAM 2.1 vs CARE.

Tests the critical novelty / confounder question on identical Meta official SAM 2.1 Tiny
(same weights, same Meta facebookresearch/sam2 code, same FP16 autocast, same allow_all SDPA,
same deterministic seeds, same official vot-toolkit 0.9.0 & dual-protocol LaSOT evaluator):

Arms:
  - vanilla                 : unmodified Meta SAM 2.1 Tiny
  - gate_skip_absent        : 5-line absent-frame exclusion — skips obj<=0 frames from BOTH
                              spatial memory (last 6 visible) and pointers (last 15 visible).
                              Zero Kalman filter, zero DRM, zero CARE probe/verifier/rollback.
  - gate_skip_absent_damptr : same 6 visible spatial slots + DAM4SAM's local-window (1..15)
                              visible-only pointers (falls back to cond_0 ptr after >=15 silent).
  - gate_skip_absent_keept1 : SAMURAI-style spatial selection (5 visible + forced t-1) with
                              unfiltered recent 15 pointers (NO Kalman filter, NO min_obj change).
  - care_native             : Meta SAM 2.1 Tiny + frozen CAREController (sam2_native verifier)
  - care_none               : Meta SAM 2.1 Tiny + frozen CAREController (none verifier)

Supports datasets: hoot_dev, hoot_transfer, lasot_ext, vot_lt22_journal.
Strictly causal: zero post-initialization label reads before all arms finish tracking.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
from statistics import median, quantiles
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from audit_care_sam2_plugin import (
    _independent_lasot_metrics,
    _independent_macro_prf,
    _independent_vot_toolkit,
    paired_deltas,
)
from care_sam2_plugin import (
    CAREController,
    DEFAULT_MEMORY_KEEP_WINDOW,
    SAM21HostAdapter,
    _clone_mem_entry,
    _iter_spec_frames,
    _parse_vot_gt_line,
    _scalar,
    _sigmoid,
    build_verifier,
    load_dataset_spec,
)
from gonogo_a6_decomposed import _patch_memory_prep
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_paired_query_continuous import combine


ALL_ARMS = (
    "vanilla",
    "gate_skip_absent",
    "gate_skip_absent_damptr",
    "gate_skip_absent_keept1",
    "care_native",
    "care_none",
)


def load_specs_extended(dataset: str, data_root: Path, clips_filter: set[str] | None = None) -> list[dict]:
    if dataset != "vot_lt22_journal":
        return load_dataset_spec(dataset, data_root, clips_filter=clips_filter)
    data_root = Path(data_root)
    plan = json.loads((data_root / "plan.json").read_text(encoding="utf-8"))
    specs = []
    for name, meta in plan["videos"].items():
        if clips_filter and name not in clips_filter:
            continue
        img_zip = data_root / "assets" / f"{name}-rgb.zip"
        ann_zip = data_root / "assets" / f"{name}-annotations.zip"
        length = int(meta["length"])
        reads = {"count": 0}

        def _make_labels(az=ann_zip, ln=length, rc=reads):
            rc["count"] += 1
            with ZipFile(az) as z:
                gt_name = [n for n in z.namelist() if n.endswith("groundtruth.txt")][0]
                lines = z.read(gt_name).decode("utf-8").splitlines()
            if len(lines) != ln:
                raise ValueError(f"VOT-LT22 length mismatch: {az} ({len(lines)} vs {ln})")
            return [_parse_vot_gt_line(line) for line in lines]

        specs.append({
            "name": name,
            "storage": "zip",
            "archive_path": img_zip,
            "archive_sha256": meta["image_sha256"],
            "length": length,
            "init_box": [float(v) for v in meta["initial_box"]],
            "suffix": ".jpg",
            "load_labels": _make_labels,
            "label_reads": reads,
        })
    return specs


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_arm_on_clip(
    adapter: SAM21HostAdapter,
    spec: dict,
    arm: str,
    trace_path: Path,
) -> dict:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    care: CAREController | None = None
    if arm == "care_native":
        care = CAREController(verifier=build_verifier("sam2_native"))
    elif arm == "care_none":
        care = CAREController(verifier=build_verifier("none"))

    stats: Counter = Counter()
    sources: Counter = Counter()
    step_ms: list[float] = []
    vis_spatial_ring: list[tuple[int, dict]] = []
    vis_ptr_ring: list[tuple[int, torch.Tensor]] = []
    init_box = spec["init_box"]
    width, height = 0, 0

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as fh, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for frame_idx, _, encoded in _iter_spec_frames(spec, None):
            frame_sha = sha256(encoded).hexdigest()
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
                rgb_np = np.asarray(pil_rgb)
            height, width = rgb_np.shape[:2]

            _sync_cuda()
            t0 = perf_counter()

            if frame_idx == 0:
                init_res = adapter.initialize(pil_rgb, init_box)
                if care is not None:
                    care.initialize(rgb_np, init_box, init_res["pred_mask"], adapter)
                _sync_cuda()
                dt_ms = (perf_counter() - t0) * 1000.0
                row = {
                    "frame": 0,
                    "sha256": frame_sha,
                    "bbox": init_box,
                    "native_bbox": init_res["pred_bbox"],
                    "object_score": None,
                    "confidence": 1.0,
                    "source": "init",
                    "memory_update": False,
                    "synchronized_model_ms": dt_ms,
                    "diag": {},
                }
                sources["init"] += 1
                fh.write(json.dumps(row) + "\n")
                continue

            # Track step with arm-specific memory conditioning
            if arm == "vanilla" or care is not None:
                res = adapter.track(pil_rgb)
            elif arm == "gate_skip_absent":
                sp_list = [entry for _, entry in vis_spatial_ring[-6:]]
                pt_list = [(frame_idx - t_v, ptr) for (t_v, ptr) in reversed(vis_ptr_ring[-15:])]
                with _patch_memory_prep(
                    adapter.predictor,
                    visible_spatial_list=sp_list,
                    visible_ptr_list=pt_list,
                ):
                    res = adapter.track(pil_rgb)
            elif arm == "gate_skip_absent_damptr":
                sp_list = [entry for _, entry in vis_spatial_ring[-6:]]
                max_ptrs = int(adapter.predictor.max_obj_ptrs_in_encoder)
                pt_list = [
                    (frame_idx - t_v, ptr)
                    for (t_v, ptr) in reversed(vis_ptr_ring)
                    if 1 <= (frame_idx - t_v) < max_ptrs
                ]
                with _patch_memory_prep(
                    adapter.predictor,
                    visible_spatial_list=sp_list,
                    visible_ptr_list=pt_list,
                ):
                    res = adapter.track(pil_rgb)
            elif arm == "gate_skip_absent_keept1":
                prev_slot = adapter.get_non_cond_slot(frame_idx - 1)
                older_vis = [entry for (t_v, entry) in vis_spatial_ring if t_v < frame_idx - 1][-5:]
                sp_list = older_vis + ([prev_slot] if prev_slot is not None else [])
                # Keep pointers unfiltered exactly like SAMURAI lines 762-770
                with _patch_memory_prep(
                    adapter.predictor,
                    visible_spatial_list=sp_list,
                    skip_non_cond_ptrs=False,
                ):
                    res = adapter.track(pil_rgb)
            else:
                raise ValueError(f"Unknown arm: {arm}")

            native_bbox = res["pred_bbox"]
            native_mask = res["pred_mask"]
            obj_score = res["object_score"]

            if care is not None:
                bbox, source, mem_upd, diag = care.step(
                    frame_idx, rgb_np, adapter, native_bbox, native_mask, obj_score, stats
                )
                if source == "recovered_care":
                    obj_out = diag.get("obj_probe", obj_score)
                elif source == "rollback":
                    obj_out = diag.get("restored_obj", obj_score)
                else:
                    obj_out = obj_score
            else:
                bbox = native_bbox
                source = "baseline"
                mem_upd = False
                diag = {}
                obj_out = obj_score
                # Update visible ring for gating arms (strictly causal: uses only current predicted obj_score > 0)
                if obj_score is not None and obj_score > 0.0 and native_bbox is not None:
                    slot = adapter.get_non_cond_slot(frame_idx)
                    if slot is not None:
                        cloned = _clone_mem_entry(slot)
                        vis_spatial_ring.append((frame_idx, cloned))
                        if len(vis_spatial_ring) > 6:
                            vis_spatial_ring.pop(0)
                        if cloned.get("obj_ptr") is not None:
                            vis_ptr_ring.append((frame_idx, cloned["obj_ptr"]))
                            if len(vis_ptr_ring) > 15:
                                vis_ptr_ring.pop(0)

            _sync_cuda()
            dt_ms = (perf_counter() - t0) * 1000.0
            step_ms.append(dt_ms)
            sources[source] += 1
            conf = _sigmoid(obj_out) if (bbox is not None and obj_out is not None) else (0.0 if bbox is None else 0.5)
            row = {
                "frame": frame_idx,
                "sha256": frame_sha,
                "bbox": bbox,
                "native_bbox": native_bbox,
                "object_score": obj_out,
                "confidence": round(float(conf), 6),
                "source": source,
                "memory_update": bool(mem_upd),
                "synchronized_model_ms": dt_ms,
                "diag": diag,
            }
            fh.write(json.dumps(row) + "\n")
            adapter.prune_old_memory(DEFAULT_MEMORY_KEEP_WINDOW)

    peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    adapter.release_clip()
    torch.cuda.empty_cache()

    return {
        "frames": spec["length"],
        "width": width,
        "height": height,
        "trace_sha256": digest(trace_path),
        "sources": dict(sources),
        "stats": dict(stats),
        "median_ms": median(step_ms) if step_ms else 0.0,
        "p95_ms": quantiles(step_ms, n=20)[18] if len(step_ms) >= 20 else (max(step_ms) if step_ms else 0.0),
        "mean_ms": sum(step_ms) / len(step_ms) if step_ms else 0.0,
        "peak_cuda_bytes": peak_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hoot_dev", "hoot_transfer", "lasot_ext", "vot_lt22_journal"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--deps", type=Path, nargs="*", default=None)
    parser.add_argument("--vot-toolkit-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False

    clips_filter = set(args.clips) if args.clips else None
    specs = load_specs_extended(args.dataset, args.data, clips_filter=clips_filter)
    args.output.mkdir(parents=True, exist_ok=True)

    adapter = SAM21HostAdapter(
        weights=args.weights,
        backbone="tiny",
        sam2_root=args.host_source,
        deps_dirs=[Path(d) for d in (args.deps or [])],
        use_sentry_runtime=False,
        device=args.device,
    )
    tmod = sys.modules.get("sam2.modeling.sam.transformer")
    if tmod is not None and hasattr(tmod, "ALLOW_ALL_KERNELS"):
        tmod.ALLOW_ALL_KERNELS = True

    freeze = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(args.weights),
        "host_provenance": adapter.provenance,
        "arms": list(args.arms),
        "clips": {s["name"]: {"length": s["length"], "archive_sha256": s["archive_sha256"]} for s in specs},
        "post_initialization_label_reads": 0,
    }
    save(args.output / "freeze.json", freeze)

    ops: dict[str, dict[str, dict]] = {a: {} for a in args.arms}
    for spec in specs:
        nm = spec["name"]
        for arm in args.arms:
            tp = args.output / nm / f"{arm}.jsonl"
            print(f"[{args.dataset}] {nm} ({spec['length']}f) :: arm={arm}", flush=True)
            info = run_arm_on_clip(adapter, spec, arm, tp)
            ops[arm][nm] = info
            print(
                f"    done: med={info['median_ms']:.1f}ms, peak={info['peak_cuda_bytes']/1048576:.0f}MiB, "
                f"stats={info['stats']}",
                flush=True,
            )

    assert sum(s["label_reads"]["count"] for s in specs) == 0, "Causality violated!"

    # Strictly post-hoc independent scoring
    labels_by_vid = {s["name"]: s["load_labels"]() for s in specs}
    arm_reports = {}
    for arm in args.arms:
        videos_data = {}
        toolkit_data = {}
        ev_by_vid = {}
        all_ms = []
        interventions = {}
        for spec in specs:
            nm = spec["name"]
            tp = args.output / nm / f"{arm}.jsonl"
            rows = [json.loads(l) for l in tp.open(encoding="utf-8")]
            boxes = [r["bbox"] for r in rows]
            confs = [r["confidence"] for r in rows]
            labels = labels_by_vid[nm]
            videos_data[nm] = (labels, boxes)
            toolkit_data[nm] = (labels, boxes, confs, (ops[arm][nm]["width"], ops[arm][nm]["height"]))
            ev_by_vid[nm] = assess(labels, boxes)
            all_ms.extend(r["synchronized_model_ms"] for r in rows[1:])
            interventions[nm] = {
                "probe_calls": ops[arm][nm]["stats"].get("probe_calls", 0),
                "verified_probes": ops[arm][nm]["stats"].get("verified_probes", 0),
                "memory_updates": sum(bool(r["memory_update"]) for r in rows),
                "rollbacks": sum(r["source"] == "rollback" for r in rows),
                "silent_frames": sum(r["bbox"] is None for r in rows[1:]),
            }

        totals = combine(ev_by_vid)
        macro_prf, fp_by_vid = _independent_macro_prf(videos_data)
        vtk = _independent_vot_toolkit(toolkit_data, toolkit_path=args.vot_toolkit_path)
        ls = _independent_lasot_metrics(videos_data)
        arm_reports[arm] = {
            "totals": totals,
            "fixed_point_macro_prf": macro_prf,
            "fixed_point_by_video": fp_by_vid,
            "vot_lt_toolkit": vtk,
            "lasot_metrics": ls,
            "interventions": interventions,
            "ms": {
                "median": median(all_ms) if all_ms else 0.0,
                "p95": quantiles(all_ms, n=20)[18] if len(all_ms) >= 20 else 0.0,
                "mean": sum(all_ms) / len(all_ms) if all_ms else 0.0,
            },
            "videos": {
                nm: {
                    "visible": ev_by_vid[nm]["visible"],
                    "absent": ev_by_vid[nm]["absent"],
                    "vis_iou": round(ev_by_vid[nm]["visible_iou_sum"] / max(1, ev_by_vid[nm]["visible"]), 4),
                    "fp": ev_by_vid[nm]["false_reports"],
                    "timely": sum(1 for e in ev_by_vid[nm]["windows"]["0.5"] if e.get("eligible") and e.get("correct_recovered")),
                    "eligible": sum(1 for e in ev_by_vid[nm]["windows"]["0.5"] if e.get("eligible")),
                    "first": sum(1 for e in ev_by_vid[nm]["windows"]["0.5"] if e.get("eligible") and e.get("category") == "correct_first"),
                }
                for nm in ev_by_vid
            },
        }

    # Paired deltas vs vanilla (if vanilla was run)
    paired_vs_vanilla = {}
    if "vanilla" in arm_reports:
        base_ev = {
            s["name"]: assess(labels_by_vid[s["name"]], [json.loads(l)["bbox"] for l in (args.output / s["name"] / "vanilla.jsonl").open(encoding="utf-8")])
            for s in specs
        }
        for arm in args.arms:
            if arm == "vanilla":
                continue
            tr_ev = {
                s["name"]: assess(labels_by_vid[s["name"]], [json.loads(l)["bbox"] for l in (args.output / s["name"] / f"{arm}.jsonl").open(encoding="utf-8")])
                for s in specs
            }
            paired_vs_vanilla[arm] = paired_deltas(base_ev, tr_ev)

    results = {
        "freeze_sha256": digest(args.output / "freeze.json"),
        "dataset": args.dataset,
        "arms": arm_reports,
        "paired_vs_vanilla": paired_vs_vanilla,
        "ops": ops,
    }
    save(args.output / "results.json", results)

    print(f"\n================ SIMPLE GATING SUMMARY ({args.dataset}) ================")
    print(f"{'Arm':26s} | {'Timely':>6s} | {'First':>5s} | {'FP':>5s} | {'VisIoU':>6s} | {'MacP':>6s} | {'MacR':>6s} | {'MacF1':>6s} | {'VOTtk':>6s} | {'AUC_m':>6s} | {'MedMs':>6s}")
    print("-" * 115)
    for arm in args.arms:
        a = arm_reports[arm]
        t = a["totals"]
        p = a["fixed_point_macro_prf"]
        v = a["vot_lt_toolkit"]["f_score"] if a["vot_lt_toolkit"].get("available") else 0.0
        au = a["lasot_metrics"]["matlab_conf"]["auc"]
        print(
            f"{arm:26s} | {t['timely']:6d} | {t['first']:5d} | {t['fp']:5d} | {t['visible_mean_iou']:6.4f} | "
            f"{p['precision']:6.4f} | {p['recall']:6.4f} | {p['f1']:6.4f} | {v:6.4f} | {au:6.4f} | {a['ms']['median']:6.1f}"
        )
    print("\nPer-Video Breakdown (VisIoU / FP / Timely):")
    for spec in specs:
        nm = spec["name"]
        print(f"  -- {nm} --")
        for arm in args.arms:
            v = arm_reports[arm]["videos"][nm]
            iv = arm_reports[arm]["interventions"][nm]
            print(
                f"    {arm:26s}: VisIoU={v['vis_iou']:.4f}, FP={v['fp']:3d}/{v['absent']:3d}, "
                f"Timely={v['timely']}/{v['eligible']}, First={v['first']}/{v['eligible']}, "
                f"silent={iv['silent_frames']:4d}, upd={iv['memory_updates']}, rb={iv['rollbacks']}"
            )
    print("DONE:", args.output)


if __name__ == "__main__":
    main()
