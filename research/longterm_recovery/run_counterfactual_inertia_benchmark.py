"""Sharded, Resumable Cloud Runner for Full-Benchmark 9-Arm Counterfactual Memory-Inertia Probe.

Scales the 9-arm single-factor counterfactual memory decomposition (`gonogo_a6_decomposed.py`)
to complete official benchmarks (`vot_lt_dir` 50 seqs / `lasot_ext` 150 seqs / `hoot_dev` /
`vot_lt22_journal`) across `tiny`, `base_plus`, or `large` backbones with 4-shard multi-process
parallelism, crash-safe `--resume`, and `--merge-shards`.

Reports BOTH:
  - Pooled frame statistics across all silent visible & silent absent frames;
  - Video-level macro-averaged statistics (weighting each affected video equally so long videos
    cannot dominate correlated frame counts);
  - Per-qualifying-event recovery statistics across all qualifying reappearance windows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from care_sam2_plugin import (
    DEFAULT_CLEAN_ANCHOR_OBJ,
    DEFAULT_CLEAN_ANCHOR_STRIDE,
    DEFAULT_MEMORY_KEEP_WINDOW,
    DEFAULT_RHO_MAX,
    DEFAULT_RHO_MIN,
    DEFAULT_SDPA_KERNEL_MODE,
    SAM21HostAdapter,
    SAM2NativeVerifier,
    _clone_mem_entry,
    _iter_spec_frames,
    _scalar,
)
from evaluation import box_iou
from gonogo_a6_decomposed import (
    ARMS,
    _eval_arm,
    _verify_strip_tensor_exactness,
)
from gonogo_simple_gating import load_specs_extended
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_scale_consistent_recovery import BORDER_MARGIN, border_gamma


def _run_probe_clip(
    adapter: SAM21HostAdapter,
    spec: dict,
    out_dir: Path,
    silent_stride: int = 1,
    keep_window: int = DEFAULT_MEMORY_KEEP_WINDOW,
    max_frames: int | None = None,
    resume: bool = False,
) -> dict:
    name = spec["name"]
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    trace_path = clip_dir / "probe.jsonl"
    status_path = clip_dir / "probe.status.json"

    expected_len = min(spec["length"], max_frames) if max_frames else spec["length"]
    if resume and trace_path.exists() and status_path.exists():
        try:
            op = json.loads(status_path.read_text(encoding="utf-8"))
            if op.get("frames") == expected_len and op.get("trace_sha256") == digest(trace_path):
                print(f"  [RESUME] probe {name} ({expected_len}f) already verified.", flush=True)
                return op
        except Exception:
            pass

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()

    verifier = SAM2NativeVerifier()
    init_box = spec["init_box"]
    a0 = max(1.0, init_box[2] * init_box[3])
    last_clean_out = None
    last_clean_frame = 0
    vis_spatial_ring: list[dict] = []
    vis_ptr_ring: list[tuple[int, torch.Tensor]] = []
    silent_streak = 0
    silent_evals = 0
    tensor_sanity = None
    mirror_sanity = None

    tmp_path = trace_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for frame_idx, _, encoded in _iter_spec_frames(spec, max_frames):
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
                rgb_np = np.asarray(pil_rgb)
            h, w = rgb_np.shape[:2]

            if frame_idx == 0:
                init_res = adapter.initialize(pil_rgb, init_box)
                verifier.initialize(rgb_np, init_box, init_res["pred_mask"], adapter.get_cached_feature_map(), w, h)
                fh.write(json.dumps({"frame": 0, "native_silent": False, "silent_streak": 0, "native_bbox": init_box, "native_obj": None}) + "\n")
                continue

            res = adapter.track(pil_rgb)
            native_bbox = res["pred_bbox"]
            native_mask = res["pred_mask"]
            native_obj = res["object_score"]

            if native_bbox is not None:
                silent_streak = 0
                slot = adapter.get_non_cond_slot(frame_idx)
                if slot is not None:
                    if native_obj is not None and native_obj > 0.0:
                        cloned = _clone_mem_entry(slot)
                        vis_spatial_ring.append(cloned)
                        if len(vis_spatial_ring) > 6:
                            vis_spatial_ring.pop(0)
                        if cloned.get("obj_ptr") is not None:
                            vis_ptr_ring.append((frame_idx, cloned["obj_ptr"]))
                            if len(vis_ptr_ring) > 15:
                                vis_ptr_ring.pop(0)
                    if (
                        native_obj is not None
                        and native_obj >= DEFAULT_CLEAN_ANCHOR_OBJ
                        and (last_clean_out is None or (frame_idx - last_clean_frame) >= DEFAULT_CLEAN_ANCHOR_STRIDE)
                    ):
                        gamma = border_gamma(native_bbox, init_box, w, h, margin=BORDER_MARGIN)
                        rho_tilde = ((native_bbox[2] * native_bbox[3]) / a0) / gamma
                        if DEFAULT_RHO_MIN <= rho_tilde <= DEFAULT_RHO_MAX:
                            if verifier.verify_anchor(rgb_np, native_bbox, native_mask, adapter.get_cached_feature_map(), w, h):
                                last_clean_out = _clone_mem_entry(slot)
                                last_clean_frame = frame_idx
                fh.write(json.dumps({
                    "frame": frame_idx,
                    "native_silent": False,
                    "silent_streak": 0,
                    "native_bbox": native_bbox,
                    "native_obj": native_obj,
                }) + "\n")
            else:
                silent_streak += 1
                do_eval = (silent_stride <= 1) or ((silent_streak - 1) % silent_stride == 0)
                if not do_eval:
                    fh.write(json.dumps({
                        "frame": frame_idx,
                        "native_silent": True,
                        "evaluated": False,
                        "silent_streak": silent_streak,
                        "native_bbox": None,
                        "native_obj": native_obj,
                    }) + "\n")
                else:
                    silent_evals += 1
                    if tensor_sanity is None and silent_streak >= 2:
                        tensor_sanity = _verify_strip_tensor_exactness(adapter, frame_idx)

                    obj_idx = adapter.state["obj_id_to_idx"][0]
                    obj_dict = adapter.state["output_dict_per_obj"][obj_idx]
                    cond_outputs = obj_dict["cond_frame_outputs"]
                    non_cond = obj_dict["non_cond_frame_outputs"]
                    cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
                    num_spatial = int(adapter.predictor.num_maskmem) - 1
                    num_ptrs = int(adapter.predictor.max_obj_ptrs_in_encoder) - 1
                    spatial_frames = [k for k in range(max(1, frame_idx - num_spatial), frame_idx) if k in non_cond]
                    ptr_frames = [k for k in range(max(1, frame_idx - num_ptrs), frame_idx) if k in non_cond]

                    if mirror_sanity is None:
                        ref_o = _eval_arm(adapter, frame_idx, {"cond_frame_outputs": cond_outputs, "non_cond_frame_outputs": non_cond})
                        mir_o = _eval_arm(
                            adapter,
                            frame_idx,
                            {"cond_frame_outputs": cond_outputs, "non_cond_frame_outputs": non_cond},
                            patch_kwargs={"skip_non_cond_spatial": False, "skip_non_cond_ptrs": False},
                        )
                        mirror_sanity = {
                            "frame_idx": frame_idx,
                            "video": name,
                            "obj_abs_diff": abs((ref_o["obj_logit"] or 0.0) - (mir_o["obj_logit"] or 0.0)),
                            "bbox_equal": ref_o["bbox"] == mir_o["bbox"],
                        }

                    no_obj_embed = adapter.predictor.no_obj_embed_spatial
                    strip_non_cond = {}
                    for k in sorted(set(spatial_frames) | set(ptr_frames)):
                        orig = non_cond[k]
                        s = _scalar(orig.get("object_score_logits"))
                        entry = dict(orig)
                        if (s is None or s <= 0.0) and no_obj_embed is not None and entry.get("maskmem_features") is not None:
                            feats = entry["maskmem_features"]
                            delta = no_obj_embed.to(device=feats.device, dtype=feats.dtype)[..., None, None].expand(*feats.shape)
                            entry["maskmem_features"] = feats - delta
                        strip_non_cond[k] = entry

                    prev1_slot = non_cond.get(frame_idx - 1)
                    live_dict = {"cond_frame_outputs": cond_outputs, "non_cond_frame_outputs": non_cond}
                    cond0_only_dict = {"cond_frame_outputs": cond_0, "non_cond_frame_outputs": {}}
                    vis_ptrs_tdiff = [(frame_idx - t_v, ptr) for (t_v, ptr) in reversed(vis_ptr_ring)]

                    arm_specs = {
                        "orig_memory": (live_dict, None),
                        "strip_no_obj_spatial_only": ({"cond_frame_outputs": cond_0, "non_cond_frame_outputs": strip_non_cond}, None),
                        "drop_recent_spatial_keep_ptrs": (live_dict, {"skip_non_cond_spatial": True, "skip_non_cond_ptrs": False}),
                        "drop_recent_ptrs_keep_spatial": (live_dict, {"skip_non_cond_spatial": False, "skip_non_cond_ptrs": True}),
                        "m0_only": (cond0_only_dict, None),
                        "m0_plus_1empty": (
                            {"cond_frame_outputs": cond_0, "non_cond_frame_outputs": {frame_idx - 1: prev1_slot} if prev1_slot is not None else {}},
                            None,
                        ),
                        "m0_plus_mclean": (
                            {"cond_frame_outputs": cond_0, "non_cond_frame_outputs": {frame_idx - 1: last_clean_out} if last_clean_out is not None else {}},
                            None,
                        ),
                        "m0_plus_6vis_condptr": (
                            cond0_only_dict,
                            {"visible_spatial_list": list(vis_spatial_ring), "skip_non_cond_ptrs": True},
                        ),
                        "m0_plus_6vis_15visptr": (
                            cond0_only_dict,
                            {"visible_spatial_list": list(vis_spatial_ring), "visible_ptr_list": vis_ptrs_tdiff},
                        ),
                    }
                    arm_res = {a: _eval_arm(adapter, frame_idx, md, pk) for a, (md, pk) in arm_specs.items()}
                    fh.write(json.dumps({
                        "frame": frame_idx,
                        "native_silent": True,
                        "evaluated": True,
                        "silent_streak": silent_streak,
                        "native_bbox": None,
                        "native_obj": native_obj,
                        "arms": arm_res,
                    }) + "\n")

            adapter.prune_old_memory(keep_window)

    tmp_path.replace(trace_path)
    adapter.release_clip()
    torch.cuda.empty_cache()

    op = {
        "frames": expected_len,
        "silent_evals": silent_evals,
        "trace_sha256": digest(trace_path),
        "tensor_sanity": tensor_sanity,
        "mirror_sanity": mirror_sanity,
    }
    status_path.write_text(json.dumps(op, indent=2) + "\n", encoding="utf-8")
    print(f"  [probe {name}] {expected_len}f, silent_evals={silent_evals}", flush=True)
    return op


def _merge_and_score_probe(dataset: str, all_specs: list[dict], out_dir: Path, operations: dict) -> dict:
    freeze_path = out_dir / "freeze.json"
    complete_path = out_dir / "complete.json"
    complete_path.write_text(
        json.dumps({
            "freeze_sha256": digest(freeze_path),
            "post_initialization_label_reads_before_scoring": 0,
            "operations": operations,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    def _evis():
        return {a: {"n": 0, "pos_obj": 0, "iou_ge_05": 0, "obj_sum": 0.0, "iou_sum": 0.0} for a in ARMS}

    def _eabs():
        return {a: {"n": 0, "pos_obj": 0, "obj_sum": 0.0} for a in ARMS}

    pooled_vis = _evis()
    pooled_abs = _eabs()
    by_video_vis = {}
    by_video_abs = {}
    event_summaries = []
    verified_sha = {str(freeze_path): digest(freeze_path), str(complete_path): digest(complete_path)}

    for spec in all_specs:
        nm = spec["name"]
        tpath = out_dir / nm / "probe.jsonl"
        actual_sha = digest(tpath)
        if actual_sha != operations[nm]["trace_sha256"]:
            raise ValueError(f"Probe trace SHA mismatch for {nm}")
        verified_sha[str(tpath)] = actual_sha

        with tpath.open(encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh]
        labels = spec["load_labels"]()[: len(rows)]
        native_boxes = [r["native_bbox"] for r in rows]
        ev_info = assess(labels, native_boxes)["windows"]["0.5"]

        vv = _evis()
        va = _eabs()
        for r in rows:
            if not (r.get("native_silent") and r.get("evaluated", True) and "arms" in r):
                continue
            gt = labels[r["frame"]]
            for a in ARMS:
                ao = r["arms"][a]
                obj = ao["obj_logit"]
                if gt is not None:
                    iou = box_iou(ao["bbox"], gt) if ao["bbox"] is not None else 0.0
                    for d in (pooled_vis[a], vv[a]):
                        d["n"] += 1
                        d["pos_obj"] += int(obj is not None and obj > 0.0)
                        d["iou_ge_05"] += int(iou >= 0.50)
                        d["obj_sum"] += float(obj or 0.0)
                        d["iou_sum"] += iou
                else:
                    for d in (pooled_abs[a], va[a]):
                        d["n"] += 1
                        d["pos_obj"] += int(obj is not None and obj > 0.0)
                        d["obj_sum"] += float(obj or 0.0)

        by_video_vis[nm] = {
            a: {
                "n": s["n"],
                "pos_obj": s["pos_obj"],
                "iou_ge_05": s["iou_ge_05"],
                "pos_obj_rate": round(s["pos_obj"] / s["n"], 4) if s["n"] > 0 else None,
                "iou_ge_05_rate": round(s["iou_ge_05"] / s["n"], 4) if s["n"] > 0 else None,
                "mean_obj": round(s["obj_sum"] / s["n"], 4) if s["n"] > 0 else None,
                "mean_iou": round(s["iou_sum"] / s["n"], 4) if s["n"] > 0 else None,
            }
            for a, s in vv.items()
        }
        by_video_abs[nm] = {
            a: {
                "n": s["n"],
                "pos_obj": s["pos_obj"],
                "fp_rate": round(s["pos_obj"] / s["n"], 4) if s["n"] > 0 else None,
                "mean_obj": round(s["obj_sum"] / s["n"], 4) if s["n"] > 0 else None,
            }
            for a, s in va.items()
        }

        for ev in ev_info:
            if not ev.get("eligible"):
                continue
            t0 = int(ev["t0"])
            win_rows = [r for r in rows[t0 : min(len(rows), t0 + 10)] if r.get("native_silent") and "arms" in r]
            if not win_rows:
                continue
            rec = {
                "video": nm,
                "t0": t0,
                "baseline_category": ev["category"],
                "silent_eval_frames_in_window": len(win_rows),
                "streak_at_t0": rows[t0]["silent_streak"],
                "arm_window_best_iou": {},
                "arm_window_max_obj": {},
            }
            for a in ARMS:
                best_iou = 0.0
                max_obj = -999.0
                for r in win_rows:
                    gt = labels[r["frame"]]
                    ao = r["arms"][a]
                    if ao["obj_logit"] is not None:
                        max_obj = max(max_obj, ao["obj_logit"])
                    iou = box_iou(ao["bbox"], gt) if (gt is not None and ao["bbox"] is not None) else 0.0
                    best_iou = max(best_iou, iou)
                rec["arm_window_best_iou"][a] = round(best_iou, 4)
                rec["arm_window_max_obj"][a] = round(max_obj, 3)
            event_summaries.append(rec)

    # Video-level macro average (each video with n > 0 weighted equally — controls for correlated frames!)
    vis_vids = [nm for nm in by_video_vis if by_video_vis[nm]["orig_memory"]["n"] > 0]
    abs_vids = [nm for nm in by_video_abs if by_video_abs[nm]["orig_memory"]["n"] > 0]
    video_macro_vis = {
        a: {
            "affected_videos": len(vis_vids),
            "mean_pos_obj_rate": round(float(np.mean([by_video_vis[nm][a]["pos_obj_rate"] for nm in vis_vids])), 4) if vis_vids else 0.0,
            "mean_iou_ge_05_rate": round(float(np.mean([by_video_vis[nm][a]["iou_ge_05_rate"] for nm in vis_vids])), 4) if vis_vids else 0.0,
            "mean_obj": round(float(np.mean([by_video_vis[nm][a]["mean_obj"] for nm in vis_vids])), 4) if vis_vids else 0.0,
            "mean_iou": round(float(np.mean([by_video_vis[nm][a]["mean_iou"] for nm in vis_vids])), 4) if vis_vids else 0.0,
        }
        for a in ARMS
    }
    video_macro_abs = {
        a: {
            "affected_videos": len(abs_vids),
            "mean_fp_rate": round(float(np.mean([by_video_abs[nm][a]["fp_rate"] for nm in abs_vids])), 4) if abs_vids else 0.0,
            "mean_obj": round(float(np.mean([by_video_abs[nm][a]["mean_obj"] for nm in abs_vids])), 4) if abs_vids else 0.0,
        }
        for a in ARMS
    }

    # Event-level summary (how many silent qualifying events achieve best_iou >= 0.50 in window)
    event_recovered_ge_05 = {
        a: sum(1 for e in event_summaries if e["arm_window_best_iou"][a] >= 0.50)
        for a in ARMS
    }

    results = {
        "status": "AUDIT_OK",
        "freeze_sha256": digest(freeze_path),
        "complete_sha256": digest(complete_path),
        "dataset": dataset,
        "total_videos": len(all_specs),
        "affected_silent_visible_videos": len(vis_vids),
        "affected_silent_absent_videos": len(abs_vids),
        "silent_qualifying_events_count": len(event_summaries),
        "event_recovered_ge_05": event_recovered_ge_05,
        "pooled_silent_visible": {
            a: {
                "n": s["n"],
                "pos_obj": s["pos_obj"],
                "iou_ge_05": s["iou_ge_05"],
                "mean_obj": round(s["obj_sum"] / max(1, s["n"]), 4),
                "mean_iou": round(s["iou_sum"] / max(1, s["n"]), 4),
            }
            for a, s in pooled_vis.items()
        },
        "pooled_silent_absent": {
            a: {
                "n": s["n"],
                "pos_obj": s["pos_obj"],
                "mean_obj": round(s["obj_sum"] / max(1, s["n"]), 4),
            }
            for a, s in pooled_abs.items()
        },
        "video_macro_silent_visible": video_macro_vis,
        "video_macro_silent_absent": video_macro_abs,
        "by_video_silent_visible": by_video_vis,
        "by_video_silent_absent": by_video_abs,
        "qualifying_events": event_summaries,
        "verified_sha256": verified_sha,
    }
    res_path = out_dir / "results.json"
    res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"=== 9-ARM COUNTERFACTUAL INERTIA SUMMARY ({dataset}, {len(all_specs)} vids, {len(vis_vids)} affected) ===", flush=True)
    for a in ARMS:
        pv = results["pooled_silent_visible"][a]
        pa = results["pooled_silent_absent"][a]
        mv = video_macro_vis[a]
        ma = video_macro_abs[a]
        ev_r = event_recovered_ge_05[a]
        print(
            f"  {a:30s} | pooled vis iou>=.5={pv['iou_ge_05']:5d}/{pv['n']:5d} obj={pv['mean_obj']:+.3f} absFP={pa['pos_obj']:5d}/{pa['n']:5d} | "
            f"vidMacro vis_iou50={mv['mean_iou_ge_05_rate']:.3f} absFP={ma['mean_fp_rate']:.3f} | ev>0.5={ev_r}/{len(event_summaries)}",
            flush=True,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("hoot_dev", "hoot_transfer", "lasot_ext", "vot_reserve", "vot_lt_dir", "vot_lt22_journal"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backbone", choices=("tiny", "base_plus", "large"), default="tiny")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--deps", type=Path, nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--silent-stride", type=int, default=1)
    parser.add_argument("--keep-window", type=int, default=DEFAULT_MEMORY_KEEP_WINDOW)
    parser.add_argument("--sdpa-kernel-mode", choices=("allow_all", "stock"), default=DEFAULT_SDPA_KERNEL_MODE)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        parser.error("Invalid --shard-idx / --num-shards")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False

    all_specs = load_specs_extended(args.dataset, args.data, clips_filter=set(args.clips) if args.clips else None)
    if not all_specs:
        raise ValueError("No clips selected")
    if sum(s["label_reads"]["count"] for s in all_specs) != 0:
        raise RuntimeError("Causality violation before inference")

    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output / "freeze.json"

    def _build_freeze(host_prov: dict | None = None) -> dict:
        return {
            "utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "backbone": args.backbone,
            "arms": list(ARMS),
            "silent_stride": args.silent_stride,
            "max_frames": args.max_frames,
            "sdpa_kernel_mode": args.sdpa_kernel_mode,
            "checkpoint_sha256": digest(args.weights),
            "script_sha256": digest(Path(__file__)),
            "host_provenance": host_prov or {},
            "clips": {s["name"]: {"length": s["length"], "archive_sha256": s["archive_sha256"]} for s in all_specs},
            "post_initialization_label_reads": 0,
        }

    if args.merge_shards:
        if not freeze_path.exists():
            save(freeze_path, _build_freeze())
        operations = {}
        for spec in all_specs:
            nm = spec["name"]
            tpath = args.output / nm / "probe.jsonl"
            spath = args.output / nm / "probe.status.json"
            if not (tpath.exists() and spath.exists()):
                raise FileNotFoundError(f"Missing probe shard output for {nm}")
            op = json.loads(spath.read_text(encoding="utf-8"))
            if op["trace_sha256"] != digest(tpath):
                raise ValueError(f"Probe trace SHA mismatch during merge for {nm}")
            operations[nm] = op
        if sum(s["label_reads"]["count"] for s in all_specs) != 0:
            raise RuntimeError("Causality violation before scoring")
        _merge_and_score_probe(args.dataset, all_specs, args.output, operations)
        return

    shard_specs = [s for idx, s in enumerate(all_specs) if idx % args.num_shards == args.shard_idx]
    if not shard_specs:
        print(f"Probe shard {args.shard_idx}/{args.num_shards} has 0 clips; exiting cleanly.", flush=True)
        return

    adapter = SAM21HostAdapter(
        weights=args.weights,
        backbone=args.backbone,
        sam2_root=args.host_source,
        deps_dirs=[Path(d) for d in (args.deps or [])],
        use_sentry_runtime=False,
        device=args.device,
    )
    tmod = sys.modules.get("sam2.modeling.sam.transformer")
    if tmod is not None and hasattr(tmod, "ALLOW_ALL_KERNELS"):
        tmod.ALLOW_ALL_KERNELS = (args.sdpa_kernel_mode == "allow_all")

    if args.num_shards == 1 or (args.shard_idx == 0 and not freeze_path.exists()):
        if not freeze_path.exists():
            save(freeze_path, _build_freeze(adapter.provenance))

    operations = {}
    for spec in shard_specs:
        operations[spec["name"]] = _run_probe_clip(
            adapter=adapter,
            spec=spec,
            out_dir=args.output,
            silent_stride=args.silent_stride,
            keep_window=args.keep_window,
            max_frames=args.max_frames,
            resume=args.resume,
        )

    if args.num_shards > 1:
        (args.output / f"shard_{args.shard_idx}_of_{args.num_shards}.json").write_text(
            json.dumps({"shard_idx": args.shard_idx, "num_shards": args.num_shards, "operations": operations}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"PROBE_SHARD_DONE {args.shard_idx}/{args.num_shards} ({len(shard_specs)} clips)", flush=True)
        return

    if sum(s["label_reads"]["count"] for s in shard_specs) != 0:
        raise RuntimeError("Causality violation during probe")
    _merge_and_score_probe(args.dataset, shard_specs, args.output, operations)


if __name__ == "__main__":
    main()
