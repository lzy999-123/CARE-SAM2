"""Clean 2x2 Factorial Single-Slot Counterfactual Control (A3 + A4 Tensor Check).

Isolates the causal effect of a single verified-empty `t-1` spatial memory map vs. a single
`t-1` `no_obj_ptr` object pointer when added to frame-0 memory `M_0`:
  1. m0_only                     : cond_0 spatial + cond_0 ptr only
  2. m0_plus_1empty_spatial_only : cond_0 + t-1 spatial map ONLY (skip_non_cond_ptrs=True)
  3. m0_plus_1empty_ptr_only     : cond_0 + t-1 object pointer ONLY (skip_non_cond_spatial=True)
  4. m0_plus_1empty_both         : cond_0 + t-1 spatial map AND t-1 pointer (matches original m0_plus_1empty)

Reports metrics stratified by:
  - all_silent      : all native-silent frames (k_t >= 1)
  - prev1_empty     : verified-empty t-1 slot (prev1_obj_logit <= 0.0, strictly k_t >= 2)
  - sustained_k6    : sustained silence (k_t >= 6, all 6 recent spatial slots empty)

Also runs the A4 tensor exactness check comparing stored `(maskmem_features - no_obj_embed_spatial)`
against fresh memory_encoder output using BOTH Meta's true sentinel `NO_OBJ_SCORE = -1024.0`
(exact post-bias mask_for_mem = -10.0) and legacy `-10.0` raw logit.
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

import numpy as np
from PIL import Image
import torch

from care_sam2_plugin import (
    DEFAULT_MEMORY_KEEP_WINDOW,
    DEFAULT_SDPA_KERNEL_MODE,
    SAM21HostAdapter,
    _iter_spec_frames,
    _scalar,
)
from evaluation import box_iou
from gonogo_a6_decomposed import _eval_arm
from gonogo_simple_gating import load_specs_extended
from prepare import digest
from tip_hoot_incumbent_competition import assess


FACTORIAL_ARMS = (
    "m0_only",
    "m0_plus_1empty_spatial_only",
    "m0_plus_1empty_ptr_only",
    "m0_plus_1empty_both",
)


def _verify_tensor_exactness_both_sentinels(adapter: SAM21HostAdapter, frame_idx: int) -> dict:
    """Compare stripped stored spatial memory against fresh encoding under NO_OBJ_SCORE=-1024.0 vs -10.0."""
    obj_idx = adapter.state["obj_id_to_idx"][0]
    slot = adapter.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx)
    if slot is None or slot.get("maskmem_features") is None:
        return {"checked": False}

    _, _, vision_feats, _, feat_sizes = adapter.predictor._get_image_feature(adapter.state, frame_idx, 1)
    B = vision_feats[-1].size(1)
    C = adapter.predictor.hidden_dim
    H, W = feat_sizes[-1]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

    stored = slot["maskmem_features"].to(pix_feat.device)
    no_obj_embed = adapter.predictor.no_obj_embed_spatial.to(device=stored.device, dtype=stored.dtype)
    delta = no_obj_embed[..., None, None].expand(*stored.shape)
    stripped = stored - delta

    out = {
        "checked": True,
        "frame_idx": frame_idx,
        "stored_dtype": str(stored.dtype),
        "sigmoid_scale_for_mem_enc": float(adapter.predictor.sigmoid_scale_for_mem_enc),
        "sigmoid_bias_for_mem_enc": float(adapter.predictor.sigmoid_bias_for_mem_enc),
    }
    for tag, raw_logit in (("sentinel_neg1024", -1024.0), ("legacy_neg10", -10.0)):
        empty_high_res = torch.full(
            (B, 1, adapter.predictor.image_size, adapter.predictor.image_size),
            raw_logit,
            device=pix_feat.device,
            dtype=torch.float32,
        )
        mask_for_mem = torch.sigmoid(empty_high_res)
        if adapter.predictor.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * adapter.predictor.sigmoid_scale_for_mem_enc
        if adapter.predictor.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + adapter.predictor.sigmoid_bias_for_mem_enc
        raw_enc = adapter.predictor.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)["vision_features"]
        diff_stripped = (raw_enc.float() - stripped.float()).abs()
        diff_stored = (raw_enc.float() - stored.float()).abs()
        out[tag] = {
            "raw_logit": raw_logit,
            "mask_for_mem_val": round(float(mask_for_mem[0, 0, 0, 0].item()), 8),
            "raw_enc_dtype": str(raw_enc.dtype),
            "mean_abs_diff_stripped_vs_fresh": round(float(diff_stripped.mean().item()), 6),
            "max_abs_diff_stripped_vs_fresh": round(float(diff_stripped.max().item()), 6),
            "mean_abs_diff_stored_vs_fresh": round(float(diff_stored.mean().item()), 6),
            "max_abs_diff_stored_vs_fresh": round(float(diff_stored.max().item()), 6),
        }
    return out


def _empty_vis():
    return {a: {"n": 0, "pos_obj": 0, "iou_ge_05": 0, "obj_sum": 0.0, "iou_sum": 0.0} for a in FACTORIAL_ARMS}


def _empty_abs():
    return {a: {"n": 0, "pos_obj": 0, "obj_sum": 0.0} for a in FACTORIAL_ARMS}


def _finalize_vis(stats_by_vid: dict) -> tuple[dict, dict]:
    pooled = _empty_vis()
    for vn, arm_d in stats_by_vid.items():
        for a in FACTORIAL_ARMS:
            for k in ("n", "pos_obj", "iou_ge_05", "obj_sum", "iou_sum"):
                pooled[a][k] += arm_d[a][k]

    pooled_out = {}
    for a, s in pooled.items():
        n = max(1, s["n"])
        pooled_out[a] = {
            "n": s["n"],
            "pos_obj": s["pos_obj"],
            "iou_ge_05": s["iou_ge_05"],
            "iou_ge_05_rate": round(s["iou_ge_05"] / n, 4) if s["n"] > 0 else 0.0,
            "mean_obj": round(s["obj_sum"] / n, 4) if s["n"] > 0 else 0.0,
            "mean_iou": round(s["iou_sum"] / n, 4) if s["n"] > 0 else 0.0,
        }

    aff_vids = [vn for vn, d in stats_by_vid.items() if d["m0_only"]["n"] > 0]
    macro_out = {}
    for a in FACTORIAL_ARMS:
        if aff_vids:
            rates = [stats_by_vid[vn][a]["iou_ge_05"] / stats_by_vid[vn][a]["n"] for vn in aff_vids]
            objs = [stats_by_vid[vn][a]["obj_sum"] / stats_by_vid[vn][a]["n"] for vn in aff_vids]
            macro_out[a] = {
                "affected_videos": len(aff_vids),
                "mean_iou_ge_05_rate": round(float(np.mean(rates)), 4),
                "mean_obj": round(float(np.mean(objs)), 4),
            }
        else:
            macro_out[a] = {"affected_videos": 0, "mean_iou_ge_05_rate": 0.0, "mean_obj": 0.0}
    return pooled_out, macro_out


def _finalize_abs(stats_by_vid: dict) -> tuple[dict, dict]:
    pooled = _empty_abs()
    for vn, arm_d in stats_by_vid.items():
        for a in FACTORIAL_ARMS:
            for k in ("n", "pos_obj", "obj_sum"):
                pooled[a][k] += arm_d[a][k]

    pooled_out = {}
    for a, s in pooled.items():
        n = max(1, s["n"])
        pooled_out[a] = {
            "n": s["n"],
            "pos_obj": s["pos_obj"],
            "fp_rate": round(s["pos_obj"] / n, 4) if s["n"] > 0 else 0.0,
            "mean_obj": round(s["obj_sum"] / n, 4) if s["n"] > 0 else 0.0,
        }

    aff_vids = [vn for vn, d in stats_by_vid.items() if d["m0_only"]["n"] > 0]
    macro_out = {}
    for a in FACTORIAL_ARMS:
        if aff_vids:
            rates = [stats_by_vid[vn][a]["pos_obj"] / stats_by_vid[vn][a]["n"] for vn in aff_vids]
            macro_out[a] = {
                "affected_videos": len(aff_vids),
                "mean_fp_rate": round(float(np.mean(rates)), 4),
            }
        else:
            macro_out[a] = {"affected_videos": 0, "mean_fp_rate": 0.0}
    return pooled_out, macro_out


def run_clip(adapter: SAM21HostAdapter, spec: dict, out_dir: Path, keep_window: int = DEFAULT_MEMORY_KEEP_WINDOW) -> dict:
    name = spec["name"]
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    trace_path = clip_dir / "factorial_probe.jsonl"
    status_path = clip_dir / "factorial_probe.status.json"

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()

    init_box = spec["init_box"]
    silent_streak = 0
    silent_evals = 0
    tensor_sanity = None
    mirror_sanity = None

    tmp_path = trace_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for frame_idx, _, encoded in _iter_spec_frames(spec, None):
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
            if frame_idx == 0:
                adapter.initialize(pil_rgb, init_box)
                fh.write(json.dumps({
                    "frame": 0, "native_silent": False, "silent_streak": 0,
                    "native_bbox": init_box, "native_obj": None,
                }) + "\n")
                continue

            res = adapter.track(pil_rgb)
            native_bbox = res["pred_bbox"]
            native_obj = res["object_score"]

            if native_bbox is not None:
                silent_streak = 0
                fh.write(json.dumps({
                    "frame": frame_idx, "native_silent": False, "silent_streak": 0,
                    "native_bbox": native_bbox, "native_obj": native_obj,
                }) + "\n")
            else:
                silent_streak += 1
                silent_evals += 1
                if tensor_sanity is None and silent_streak >= 2:
                    tensor_sanity = _verify_tensor_exactness_both_sentinels(adapter, frame_idx)

                obj_idx = adapter.state["obj_id_to_idx"][0]
                obj_dict = adapter.state["output_dict_per_obj"][obj_idx]
                cond_outputs = obj_dict["cond_frame_outputs"]
                non_cond = obj_dict["non_cond_frame_outputs"]
                cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
                prev1_slot = non_cond.get(frame_idx - 1)
                prev1_obj = _scalar(prev1_slot.get("object_score_logits")) if prev1_slot is not None else None
                prev1_is_empty = bool(prev1_slot is not None and (prev1_obj is None or prev1_obj <= 0.0))

                cond0_only_dict = {"cond_frame_outputs": cond_0, "non_cond_frame_outputs": {}}
                prev1_dict = {
                    "cond_frame_outputs": cond_0,
                    "non_cond_frame_outputs": {frame_idx - 1: prev1_slot} if prev1_slot is not None else {},
                }

                if mirror_sanity is None and prev1_slot is not None:
                    ref_o = _eval_arm(adapter, frame_idx, prev1_dict, None)
                    mir_o = _eval_arm(
                        adapter, frame_idx, prev1_dict,
                        {"skip_non_cond_spatial": False, "skip_non_cond_ptrs": False},
                    )
                    mirror_sanity = {
                        "frame_idx": frame_idx,
                        "video": name,
                        "obj_abs_diff": abs((ref_o["obj_logit"] or 0.0) - (mir_o["obj_logit"] or 0.0)),
                        "bbox_equal": ref_o["bbox"] == mir_o["bbox"],
                    }

                arm_specs = {
                    "m0_only": (cond0_only_dict, None),
                    "m0_plus_1empty_spatial_only": (prev1_dict, {"skip_non_cond_spatial": False, "skip_non_cond_ptrs": True}),
                    "m0_plus_1empty_ptr_only": (prev1_dict, {"skip_non_cond_spatial": True, "skip_non_cond_ptrs": False}),
                    "m0_plus_1empty_both": (prev1_dict, None),
                }
                arm_res = {a: _eval_arm(adapter, frame_idx, md, pk) for a, (md, pk) in arm_specs.items()}
                fh.write(json.dumps({
                    "frame": frame_idx,
                    "native_silent": True,
                    "silent_streak": silent_streak,
                    "prev1_exists": prev1_slot is not None,
                    "prev1_obj": prev1_obj,
                    "prev1_is_empty": prev1_is_empty,
                    "native_bbox": None,
                    "native_obj": native_obj,
                    "arms": arm_res,
                }) + "\n")

            adapter.prune_old_memory(keep_window)

    tmp_path.replace(trace_path)
    adapter.release_clip()
    torch.cuda.empty_cache()

    op = {
        "frames": spec["length"],
        "silent_evals": silent_evals,
        "trace_sha256": digest(trace_path),
        "tensor_sanity": tensor_sanity,
        "mirror_sanity": mirror_sanity,
    }
    status_path.write_text(json.dumps(op, indent=2) + "\n", encoding="utf-8")
    print(f"  [A3 factorial {name}] {spec['length']}f, silent_evals={silent_evals}", flush=True)
    return op


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["hoot_dev", "vot_lt22_journal", "vot_lt_dir"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--deps", type=Path, nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False

    args.output.mkdir(parents=True, exist_ok=True)
    specs = load_specs_extended(args.dataset, args.data)
    adapter = SAM21HostAdapter(
        weights=args.weights,
        backbone=args.backbone,
        sam2_root=args.host_source,
        deps_dirs=[Path(d) for d in (args.deps or [])],
        use_sentry_runtime=False,
        device=args.device,
    )

    freeze = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "backbone": args.backbone,
        "clips": [s["name"] for s in specs],
        "arms": list(FACTORIAL_ARMS),
        "script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(args.weights),
        "host_provenance": adapter.provenance,
        "sdpa_kernel_mode": DEFAULT_SDPA_KERNEL_MODE,
        "post_initialization_label_reads": 0,
    }
    freeze_path = args.output / "freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    operations = {}
    for spec in specs:
        operations[spec["name"]] = run_clip(adapter, spec, args.output)

    complete_path = args.output / "complete.json"
    complete_path.write_text(
        json.dumps({
            "freeze_sha256": digest(freeze_path),
            "post_initialization_label_reads_before_scoring": 0,
            "operations": operations,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    strata = ("all_silent", "prev1_empty", "sustained_k6")
    vis_by_vid = {st: {} for st in strata}
    abs_by_vid = {st: {} for st in strata}
    event_summaries = []

    for spec in specs:
        nm = spec["name"]
        rows = [json.loads(l) for l in (args.output / nm / "factorial_probe.jsonl").read_text(encoding="utf-8").splitlines()]
        labels = spec["load_labels"]()[: len(rows)]
        native_boxes = [r["native_bbox"] for r in rows]
        ev_info = assess(labels, native_boxes)["windows"]["0.5"]

        for st in strata:
            vis_by_vid[st][nm] = _empty_vis()
            abs_by_vid[st][nm] = _empty_abs()

        for r in rows:
            if not (r.get("native_silent") and "arms" in r):
                continue
            gt = labels[r["frame"]]
            k = int(r["silent_streak"])
            p1_empty = bool(r.get("prev1_is_empty"))
            active_strata = ["all_silent"]
            if p1_empty:
                active_strata.append("prev1_empty")
            if k >= 6 and p1_empty:
                active_strata.append("sustained_k6")

            for st in active_strata:
                for a in FACTORIAL_ARMS:
                    ao = r["arms"][a]
                    obj = ao["obj_logit"]
                    if gt is not None:
                        iou = box_iou(ao["bbox"], gt) if ao["bbox"] is not None else 0.0
                        d = vis_by_vid[st][nm][a]
                        d["n"] += 1
                        d["pos_obj"] += int(obj is not None and obj > 0.0)
                        d["iou_ge_05"] += int(iou >= 0.50)
                        d["obj_sum"] += float(obj or 0.0)
                        d["iou_sum"] += iou
                    else:
                        d = abs_by_vid[st][nm][a]
                        d["n"] += 1
                        d["pos_obj"] += int(obj is not None and obj > 0.0)
                        d["obj_sum"] += float(obj or 0.0)

        for ev in ev_info:
            if not ev.get("eligible"):
                continue
            t0 = int(ev["t0"])
            win_rows = [r for r in rows[t0 : min(len(rows), t0 + 10)] if r.get("native_silent") and r.get("prev1_is_empty") and "arms" in r]
            if not win_rows:
                continue
            rec = {
                "video": nm,
                "t0": t0,
                "streak_at_t0": rows[t0]["silent_streak"],
                "prev1_empty_frames_in_window": len(win_rows),
                "arm_window_best_iou": {},
            }
            for a in FACTORIAL_ARMS:
                best_iou = 0.0
                for r in win_rows:
                    gt = labels[r["frame"]]
                    ao = r["arms"][a]
                    iou = box_iou(ao["bbox"], gt) if (gt is not None and ao["bbox"] is not None) else 0.0
                    best_iou = max(best_iou, iou)
                rec["arm_window_best_iou"][a] = round(best_iou, 4)
            event_summaries.append(rec)

    summary_by_stratum = {}
    for st in strata:
        pv, mv = _finalize_vis(vis_by_vid[st])
        pa, ma = _finalize_abs(abs_by_vid[st])
        summary_by_stratum[st] = {
            "pooled_silent_visible": pv,
            "video_macro_silent_visible": mv,
            "pooled_silent_absent": pa,
            "video_macro_silent_absent": ma,
        }

    results = {
        "status": "AUDIT_OK",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "freeze_sha256": digest(freeze_path),
        "complete_sha256": digest(complete_path),
        "strata": summary_by_stratum,
        "prev1_empty_qualifying_events_count": len(event_summaries),
        "prev1_empty_event_recovered_ge_05": {
            a: sum(1 for e in event_summaries if e["arm_window_best_iou"][a] >= 0.50)
            for a in FACTORIAL_ARMS
        },
        "qualifying_events": event_summaries,
        "tensor_sanity_by_video": {nm: op["tensor_sanity"] for nm, op in operations.items() if op["tensor_sanity"]},
        "mirror_sanity_by_video": {nm: op["mirror_sanity"] for nm, op in operations.items() if op["mirror_sanity"]},
    }
    res_path = args.output / "results.json"
    res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n=== A3 SINGLE-SLOT FACTORIAL CONTROL SUMMARY ===", flush=True)
    for st in strata:
        print(f"\nStratum: {st}", flush=True)
        for a in FACTORIAL_ARMS:
            pv = summary_by_stratum[st]["pooled_silent_visible"][a]
            mv = summary_by_stratum[st]["video_macro_silent_visible"][a]
            pa = summary_by_stratum[st]["pooled_silent_absent"][a]
            print(
                f"  {a:30s}: vis={pv['iou_ge_05']:4d}/{pv['n']} ({100*pv['iou_ge_05_rate']:5.2f}%) "
                f"macro={100*mv['mean_iou_ge_05_rate']:5.2f}% obj={pv['mean_obj']:+.2f} | "
                f"abs_fp={pa['pos_obj']:4d}/{pa['n']} ({100*pa['fp_rate']:5.2f}%)",
                flush=True,
            )


if __name__ == "__main__":
    main()
