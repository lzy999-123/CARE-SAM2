"""Standalone Sharded Official SAM2Long (ICCV 2025) Benchmark Runner for VOT-LT2022.

Strictly separate from frozen `care_sam2_plugin.py` (which remains untouched).
Runs official `Mark12Ding/SAM2Long` on identical `sam2.1_hiera_tiny.pt` weights,
identical VOT-LT2022 50-sequence dataset, identical FP16 autocast / `allow_all`
SDPA / `seed=0` determinism, and identical post-decode synchronized GPU timing.

Outputs BOTH arms from a single official SAM2Long forward pass:
  - `baseline`  : strictly causal online top-1 pathway at frame t (`pathway_id = 0`)
  - `backtrack` : official SAM2Long full-video backtracked pathway (`mem_pick_indexs[0][t]`)

Fully compatible with `audit_care_sam2_plugin.py` (`status == "AUDIT_OK"`).
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import random
from statistics import median, quantiles
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as vision_f

from audit_care_sam2_plugin import (
    _independent_lasot_metrics,
    _independent_macro_prf,
    _independent_vot_toolkit,
)
from care_sam2_plugin import (
    _git_head_sha,
    _iter_spec_frames,
    _scalar,
    _sigmoid,
    load_dataset_spec,
    mask_to_bbox,
)
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_paired_query_continuous import combine


def _prepare_image(image: Image.Image, image_size: int, mean: torch.Tensor, std: torch.Tensor, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).to(device)
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    tensor = vision_f.resize(tensor, (image_size, image_size))
    return (tensor - mean) / std


def _prune_unreachable_maskmem(output_dict: dict, mem_pick_indexs, frame_idx: int, iou_thre: float, max_keep_valid: int = 16):
    """Exact bit-identical RAM pruning: drop `maskmem_features` strictly older than all active pathways' 16th valid frame."""
    if not isinstance(mem_pick_indexs, list) or frame_idx < 32:
        return
    non_cond = output_dict["non_cond_frame_outputs"]
    min_needed = frame_idx
    for pick_map in mem_pick_indexs:
        valid = 0
        oldest_for_path = 2
        for i in range(frame_idx - 1, 1, -1):
            slot = non_cond.get(i)
            if slot is None:
                continue
            pidx = pick_map.get(i, 0)
            obj_t = slot["object_score_logits"][..., pidx]
            iou_t = slot["ious"][..., pidx]
            obj_s = float(obj_t.max().item() if obj_t.numel() > 1 else obj_t.item())
            iou_s = float(iou_t.max().item() if iou_t.numel() > 1 else iou_t.item())
            if iou_s > iou_thre and obj_s > 0.0:
                valid += 1
                oldest_for_path = i
                if valid >= max_keep_valid:
                    break
        if valid < max_keep_valid:
            return  # pathway still scanning back to frame 1
        min_needed = min(min_needed, oldest_for_path)
    # Safe margin of 2 frames; never touch frame 1
    cutoff = min_needed - 2
    if cutoff <= 2:
        return
    for k in list(non_cond.keys()):
        if 2 <= k < cutoff and non_cond[k].get("maskmem_features") is not None:
            non_cond[k]["maskmem_features"] = None
            non_cond[k]["maskmem_pos_enc"] = None


def build_sam2long_predictor(sam2long_root: Path, weights: Path, backbone: str = "tiny", device: str = "cuda:0"):
    root_str = str(sam2long_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import importlib
    import sam2 as sam2_pkg
    from sam2.build_sam import build_sam2_video_predictor
    pred_mod = importlib.import_module("sam2.sam2_video_predictor")
    pred_mod.tqdm = lambda it, **_: it

    cfg_map = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "baseplus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    }
    cfg = cfg_map[backbone]
    cfg_path = Path(sam2_pkg.__file__).resolve().parent / cfg
    predictor = build_sam2_video_predictor(cfg, str(weights), device=device)
    prov = {
        "adapter": "SAM2LongOfficialAdapter",
        "host_source_root": root_str,
        "host_git_commit": _git_head_sha(sam2long_root),
        "host_module_path": str(Path(sam2_pkg.__file__).resolve()),
        "sam2_module_path": str(Path(sam2_pkg.__file__).resolve()),
        "config_name": cfg,
        "config_path": str(cfg_path.resolve()) if cfg_path.exists() else None,
        "config_sha256": digest(cfg_path) if cfg_path.exists() else None,
    }
    return predictor, prov


def run_clip_sam2long(
    spec: dict,
    predictor,
    out_dir: Path,
    num_pathway: int = 3,
    iou_thre: float = 0.1,
    uncertainty: float = 2.0,
    sdpa_kernel_mode: str = "allow_all",
    device: str = "cuda:0",
    max_frames: int | None = None,
    resume: bool = False,
) -> dict[str, dict]:
    name = spec["name"]
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    p_base = clip_dir / "baseline.jsonl"
    p_back = clip_dir / "backtrack.jsonl"
    s_base = clip_dir / "baseline.status.json"
    s_back = clip_dir / "backtrack.status.json"

    if resume and p_base.exists() and p_back.exists() and s_base.exists() and s_back.exists():
        sb = json.loads(s_base.read_text(encoding="utf-8"))
        sk = json.loads(s_back.read_text(encoding="utf-8"))
        if sb.get("trace_sha256") == digest(p_base) and sk.get("trace_sha256") == digest(p_back):
            print(f"SAM2Long {name} RESUMED (verified SHA-256)", flush=True)
            return {"baseline": sb, "backtrack": sk}

    tmod = sys.modules.get("sam2.modeling.sam.transformer")
    if tmod is not None and hasattr(tmod, "ALLOW_ALL_KERNELS"):
        tmod.ALLOW_ALL_KERNELS = (sdpa_kernel_mode == "allow_all")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(device)
    image_size = int(getattr(predictor, "image_size", 1024))
    torch.set_num_threads(2)

    state = None
    mem_pick_indexs = 0
    online_rows = []
    frame_shas = []
    frame_ms = []
    img_width = 0
    img_height = 0
    total_frames = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i, limit, encoded in _iter_spec_frames(spec, max_frames):
            total_frames = limit
            sha = sha256(encoded).hexdigest()
            frame_shas.append(sha)
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
                if i == 0:
                    img_width, img_height = pil_rgb.size

            img_tensor = _prepare_image(pil_rgb, image_size, mean, std, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = perf_counter()

            if i == 0:
                state = {
                    "images": {0: img_tensor},
                    "num_frames": limit,
                    "offload_video_to_cpu": False,
                    "offload_state_to_cpu": False,
                    "video_height": img_height,
                    "video_width": img_width,
                    "device": torch.device(device),
                    "storage_device": torch.device(device),
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
                    "num_pathway": num_pathway,
                    "iou_thre": iou_thre,
                    "uncertainty": uncertainty,
                }
                predictor._get_image_feature(state, frame_idx=0, batch_size=1)
                x, y, w, h = spec["init_box"]
                _, _, logits = predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=0, box=[x, y, x + w, y + h]
                )
                predictor.propagate_in_video_preflight(state)
                pred_mask = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
                state["images"].pop(0, None)
                final_bbox = [float(v) for v in spec["init_box"]]
                native_bbox = mask_to_bbox(pred_mask)
                obj_score = None
                confidence = 1.0
                source = "init"
            else:
                state["images"][i] = img_tensor
                current_out, _, mem_pick_indexs = predictor._run_single_frame_inference(
                    inference_state=state,
                    output_dict=state["output_dict"],
                    frame_idx=i,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    mem_pick_indexs=mem_pick_indexs,
                    start_frame_idx=0,
                )
                state["output_dict"]["non_cond_frame_outputs"][i] = current_out
                state["images"].pop(i, None)

                # Strictly causal online top-1 pathway (pathway index 0)
                pm0 = current_out["pred_masks"][..., 0].to(device, non_blocking=True)
                _, video_mask = predictor._get_orig_video_res_output(state, pm0)
                mask_np = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
                native_bbox = mask_to_bbox(mask_np)
                obj_score = _scalar(current_out["object_score_logits"][..., 0])
                final_bbox = native_bbox
                confidence = _sigmoid(obj_score) if final_bbox is not None else 0.0
                source = "baseline"
                # Move historical pred_masks to CPU (only read post-video for backtrack) while keeping maskmem on GPU
                current_out["pred_masks"] = current_out["pred_masks"].to("cpu", non_blocking=True)

                if i % 16 == 0:
                    _prune_unreachable_maskmem(state["output_dict"], mem_pick_indexs, i, iou_thre=iou_thre)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = 1000.0 * (perf_counter() - t0)
            if i > 0:
                frame_ms.append(elapsed_ms)

            online_rows.append({
                "frame": i,
                "sha256": sha,
                "bbox": final_bbox,
                "native_bbox": native_bbox,
                "object_score": obj_score,
                "confidence": round(confidence, 6),
                "source": source,
                "memory_update": False,
                "synchronized_model_ms": elapsed_ms,
                "diag": {},
            })

            if (i + 1) % 100 == 0 or (i + 1) == limit:
                print(
                    f"SAM2Long {name} {i + 1}/{limit} med_ms={median(frame_ms) if frame_ms else 0.0:.1f}",
                    flush=True,
                )

        # Official full-video backtracked pathway (mem_pick_indexs[0][t]) from the exact same run
        back_rows = [dict(online_rows[0])]
        non_cond = state["output_dict"]["non_cond_frame_outputs"]
        for i in range(1, total_frames):
            pidx = mem_pick_indexs[0][i] if isinstance(mem_pick_indexs, list) else 0
            slot = non_cond[i]
            pm = slot["pred_masks"][..., pidx].to(device, non_blocking=True)
            _, video_mask = predictor._get_orig_video_res_output(state, pm)
            mask_np = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
            bb = mask_to_bbox(mask_np)
            os_val = _scalar(slot["object_score_logits"][..., pidx])
            conf_val = _sigmoid(os_val) if bb is not None else 0.0
            back_rows.append({
                "frame": i,
                "sha256": frame_shas[i],
                "bbox": bb,
                "native_bbox": bb,
                "object_score": os_val,
                "confidence": round(conf_val, 6),
                "source": "backtrack",
                "memory_update": False,
                "synchronized_model_ms": online_rows[i]["synchronized_model_ms"],
                "diag": {"pathway_idx": int(pidx)},
            })

    # Release memory before writing
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ops = {}
    for arm_name, rows, final_p, stat_p in [
        ("baseline", online_rows, p_base, s_base),
        ("backtrack", back_rows, p_back, s_back),
    ]:
        part_p = clip_dir / f"{arm_name}.partial.jsonl"
        with part_p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
        part_p.replace(final_p)
        op = {
            "frames": total_frames,
            "width": img_width,
            "height": img_height,
            "trace_sha256": digest(final_p),
            "median_ms": median(frame_ms) if frame_ms else 0.0,
            "p95_ms": quantiles(frame_ms, n=100)[94] if len(frame_ms) >= 20 else (max(frame_ms) if frame_ms else 0.0),
            "sources": dict(Counter(r["source"] for r in rows)),
            "stats": {},
        }
        save(stat_p, op)
        ops[arm_name] = op

    return ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="vot_lt_dir")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sam2long-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", choices=("tiny", "baseplus"), default="tiny")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-pathway", type=int, default=3)
    parser.add_argument("--iou-thre", type=float, default=0.1)
    parser.add_argument("--uncertainty", type=float, default=2.0)
    parser.add_argument("--clips", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()

    clips_filter = {c.strip() for c in args.clips.split(",") if c.strip()} if args.clips else None
    specs = load_dataset_spec(args.dataset, args.data, clips_filter=clips_filter)
    args.output.mkdir(parents=True, exist_ok=True)
    arms = ["baseline", "backtrack"]

    if not args.merge_shards:
        predictor, prov = build_sam2long_predictor(args.sam2long_root, args.weights, backbone=args.backbone)
        shard_specs = [s for idx, s in enumerate(specs) if idx % args.num_shards == args.shard_idx]
        shard_ops = {}
        for spec in shard_specs:
            shard_ops[spec["name"]] = run_clip_sam2long(
                spec, predictor, args.output,
                num_pathway=args.num_pathway,
                iou_thre=args.iou_thre,
                uncertainty=args.uncertainty,
                max_frames=args.max_frames,
                resume=args.resume,
            )
        save(args.output / f"shard_{args.shard_idx:02d}_of_{args.num_shards:02d}.json", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "host_provenance": prov,
            "operations": shard_ops,
        })
        if args.num_shards > 1:
            print(f"Shard {args.shard_idx}/{args.num_shards} complete ({len(shard_specs)} clips).", flush=True)
            return

    # Merge shards & score post-hoc
    _, prov = build_sam2long_predictor(args.sam2long_root, args.weights, backbone=args.backbone)
    freeze_obj = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "host": "sam2long",
        "backbone": args.backbone,
        "verifier": "none",
        "arms": arms,
        "script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(args.weights),
        "host_provenance": prov,
        "sam2long_params": {
            "num_pathway": args.num_pathway,
            "iou_thre": args.iou_thre,
            "uncertainty": args.uncertainty,
        },
        "sdpa_kernel_mode": "allow_all",
        "post_initialization_label_reads": 0,
        "clips": {s["name"]: {"length": s["length"], "archive_sha256": s["archive_sha256"]} for s in specs},
    }
    freeze_path = args.output / "freeze.json"
    save(freeze_path, freeze_obj)
    freeze_sha = digest(freeze_path)

    operations = {}
    for s in specs:
        name = s["name"]
        operations[name] = {}
        for arm in arms:
            sp = args.output / name / f"{arm}.status.json"
            if not sp.exists():
                raise FileNotFoundError(f"Missing {sp}")
            operations[name][arm] = json.loads(sp.read_text(encoding="utf-8"))

    complete_path = args.output / "complete.json"
    save(complete_path, {
        "utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "post_initialization_label_reads_before_scoring": 0,
        "operations": operations,
    })
    complete_sha = digest(complete_path)

    # Strictly post-hoc evaluation (after all traces are frozen)
    results_arms = {}
    for arm in arms:
        videos_eval = {}
        prf_inputs = {}
        tk_inputs = {}
        all_ms = []
        for s in specs:
            name = s["name"]
            labels = s["load_labels"]()
            rows = [json.loads(line) for line in (args.output / name / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            lbl = labels[:len(rows)]
            boxes = [r["bbox"] for r in rows]
            confs = [r["confidence"] for r in rows]
            videos_eval[name] = assess(lbl, boxes)
            prf_inputs[name] = (lbl, boxes)
            op = operations[name][arm]
            tk_inputs[name] = (lbl, boxes, confs, (int(op["width"]), int(op["height"])))
            all_ms.extend(r["synchronized_model_ms"] for r in rows[1:])
        totals = combine(videos_eval)
        prf, prf_by_vid = _independent_macro_prf(prf_inputs)
        vtk = _independent_vot_toolkit(tk_inputs)
        ls = _independent_lasot_metrics(prf_inputs)
        results_arms[arm] = {
            "totals": totals,
            "fixed_point_macro_prf": prf,
            "fixed_point_by_video": prf_by_vid,
            "vot_lt_toolkit": vtk,
            "lasot_metrics": ls,
            "ms": {
                "median": median(all_ms) if all_ms else 0.0,
                "p95": quantiles(all_ms, n=100)[94] if len(all_ms) >= 20 else (max(all_ms) if all_ms else 0.0),
                "mean": sum(all_ms) / len(all_ms) if all_ms else 0.0,
            },
            "videos": videos_eval,
        }

    results_path = args.output / "results.json"
    save(results_path, {
        "utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "complete_sha256": complete_sha,
        "arms": results_arms,
    })
    print(f"MERGED_OK {results_path}", flush=True)


if __name__ == "__main__":
    main()
