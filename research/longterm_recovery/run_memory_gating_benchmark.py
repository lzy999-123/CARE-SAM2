"""Sharded, Resumable, Independently Audited Cloud Runner for Simple Memory-Gating Baselines.

Runs controlled absent-frame memory-gating baselines on Meta official SAM 2.1 (`tiny`, `base_plus`,
`large`) across full official benchmarks (`vot_lt_dir`, `lasot_ext`, `hoot_dev`, `hoot_transfer`,
`vot_reserve`, `vot_lt22_journal`) with 4-shard multi-process execution, crash-safe resume, strict
causality (zero post-initialization label reads before merge/scoring), and automatic independent
official `vot-toolkit` + dual-protocol LaSOT auditing (`AUDIT_OK`).

Arms:
  - vanilla                 : unmodified Meta SAM 2.1
  - gate_skip_absent        : skip obj<=0 frames from BOTH spatial memory (6) & pointers (15)
  - gate_skip_absent_damptr : skip obj<=0 from spatial (6) + DAM4SAM local-window (1..15) ptrs
  - gate_skip_absent_keept1 : SAMURAI-style spatial (5 visible + forced t-1) + unfiltered 15 ptrs
"""

from __future__ import annotations

import argparse
from collections import Counter
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

import cv2
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
    DEFAULT_MEMORY_KEEP_WINDOW,
    DEFAULT_SDPA_KERNEL_MODE,
    SAM21HostAdapter,
    _clone_mem_entry,
    _iter_spec_frames,
    _sigmoid,
)
from gonogo_a6_decomposed import _patch_memory_prep
from gonogo_simple_gating import load_specs_extended
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_paired_query_continuous import combine


GATING_ARMS = (
    "vanilla",
    "gate_skip_absent",
    "gate_skip_absent_damptr",
    "gate_skip_absent_keept1",
)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_sdpa_mode(mode: str):
    tmod = sys.modules.get("sam2.modeling.sam.transformer")
    if tmod is not None and hasattr(tmod, "ALLOW_ALL_KERNELS"):
        tmod.ALLOW_ALL_KERNELS = (mode == "allow_all")


def run_gating_clip_arm(
    adapter: SAM21HostAdapter,
    spec: dict,
    arm: str,
    out_dir: Path,
    keep_window: int = DEFAULT_MEMORY_KEEP_WINDOW,
    max_frames: int | None = None,
    resume: bool = False,
) -> tuple[dict, Path]:
    name = spec["name"]
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    trace_path = clip_dir / f"{arm}.jsonl"
    status_path = clip_dir / f"{arm}.status.json"

    expected_len = min(spec["length"], max_frames) if max_frames else spec["length"]
    if resume and trace_path.exists() and status_path.exists():
        try:
            op = json.loads(status_path.read_text(encoding="utf-8"))
            if op.get("frames") == expected_len and op.get("trace_sha256") == digest(trace_path):
                print(f"  [RESUME] {name}/{arm} ({expected_len}f) already verified.", flush=True)
                return op, trace_path
        except Exception:
            pass

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sources: Counter = Counter()
    step_ms: list[float] = []
    vis_spatial_ring: list[tuple[int, dict]] = []
    vis_ptr_ring: list[tuple[int, torch.Tensor]] = []
    init_box = spec["init_box"]
    width, height = 0, 0

    tmp_path = trace_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for frame_idx, _, encoded in _iter_spec_frames(spec, max_frames):
            frame_sha = sha256(encoded).hexdigest()
            with Image.open(BytesIO(encoded)) as img:
                pil_rgb = img.convert("RGB")
                rgb_np = np.asarray(pil_rgb)
            height, width = rgb_np.shape[:2]

            _sync_cuda()
            t0 = perf_counter()

            if frame_idx == 0:
                init_res = adapter.initialize(pil_rgb, init_box)
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

            if arm == "vanilla":
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
                with _patch_memory_prep(
                    adapter.predictor,
                    visible_spatial_list=sp_list,
                    skip_non_cond_ptrs=False,
                ):
                    res = adapter.track(pil_rgb)
            else:
                raise ValueError(f"Unknown gating arm: {arm}")

            native_bbox = res["pred_bbox"]
            obj_score = res["object_score"]

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
            sources["baseline"] += 1
            conf = _sigmoid(obj_score) if (native_bbox is not None and obj_score is not None) else (0.0 if native_bbox is None else 0.5)
            row = {
                "frame": frame_idx,
                "sha256": frame_sha,
                "bbox": native_bbox,
                "native_bbox": native_bbox,
                "object_score": obj_score,
                "confidence": round(float(conf), 6),
                "source": "baseline",
                "memory_update": False,
                "synchronized_model_ms": dt_ms,
                "diag": {},
            }
            fh.write(json.dumps(row) + "\n")
            adapter.prune_old_memory(keep_window)

    tmp_path.replace(trace_path)
    peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    adapter.release_clip()
    torch.cuda.empty_cache()

    op = {
        "frames": expected_len,
        "width": width,
        "height": height,
        "trace_sha256": digest(trace_path),
        "sources": dict(sources),
        "stats": {},
        "median_synchronized_step_ms": median(step_ms) if step_ms else 0.0,
        "p95_synchronized_step_ms": quantiles(step_ms, n=20)[18] if len(step_ms) >= 20 else (max(step_ms) if step_ms else 0.0),
        "mean_synchronized_step_ms": sum(step_ms) / len(step_ms) if step_ms else 0.0,
        "peak_allocated_cuda_bytes": peak_bytes,
    }
    status_path.write_text(json.dumps(op, indent=2) + "\n", encoding="utf-8")
    print(
        f"  [{name}/{arm}] {expected_len}f med={op['median_synchronized_step_ms']:.1f}ms "
        f"peak={peak_bytes/1048576:.0f}MiB",
        flush=True,
    )
    return op, trace_path


def _score_and_audit_merged(
    dataset: str,
    all_specs: list[dict],
    out_dir: Path,
    arms: list[str],
    operations: dict,
    vot_toolkit_path: Path | None,
) -> dict:
    freeze_path = out_dir / "freeze.json"
    complete_path = out_dir / "complete.json"
    results_path = out_dir / "results.json"

    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    labels_by_vid = {s["name"]: s["load_labels"]() for s in all_specs}

    verified_sha = {
        str(freeze_path): digest(freeze_path),
        str(complete_path): digest(complete_path),
    }

    # Verify identical pixel SHA streams across arms for each video
    arm_reports = {}
    ref_pixel_shas: dict[str, list[str]] = {}
    ev_by_arm: dict[str, dict] = {}
    for arm in arms:
        videos_data = {}
        toolkit_data = {}
        ev_by_vid = {}
        all_ms = []
        interventions = {}
        silence_prev = {}
        for spec in all_specs:
            nm = spec["name"]
            tpath = out_dir / nm / f"{arm}.jsonl"
            actual_sha = digest(tpath)
            if actual_sha != operations[nm][arm]["trace_sha256"]:
                raise ValueError(f"Trace SHA mismatch during audit for {nm}/{arm}")
            verified_sha[str(tpath)] = actual_sha

            with tpath.open(encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh]
            pix_shas = [r["sha256"] for r in rows]
            if nm not in ref_pixel_shas:
                ref_pixel_shas[nm] = pix_shas
            elif ref_pixel_shas[nm] != pix_shas:
                raise ValueError(f"Pixel SHA stream mismatch across arms in {nm}/{arm}")

            labels = labels_by_vid[nm][: len(rows)]
            boxes = [r["bbox"] for r in rows]
            confs = [r["confidence"] for r in rows]
            videos_data[nm] = (labels, boxes)
            toolkit_data[nm] = (labels, boxes, confs, (operations[nm][arm]["width"], operations[nm][arm]["height"]))
            ev_by_vid[nm] = assess(labels, boxes)
            all_ms.extend(r["synchronized_model_ms"] for r in rows[1:])

            # Silence-while-visible prevalence (streak >= 6)
            streak = sv = sa = eps = 0
            in_sv = False
            for r, g in zip(rows[1:], labels[1:]):
                if r["bbox"] is None:
                    streak += 1
                else:
                    streak = 0
                    in_sv = False
                if streak >= 6:
                    if g is not None:
                        sv += 1
                        if not in_sv:
                            eps += 1
                            in_sv = True
                    else:
                        sa += 1
            silence_prev[nm] = {
                "frames": len(rows) - 1,
                "silent_ge6_visible": sv,
                "silent_ge6_visible_episodes": eps,
                "silent_ge6_absent": sa,
            }
            interventions[nm] = {
                "probe_calls": 0,
                "verified_probes": 0,
                "memory_updates": 0,
                "rollbacks": 0,
                "silent_frames": sum(r["bbox"] is None for r in rows[1:]),
            }

        ev_by_arm[arm] = ev_by_vid
        totals = combine(ev_by_vid)
        macro_prf, fp_by_vid = _independent_macro_prf(videos_data)
        vtk = _independent_vot_toolkit(toolkit_data, toolkit_path=vot_toolkit_path)
        ls = _independent_lasot_metrics(videos_data)
        arm_reports[arm] = {
            "totals": totals,
            "fixed_point_macro_prf": macro_prf,
            "fixed_point_by_video": fp_by_vid,
            "vot_lt_toolkit": vtk,
            "lasot_metrics": ls,
            "interventions": interventions,
            "silence_prevalence": silence_prev,
            "ms": {
                "median": median(all_ms) if all_ms else 0.0,
                "p95": quantiles(all_ms, n=20)[18] if len(all_ms) >= 20 else 0.0,
                "mean": sum(all_ms) / len(all_ms) if all_ms else 0.0,
            },
            "videos": ev_by_vid,
        }

    paired = {}
    if "vanilla" in ev_by_arm:
        for arm in arms:
            if arm != "vanilla":
                paired[arm] = paired_deltas(ev_by_arm["vanilla"], ev_by_arm[arm])

    results_payload = {
        "freeze_sha256": digest(freeze_path),
        "complete_sha256": digest(complete_path),
        "dataset": dataset,
        "arms": arm_reports,
        "paired_vs_vanilla": paired,
    }
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
    verified_sha[str(results_path)] = digest(results_path)

    audit_payload = {
        "status": "AUDIT_OK",
        "dataset": dataset,
        "host": "sam21_gating",
        "backbone": freeze.get("backbone"),
        "verifier": "none",
        "script_sha256": freeze.get("script_sha256"),
        "host_provenance": freeze.get("host_provenance", {}),
        "verified_sha256": verified_sha,
        "arms": arm_reports,
        "paired_vs_vanilla": paired,
    }
    (out_dir / "independent-audit.json").write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
    return audit_payload


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
    parser.add_argument("--vot-toolkit-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=GATING_ARMS, default=["gate_skip_absent", "gate_skip_absent_keept1"])
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
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
            "host": "sam21_gating",
            "backbone": args.backbone,
            "arms": list(args.arms),
            "max_frames": args.max_frames,
            "keep_window": args.keep_window,
            "sdpa_kernel_mode": args.sdpa_kernel_mode,
            "checkpoint_sha256": digest(args.weights),
            "script_sha256": digest(Path(__file__)),
            "host_provenance": host_prov or {},
            "precision": "matched CUDA FP16 autocast",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "clips": {
                s["name"]: {"length": s["length"], "archive_sha256": s["archive_sha256"], "init_box": s["init_box"]}
                for s in all_specs
            },
            "post_initialization_label_reads": 0,
        }

    if args.merge_shards:
        if not freeze_path.exists():
            save(freeze_path, _build_freeze())
        operations = {}
        for spec in all_specs:
            nm = spec["name"]
            operations[nm] = {}
            for arm in args.arms:
                tpath = args.output / nm / f"{arm}.jsonl"
                spath = args.output / nm / f"{arm}.status.json"
                if not (tpath.exists() and spath.exists()):
                    raise FileNotFoundError(f"Missing shard output for {nm}/{arm}")
                op = json.loads(spath.read_text(encoding="utf-8"))
                if op["trace_sha256"] != digest(tpath):
                    raise ValueError(f"Trace SHA mismatch during merge for {nm}/{arm}")
                operations[nm][arm] = op
        if sum(s["label_reads"]["count"] for s in all_specs) != 0:
            raise RuntimeError("Causality violation before scoring")
        complete_path = args.output / "complete.json"
        complete_path.write_text(
            json.dumps({
                "freeze_sha256": digest(freeze_path),
                "post_initialization_label_reads_before_scoring": 0,
                "operations": operations,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        audit = _score_and_audit_merged(args.dataset, all_specs, args.output, args.arms, operations, args.vot_toolkit_path)
        _print_summary(args, audit)
        return

    shard_specs = [s for idx, s in enumerate(all_specs) if idx % args.num_shards == args.shard_idx]
    if not shard_specs:
        print(f"Shard {args.shard_idx}/{args.num_shards} has 0 clips; exiting cleanly.", flush=True)
        return

    adapter = SAM21HostAdapter(
        weights=args.weights,
        backbone=args.backbone,
        sam2_root=args.host_source,
        deps_dirs=[Path(d) for d in (args.deps or [])],
        use_sentry_runtime=False,
        device=args.device,
    )
    _configure_sdpa_mode(args.sdpa_kernel_mode)

    if args.num_shards == 1 or (args.shard_idx == 0 and not freeze_path.exists()):
        if not freeze_path.exists():
            save(freeze_path, _build_freeze(adapter.provenance))

    operations = {}
    for spec in shard_specs:
        nm = spec["name"]
        operations[nm] = {}
        for arm in args.arms:
            op, _ = run_gating_clip_arm(
                adapter=adapter,
                spec=spec,
                arm=arm,
                out_dir=args.output,
                keep_window=args.keep_window,
                max_frames=args.max_frames,
                resume=args.resume,
            )
            operations[nm][arm] = op

    if args.num_shards > 1:
        (args.output / f"shard_{args.shard_idx}_of_{args.num_shards}.json").write_text(
            json.dumps({"shard_idx": args.shard_idx, "num_shards": args.num_shards, "operations": operations}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"GATING_SHARD_DONE {args.shard_idx}/{args.num_shards} ({len(shard_specs)} clips)", flush=True)
        return

    if sum(s["label_reads"]["count"] for s in shard_specs) != 0:
        raise RuntimeError("Causality violation during inference")
    complete_path = args.output / "complete.json"
    complete_path.write_text(
        json.dumps({
            "freeze_sha256": digest(freeze_path),
            "post_initialization_label_reads_before_scoring": 0,
            "operations": operations,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = _score_and_audit_merged(args.dataset, shard_specs, args.output, args.arms, operations, args.vot_toolkit_path)
    _print_summary(args, audit)


def _print_summary(args, audit: dict):
    print(f"=== MEMORY GATING BENCHMARK SUMMARY ({args.backbone} on {args.dataset}) [{audit['status']}] ===", flush=True)
    for arm in args.arms:
        a = audit["arms"][arm]
        t = a["totals"]
        prf = a["fixed_point_macro_prf"]
        vtk = a["vot_lt_toolkit"]
        ls = a["lasot_metrics"]
        ms = a["ms"]
        vtk_str = f"{vtk['f_score']:.4f} (th={vtk['best_threshold']})" if vtk.get("available") else "N/A"
        print(
            f"  [{arm:24s}] timely={t['timely']} first={t['first']} fp={t['fp']} "
            f"vis_iou={t['visible_mean_iou']:.4f} | MacroP={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f} | "
            f"VOT-tk={vtk_str} | AUC_m={ls['matlab_conf']['auc']:.4f} | med={ms['median']:.1f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
