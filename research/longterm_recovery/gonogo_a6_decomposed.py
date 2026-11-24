"""Go/No-Go Experiment 2 & 4: Full Single-Factor Counterfactual Decomposition of SAM 2.1
Post-Absence Memory Inertia + Tensor Sanity Check + Exact Determinism Verification.

Evaluates 9 strictly controlled counterfactual memory conditions on the EXACT same silent frame
and cached image features in vanilla Meta SAM 2.1 Tiny (HOOT Dev, 4 clips, 3,025 frames):
  1. orig_memory                 : unmodified live memory (control; verified 0.0 diff vs native)
  2. strip_no_obj_spatial_only   : subtract no_obj_embed_spatial from silent spatial slots only
  3. drop_recent_spatial_keep_ptrs : drop ALL 6 non-cond spatial slots; KEEP all 15 recent ptrs
  4. drop_recent_ptrs_keep_spatial : KEEP all 6 recent empty spatial slots; drop ALL 15 non-cond ptrs
  5. m0_only                     : cond_0 only (0 non-cond spatial, 0 non-cond ptrs)
  6. m0_plus_1empty              : cond_0 + ONLY 1 most-recent silent slot (frame_idx - 1)
  7. m0_plus_mclean              : cond_0 + last certified clean visible anchor (CARE 1+1)
  8. m0_plus_6vis_condptr        : cond_0 + last 6 visible (obj>0) spatial slots + cond_0 ptr only
  9. m0_plus_6vis_15visptr       : cond_0 + last 6 visible spatial slots + last 15 visible ptrs

Also performs:
  - Direct tensor sanity check comparing `(stored_silent_maskmem - no_obj_embed_spatial)` against
    freshly encoded `memory_encoder(pix_feat, empty_mask)` without `no_obj_embed_spatial`, confirming
    whether the `-2.05 -> -4.05` logit drop under `strip_no_obj_spatial_only` is genuine model behavior.
  - Exact frame-by-frame determinism check against the frozen 2026-11-22 Meta SAM 2.1 baseline trace.
  - Reports pooled, per-video, AND per-qualifying-event breakdowns (never hiding video concentration).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
    DEFAULT_CLEAN_ANCHOR_OBJ,
    DEFAULT_CLEAN_ANCHOR_STRIDE,
    DEFAULT_MEMORY_KEEP_WINDOW,
    DEFAULT_RHO_MAX,
    DEFAULT_RHO_MIN,
    SAM21HostAdapter,
    SAM2NativeVerifier,
    _clone_mem_entry,
    _iter_spec_frames,
    _scalar,
    load_dataset_spec,
    mask_to_bbox,
)
from evaluation import box_iou
from prepare import digest, save
from tip_hoot_incumbent_competition import assess
from tip_scale_consistent_recovery import BORDER_MARGIN, border_gamma


ARMS = (
    "orig_memory",
    "strip_no_obj_spatial_only",
    "drop_recent_spatial_keep_ptrs",
    "drop_recent_ptrs_keep_spatial",
    "m0_only",
    "m0_plus_1empty",
    "m0_plus_mclean",
    "m0_plus_6vis_condptr",
    "m0_plus_6vis_15visptr",
)


def _custom_prepare_memory(
    predictor,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    output_dict,
    num_frames,
    track_in_reverse=False,
    *,
    skip_non_cond_spatial: bool = False,
    skip_non_cond_ptrs: bool = False,
    visible_spatial_list: list[dict] | None = None,
    visible_ptr_list: list[tuple[int, torch.Tensor]] | None = None,
):
    """Exact mirror of Meta SAM2Base._prepare_memory_conditioned_features with explicit factor switches."""
    from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

    B = current_vision_feats[-1].size(1)
    C = predictor.hidden_dim
    H, W = feat_sizes[-1]
    device = current_vision_feats[-1].device

    if predictor.num_maskmem == 0:
        return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

    num_obj_ptr_tokens = 0
    tpos_sign_mul = -1 if track_in_reverse else 1
    assert not is_init_cond_frame

    to_cat_memory, to_cat_memory_pos_embed = [], []
    cond_outputs = output_dict["cond_frame_outputs"]
    assert len(cond_outputs) > 0
    selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
        frame_idx, cond_outputs, predictor.max_cond_frames_in_attn
    )
    t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

    if visible_spatial_list is not None:
        # visible_spatial_list is chronological (oldest -> newest), length <= num_maskmem - 1
        n_vis = len(visible_spatial_list)
        for t_pos in range(1, predictor.num_maskmem):
            idx = t_pos - predictor.num_maskmem  # -(num_maskmem-1) .. -1
            if idx < -n_vis:
                continue
            t_pos_and_prevs.append((t_pos, visible_spatial_list[idx]))
    elif not skip_non_cond_spatial:
        stride = 1 if predictor.training else predictor.memory_temporal_stride_for_eval
        for t_pos in range(1, predictor.num_maskmem):
            t_rel = predictor.num_maskmem - t_pos
            if t_rel == 1:
                prev_frame_idx = frame_idx - t_rel if not track_in_reverse else frame_idx + t_rel
            else:
                if not track_in_reverse:
                    prev_frame_idx = ((frame_idx - 2) // stride) * stride - (t_rel - 2) * stride
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // stride) * stride + (t_rel - 2) * stride
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, out))

    for t_pos, prev in t_pos_and_prevs:
        if prev is None:
            continue
        feats = prev["maskmem_features"].to(device, non_blocking=True)
        to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
        maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
        maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
        maskmem_enc = maskmem_enc + predictor.maskmem_tpos_enc[predictor.num_maskmem - t_pos - 1]
        to_cat_memory_pos_embed.append(maskmem_enc)

    if predictor.use_obj_ptrs_in_encoder:
        max_obj_ptrs_in_encoder = min(num_frames, predictor.max_obj_ptrs_in_encoder)
        if not predictor.training and predictor.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond_outputs = {
                t: out
                for t, out in selected_cond_outputs.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
        else:
            ptr_cond_outputs = selected_cond_outputs
        pos_and_ptrs = [
            (
                (frame_idx - t) * tpos_sign_mul if predictor.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t),
                out["obj_ptr"],
            )
            for t, out in ptr_cond_outputs.items()
        ]
        if visible_ptr_list is not None:
            for t_diff, ptr in visible_ptr_list:
                pos_and_ptrs.append((t_diff, ptr))
        elif not skip_non_cond_ptrs:
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=0)
            if predictor.add_tpos_enc_to_obj_ptrs:
                t_diff_max = max_obj_ptrs_in_encoder - 1
                tpos_dim = C if predictor.proj_tpos_enc_in_obj_ptrs else predictor.mem_dim
                obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
                obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                obj_pos = predictor.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, predictor.mem_dim)
            else:
                obj_pos = obj_ptrs.new_zeros(len(pos_list), B, predictor.mem_dim)
            if predictor.mem_dim < C:
                obj_ptrs = obj_ptrs.reshape(-1, B, C // predictor.mem_dim, predictor.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = obj_pos.repeat_interleave(C // predictor.mem_dim, dim=0)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos_embed.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]
        else:
            num_obj_ptr_tokens = 0

    memory = torch.cat(to_cat_memory, dim=0)
    memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
    pix_feat_with_mem = predictor.memory_attention(
        curr=current_vision_feats,
        curr_pos=current_vision_pos_embeds,
        memory=memory,
        memory_pos=memory_pos_embed,
        num_obj_ptr_tokens=num_obj_ptr_tokens,
    )
    return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)


@contextmanager
def _patch_memory_prep(predictor, **kwargs):
    orig = predictor._prepare_memory_conditioned_features

    def _wrapped(
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        return _custom_prepare_memory(
            predictor,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse=track_in_reverse,
            **kwargs,
        )

    predictor._prepare_memory_conditioned_features = _wrapped
    try:
        yield
    finally:
        predictor._prepare_memory_conditioned_features = orig


def _eval_arm(adapter: SAM21HostAdapter, frame_idx: int, mem_dict: dict, patch_kwargs: dict | None = None) -> dict:
    _, _, vision_feats, vision_pos, feat_sizes = adapter.predictor._get_image_feature(
        adapter.state, frame_idx, 1
    )
    if patch_kwargs:
        with _patch_memory_prep(adapter.predictor, **patch_kwargs):
            cur_out = adapter.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=mem_dict,
                num_frames=frame_idx + 1,
                run_mem_encoder=False,
            )
    else:
        cur_out = adapter.predictor.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=mem_dict,
            num_frames=frame_idx + 1,
            run_mem_encoder=False,
        )
    _, video_mask = adapter.predictor._get_orig_video_res_output(adapter.state, cur_out["pred_masks"])
    binary = (video_mask[0, 0] > 0).to(torch.uint8).cpu().numpy()
    return {
        "bbox": mask_to_bbox(binary),
        "obj_logit": _scalar(cur_out.get("object_score_logits")),
    }


def _verify_strip_tensor_exactness(adapter: SAM21HostAdapter, frame_idx: int) -> dict:
    """Verify that `stored_silent_maskmem - no_obj_embed_spatial` equals freshly encoded
    `memory_encoder(pix_feat, empty_mask)` with NO `no_obj_embed_spatial` added."""
    obj_idx = adapter.state["obj_id_to_idx"][0]
    slot = adapter.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx)
    if slot is None or slot.get("maskmem_features") is None:
        return {"checked": False}

    _, _, vision_feats, _, feat_sizes = adapter.predictor._get_image_feature(adapter.state, frame_idx, 1)
    B = vision_feats[-1].size(1)
    C = adapter.predictor.hidden_dim
    H, W = feat_sizes[-1]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
    # Build exact empty mask logits (-10.0) passed to _encode_new_memory when object_score_logits <= 0
    empty_high_res = torch.full((B, 1, adapter.predictor.image_size, adapter.predictor.image_size), -10.0, device=pix_feat.device)
    mask_for_mem = torch.sigmoid(empty_high_res)
    if adapter.predictor.sigmoid_scale_for_mem_enc != 1.0:
        mask_for_mem = mask_for_mem * adapter.predictor.sigmoid_scale_for_mem_enc
    if adapter.predictor.sigmoid_bias_for_mem_enc != 0.0:
        mask_for_mem = mask_for_mem + adapter.predictor.sigmoid_bias_for_mem_enc
    raw_enc = adapter.predictor.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)["vision_features"]

    stored = slot["maskmem_features"].to(raw_enc.device)
    no_obj_embed = adapter.predictor.no_obj_embed_spatial.to(device=stored.device, dtype=stored.dtype)
    delta = no_obj_embed[..., None, None].expand(*stored.shape)
    stripped = stored - delta

    diff_raw_vs_stripped = (raw_enc.float() - stripped.float()).abs()
    diff_raw_vs_stored = (raw_enc.float() - stored.float()).abs()
    return {
        "checked": True,
        "frame_idx": frame_idx,
        "stored_dtype": str(stored.dtype),
        "raw_enc_dtype": str(raw_enc.dtype),
        "stored_shape": list(stored.shape),
        "no_obj_embed_shape": list(no_obj_embed.shape),
        "max_abs_diff_stripped_vs_fresh_raw": round(float(diff_raw_vs_stripped.max().item()), 6),
        "mean_abs_diff_stripped_vs_fresh_raw": round(float(diff_raw_vs_stripped.mean().item()), 6),
        "max_abs_diff_stored_vs_fresh_raw": round(float(diff_raw_vs_stored.max().item()), 6),
        "mean_abs_diff_stored_vs_fresh_raw": round(float(diff_raw_vs_stored.mean().item()), 6),
        "raw_enc_l2_norm": round(float(raw_enc.float().norm().item()), 4),
        "stored_l2_norm": round(float(stored.float().norm().item()), 4),
        "delta_l2_norm": round(float(delta.float().norm().item()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--deps", type=Path, nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref-baseline-dir", type=Path, default=None, help="Frozen 2026-11-22 Meta SAM 2.1 baseline dir for determinism check")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False

    specs = load_dataset_spec("hoot_dev", args.data)
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
        "script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(args.weights),
        "host_provenance": adapter.provenance,
        "arms": list(ARMS),
        "clips": {s["name"]: {"length": s["length"], "archive_sha256": s["archive_sha256"]} for s in specs},
        "post_initialization_label_reads": 0,
    }
    save(args.output / "freeze.json", freeze)

    verifier = SAM2NativeVerifier()
    per_video_rows: dict[str, list[dict]] = {}
    tensor_sanity: dict | None = None
    mirror_sanity: dict | None = None

    for spec in specs:
        name = spec["name"]
        print(f"\n[A6-Decomposed] Running {name} ({spec['length']} frames)...", flush=True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        out_path = args.output / f"{name}.jsonl"
        rows = []
        last_clean_out = None
        last_clean_frame = 0
        # Keep chronological history of visible (obj>0) memory entries (spatial up to 6, ptrs up to 15)
        vis_spatial_ring: list[dict] = []
        vis_ptr_ring: list[tuple[int, torch.Tensor]] = []
        silent_streak = 0
        init_box = spec["init_box"]
        a0 = max(1.0, init_box[2] * init_box[3])

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            with out_path.open("w", encoding="utf-8") as fh:
                for frame_idx, _, encoded in _iter_spec_frames(spec, None):
                    with Image.open(BytesIO(encoded)) as img:
                        pil_rgb = img.convert("RGB")
                        rgb_np = np.asarray(pil_rgb)
                    h, w = rgb_np.shape[:2]
                    if frame_idx == 0:
                        init_res = adapter.initialize(pil_rgb, init_box)
                        verifier.initialize(rgb_np, init_box, init_res["pred_mask"], adapter.get_cached_feature_map(), w, h)
                        row0 = {"frame": 0, "native_silent": False, "silent_streak": 0, "native_bbox": init_box, "native_obj": None}
                        rows.append(row0)
                        fh.write(json.dumps(row0) + "\n")
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
                            # Update certified clean anchor identically to CARE
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

                        rec = {
                            "frame": frame_idx,
                            "native_silent": False,
                            "silent_streak": 0,
                            "native_bbox": native_bbox,
                            "native_obj": native_obj,
                        }
                    else:
                        silent_streak += 1
                        if tensor_sanity is None and silent_streak >= 2:
                            tensor_sanity = _verify_strip_tensor_exactness(adapter, frame_idx)
                            print(f"  [Tensor Sanity] {json.dumps(tensor_sanity)}", flush=True)

                        obj_idx = adapter.state["obj_id_to_idx"][0]
                        obj_dict = adapter.state["output_dict_per_obj"][obj_idx]
                        cond_outputs = obj_dict["cond_frame_outputs"]
                        non_cond = obj_dict["non_cond_frame_outputs"]
                        cond_0 = {0: cond_outputs[0]} if 0 in cond_outputs else cond_outputs
                        num_spatial = int(adapter.predictor.num_maskmem) - 1
                        num_ptrs = int(adapter.predictor.max_obj_ptrs_in_encoder) - 1
                        spatial_frames = [k for k in range(max(1, frame_idx - num_spatial), frame_idx) if k in non_cond]
                        ptr_frames = [k for k in range(max(1, frame_idx - num_ptrs), frame_idx) if k in non_cond]

                        # Verify custom_prepare_memory exactness on very first silent frame
                        if mirror_sanity is None:
                            ref_out = _eval_arm(adapter, frame_idx, {"cond_frame_outputs": cond_outputs, "non_cond_frame_outputs": non_cond})
                            mir_out = _eval_arm(
                                adapter,
                                frame_idx,
                                {"cond_frame_outputs": cond_outputs, "non_cond_frame_outputs": non_cond},
                                patch_kwargs={"skip_non_cond_spatial": False, "skip_non_cond_ptrs": False},
                            )
                            mirror_sanity = {
                                "frame_idx": frame_idx,
                                "video": name,
                                "ref_obj": ref_out["obj_logit"],
                                "mir_obj": mir_out["obj_logit"],
                                "obj_abs_diff": abs((ref_out["obj_logit"] or 0.0) - (mir_out["obj_logit"] or 0.0)),
                                "bbox_equal": ref_out["bbox"] == mir_out["bbox"],
                            }
                            print(f"  [Mirror Sanity] {json.dumps(mirror_sanity)}", flush=True)

                        # Build strip_no_obj_spatial_only dict
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

                        # Visible pointers with actual temporal distance (frame_idx - t)
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

                        no_obj_sp = sum(
                            1 for k in spatial_frames if (_scalar(non_cond[k].get("object_score_logits")) or -1.0) <= 0.0
                        )
                        no_obj_pt = sum(
                            1 for k in ptr_frames if (_scalar(non_cond[k].get("object_score_logits")) or -1.0) <= 0.0
                        )
                        rec = {
                            "frame": frame_idx,
                            "native_silent": True,
                            "silent_streak": silent_streak,
                            "native_bbox": None,
                            "native_obj": native_obj,
                            "slots": {
                                "spatial_present": len(spatial_frames),
                                "spatial_no_obj": no_obj_sp,
                                "ptr_present": len(ptr_frames),
                                "ptr_no_obj": no_obj_pt,
                                "vis_spatial_avail": len(vis_spatial_ring),
                                "vis_ptr_avail": len(vis_ptr_ring),
                                "has_last_clean": last_clean_out is not None,
                            },
                            "arms": arm_res,
                        }
                    rows.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    adapter.prune_old_memory(DEFAULT_MEMORY_KEEP_WINDOW)

        per_video_rows[name] = rows
        adapter.release_clip()
        torch.cuda.empty_cache()

    # Strict causality assertion
    assert sum(s["label_reads"]["count"] for s in specs) == 0, "Causality violated!"

    # Determinism check vs frozen 2026-11-22 Meta SAM 2.1 baseline
    determinism = {}
    if args.ref_baseline_dir is not None and args.ref_baseline_dir.exists():
        for spec in specs:
            nm = spec["name"]
            ref_p = args.ref_baseline_dir / nm / "baseline.jsonl"
            if ref_p.exists():
                ref_rows = [json.loads(l) for l in ref_p.open(encoding="utf-8")]
                cur_rows = per_video_rows[nm]
                box_diffs = sum(a["bbox"] != b["native_bbox"] for a, b in zip(ref_rows, cur_rows))
                obj_max_diff = max(
                    (abs((a["object_score"] or 0.0) - (b["native_obj"] or 0.0)) for a, b in zip(ref_rows[1:], cur_rows[1:])),
                    default=0.0,
                )
                determinism[nm] = {"frames": len(cur_rows), "bbox_diff_frames": box_diffs, "max_obj_abs_diff": obj_max_diff}

    # Strictly post-hoc evaluation (overall + per-video + per-event)
    def _empty_vis():
        return {a: {"n": 0, "pos_obj": 0, "iou_ge_05": 0, "obj_sum": 0.0, "iou_sum": 0.0} for a in ARMS}

    def _empty_abs():
        return {a: {"n": 0, "pos_obj": 0, "obj_sum": 0.0} for a in ARMS}

    pooled_vis = _empty_vis()
    pooled_abs = _empty_abs()
    by_video_vis = {}
    by_video_abs = {}
    event_summaries = []

    for spec in specs:
        nm = spec["name"]
        labels = spec["load_labels"]()
        rows = per_video_rows[nm]
        native_boxes = [r["native_bbox"] for r in rows]
        ev_info = assess(labels, native_boxes)["windows"]["0.5"]

        vv = _empty_vis()
        va = _empty_abs()
        for r in rows:
            if not r["native_silent"]:
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
                "mean_obj": round(s["obj_sum"] / max(1, s["n"]), 4),
                "mean_iou": round(s["iou_sum"] / max(1, s["n"]), 4),
            }
            for a, s in vv.items()
        }
        by_video_abs[nm] = {
            a: {
                "n": s["n"],
                "pos_obj": s["pos_obj"],
                "mean_obj": round(s["obj_sum"] / max(1, s["n"]), 4),
            }
            for a, s in va.items()
        }

        for ev in ev_info:
            if not ev.get("eligible"):
                continue
            t0 = int(ev["t0"])
            win_rows = rows[t0 : min(len(rows), t0 + 10)]
            silent_win = [r for r in win_rows if r["native_silent"]]
            rec = {
                "video": nm,
                "t0": t0,
                "baseline_category": ev["category"],
                "silent_frames_in_window": len(silent_win),
                "streak_at_t0": rows[t0]["silent_streak"],
                "slots_at_t0": rows[t0].get("slots"),
                "arm_window_best_iou": {},
                "arm_window_max_obj": {},
            }
            for a in ARMS:
                best_iou = 0.0
                max_obj = -999.0
                for r in silent_win:
                    gt = labels[r["frame"]]
                    ao = r["arms"][a]
                    if ao["obj_logit"] is not None:
                        max_obj = max(max_obj, ao["obj_logit"])
                    iou = box_iou(ao["bbox"], gt) if (gt is not None and ao["bbox"] is not None) else 0.0
                    best_iou = max(best_iou, iou)
                rec["arm_window_best_iou"][a] = round(best_iou, 4)
                rec["arm_window_max_obj"][a] = round(max_obj, 3) if silent_win else None
            event_summaries.append(rec)

    results = {
        "freeze_sha256": digest(args.output / "freeze.json"),
        "tensor_sanity": tensor_sanity,
        "mirror_sanity": mirror_sanity,
        "determinism_vs_ref": determinism,
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
        "by_video_silent_visible": by_video_vis,
        "by_video_silent_absent": by_video_abs,
        "qualifying_events": event_summaries,
    }
    save(args.output / "results.json", results)

    print("\n================ A6 DECOMPOSED SUMMARY ================")
    print("Determinism vs 2026-11-22 Meta baseline:", json.dumps(determinism))
    print("Tensor sanity:", json.dumps(tensor_sanity))
    print("Mirror sanity:", json.dumps(mirror_sanity))
    print("\nPooled Silent VISIBLE frames:")
    for a, d in results["pooled_silent_visible"].items():
        sa = results["pooled_silent_absent"][a]
        print(
            f"  {a:30s} | vis(n={d['n']}): obj>0={d['pos_obj']:3d}, iou>=.5={d['iou_ge_05']:3d}, "
            f"mean_obj={d['mean_obj']:+.4f}, mean_iou={d['mean_iou']:.4f} | abs_obj>0={sa['pos_obj']:3d}/{sa['n']}"
        )
    print("\nPer-Video Silent VISIBLE (iou>=0.5 / n) & ABSENT (obj>0 / n):")
    for nm in sorted(by_video_vis):
        print(f"  -- {nm} (vis_n={by_video_vis[nm]['orig_memory']['n']}, abs_n={by_video_abs[nm]['orig_memory']['n']}) --")
        for a in ARMS:
            v = by_video_vis[nm][a]
            ab = by_video_abs[nm][a]
            print(f"    {a:30s}: vis_obj>0={v['pos_obj']:3d}, iou>=.5={v['iou_ge_05']:3d}, mean_obj={v['mean_obj']:+.3f}, mean_iou={v['mean_iou']:.4f} | abs_FP={ab['pos_obj']:3d}/{ab['n']}")
    print("DONE:", args.output)


if __name__ == "__main__":
    main()
