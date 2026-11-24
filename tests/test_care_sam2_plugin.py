"""Unit tests for CARE plug-and-play SAM 2.1 / SAMURAI / DAM4SAM recovery plugin."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/longterm_recovery"))

from audit_care_sam2_plugin import audit_run
from care_sam2_plugin import (
    BaseHostAdapter,
    CAREController,
    NoneVerifier,
    SAM2NativeVerifier,
    SAMURAIHostAdapter,
    _dir_frames_manifest_sha,
    lasot_metrics,
    load_dataset_spec,
    macro_prf,
    mask_to_bbox,
    score_traces,
    vot_lt_curve_metrics,
    vot_lt_toolkit_metrics,
)

LOCAL_VOT_TOOLKIT = Path(os.environ["VOT_TOOLKIT_PYTHON"]) if os.environ.get("VOT_TOOLKIT_PYTHON") else Path("__no_local_vot_toolkit__")


class _MockHostAdapter(BaseHostAdapter):
    host_name = "mock"

    def __init__(self, feat_map, probe_plan, rollback_result=(None, -2.0)):
        super().__init__(device="cpu")
        self._feat_map = feat_map
        self._probe_plan = probe_plan
        self._rollback_result = rollback_result
        self.slots = {}
        self.committed = []
        self.rolled_back = []
        self.probe_calls = []

    def get_cached_feature_map(self):
        return self._feat_map

    def get_non_cond_slot(self, frame_idx):
        return self.slots.get(frame_idx)

    def dual_anchor_probe(self, last_clean_out, use_dual_anchor=True):
        self.probe_calls.append((self.frame_idx, use_dual_anchor, last_clean_out is not None))
        return self._probe_plan[self.frame_idx]

    def commit_reentry(self, probe, purge_window):
        self.committed.append((self.frame_idx, probe["box"], purge_window))
        return {"applied": True, "frame_idx": self.frame_idx, "evicted_no_obj": 6, "saved_orig": {}, "saved_evicted": {}}

    def rollback_reentry(self, inj_frame, snapshot):
        self.rolled_back.append((self.frame_idx, inj_frame))
        return self._rollback_result

    def prune_old_memory(self, keep_window):
        pass


class CAREPluginTests(unittest.TestCase):
    def _make_synthetic_feature_scene(self, h=320, w=480, hf=64, wf=64, c=64):
        rng = np.random.default_rng(42)
        bg_vec = rng.normal(size=(c,)).astype(np.float32)
        bg_vec /= np.linalg.norm(bg_vec)
        fg_vec = rng.normal(size=(c,)).astype(np.float32)
        fg_vec -= (fg_vec @ bg_vec) * bg_vec
        fg_vec /= np.linalg.norm(fg_vec)
        dist_vec = rng.normal(size=(c,)).astype(np.float32)
        dist_vec -= (dist_vec @ bg_vec) * bg_vec + (dist_vec @ fg_vec) * fg_vec
        dist_vec /= np.linalg.norm(dist_vec)
        bias = rng.normal(size=(c, 1, 1)).astype(np.float32) * 0.05

        feat0 = np.broadcast_to(bg_vec[:, None, None], (c, hf, wf)).copy() + bias
        feat0 += rng.normal(scale=0.05, size=(c, hf, wf)).astype(np.float32)
        x0_f, x1_f = int(60 / w * wf), int(140 / w * wf)
        y0_f, y1_f = int(80 / h * hf), int(160 / h * hf)
        feat0[:, y0_f:y1_f, x0_f:x1_f] = fg_vec[:, None, None] + bias

        mask0 = np.zeros((h, w), dtype=np.uint8)
        mask0[80:160, 60:140] = 1
        box0 = [60.0, 80.0, 80.0, 80.0]

        featt = np.broadcast_to(bg_vec[:, None, None], (c, hf, wf)).copy() + bias * 2.0
        featt += rng.normal(scale=0.05, size=(c, hf, wf)).astype(np.float32)
        tx0, tx1 = int(300 / w * wf), int(380 / w * wf)
        ty0, ty1 = int(180 / h * hf), int(260 / h * hf)
        featt[:, ty0:ty1, tx0:tx1] = fg_vec[:, None, None] + bias * 2.0
        dx0, dx1 = int(200 / w * wf), int(280 / w * wf)
        dy0, dy1 = int(40 / h * hf), int(120 / h * hf)
        featt[:, dy0:dy1, dx0:dx1] = dist_vec[:, None, None] + bias * 2.0

        mask_true = np.zeros((h, w), dtype=np.uint8)
        mask_true[180:260, 300:380] = 1
        box_true = [300.0, 180.0, 80.0, 80.0]

        mask_dist = np.zeros((h, w), dtype=np.uint8)
        mask_dist[40:120, 200:280] = 1
        box_dist = [200.0, 40.0, 80.0, 80.0]

        return (
            torch.from_numpy(feat0[None]),
            mask0,
            box0,
            torch.from_numpy(featt[None]),
            mask_true,
            box_true,
            mask_dist,
            box_dist,
            h,
            w,
        )

    def test_sam2_native_verifier_discriminates_target_vs_distractor_under_bias_shift(self):
        feat0, mask0, box0, featt, mask_true, box_true, mask_dist, box_dist, h, w = self._make_synthetic_feature_scene()
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        v = SAM2NativeVerifier()
        cal = v.initialize(rgb, box0, mask0, feat0, w, h)
        self.assertTrue(cal["calibrated"])
        self.assertGreater(cal["neg_count"], 0)

        ok_true, diag_true = v.verify_probe(rgb, box_true, mask_true, featt, w, h, obj_score=3.0, gamma=1.0, tau_obj_border=3.5)
        ok_dist, diag_dist = v.verify_probe(rgb, box_dist, mask_dist, featt, w, h, obj_score=3.0, gamma=1.0, tau_obj_border=3.5)
        self.assertTrue(ok_true, f"Expected true shifted target to pass: {diag_true}, floor={cal['fgc_floor']}")
        self.assertFalse(ok_dist, f"Expected distractor to fail: {diag_dist}, floor={cal['fgc_floor']}")
        self.assertGreater(diag_true["fgc"], diag_dist["fgc"] + 0.3)

    def test_samurai_configure_arm_defaults_matched_and_supports_2x2_factorial(self):
        class _DummyPred:
            min_obj_score_logits = -1.0

        adapter = SAMURAIHostAdapter.__new__(SAMURAIHostAdapter)
        adapter.predictor = _DummyPred()
        adapter._stock_min_obj = -1.0

        # Default (None, None) MUST match stock -1.0 on BOTH arms (no silent threshold confounding!)
        default_params = {"samurai_baseline_min_obj": None, "samurai_care_min_obj": None}
        adapter.configure_arm("baseline", default_params)
        self.assertEqual(adapter.predictor.min_obj_score_logits, -1.0)
        adapter.configure_arm("care", default_params)
        self.assertEqual(adapter.predictor.min_obj_score_logits, -1.0)

        # Explicit 2x2 matched 0.0 cell
        cell_0 = {"samurai_baseline_min_obj": 0.0, "samurai_care_min_obj": 0.0}
        adapter.configure_arm("baseline", cell_0)
        self.assertEqual(adapter.predictor.min_obj_score_logits, 0.0)
        adapter.configure_arm("care", cell_0)
        self.assertEqual(adapter.predictor.min_obj_score_logits, 0.0)

    def test_care_controller_two_frame_confirm_and_one_step_rollback(self):
        h, w = 200, 300
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        init_box = [50.0, 50.0, 60.0, 60.0]
        init_mask = np.zeros((h, w), dtype=np.uint8)
        init_mask[50:110, 50:110] = 1

        probe_box = [100.0, 60.0, 60.0, 60.0]
        probe_mask = np.zeros((h, w), dtype=np.uint8)
        probe_mask[60:120, 100:160] = 1
        good_probe = {"cur_out": {}, "mask": probe_mask, "box": probe_box, "obj_score": 3.5, "quality": 0.85}

        adapter = _MockHostAdapter(feat_map=None, probe_plan={7: good_probe, 8: good_probe}, rollback_result=(None, -2.5))
        adapter.slots[1] = {"obj_ptr": torch.zeros(1, 4)}
        ctrl = CAREController(verifier=NoneVerifier(), k_sil=6, enable_rollback=True)
        ctrl.initialize(rgb, init_box, init_mask, adapter)
        stats = {k: 0 for k in ("anchor_checks", "anchor_updates", "probe_calls", "verified_probes", "memory_updates", "rollbacks")}

        adapter.frame_idx = 1
        box, src, upd, _ = ctrl.step(1, rgb, adapter, init_box, init_mask, 5.0, stats)
        self.assertEqual(src, "baseline")
        self.assertEqual(stats["anchor_updates"], 1)

        for f in range(2, 7):
            adapter.frame_idx = f
            box, src, upd, _ = ctrl.step(f, rgb, adapter, None, None, -3.0, stats)
            self.assertIsNone(box)
            self.assertEqual(stats["probe_calls"], 0)

        adapter.frame_idx = 7
        box, src, upd, diag = ctrl.step(7, rgb, adapter, None, None, -3.0, stats)
        self.assertIsNone(box)
        self.assertTrue(diag.get("pending"))
        self.assertEqual(stats["memory_updates"], 0)

        adapter.frame_idx = 8
        box, src, upd, _ = ctrl.step(8, rgb, adapter, None, None, -3.0, stats)
        self.assertEqual(box, probe_box)
        self.assertEqual(src, "recovered_care")
        self.assertTrue(upd)
        self.assertEqual(stats["memory_updates"], 1)
        self.assertEqual(len(adapter.committed), 1)

        adapter.frame_idx = 9
        box, src, upd, diag = ctrl.step(9, rgb, adapter, probe_box, probe_mask, 0.2, stats)
        self.assertIsNone(box)
        self.assertEqual(src, "rollback")
        self.assertEqual(stats["rollbacks"], 1)
        self.assertEqual(adapter.rolled_back, [(9, 8)])

        adapter.frame_idx = 10
        box, src, upd, _ = ctrl.step(10, rgb, adapter, None, None, -3.0, stats)
        self.assertEqual(stats["probe_calls"], 2)

    def test_custom_and_official_vot_lt_and_dual_protocol_lasot_metrics(self):
        # 4 frames: frames 0,1,3 visible exact match; frame 2 absent with low-conf FP
        gt = [[0, 0, 20, 20], [0, 0, 20, 20], None, [0, 0, 20, 20]]
        pred = [[0, 0, 20, 20], [0, 0, 20, 20], [100, 100, 20, 20], [0, 0, 20, 20]]
        confs = [1.0, 0.95, 0.10, 0.90]

        vcu = vot_lt_curve_metrics({"seq1": (gt, pred, confs)})
        self.assertAlmostEqual(vcu["f_score"], 1.0, places=5)
        self.assertGreater(vcu["best_threshold"], 0.10)

        # Dual-protocol LaSOT:
        # Visible frames = 3/4, all IoU=1.0 -> vis_ge AUC = 1.0, vis_strict (t=1.0 -> 0) = 20/21,
        # matlab_conf (strict >, denom = 4 total frames) = (20/21) * (3/4) = 15/21
        ls = lasot_metrics({"seq1": (gt, pred)})
        self.assertAlmostEqual(ls["auc"], 1.0, places=5)
        self.assertAlmostEqual(ls["vis_strict"]["auc"], 20.0 / 21.0, places=5)
        self.assertAlmostEqual(ls["matlab_conf"]["auc"], (20.0 / 21.0) * (3.0 / 4.0), places=5)
        self.assertAlmostEqual(ls["p20"], 1.0, places=5)
        self.assertAlmostEqual(ls["matlab_conf"]["p20"], 0.75, places=5)

        # Official vot-toolkit 0.9.0 test + anti-inflation check on abstained absent frames
        if LOCAL_VOT_TOOLKIT.exists():
            # Case with many absent frames where tracker correctly abstains (None):
            # Recall MUST NOT exceed 1.0 and empty-empty overlap=1.0 MUST NOT inflate Recall!
            gt2 = [[10, 10, 20, 20], [10, 10, 20, 20], None, None, None, None, [10, 10, 20, 20]]
            pred2 = [[10, 10, 20, 20], [10, 10, 20, 20], None, None, None, None, None]
            confs2 = [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
            vtk = vot_lt_toolkit_metrics({"seq2": (gt2, pred2, confs2, (200, 200))}, toolkit_path=LOCAL_VOT_TOOLKIT)
            self.assertTrue(vtk["available"])
            self.assertEqual(vtk["toolkit_version"], "0.9.0")
            # Visible frames in gt2 = 3 (frames 0,1,6); only frame 1 predicted -> Recall = 1/3, Precision = 1.0
            self.assertLessEqual(vtk["recall"], 0.34)
            self.assertAlmostEqual(vtk["recall"], 1.0 / 3.0, places=3)
            self.assertAlmostEqual(vtk["precision"], 1.0, places=3)

    def test_dir_loaders_strictly_causal_line_zero_only(self):
        """Modifying post-init lines in groundtruth.txt MUST NOT affect load_dataset_spec() or archive_sha256."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ds_kind in ("vot_lt_dir", "lasot_ext"):
                ds = root / ds_kind
                seq = ds / "clip1"
                img_sub = (seq / "color") if ds_kind == "vot_lt_dir" else (seq / "img")
                img_sub.mkdir(parents=True)
                for i in range(4):
                    (img_sub / f"{i:06d}.jpg").write_bytes(f"frame-{i}".encode("ascii"))
                gt_path = seq / "groundtruth.txt"
                gt_path.write_text("10,20,30,40\nCORRUPT_POST_INIT_LINE_1\nCORRUPT_2\nCORRUPT_3\n", encoding="utf-8")
                specs_a = load_dataset_spec(ds_kind, ds)
                self.assertEqual(len(specs_a), 1)
                self.assertEqual(specs_a[0]["init_box"], [10.0, 20.0, 30.0, 40.0])
                self.assertEqual(specs_a[0]["label_reads"]["count"], 0)
                sha_a = specs_a[0]["archive_sha256"]

                # Mutate post-init labels -> manifest hash MUST remain identical (proves post-init lines weren't read!)
                gt_path.write_text("10,20,30,40\n1,2,3,4\n5,6,7,8\n9,10,11,12\n", encoding="utf-8")
                specs_b = load_dataset_spec(ds_kind, ds)
                self.assertEqual(specs_b[0]["archive_sha256"], sha_a)
                self.assertEqual(specs_b[0]["label_reads"]["count"], 0)

    def test_auditor_verifies_valid_run_and_catches_tampering(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "vot_dir"
            seq = ds / "toy"
            (seq / "color").mkdir(parents=True)
            (ds / "list.txt").write_text("toy\n", encoding="utf-8")
            gt_lines = ["0,0,10,10"] + ["nan,nan,nan,nan"] * 5 + ["0,0,10,10"] * 12
            (seq / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
            frame_paths = []
            frame_shas = []
            for i in range(len(gt_lines)):
                p = seq / "color" / f"{i:06d}.jpg"
                im = Image.new("RGB", (64, 64), color=(i * 10 % 255, 0, 0))
                im.save(p, format="JPEG")
                frame_paths.append(p)
                frame_shas.append(hashlib.sha256(p.read_bytes()).hexdigest())

            run_dir = root / "run"
            (run_dir / "toy").mkdir(parents=True)
            manifest_hash = _dir_frames_manifest_sha("toy", frame_paths)

            trace_path = run_dir / "toy" / "baseline.jsonl"
            rows = []
            for i in range(len(gt_lines)):
                box = [0.0, 0.0, 10.0, 10.0] if (i == 0 or i >= 6) else None
                rows.append(json.dumps({
                    "frame": i,
                    "sha256": frame_shas[i],
                    "bbox": box,
                    "native_bbox": box,
                    "object_score": 5.0 if box else -5.0,
                    "confidence": 0.99 if box else 0.0,
                    "source": "init" if i == 0 else "baseline",
                    "memory_update": False,
                    "synchronized_model_ms": 10.0,
                    "diag": {},
                }))
            trace_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()

            freeze_path = run_dir / "freeze.json"
            freeze_path.write_text(json.dumps({
                "dataset": "vot_lt_dir",
                "host": "sam21",
                "backbone": "tiny",
                "verifier": "sam2_native",
                "clips": {"toy": {"length": len(gt_lines), "archive_sha256": manifest_hash, "init_box": [0.0, 0.0, 10.0, 10.0]}},
                "arms": ["baseline"],
                "post_initialization_label_reads": 0,
            }), encoding="utf-8")
            freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()

            ops = {"toy": {"baseline": {"frames": len(gt_lines), "width": 64, "height": 64, "trace_sha256": trace_sha, "stats": {}}}}
            complete_path = run_dir / "complete.json"
            complete_path.write_text(json.dumps({
                "freeze_sha256": freeze_sha,
                "post_initialization_label_reads_before_scoring": 0,
                "operations": ops,
            }), encoding="utf-8")
            complete_sha = hashlib.sha256(complete_path.read_bytes()).hexdigest()

            specs = load_dataset_spec("vot_lt_dir", ds)
            tk = LOCAL_VOT_TOOLKIT if LOCAL_VOT_TOOLKIT.exists() else None
            scored = score_traces(specs, {"toy": {"baseline": trace_path}}, ["baseline"], operations=ops, vot_toolkit_path=tk)
            results_path = run_dir / "results.json"
            results_path.write_text(json.dumps({
                "freeze_sha256": freeze_sha,
                "complete_sha256": complete_sha,
                **scored,
            }), encoding="utf-8")

            report = audit_run("vot_lt_dir", ds, run_dir, vot_toolkit_path=tk)
            self.assertEqual(report["status"], "AUDIT_OK")
            self.assertAlmostEqual(report["arms"]["baseline"]["fixed_point_macro_prf"]["f1"], 1.0)
            if tk is not None:
                self.assertTrue(report["arms"]["baseline"]["vot_lt_toolkit"]["available"])

            # Tamper with results.json -> auditor MUST reject!
            bad = json.loads(results_path.read_text(encoding="utf-8"))
            bad["arms"]["baseline"]["totals"]["visible_mean_iou"] = 0.123
            results_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_run("vot_lt_dir", ds, run_dir, vot_toolkit_path=tk)


if __name__ == "__main__":
    unittest.main()
