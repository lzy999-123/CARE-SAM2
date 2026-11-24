"""Unit tests for the 2026-11-22 positioning-switch cloud runners, merge/audit pipelines,
paired video bootstrap CIs, and cloud shell script hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "research" / "longterm_recovery"))

from gonogo_a6_decomposed import ARMS as A6_ARMS
from prepare import digest
from run_counterfactual_inertia_benchmark import _merge_and_score_probe
from run_memory_gating_benchmark import _score_and_audit_merged
from summarize_cloud_results import build_report, paired_video_bootstrap_ci


class CloudPositioningRunnersTests(unittest.TestCase):
    def test_cloud_shell_scripts_are_pure_lf_and_remove_wasted_care_hosts(self):
        cloud_dir = R / "research" / "longterm_recovery" / "cloud"
        for sh in sorted(cloud_dir.glob("*.sh")):
            raw = sh.read_bytes()
            self.assertNotIn(b"\r", raw, f"CRLF found in {sh.name} — Linux bash will fail")
        suite = (cloud_dir / "run_full_suite.sh").read_text(encoding="utf-8")
        # Must include new gating & 9-arm counterfactual runs
        self.assertIn("run_memory_gating_benchmark.py", suite)
        self.assertIn("run_counterfactual_inertia_benchmark.py", suite)
        self.assertIn("summarize_cloud_results.py", suite)
        self.assertIn("vot22_sam21_tiny_gating", suite)
        self.assertIn("vot22_sam21_tiny_a6_decomposed", suite)
        # Must NOT run wasted +care arms on SAMURAI or DAM4SAM
        for line in suite.splitlines():
            if "samurai" in line and "run_care_sharded" in line:
                self.assertNotIn('"baseline care"', line)
            if "dam4sam" in line and "run_care_sharded" in line:
                self.assertNotIn('"baseline care"', line)

    def test_memory_gating_audit_verifies_and_catches_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            labels = [[10.0, 10.0, 20.0, 20.0]] * 10
            spec = {
                "name": "clip_a",
                "length": 10,
                "archive_sha256": "abc",
                "init_box": labels[0],
                "load_labels": lambda: labels,
            }
            (out_dir / "freeze.json").write_text(json.dumps({"backbone": "tiny", "script_sha256": "sha1"}), encoding="utf-8")
            clip_dir = out_dir / "clip_a"
            clip_dir.mkdir()

            ops = {"clip_a": {}}
            for arm in ("vanilla", "gate_skip_absent"):
                tp = clip_dir / f"{arm}.jsonl"
                rows = []
                for f in range(10):
                    is_sil = (arm == "vanilla" and f >= 3)
                    rows.append({
                        "frame": f,
                        "sha256": f"pix_{f}",
                        "bbox": None if is_sil else [10.0, 10.0, 20.0, 20.0],
                        "native_bbox": None if is_sil else [10.0, 10.0, 20.0, 20.0],
                        "object_score": None if f == 0 else (-2.0 if is_sil else 5.0),
                        "confidence": 1.0 if f == 0 else (0.0 if is_sil else 0.99),
                        "source": "init" if f == 0 else "baseline",
                        "memory_update": False,
                        "synchronized_model_ms": 10.0,
                        "diag": {},
                    })
                tp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                ops["clip_a"][arm] = {"frames": 10, "width": 100, "height": 100, "trace_sha256": digest(tp)}

            (out_dir / "complete.json").write_text(json.dumps({"operations": ops}), encoding="utf-8")
            audit = _score_and_audit_merged("hoot_dev", [spec], out_dir, ["vanilla", "gate_skip_absent"], ops, None)
            self.assertEqual(audit["status"], "AUDIT_OK")
            # Vanilla silent frames 3..9 -> streak reaches 6 at frames 8 and 9 (2 frames)
            self.assertEqual(audit["arms"]["vanilla"]["silence_prevalence"]["clip_a"]["silent_ge6_visible"], 2)
            self.assertEqual(audit["arms"]["gate_skip_absent"]["silence_prevalence"]["clip_a"]["silent_ge6_visible"], 0)
            self.assertGreater(
                audit["arms"]["gate_skip_absent"]["fixed_point_macro_prf"]["f1"],
                audit["arms"]["vanilla"]["fixed_point_macro_prf"]["f1"],
            )

            # Tamper with trace -> must fail audit
            tp = clip_dir / "vanilla.jsonl"
            tp.write_text(tp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _score_and_audit_merged("hoot_dev", [spec], out_dir, ["vanilla", "gate_skip_absent"], ops, None)

    def test_counterfactual_merge_weights_videos_equally_in_video_macro(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "freeze.json").write_text(json.dumps({"backbone": "tiny"}), encoding="utf-8")

            # Short video (2 silent visible frames, 100% recovered under drop_recent_spatial_keep_ptrs)
            # Long video (8 silent visible frames, 0% recovered)
            # Pooled rate = 2/10 = 0.20, Video-macro rate = (1.0 + 0.0)/2 = 0.50
            gt_box = [10.0, 10.0, 20.0, 20.0]
            specs = []
            ops = {}
            for nm, n_sil, hits in [("short_v", 2, True), ("long_v", 8, False)]:
                cdir = out_dir / nm
                cdir.mkdir()
                rows = [{"frame": 0, "native_silent": False, "silent_streak": 0, "native_bbox": gt_box, "native_obj": None}]
                for f in range(1, n_sil + 1):
                    arms = {
                        a: {
                            "bbox": gt_box if (hits and a == "drop_recent_spatial_keep_ptrs") else None,
                            "obj_logit": 3.0 if (hits and a == "drop_recent_spatial_keep_ptrs") else -2.0,
                        }
                        for a in A6_ARMS
                    }
                    rows.append({
                        "frame": f,
                        "native_silent": True,
                        "evaluated": True,
                        "silent_streak": f,
                        "native_bbox": None,
                        "native_obj": -2.0,
                        "arms": arms,
                    })
                tp = cdir / "probe.jsonl"
                tp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                ops[nm] = {"frames": len(rows), "silent_evals": n_sil, "trace_sha256": digest(tp)}
                specs.append({"name": nm, "length": len(rows), "load_labels": lambda n=len(rows): [gt_box] * n})

            res = _merge_and_score_probe("hoot_dev", specs, out_dir, ops)
            self.assertEqual(res["status"], "AUDIT_OK")
            self.assertEqual(res["affected_silent_visible_videos"], 2)
            # Pooled: 2/10 = 20%; Video-macro: (1.0 + 0.0) / 2 = 50%
            self.assertEqual(res["pooled_silent_visible"]["drop_recent_spatial_keep_ptrs"]["iou_ge_05"], 2)
            self.assertEqual(res["pooled_silent_visible"]["drop_recent_spatial_keep_ptrs"]["n"], 10)
            self.assertAlmostEqual(res["video_macro_silent_visible"]["drop_recent_spatial_keep_ptrs"]["mean_iou_ge_05_rate"], 0.5, places=4)

    def test_bootstrap_ci_and_summary_report_generation(self):
        base = {"v1": 0.50, "v2": 0.60, "v3": 0.70, "v4": 0.65}
        trial = {"v1": 0.55, "v2": 0.60, "v3": 0.78, "v4": 0.71}
        ci = paired_video_bootstrap_ci(base, trial, n_boot=2000, seed=42)
        self.assertEqual(ci["n_videos"], 4)
        self.assertEqual((ci["wins"], ci["ties"], ci["losses"]), (3, 1, 0))
        self.assertGreater(ci["mean_delta"], 0.0)
        self.assertGreaterEqual(ci["ci_lo"], 0.0)

        # Verify build_report runs cleanly on a synthetic audited runs dir
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            arm_stub = {
                "totals": {"timely": 10, "first": 8, "fp": 20, "visible_mean_iou": 0.65},
                "fixed_point_macro_prf": {"precision": 0.70, "recall": 0.64, "f1": 0.6687},
                "fixed_point_by_video": {"v1": {"precision": 0.70, "recall": 0.64}},
                "vot_lt_toolkit": {"available": True, "f_score": 0.67, "best_threshold": "-inf"},
                "lasot_metrics": {"auc": 0.62, "matlab_conf": {"auc": 0.58}},
                "interventions": {"v1": {"memory_updates": 2, "rollbacks": 1}},
                "ms": {"median": 30.0},
                "videos": {"v1": {"visible": 100, "visible_iou_sum": 65.0, "false_reports": 20}},
            }
            d1 = rdir / "vot22_sam21_tiny_native"
            d1.mkdir()
            (d1 / "independent-audit.json").write_text(
                json.dumps({"status": "AUDIT_OK", "arms": {"baseline": arm_stub, "care": arm_stub}}),
                encoding="utf-8",
            )
            md = build_report(rdir)
            self.assertIn("Full VOT-LT2022", md)
            self.assertIn("Paired Video-Level Bootstrap", md)
            self.assertIn("AUDIT_OK", md)


if __name__ == "__main__":
    unittest.main()
