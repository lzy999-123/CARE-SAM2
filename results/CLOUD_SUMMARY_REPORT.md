# Cloud Full-Benchmark Programmatic Summary Report

Source: Full 50-sequence official VOT-LT2022 audited runs (parsed strictly from `independent-audit.json` / `results.json`).

## 1. Full VOT-LT2022 (50 Sequences) Main Pareto Table

| Configuration | Status | Timely | First | Absent FP | Upd / RB | Vis.IoU | Online P | Online R | Online F1 | VOT-tk F | `best_th` | LaSOT AUC (vis) | LaSOT AUC (MATLAB) | Med ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SAM 2.1 Tiny — `vanilla` | `AUDIT_OK` | 87 | 81 | 3229 | 0 / 0 | 0.5145 | 0.5750 | 0.5068 | **0.5388** | **0.5413** | `-inf` | 0.5018 | 0.4547 | 94.4 |
| SAM 2.1 Tiny — `gate_skip_absent` (5-line) | `AUDIT_OK` | 114 | 99 | 5481 | 0 / 0 | 0.5589 | 0.5378 | 0.5472 | **0.5425** | **0.5472** | `0.8640` | 0.5418 | 0.4894 | 114.4 |
| SAM 2.1 Tiny — `gate_skip_absent_keept1` (SAMURAI rule, no KF) | `AUDIT_OK` | 113 | 96 | 7774 | 0 / 0 | 0.5638 | 0.5464 | 0.5585 | **0.5524** | **0.5570** | `0.8854` | 0.5530 | 0.4999 | 114.0 |
| SAM 2.1 Tiny — `CARE (sam2_native)` | `AUDIT_OK` | 96 | 89 | 3763 | 68 / 10 | 0.5671 | 0.5709 | 0.5518 | **0.5612** | **0.5638** | `-inf` | 0.5461 | 0.4940 | 93.5 |
| SAM 2.1 Base+ — `vanilla` | `AUDIT_OK` | 76 | 66 | 2859 | 0 / 0 | 0.4886 | 0.5836 | 0.4869 | **0.5309** | **0.5349** | `0.9097` | 0.4814 | 0.4387 | 147.3 |
| SAM 2.1 Base+ — `gate_skip_absent` (5-line) | `AUDIT_OK` | 115 | 102 | 6459 | 0 / 0 | 0.5586 | 0.5414 | 0.5499 | **0.5456** | **0.5530** | `0.9957` | 0.5441 | 0.4908 | 161.6 |
| SAM 2.1 Base+ — `gate_skip_absent_keept1` | `AUDIT_OK` | 101 | 90 | 6056 | 0 / 0 | 0.5206 | 0.5370 | 0.5218 | **0.5293** | **0.5345** | `0.9786` | 0.5161 | 0.4697 | 161.3 |
| SAM 2.1 Base+ — `CARE (sam2_native)` | `AUDIT_OK` | 100 | 88 | 3142 | 93 / 4 | 0.5671 | 0.5839 | 0.5583 | **0.5708** | **0.5750** | `0.9232` | 0.5519 | 0.4996 | 143.0 |
| SAMURAI Tiny — stock (`min_obj=-1.0`) | `AUDIT_OK` | 113 | 96 | 9856 | 0 / 0 | 0.5682 | 0.5324 | 0.5549 | **0.5434** | **0.5510** | `0.8056` | 0.5495 | 0.4980 | 190.4 |
| SAMURAI Tiny — calibrated (`min_obj=0.0`) | `AUDIT_OK` | 115 | 101 | 6411 | 0 / 0 | 0.5712 | 0.5480 | 0.5598 | **0.5539** | **0.5588** | `0.8328` | 0.5543 | 0.5004 | 187.7 |
| DAM4SAM Tiny — stock | `AUDIT_OK` | 101 | 82 | 6044 | 0 / 0 | 0.5308 | 0.5188 | 0.5227 | **0.5207** | **0.5262** | `0.9619` | 0.5170 | 0.4703 | 132.1 |

## 2. Full VOT-LT2022 Component & Verifier Ablation Table

| Configuration | Status | Timely | First | Absent FP | Upd / RB | Vis.IoU | Online P | Online R | Online F1 | VOT-tk F | `best_th` | LaSOT AUC (vis) | LaSOT AUC (MATLAB) | Med ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SAM 2.1 Tiny — `vanilla` | `AUDIT_OK` | 87 | 81 | 3229 | 0 / 0 | 0.5145 | 0.5750 | 0.5068 | **0.5388** | **0.5413** | `-inf` | 0.5018 | 0.4547 | 94.4 |
| CARE — full (`sam2_native`, dual-anchor + scale-norm + 1-step rollback) | `AUDIT_OK` | 96 | 89 | 3763 | 68 / 10 | 0.5671 | 0.5709 | 0.5518 | **0.5612** | **0.5638** | `-inf` | 0.5461 | 0.4940 | 93.5 |
| CARE — verifier `none` (no appearance verifier) | `AUDIT_OK` | 95 | 88 | 6515 | 73 / 12 | 0.5634 | 0.5656 | 0.5500 | **0.5577** | **0.5603** | `-inf` | 0.5443 | 0.4930 | 99.7 |
| CARE — `--no-dual-anchor` (`m0` only) | `AUDIT_OK` | 88 | 81 | 4263 | 70 / 11 | 0.5461 | 0.5615 | 0.5347 | **0.5478** | **0.5504** | `0.7400` | 0.5293 | 0.4778 | 100.2 |
| CARE — `--no-certify-anchor` | `AUDIT_OK` | 96 | 89 | 3422 | 69 / 12 | 0.5671 | 0.5716 | 0.5518 | **0.5615** | **0.5642** | `-inf` | 0.5462 | 0.4941 | 99.0 |
| CARE — `--no-scale-norm` | `AUDIT_OK` | 96 | 89 | 3632 | 68 / 14 | 0.5594 | 0.5716 | 0.5478 | **0.5595** | **0.5620** | `-inf` | 0.5425 | 0.4897 | 91.1 |
| CARE — `--no-rollback` (no 1-step self-verification rollback) | `AUDIT_OK` | 96 | 89 | 3760 | 62 / 0 | 0.5672 | 0.5701 | 0.5521 | **0.5610** | **0.5636** | `-inf` | 0.5465 | 0.4943 | 90.8 |

## 3. Paired Video-Level Bootstrap 95% CIs (10,000 Resamples)

| Pair | Videos | Metric | Mean Delta | 95% Bootstrap CI | Win / Tie / Loss |
| :--- | :---: | :--- | :---: | :---: | :---: |
| CARE vs Vanilla (Tiny) | 50 | Online Video F1 | +0.0303 | [+0.0101, +0.0549] | 18 / 28 / 4 |
| CARE vs Vanilla (Tiny) | 50 | Visible IoU | +0.0449 | [+0.0182, +0.0770] | 20 / 28 / 2 |
| CARE vs Vanilla (Tiny) | 50 | Absent FP (lower better) | +10.68 | [-0.50, +28.94] | 12 / 36 / 2 |
| gate_skip_absent vs Vanilla (Tiny) | 50 | Online Video F1 | +0.0131 | [-0.0145, +0.0444] | 22 / 7 / 21 |
| gate_skip_absent vs Vanilla (Tiny) | 50 | Visible IoU | +0.0404 | [+0.0054, +0.0805] | 28 / 7 / 15 |
| gate_skip_absent vs Vanilla (Tiny) | 50 | Absent FP (lower better) | +45.04 | [+22.48, +72.84] | 32 / 16 / 2 |
| CARE vs gate_skip_absent (Tiny) | 50 | Online Video F1 | +0.0172 | [+0.0019, +0.0341] | 28 / 7 / 15 |
| CARE vs gate_skip_absent (Tiny) | 50 | Visible IoU | +0.0046 | [-0.0133, +0.0223] | 18 / 7 / 25 |
| CARE vs gate_skip_absent (Tiny) | 50 | Absent FP (lower better) | -34.36 | [-58.00, -16.06] | 3 / 16 / 31 |

## 4. Full VOT-LT2022 9-Arm Counterfactual Memory Decomposition (34/50 affected videos, 69 silent qualifying events)

| Counterfactual Memory Condition | Pooled Vis `IoU>=0.5` | Pooled Vis Mean `obj` | Pooled Absent FP (`obj>0`) | Video-Macro Vis `IoU>=0.5` Rate | Video-Macro Absent FP Rate | Silent Events `IoU>=0.5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `orig_memory` | 0/22900 (0.0%) | -2.1394 | 0/13504 (0.0%) | 0.00% | 0.00% | 0/69 |
| `strip_no_obj_spatial_only` | 41/22900 (0.2%) | -3.9806 | 35/13504 (0.3%) | 0.94% | 1.02% | 2/69 |
| `drop_recent_spatial_keep_ptrs` | 8472/22900 (37.0%) | +0.3164 | 3079/13504 (22.8%) | 35.74% | 21.43% | 30/69 |
| `drop_recent_ptrs_keep_spatial` | 62/22900 (0.3%) | -2.9166 | 109/13504 (0.8%) | 3.34% | 2.73% | 1/69 |
| `m0_only` | 7134/22900 (31.2%) | -0.2597 | 2439/13504 (18.1%) | 29.57% | 20.16% | 25/69 |
| `m0_plus_1empty` | 1123/22900 (4.9%) | -1.5683 | 450/13504 (3.3%) | 4.18% | 5.50% | 5/69 |
| `m0_plus_mclean` | 7460/22900 (32.6%) | +0.1105 | 1609/13504 (11.9%) | 27.31% | 15.56% | 32/69 |
| `m0_plus_6vis_condptr` | 9266/22900 (40.5%) | +1.1506 | 2640/13504 (19.5%) | 32.30% | 22.33% | 36/69 |
| `m0_plus_6vis_15visptr` | 8570/22900 (37.4%) | +0.9048 | 2234/13504 (16.5%) | 27.25% | 18.10% | 35/69 |

