"""Per-video long-term tracking assessment helper."""
import math
from audit_journal_vot_results import qualify
from evaluation import box_iou, score_event

LEVELS = (.30, .50, .70)


def assess(labels, boxes):
    visible, absent = len(labels) - labels.count(None), labels.count(None)
    output = dict(enumerate(boxes))
    starts = [t for t, _ in qualify(labels)]
    return {"visible": visible, "absent": absent,
            "visible_iou_sum": math.fsum(box_iou(a, b) for a, b in zip(boxes, labels, strict=True)
                                          if a is not None and b is not None),
            "false_reports": sum(a is not None and b is None for a, b in zip(boxes, labels, strict=True)),
            "windows": {str(level): [dict(t0=t, **score_event(labels, output, t, threshold=level))
                                     for t in starts] for level in LEVELS}}
