"""Explicit frame-output scoring; labels are used here, never to select a box."""
import math


def box_iou(a, b):
    for box in (a, b):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or min(box[2:]) <= 0:
            raise ValueError("Expected finite positive xywh box")
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)


def score_event(labels, outputs, t0, window=10, threshold=.5):
    """outputs maps every evaluated frame to an accepted box or explicit None.

    A continuously emitting tracker is scored using its output at t0.
    Missing rows raise, rather than masquerading as abstention.
    No event with a truncated/non-visible window enters the denominator.
    """
    if t0 < 0 or window <= 0 or not 0 < threshold <= 1:
        raise ValueError("Invalid event configuration")
    stop = t0+window
    if stop > len(labels):
        return {"eligible": False, "reason": "truncated_window"}
    if any(label is None for label in labels[t0:stop]):
        return {"eligible": False, "reason": "nonvisible_window"}
    if any(frame not in outputs for frame in range(t0, stop)):
        raise ValueError("Incomplete output timeline")
    accepted = []
    for frame in range(t0, stop):
        box = outputs[frame]
        if box is not None:
            accepted.append((frame, box_iou(box, labels[frame])))
    correct = [row for row in accepted if row[1] >= threshold]
    first = accepted[0] if accepted else None
    category = ("unconfirmed" if first is None else
                "correct_first" if first[1] >= threshold else "wrong_first")
    return {"eligible": True, "category": category,
            "first_accept_frame": first[0] if first else None,
            "first_accept_iou": first[1] if first else None,
            "correct_recovered": bool(correct),
            "first_correct_delay_frames": correct[0][0]-t0 if correct else None,
            "accepted_wrong_frames": sum(overlap < threshold for _, overlap in accepted)}
