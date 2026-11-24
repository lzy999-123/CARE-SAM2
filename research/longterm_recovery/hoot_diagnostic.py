"""HOOT label parsing helper."""
import json
from evaluation import box_iou


def labels_from(raw, length, name):
    data = json.loads(raw)
    if data["video_key"] != name or len(data["frames"]) != length:
        raise ValueError(f"HOOT labels do not match archive identity for {name}")
    labels, cause = [], []
    for i, row in enumerate(data["frames"]):
        if row["frame_id"] != i:
            raise ValueError(f"Noncontiguous labels in {name}:{i}")
        attr = row["attributes"]
        if type(attr["absent"]) is not bool or type(attr["full_occlusion"]) is not bool:
            raise ValueError(f"Invalid visibility flags in {name}:{i}")
        reason = "out_of_view" if attr["absent"] else "full_occlusion" if attr["full_occlusion"] else None
        cause.append(reason)
        box = None
        if reason is None:
            points = row["aa_bb"]
            if len(points) != 4 or any(len(p) != 2 for p in points):
                raise ValueError(f"Invalid visible polygon in {name}:{i}")
            x, y = zip(*points)
            box = [min(x), min(y), max(x)-min(x), max(y)-min(y)]
            box_iou(box, box)
        labels.append(box)
    return labels, cause
