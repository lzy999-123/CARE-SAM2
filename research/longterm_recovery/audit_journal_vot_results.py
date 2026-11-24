"""Long-term tracking overlap and qualifying reappearance event helpers."""
import numpy as np


def overlap(box, target):
    if box is None:
        return None
    left, top, width, height = box
    tx, ty, tw, th = target
    iw = max(0, min(left+width, tx+tw)-max(left, tx))
    ih = max(0, min(top+height, ty+th)-max(top, ty))
    return iw*ih/(width*height+tw*th-iw*ih)


def qualify(labels):
    visible = np.array([row is not None for row in labels], dtype=bool)
    for at in np.flatnonzero(~visible[:-1] & visible[1:])+1:
        i = int(at)
        before = i-1
        while before >= 0 and not visible[before]:
            before -= 1
        absent = i-before-1
        if absent >= 15 and i+10 <= len(visible) and visible[i:i+10].all():
            yield i, absent
