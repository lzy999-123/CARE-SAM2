"""Foreground-restricted token matching helpers."""
import numpy as np
from tip_token_identity import CENTERS, INSIDE

METHODS = ("rectangle", "foreground", "foreground_contrast")
MINIMUM = 4


def silhouette(binary, box):
    x, y, w, h = box
    h_img, w_img = binary.shape
    positions = np.floor(CENTERS * [2 * w, 2 * h] + [x - w / 2, y - h / 2]).astype(int)
    ix, iy = positions.T
    valid = (ix >= 0) & (ix < w_img) & (iy >= 0) & (iy < h_img)
    result = np.zeros(len(CENTERS), dtype=bool)
    result[valid] = binary[iy[valid], ix[valid]]
    return result & INSIDE


def pooled(token, selected):
    if token is None:
        return None
    vector, valid = token
    active = valid & selected
    if active.sum() < MINIMUM:
        return None
    values = vector[1:][active].mean(axis=0)
    length = np.linalg.norm(values)
    return values / length if length > 0 else None


def descriptors(token, mask):
    return pooled(token, INSIDE), pooled(token, mask), pooled(token, ~mask)


def similarity(reference, observation):
    rect, fg, bg = reference
    box, mask, context = observation
    whole = None if rect is None or box is None else float(rect @ box)
    core = None if fg is None or mask is None else float(fg @ mask)
    contrast = None if core is None or bg is None or context is None else core - max(
        float(fg @ context), float(bg @ mask))
    return dict(zip(METHODS, (whole, core, contrast), strict=True))
