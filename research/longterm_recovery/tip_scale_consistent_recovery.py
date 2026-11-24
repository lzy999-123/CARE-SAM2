"""Boundary-truncated support normalization constants and helper."""

LINK = 0.30
QUALITY = 0.50
BORDER_MARGIN = 4
GAMMA_MIN = 0.25


def border_gamma(box, init_box, width, height, margin=BORDER_MARGIN):
    if box is None:
        return 1.0
    x, y, w, h = box
    w0, h0 = max(1.0, float(init_box[2])), max(1.0, float(init_box[3]))
    gx = (w / w0) if (x <= margin or x + w >= width - margin) else 1.0
    gy = (h / h0) if (y <= margin or y + h >= height - margin) else 1.0
    return float(max(GAMMA_MIN, min(1.0, gx, gy)))
