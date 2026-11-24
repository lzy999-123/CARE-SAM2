"""First-frame jitter and background calibration box helpers."""

GRID = (.10, .30, .50, .70, .90)


def offset_boxes(box, offsets):
    x, y, w, h = box
    return [[x + (dx + (1 - size) / 2) * w, y + (dy + (1 - size) / 2) * h, w * size, h * size]
            for dx, dy, size in offsets]


def global_boxes(box, width, height):
    x, y, w, h = box
    boxes = set()
    for size in (.75, 1., 1.25):
        bw, bh = w * size, h * size
        if bw > width or bh > height:
            continue
        for u in GRID:
            for v in GRID:
                bx, by = (width - bw) * u, (height - bh) * v
                if bx + bw <= x or bx >= x + w or by + bh <= y or by >= y + h:
                    boxes.add((bx, by, bw, bh))
    return sorted(boxes)
