"""Token-grid cropping and DINOv2 token encoding helpers."""
import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

from appearance_features import Descriptor, SIDE

SIDE_TOKENS = 8
MIN_RING_TOKENS = 4
MIN_CORE_TOKENS = 8
PATCH_CHUNK = 16
SURROUND = [(dx, dy, 1.) for dy in (-1.10, 0., 1.10) for dx in (-1.10, 0., 1.10) if dx or dy]
YY, XX = np.meshgrid(np.arange(SIDE_TOKENS), np.arange(SIDE_TOKENS), indexing="ij")
CENTERS = (np.column_stack([XX.ravel(), YY.ravel()]) + .5) / SIDE_TOKENS
INSIDE = ((CENTERS >= .25) & (CENTERS < .75)).all(axis=1)


def crop(image, box):
    x, y, w, h = box
    left, top = x - .50 * w, y - .50 * h
    dx, dy = 2 * w / SIDE, 2 * h / SIDE
    height, width = image.shape[:2]
    coords = CENTERS * [2 * w, 2 * h] + [left, top]
    valid = ((coords >= 0.) & (coords < [width, height])).all(axis=1)
    if (np.count_nonzero(valid & INSIDE) < MIN_CORE_TOKENS
            or np.count_nonzero(valid & ~INSIDE) < MIN_RING_TOKENS):
        return None
    matrix = np.array([[dx, 0, left + .50 * dx - .50], [0, dy, top + .50 * dy - .50]])
    pixels = cv2.warpAffine(image, matrix, (SIDE, SIDE),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return pixels, valid


class TokenEncoder(Descriptor):
    @torch.inference_mode()
    def tokens(self, crops):
        output = [None] * len(crops)
        valid = [i for i, item in enumerate(crops) if item is not None]
        for start in range(0, len(valid), PATCH_CHUNK):
            indices = valid[start:start + PATCH_CHUNK]
            value = torch.from_numpy(np.stack([crops[i][0] for i in indices])).to("cuda")
            value = (value.permute(0, 3, 1, 2).float() / 255. - self.mean) / self.std
            features = normalize(self.model(pixel_values=value).last_hidden_state, dim=-1).cpu().numpy()
            for i, item in zip(indices, features, strict=True):
                output[i] = item, crops[i][1]
            self.images_encoded += len(indices)
            self.batch_calls += 1
        return output
