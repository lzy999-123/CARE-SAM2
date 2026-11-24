"""Fixed DINOv2 crop features for target matching with first-frame negatives."""
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

SOURCE = Path(os.environ.get("DINOV2_LOCAL_DIR", "."))
WEIGHTS = SOURCE / "model-resumed.safetensors"
SIDE = 112
NEGATIVE_OFFSETS = [
    (dx, dy) for dy in [-1.25, 0., 1.25] for dx in [-1.25, 0., 1.25] if dx or dy
]


def patch(image, box):
    x, y, w, h = box
    height, width = image.shape[:2]
    left, right = max(0, int(np.floor(x))), min(width, int(np.ceil(x + w)))
    top, bottom = max(0, int(np.floor(y))), min(height, int(np.ceil(y + h)))
    if left >= right or top >= bottom:
        return None
    return cv2.resize(image[top:bottom, left:right], (SIDE, SIDE), interpolation=cv2.INTER_AREA)


class Descriptor:
    def __init__(self):
        deps_dir = SOURCE / "deps"
        if deps_dir.is_dir():
            sys.path.insert(0, str(deps_dir))
        from safetensors.torch import load_file
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config.from_json_file(str(SOURCE / "config.json"))
        config._attn_implementation = "sdpa"
        self.model = Dinov2Model(config)
        self.model.load_state_dict(load_file(str(WEIGHTS)), strict=True)
        self.model = self.model.eval().cuda()
        processor = json.loads((SOURCE / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.mean = torch.tensor(processor["image_mean"], device="cuda")[None, :, None, None]
        self.std = torch.tensor(processor["image_std"], device="cuda")[None, :, None, None]
        self.dimension = config.hidden_size
        self.images_encoded = 0
        self.batch_calls = 0

    @torch.inference_mode()
    def extract(self, patches):
        pixels = torch.from_numpy(np.stack(patches)).cuda().permute(0, 3, 1, 2).float()
        pixels = ((pixels / 255.) - self.mean) / self.std
        hidden = self.model(pixel_values=pixels).last_hidden_state[:, 0]
        vectors = torch.nn.functional.normalize(hidden, dim=1).cpu().numpy()
        self.images_encoded += len(patches)
        self.batch_calls += 1
        return vectors

    def initialize(self, image, box):
        target = patch(image, box)
        if target is None:
            raise ValueError("Target crop is empty")
        x, y, w, h = box
        height, width = image.shape[:2]
        negative_boxes = [[x + dx * w, y + dy * h, w, h] for dx, dy in NEGATIVE_OFFSETS
                          if 0 <= x + dx * w and 0 <= y + dy * h and x + (dx + 1) * w <= width and y + (dy + 1) * h <= height]
        vectors = self.extract([target, *[patch(image, negative) for negative in negative_boxes]])
        self.positive = vectors[0]
        self.negatives = vectors[1:]
        return negative_boxes

    def scores(self, image, boxes):
        pixels = [patch(image, box) for box in boxes]
        valid = [i for i, crop_img in enumerate(pixels) if crop_img is not None]
        output = [[0., 0., 0.] for _ in boxes]
        if valid:
            vectors = self.extract([pixels[i] for i in valid])
            for i, vector in zip(valid, vectors):
                positive = float(vector @ self.positive)
                negative = float((self.negatives @ vector).max()) if len(self.negatives) else 0.
                output[i] = [positive, negative, positive - negative]
        return output
