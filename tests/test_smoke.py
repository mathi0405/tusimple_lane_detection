"""CPU smoke tests: no TuSimple data or GPU required."""
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import TuSimpleDataset  # noqa: E402
from models.model import UNet  # noqa: E402
from utils.mask_utils import create_lane_mask  # noqa: E402
from utils.metrics import compute_iou, compute_precision_recall_f1  # noqa: E402


def test_unet_output_shape_and_range():
    model = UNet().eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_lane_mask_ignores_missing_points():
    lanes = [[10, -2, 30, 40]]
    h = [5, 10, 15, 20]
    mask = create_lane_mask(lanes, h, image_size=(32, 64), width=3)
    assert mask.shape == (1, 32, 64)
    assert mask.sum() > 0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_metrics_perfect_and_empty_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 3] = 1.0
    assert abs(compute_iou(target.clone(), target) - 1.0) < 1e-4
    p, r, f1 = compute_precision_recall_f1(target.clone(), target)
    assert min(p, r, f1) > 0.999
    _, r0, _ = compute_precision_recall_f1(torch.zeros_like(target), target)
    assert r0 == 0.0


def test_iou_respects_threshold():
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 1.0
    pred = torch.zeros(1, 1, 4, 4)
    pred[..., 0, 0] = 0.3
    assert compute_iou(pred, target, threshold=0.2) > 0.99
    assert compute_iou(pred, target, threshold=0.5) < 0.01


def test_dataset_scales_lane_coordinates(tmp_path):
    img_dir = tmp_path / "clips"
    img_dir.mkdir()
    Image.new("RGB", (1280, 720)).save(img_dir / "0.jpg")
    label = {"raw_file": "clips/0.jpg", "lanes": [[640, -2]], "h_samples": [360, 700]}
    js = tmp_path / "labels.json"
    js.write_text(json.dumps(label) + "\n")
    ds = TuSimpleDataset(str(js), str(tmp_path),
                         transform=transforms.Compose([transforms.Resize((352, 640)), transforms.ToTensor()]),
                         target_size=(352, 640))
    s = ds[0]
    assert s["image"].shape == (3, 352, 640)
    assert s["lanes"].tolist() == [[320, -2]]
    assert s["h_samples"].tolist() == [176, 342]
