import torch, numpy as np
from PIL import Image, ImageDraw

def create_lane_mask(lanes, h_samples, image_size=(352, 640), width=9):
    """
    Draws white lane polylines on a black background.
    Returns tensor shape [1, H, W] with values 0.0 / 1.0.
    """
    H, W = image_size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for lane in lanes:
        pts = [(x, y) for x, y in zip(lane, h_samples) if x >= 0]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=width)
    arr  = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)      # [1,H,W]

def create_colored_mask(mask, color=(0, 255, 0)):
    mask = (mask > 0.5).astype(np.uint8)
    out  = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = color
    return out
