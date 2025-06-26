"""
dataset/dataset.py
──────────────────
TuSimpleDataset
• Scales lane x-coordinates and h_sample y-coordinates to match the
  resized image resolution so ground-truth masks align perfectly.
• Returns:
    ─ image      : Tensor [3,H,W]  (float32 0-1)
    ─ lanes      : Int16 tensor [num_lanes, num_samples]
    ─ h_samples  : Int16 tensor [num_samples]
"""

import os, json, torch
from torch.utils.data import Dataset
from PIL import Image

# ------------------------------------------------------------------ #
class TuSimpleDataset(Dataset):
    ORIG_W, ORIG_H = 1280, 720                # native TuSimple size

    def __init__(self,
                 json_path: str,
                 image_root: str,
                 transform=None,
                 target_size=(352, 640)):      # (H, W) after Resize
        """
        target_size must match the size you pass to transforms.Resize.
        """
        self.image_root  = image_root
        self.transform   = transform
        self.tH, self.tW = target_size
        self.sx = self.tW / self.ORIG_W       # scale factor width  ≈ 0.5
        self.sy = self.tH / self.ORIG_H       # scale factor height ≈ 0.489

        with open(json_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    # -------------------------------------------------------------- #
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann   = self.annotations[idx]

        img_path = os.path.join(self.image_root, ann["raw_file"])
        img      = Image.open(img_path).convert("RGB")

        # ── scale lane coordinates to resized image ────────────────
        scaled_lanes = []
        for lane in ann["lanes"]:
            scaled_lanes.append([
                int(x * self.sx) if x >= 0 else -2   # keep -2 as sentinel
                for x in lane
            ])
        scaled_h = [int(y * self.sy) for y in ann["h_samples"]]
        # ────────────────────────────────────────────────────────────

        if self.transform:
            img = self.transform(img)

        return {
            "image"     : img,                                      # [3,H,W]
            "lanes"     : torch.tensor(scaled_lanes, dtype=torch.int16),
            "h_samples" : torch.tensor(scaled_h,     dtype=torch.int16)
        }

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    """
    Quick sanity check:
      • Loads first sample
      • Creates ground-truth lane mask
      • Saves it to debug_mask.png
    Run with:
        python -m dataset.dataset
    """

    from torchvision import transforms
    from utils.mask_utils import create_lane_mask
    from torchvision.utils import save_image

    tfm = transforms.Compose([
        transforms.Resize((352, 640)),
        transforms.ToTensor()
    ])

    ds = TuSimpleDataset(
        json_path="data/test_label_new.json",
        image_root="data/tusimple",
        transform=tfm,
        target_size=(352, 640)
    )

    sample = ds[0]
    print("✅ Sample image tensor:", sample["image"].shape)
    mask = create_lane_mask(sample["lanes"], sample["h_samples"],
                            image_size=(352, 640))
    save_image(mask, "debug_mask.png")
    print("✅ Saved lane mask to debug_mask.png (white lines on black bg)")