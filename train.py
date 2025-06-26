import os, torch
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from dataset.dataset import TuSimpleDataset
from models.model import UNet
from utils.collate import lane_collate_fn
from utils.mask_utils import create_lane_mask

# ─── CONFIG ──────────────────────────────────────────────────────────
JSON_PATH  = 'data/test_label_new.json'
IMAGE_ROOT = 'data/tusimple'

EPOCHS     = 10
BATCH_SIZE = 2
LR         = 1e-4
IMG_SIZE   = (352, 640)                 # (H, W)  resize size everywhere

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"🚀  Using device: {DEVICE}")

    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    dataset = TuSimpleDataset(JSON_PATH, IMAGE_ROOT,
                              transform=tfm, target_size=IMG_SIZE)
    loader  = DataLoader(dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0,
                         collate_fn=lane_collate_fn)

    model = UNet().to(DEVICE)
    assert any(p.requires_grad for p in model.parameters()), \
        "❌ No trainable parameters!"

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs("outputs", exist_ok=True)
    print("🚦  Starting training on full dataset...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        prog = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch in prog:
            imgs       = batch['image'].to(DEVICE)
            lane_lists = batch['lanes']
            h_samples  = batch['h_samples']

            gt_masks = torch.stack([
                create_lane_mask(l, h, image_size=IMG_SIZE)
                for l, h in zip(lane_lists, h_samples)
            ]).to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, gt_masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")

        print(f"✅ Epoch {epoch+1}/{EPOCHS} finished  Total-Loss: {total_loss:.4f}")

        # ── Save checkpoint every epoch ───────────────────────────────
        ckpt = f"outputs/lane_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt)
        print("💾 Saved checkpoint:", os.path.abspath(ckpt))
        # ──────────────────────────────────────────────────────────────

    # ── Final model ──────────────────────────────────────────────────
    final_path = "outputs/lane_model.pth"
    torch.save(model.state_dict(), final_path)
    print("🎉 Training complete →", os.path.abspath(final_path))

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()