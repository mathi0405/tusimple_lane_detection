import os, torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, transforms as T
from PIL import Image

from dataset.dataset import TuSimpleDataset
from models.model import UNet
from utils.collate import lane_collate_fn
from utils.mask_utils import create_lane_mask, create_colored_mask
from utils.metrics import compute_iou, compute_precision_recall_f1

# --- CONFIG ----------------------------------------------------------
MODEL_PATH = 'outputs/lane_model.pth'
JSON_PATH   = 'data/test_label_new.json'
IMAGE_ROOT  = 'data/tusimple'
SAVE_DIR    = 'outputs/predictions'
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 1
SUBSET      = False           # set False for full test
THRESHOLDS  = [0.1, 0.2, 0.3, 0.4, 0.5]
# ---------------------------------------------------------------------

print(f"🚀  Evaluating on {DEVICE}")

tfm = transforms.Compose([T.Resize((352, 640)), T.ToTensor()])
full_ds = TuSimpleDataset(JSON_PATH, IMAGE_ROOT, tfm)
dataset = Subset(full_ds, range(10)) if SUBSET else full_ds
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, collate_fn=lane_collate_fn)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

for thr in THRESHOLDS:
    tot_iou = tot_p = tot_r = tot_f1 = 0
    n = 0
    print(f"\n🔎  Threshold = {thr:.2f}")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            img  = batch['image'].to(DEVICE)
            lanes, h_samples = batch['lanes'], batch['h_samples']

            # ground-truth mask
            gtm = torch.stack([ create_lane_mask(l, h, (352,640))
                                for l,h in zip(lanes, h_samples) ]).to(DEVICE)

            out = model(img)                 # UNet already has sigmoid
            mx  = out.max().item()
            print(f"   Batch {b_idx}: max pred = {mx:.3f}")

            # metrics
            tot_iou += compute_iou(out, gtm)
            p,r,f1  = compute_precision_recall_f1(out, gtm, threshold=thr)
            tot_p   += p; tot_r += r; tot_f1 += f1; n += 1

            # save a single blended overlay for first image in this batch
            pred_bin = (out[0] > thr).float().cpu().squeeze(0).numpy()
            color_m  = create_colored_mask(pred_bin, (0,255,0))
            orig     = T.ToPILImage()(img[0].cpu())
            blend    = Image.blend(orig.convert("RGB"),
                                   Image.fromarray(color_m).convert("RGB"),
                                   alpha=0.5)
            if b_idx == 0:          # only first batch per threshold
                blend.save(os.path.join(
                    SAVE_DIR, f"thr{int(thr*100)}_example.png"))

    print(f"   IoU: {tot_iou/n:.4f} | Precision: {tot_p/n:.4f} | "
          f"Recall: {tot_r/n:.4f} | F1: {tot_f1/n:.4f}")
