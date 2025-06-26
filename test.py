import os, cv2, torch, numpy as np
from torchvision import transforms
from dataset.dataset import TuSimpleDataset
from models.model import UNet
from utils.polynomial_fit import (
    extract_lane_points, fit_polynomial,
    draw_polynomial_lane, compute_center_path, draw_center_path
)
from utils.mask_utils import create_colored_mask

# ─── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = 'outputs/lane_model.pth'
JSON_PATH  = 'data/test_label_new.json'
IMAGE_ROOT = 'data/tusimple'
SAVE_PATH  = 'outputs/test_output.png'

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = (352, 640)               # (H, W) used in training
THRESH     = 0.30                     # mask threshold
# ─────────────────────────────────────────────────────────────────────

# transform matches training
tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# dataset / model -----------------------------------------------------
ds = TuSimpleDataset(JSON_PATH, IMAGE_ROOT, transform=tfm,
                     target_size=IMG_SIZE)
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# pick sample 0
sample = ds[0]
img_t  = sample['image'].unsqueeze(0).to(DEVICE)

# original BGR image for overlay
raw_path  = os.path.join(IMAGE_ROOT, ds.annotations[0]['raw_file'])
orig_bgr  = cv2.resize(cv2.imread(raw_path), IMG_SIZE[::-1])  # (W,H)

# forward pass --------------------------------------------------------
with torch.no_grad():
    pred = model(img_t)[0, 0].cpu().numpy()
bin_mask = (pred > THRESH).astype(np.uint8) * 255             # 0/255

# fit lanes per contour ----------------------------------------------
contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
lane_fits = []
for cnt in contours:
    tmp = np.zeros_like(bin_mask)
    cv2.drawContours(tmp, [cnt], -1, 255, -1)
    xs, ys = extract_lane_points(tmp, min_pts=40)
    coef, y0, y1 = fit_polynomial(xs, ys, degree=2)
    if coef is not None:
        lane_fits.append((coef, y0, y1))

# sort by bottom x (safer than mid-x) --------------------------------
bottom_y = IMG_SIZE[0] - 5
lane_fits.sort(key=lambda t: np.polyval(t[0], bottom_y))

# draw lanes (green)
for coef, y0, y1 in lane_fits:
    draw_polynomial_lane(orig_bgr, coef, y0, y1,
                         color=(0, 255, 0), thickness=3)

# draw center path (red) if >=2 lanes
if len(lane_fits) >= 2:
    (cL, yL0, yL1), (cR, yR0, yR1) = lane_fits[0], lane_fits[-1]
    y_min = int(max(yL0, yR0))
    y_max = int(min(yL1, yR1))
    center_pts = compute_center_path(cL, cR, IMG_SIZE[0], y_min, y_max)
    draw_center_path(orig_bgr, center_pts, color=(0, 0, 255), thickness=2)

# add semi-transparent mask overlay ----------------------------------
color_mask = create_colored_mask(bin_mask, (0, 255, 0))
blend      = cv2.addWeighted(orig_bgr, 0.8, color_mask, 0.3, 0)

# save / show ---------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(SAVE_PATH, blend)
cv2.imshow("Lane Overlay with Polynomial + Center Path", blend)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("✅ Saved overlay to", SAVE_PATH)
