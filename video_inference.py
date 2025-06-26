"""
video_inference.py  ▸  Real-time lane + center-path overlay
────────────────────────────────────────────────────────────
Key features
• Resolution-agnostic: input & output match original video size
• Robust lane tracking: keeps the same left & right IDs across frames
• Exponential smoothing of polynomial coefficients (Kalman-like)
• Fail-safe: if a lane disappears temporarily, prediction is extrapolated
• Console FPS + saved MP4

Dependencies: cv2, torch, numpy, torchvision, utils.polynomial_fit, utils.mask_utils
"""

import cv2, os, time, torch, numpy as np
from torchvision import transforms
from models.model import UNet
from utils.polynomial_fit import (
    extract_lane_points, fit_polynomial,
    draw_polylines, compute_center_path
)
from utils.mask_utils import create_colored_mask

# ───────────────────── CONFIG ───────────────────────────────────────
VIDEO_PATH   = "test_video.mp4"            # input
MODEL_PATH   = "outputs/lane_model.pth"    # trained model
OUTPUT_PATH  = "outputs/lane_output.mp4"   # output
IMG_SIZE     = (352, 640)                  # (H,W) network resolution
THRESH       = 0.30                        # mask threshold
SMOOTH_ALPHA = 0.85                        # 0=no smoothing  0.85=stable
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────

# preprocessing identical to training
to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),           # keep aspect
    transforms.ToTensor()
])

# ─── Load model ─────────────────────────────────────────────────────
net = UNet().to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

# ─── Video I/O set-up ───────────────────────────────────────────────
cap   = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30
orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs("outputs", exist_ok=True)
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (orig_w, orig_h))

# state for smoothing/tracking
prev_left_coef  = None
prev_right_coef = None
prev_time       = time.time()

print("🚦  Processing video …   (press q to quit window)")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── NN inference ────────────────────────────────────────────────
    small = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]))
    inp   = to_tensor(small).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = net(inp)[0, 0].cpu().numpy()
    mask = (pred > THRESH).astype(np.uint8) * 255      # [H,W]

    # ── contour → polynomial fits ──────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    fits = []
    for cnt in contours:
        tmp = np.zeros_like(mask)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        xs, ys = extract_lane_points(tmp, min_pts=45)
        coef, y0, y1 = fit_polynomial(xs, ys, degree=2)
        if coef is not None:
            fits.append((coef, y0, y1))

    # if no lanes found, carry-forward previous coefs
    if not fits and prev_left_coef is not None:
        fits = [(prev_left_coef, 0, IMG_SIZE[0]-1),
                (prev_right_coef, 0, IMG_SIZE[0]-1)]

    # ── identify left & right lanes via bottom-x position ───────────
    bottom_y = IMG_SIZE[0] - 5
    fits.sort(key=lambda f: np.polyval(f[0], bottom_y))   # left→right
    if len(fits) >= 2:
        left_coef,  yL0, yL1 = fits[0]
        right_coef, yR0, yR1 = fits[-1]

        # ─ smoothing ─
        if prev_left_coef is not None:
            left_coef  = SMOOTH_ALPHA * prev_left_coef  + (1-SMOOTH_ALPHA) * left_coef
            right_coef = SMOOTH_ALPHA * prev_right_coef + (1-SMOOTH_ALPHA) * right_coef

        prev_left_coef, prev_right_coef = left_coef, right_coef
        fits = [(left_coef, yL0, yL1), (right_coef, yR0, yR1)]

        # ─ center path ─
        y_min = max(yL0, yR0)
        y_max = min(yL1, yR1)
        center_pts = compute_center_path(left_coef, right_coef,
                                         IMG_SIZE[0], y_min, y_max)
    else:
        center_pts = None

    # ── create overlay on small frame ───────────────────────────────
    overlay_small = small.copy()
    draw_polylines(overlay_small, fits, color=(0,255,0), thickness=3)
    if center_pts:
        for p1, p2 in zip(center_pts[:-1], center_pts[1:]):
            cv2.line(overlay_small, p1, p2, (0,0,255), 2)
    mask_rgb = create_colored_mask(mask, (0,255,0))
    overlay_small = cv2.addWeighted(overlay_small, 0.8, mask_rgb, 0.25, 0)

    # up-scale to original resolution for saving
    overlay_full = cv2.resize(overlay_small, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

    # ── FPS text ────────────────────────────────────────────────────
    t_now = time.time()
    fps   = 1.0 / (t_now - prev_time + 1e-6)
    prev_time = t_now
    cv2.putText(overlay_full, f"{fps:.1f} FPS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 2)

    out.write(overlay_full)
    cv2.imshow("Lane Detection", overlay_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames …")

cap.release(); out.release(); cv2.destroyAllWindows()
print("✅  Saved annotated video to:", OUTPUT_PATH)
print("✅  Processed", frame_idx, "frames at", fps_in, "FPS")
print("🚦  Done!")
# ────────────────────────────────────────────────────────────────────