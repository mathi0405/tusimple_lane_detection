import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from dataset.dataset import TuSimpleDataset
from models.model import UNet
from utils.mask_utils import create_lane_mask

# --- CONFIG ---
MODEL_PATH = 'outputs/lane_model.pth'
JSON_PATH = 'data/test_label_new.json'
IMAGE_ROOT = 'data/tusimple'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load dataset
transform = transforms.Compose([
    transforms.Resize((352, 640)),
    transforms.ToTensor()
])
dataset = TuSimpleDataset(json_path=JSON_PATH, image_root=IMAGE_ROOT, transform=transform)
sample = dataset[0]
image_tensor = sample['image'].unsqueeze(0).to(DEVICE)
lanes = sample['lanes']
h_samples = sample['h_samples']

# Ground truth mask
gt_mask = create_lane_mask(lanes, h_samples, image_size=(352, 640)).squeeze().numpy() * 255

# Prediction
with torch.no_grad():
    pred = model(image_tensor)[0][0].cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

# Show side-by-side
gt_mask = cv2.resize(gt_mask, (640, 352))
pred_mask = cv2.resize(pred_mask, (640, 352))

cv2.imshow("Ground Truth Mask", gt_mask)
cv2.imshow("Predicted Mask", pred_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()