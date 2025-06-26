import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset.dataset import TuSimpleDataset  # make sure this path is correct
import numpy as np

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor()
])

# Paths
json_path = "data/TUSimple/train_set/label_data_0313.json"  # or other JSON file
image_root = "data/TUSimple/train_set"

# Load dataset
dataset = TuSimpleDataset(json_path, image_root, transform=None)

# Pick a sample
sample = dataset[0]
image = sample['image']
lanes = sample['lanes']
h_samples = sample['h_samples']

# Load raw image for plotting (without transform)
raw_img_path = os.path.join(image_root, dataset.annotations[0]['raw_file'])
raw_image = plt.imread(raw_img_path)

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(raw_image)

# Plot each lane
colors = ['r', 'g', 'b', 'y', 'c']
for i, lane in enumerate(lanes):
    xs = lane
    ys = h_samples
    xs, ys = np.array(xs), np.array(ys)
    mask = xs != -2
    plt.plot(xs[mask], ys[mask], marker='o', color=colors[i % len(colors)], label=f"Lane {i+1}")

plt.title("TuSimple Lane Visualization")
plt.axis('off')
plt.legend()
plt.show()