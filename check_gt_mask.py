from dataset.dataset import TuSimpleDataset
from utils.mask_utils import create_lane_mask
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((352, 640)),
    transforms.ToTensor()
])

dataset = TuSimpleDataset("data/test_label_new.json", "data/tusimple", transform=transform)
sample = dataset[0]
mask = create_lane_mask(sample['lanes'], sample['h_samples'], image_size=(352, 640))

plt.imshow(mask.squeeze(0), cmap='gray')
plt.title("Ground Truth Lane Mask")
plt.axis("off")
plt.show()
