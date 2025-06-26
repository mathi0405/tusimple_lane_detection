import torch
def lane_collate_fn(batch):
    images = [item['image'] for item in batch]
    lanes = [item['lanes'] for item in batch]
    h_samples = [item['h_samples'] for item in batch]

    # Stack images (same size)
    images = torch.stack(images, dim=0)

    # Leave lanes and h_samples as lists of tensors
    return {
        'image': images,
        'lanes': lanes,
        'h_samples': h_samples
    }