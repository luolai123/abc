import os
import glob
import cv2
import torch
from typing import Tuple, List
from torch.utils.data import Dataset
from torchvision import transforms


class RGBObstacleDataset(Dataset):
    """
    Dataset for RGB obstacle segmentation.

    Expected directory layout:
        root/
            rgb/  (RGB images)
            mask/ (binary masks with the same file name as rgb)
    """

    def __init__(self, root_dir: str, image_size: Tuple[int, int], mode: str = "train", val_ratio: float = 0.1):
        assert 0.0 < val_ratio < 1.0, "val_ratio must be within (0, 1)"
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_rgb_files = sorted(glob.glob(os.path.join(root_dir, "rgb", "*.png")) +
                               glob.glob(os.path.join(root_dir, "rgb", "*.jpg")) +
                               glob.glob(os.path.join(root_dir, "rgb", "*.jpeg")))
        assert len(all_rgb_files) > 0, f"No RGB images found under {root_dir}/rgb"

        split_idx = int(len(all_rgb_files) * (1 - val_ratio))
        if mode == "train":
            self.rgb_files = all_rgb_files[:split_idx]
        elif mode == "valid":
            self.rgb_files = all_rgb_files[split_idx:]
        else:
            raise ValueError("mode should be 'train' or 'valid'")

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.rgb_files)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_files[idx]
        mask_path = rgb_path.replace(os.sep + "rgb" + os.sep, os.sep + "mask" + os.sep)
        mask_path = os.path.splitext(mask_path)[0] + ".png"

        image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found for {rgb_path}, expected {mask_path}")

        image_tensor = self.transform(image)
        mask_tensor = torch.from_numpy(cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)).float()
        mask_tensor = (mask_tensor > 127).float().unsqueeze(0)  # [1, H, W]
        return image_tensor, mask_tensor


def list_dataset_statistics(root_dir: str) -> List[str]:
    summary = []
    rgb_files = sorted(glob.glob(os.path.join(root_dir, "rgb", "*.png")) +
                       glob.glob(os.path.join(root_dir, "rgb", "*.jpg")) +
                       glob.glob(os.path.join(root_dir, "rgb", "*.jpeg")))
    mask_files = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
    summary.append(f"RGB images : {len(rgb_files)}")
    summary.append(f"Mask images: {len(mask_files)}")
    if len(rgb_files) != len(mask_files):
        summary.append("Warning: mismatched RGB / mask counts. Ensure file names align.")
    return summary

