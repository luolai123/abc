import argparse
import glob
import os
from typing import Tuple

import cv2
import numpy as np


def process_sequence(rgb_dir: str, mask_dir: str, obstacle_threshold: float, image_size: Tuple[int, int]):
    """Generate binary masks from monocular RGB data.

    Depth supervision has been removed from the pipeline. If masks already exist in
    ``mask_dir`` they will be resized and binarized; otherwise an all-safe mask is
    created for each RGB frame so the segmentation network can be trained on
    monocular inputs.
    """

    os.makedirs(mask_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                       glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                       glob.glob(os.path.join(rgb_dir, "*.jpeg")))

    for rgb_path in rgb_files:
        basename = os.path.basename(rgb_path)
        mask_path = os.path.join(mask_dir, os.path.splitext(basename)[0] + ".png")

        if os.path.exists(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                print(f"Skip {rgb_path}, invalid existing mask")
                continue
            mask = (mask_raw > obstacle_threshold).astype(np.uint8) * 255
        else:
            mask = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

        if image_size is not None:
            mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build binary obstacle masks from monocular RGB data")
    parser.add_argument("--rgb_dir", required=True, help="Directory containing RGB images")
    parser.add_argument("--mask_dir", required=True, help="Output directory for generated masks")
    parser.add_argument("--obstacle_threshold", type=float, default=0.5, help="Threshold for existing grayscale masks")
    parser.add_argument("--width", type=int, default=160, help="Output image width (set 0 to keep original)")
    parser.add_argument("--height", type=int, default=96, help="Output image height (set 0 to keep original)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_size = (args.width, args.height) if args.width > 0 and args.height > 0 else None
    if target_size is None:
        raise ValueError("Width and height must be positive when generating monocular masks.")

    process_sequence(args.rgb_dir, args.mask_dir, args.obstacle_threshold, target_size)
    print("Finished generating masks. Place RGB images under <root>/rgb and masks under <root>/mask for training.")
