import os
import glob
import argparse
from typing import Tuple

import cv2
import numpy as np


def build_mask_from_depth(depth_image: np.ndarray, obstacle_threshold_m: float, min_valid_m: float = 0.2) -> np.ndarray:
    """Generate a binary mask (1 = safe, 0 = obstacle) from a depth map."""
    depth_image = depth_image.astype(np.float32)
    invalid = np.isnan(depth_image) | (depth_image < min_valid_m)
    obstacle = depth_image < obstacle_threshold_m
    mask = np.ones_like(depth_image, dtype=np.uint8)
    mask[obstacle] = 0
    mask[invalid] = 0
    return mask * 255


def process_sequence(rgb_dir: str, depth_dir: str, mask_dir: str, depth_scale: float, obstacle_threshold_m: float,
                     image_size: Tuple[int, int]):
    os.makedirs(mask_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                       glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                       glob.glob(os.path.join(rgb_dir, "*.jpeg")))

    for rgb_path in rgb_files:
        basename = os.path.basename(rgb_path)
        depth_path = os.path.join(depth_dir, os.path.splitext(basename)[0] + ".png")
        if not os.path.exists(depth_path):
            print(f"Skip {rgb_path}, missing depth {depth_path}")
            continue

        rgb = cv2.imread(rgb_path)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"Skip {rgb_path}, invalid depth")
            continue

        depth = depth_raw.astype(np.float32) * depth_scale
        mask = build_mask_from_depth(depth, obstacle_threshold_m)

        if image_size is not None:
            rgb = cv2.resize(rgb, image_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

        mask_path = os.path.join(mask_dir, os.path.splitext(basename)[0] + ".png")
        cv2.imwrite(mask_path, mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Build binary obstacle masks from paired RGB+depth data")
    parser.add_argument("--rgb_dir", required=True, help="Directory containing RGB images")
    parser.add_argument("--depth_dir", required=True, help="Directory containing depth images aligned to RGB")
    parser.add_argument("--mask_dir", required=True, help="Output directory for generated masks")
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Scale factor to convert depth units to meters")
    parser.add_argument("--obstacle_threshold", type=float, default=2.5, help="Depth threshold (m) for obstacle pixels")
    parser.add_argument("--width", type=int, default=160, help="Output image width (set 0 to keep original)")
    parser.add_argument("--height", type=int, default=96, help="Output image height (set 0 to keep original)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_size = (args.width, args.height) if args.width > 0 and args.height > 0 else None
    process_sequence(args.rgb_dir, args.depth_dir, args.mask_dir, args.depth_scale,
                     args.obstacle_threshold, target_size)
    print("Finished generating masks. Place RGB images under <root>/rgb and masks under <root>/mask for training.")

