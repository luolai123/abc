TODO:

1. Enable AMP (Automatic Mixed Precision) training.
Ensure that the CUDA versions of the virtual environment and system are consistent.

## RGB segmentation data & training

- Use `python segmentation/data_preparation.py --rgb_dir <RGB_DIR> --depth_dir <DEPTH_DIR> --mask_dir <MASK_DIR>` to convert aligned RGB/depth recordings into pixel-wise obstacle/safe masks (depth values closer than the threshold are marked as obstacles).
- Arrange the generated data as:
  - `<dataset_root>/rgb/*.png|jpg`
  - `<dataset_root>/mask/*.png`
- Train the lightweight U-Net binary classifier with `python train_segmentation.py --data_root <dataset_root> --log_dir saved/segmentation`.

## Online inference notes

- `test_yopo_ros.py` now consumes an RGB topic (`rgb_topic`) and runs the segmentation network before YOPO inference. The binary mask is used both as the perception input to the policy (replacing the depth channel) and to bias primitive selection toward the largest safe component in the view.
- Configure segmentation weights via `--segmentation_weight` when launching `test_yopo_ros.py`.
