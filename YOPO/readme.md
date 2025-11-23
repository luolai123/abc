TODO:

1. Enable AMP (Automatic Mixed Precision) training.
Ensure that the CUDA versions of the virtual environment and system are consistent.

## RGB segmentation data & training

- Collect data with the simulator: `rosrun sensor_simulator dataset_generator` (runs from `Simulator/`, saves monocular RGB to `../dataset/rgb/`).
- Build masks directly from RGB: `python segmentation/data_preparation.py --rgb_dir ../dataset/rgb --mask_dir ../dataset/mask --obstacle_threshold 5.0 --width 160 --height 96`.
- Arrange training data as:
  - `<dataset_root>/rgb/*.png|jpg`
  - `<dataset_root>/mask/*.png`
- Train the lightweight U-Net binary classifier: `python train_segmentation.py --data_root ../dataset --log_dir saved/segmentation --epochs 50 --batch_size 8`.

## Online inference notes

- `test_yopo_ros.py` now consumes an RGB topic (`rgb_topic`) and runs the segmentation network before YOPO inference. The binary mask is used both as the perception input to the policy (replacing the depth channel) and to bias primitive selection toward the largest safe component in the view. Only monocular RGB encodings (`rgb8`/`bgr8`) are accepted.
- Configure segmentation weights via `--segmentation_weight` when launching `test_yopo_ros.py`.
