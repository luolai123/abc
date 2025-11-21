# You Only Plan Once (YOPO)

YOPO is a learning-based motion planner for agile flight in obstacle-dense environments. It fuses perception, motion primitive search, and trajectory optimization into one stage, now supporting RGB-based obstacle segmentation to guide safer primitive selection.

## Features
- One-stage planner that scores and refines motion primitives for high-speed flight.
- CUDA-accelerated simulator for large-scale data generation and testing.
- RGB obstacle segmentation pipeline: depth-threshold label generation, training script, and ROS integration for selecting safe regions.
- TensorRT export for real-time deployment on NVIDIA Jetson.

## Requirements
- Ubuntu 20.04
- CUDA-capable GPU (for training/simulation)
- ROS (tested with catkin)
- Conda (recommended)
- Python 3.8

## Installation
1. **Clone the repository**
   ```bash
git clone --depth 1 https://github.com/TJU-Aerial-Robotics/YOPO.git
target_dir=YOPO
cd $target_dir
```

2. **Create and activate the environment**
   ```bash
conda create -n yopo python=3.8
conda activate yopo
pip install -r requirements.txt
```

3. **Build simulators (ROS workspaces)**
   ```bash
# Controller and dynamics
conda deactivate
cd Controller
catkin_make

# Environment and sensors (CUDA-enabled)
cd ../Simulator
catkin_make
```

## Quick Start
### Run the pre-trained planner in simulation
1. **Start controller**
   ```bash
cd Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```
2. **Start sensor/environment simulator**
   ```bash
cd ../Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda
```
3. **Launch YOPO (pretrained)**
   ```bash
cd ../YOPO
conda activate yopo
python test_yopo_ros.py --trial=1 --epoch=50
```
4. **Visualize (optional)**
   ```bash
rviz -d yopo.rviz
```

### Train YOPO policy
1. **Collect dataset**
   ```bash
cd Simulator
source devel/setup.bash
rosrun sensor_simulator dataset_generator
```
   Data is saved to `./dataset/` under the project root. Adjust sampling/environment in `Simulator/src/config/config.yaml`.

2. **Train**
   ```bash
cd ../YOPO
conda activate yopo
python train_yopo.py
```
   Configure trajectory optimization in `YOPO/config/traj_opt.yaml`. Training 50 epochs on ~100k samples typically completes within an hour on an RTX 3080.

### Train RGB obstacle segmentation
1. **Prepare RGB/depth pairs** (collected from the simulator). Generate binary masks using depth thresholds:
   ```bash
cd YOPO
conda activate yopo
python -m segmentation.data_preparation --data_root ../dataset --depth_threshold 5.0
```
2. **Train the segmentation model**
   ```bash
python train_segmentation.py --data_root ../dataset --epochs 50 --batch_size 8
```
   The script trains a lightweight U-Net with binary cross-entropy and saves checkpoints to `YOPO/saved/segmentation/`.

## Advanced Usage
- **Sensor/environment config:** adjust camera/LiDAR and maze parameters in `Simulator/src/config/config.yaml`.
- **Flight speed and penalties:** tune trajectory settings in `YOPO/config/traj_opt.yaml`.
- **TensorRT deployment:**
  ```bash
conda activate yopo
pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
cd YOPO
python yopo_trt_transfer.py --trial=1 --epoch=50
python test_yopo_ros.py --use_tensorrt=1
```
- **Real hardware:** set `env` in `test_yopo_ros.py` to your camera depth unit (e.g., `env: 435`) and update odometry topics to NWU frame. Ensure RGB camera resolution/FOV matches training setup.

## FAQ
- **Where are pretrained weights?** Default YOPO weights are at `YOPO/saved/YOPO_1/epoch50.pth`; segmentation checkpoints are saved under `YOPO/saved/segmentation/`.
- **RViz map is empty.** Confirm both controller and sensor simulators are running and `rviz` uses `yopo.rviz`.
- **CUDA build issues in Simulator.** See `Simulator/src/readme.md` for troubleshooting CUDA and environment settings.
- **Slow training on hybrid CPUs.** Pin training to performance cores, e.g., `taskset -c 1,2,3,4 python train_yopo.py`.

## Citation
- Paper: [You Only Plan Once: A Learning-Based One-Stage Planner With Guidance Learning](https://ieeexplore.ieee.org/document/10528860)
- YOPOv2-Tracker: [arXiv](https://arxiv.org/html/2505.06923v1)
