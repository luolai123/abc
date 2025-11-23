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

## 环境与编译
1. **克隆仓库并进入工作目录**
   ```bash
   git clone --depth 1 https://github.com/TJU-Aerial-Robotics/YOPO.git
   cd YOPO
   ```

2. **创建 Python 环境并安装依赖**（建议 Conda）
   ```bash
   conda create -n yopo python=3.8
   conda activate yopo
   pip install -r requirements.txt
   ```

3. **编译两个 ROS 工作空间**
   - 控制与动力学（无需 CUDA）
     ```bash
     conda deactivate     # catkin 建议在系统环境下编译
     cd Controller
     catkin_make
     ```
   - 传感器/环境模拟（需要 CUDA）
     ```bash
     cd ../Simulator
     catkin_make
     ```
   每次开新终端运行仿真或采集数据前，记得 `source devel/setup.bash`。

## 仿真运行（使用预训练权重）
1. 启动控制器
   ```bash
   cd Controller
   source devel/setup.bash
   roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
   ```
2. 启动传感器/环境模拟
   ```bash
   cd ../Simulator
   source devel/setup.bash
   rosrun sensor_simulator sensor_simulator_cuda
   ```
   - 若需要在 RViz 查看 RGB 输入，确保 `Simulator/src/config/config.yaml` 中 `render_rgb: true`，摄像头话题默认为 `/rgb_image`。
3. 运行 YOPO
   ```bash
   cd ../YOPO
   conda activate yopo
   python test_yopo_ros.py --trial=1 --epoch=50
   ```
4. 可视化（可选）
   ```bash
   rviz -d yopo.rviz
   ```

## 数据采集（YOPO 训练 & RGB 分割）
1. **配置采集范围与相机参数**：编辑 `Simulator/src/config/config.yaml`（如 `env_num`、`image_num`、`x_range`、`camera.max_depth_dist`）。
2. **运行采集器**（会重置 `../dataset/`）：
   ```bash
   cd Simulator
   source devel/setup.bash
   rosrun sensor_simulator dataset_generator
   ```
   生成内容：
   - `dataset/<map_id>/img_*.png`：每个随机环境的深度 16-bit PNG（按 `max_depth_dist` 归一化）。
   - `dataset/pose-<map_id>.csv`：对应位姿（px,py,pz,qw,qx,qy,qz）。
   - `dataset/depth/img_<map_id>_<idx>.png`：聚合后的深度（便于分割数据处理）。
   - `dataset/rgb/img_<map_id>_<idx>.png`：按深度伪彩上色的 RGB（与深度对齐，可直接用于分割训练）。

   > 说明：RGB 由深度经 `COLORMAP_TURBO` 伪彩映射生成，若需要真实纹理可自行替换相机渲染逻辑。

### 训练概览
- **运动基元偏移量训练（YOPO 主体）**：`python train_yopo.py`，使用深度图和轨迹优化标签监督网络输出的末端状态与评分。
- **RGB 像素级二分类训练**：`python train_segmentation.py --data_root ../dataset`，输入与深度对齐的 RGB，输出障碍/安全区域掩码，用于推理时约束运动基元的选择方向。

## YOPO 训练流程
1. **准备数据**：默认读取 `config/traj_opt.yaml` 中的 `dataset_path`（默认为 `../dataset`），直接使用上一步采集的深度图。
2. **启动训练**
   ```bash
   cd YOPO
   conda activate yopo
   python train_yopo.py
   ```
   典型设置：~10 张地图、共 10 万帧，RTX 3080 约 1 小时完成 50 epoch。调整轨迹/速度采样请修改 `config/traj_opt.yaml`。

## RGB 障碍分割训练
1. **生成二值掩膜**（深度→mask，对齐 RGB）：
   ```bash
   cd YOPO
   conda activate yopo
   python segmentation/data_preparation.py \
     --rgb_dir ../dataset/rgb --depth_dir ../dataset/depth --mask_dir ../dataset/mask \
     --depth_scale 0.00030518 --obstacle_threshold 5.0 --width 160 --height 96
   ```
   - `depth_scale` 应设为 `max_depth_dist / 65535`（默认 20m ≈ 0.00030518）。
   - 输出 `dataset/mask/*.png` 与 RGB 同名。

2. **训练分割网络**
   ```bash
   python train_segmentation.py --data_root ../dataset --epochs 50 --batch_size 8
   ```
   检查点保存在 `YOPO/saved/segmentation/`。

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
