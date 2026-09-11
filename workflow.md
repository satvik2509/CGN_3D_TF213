# Running `3d_inference_new.py`

This guide covers cloning the repo, setting up the environment, and running the
3D point-cloud grasp inference pipeline end to end.

## 1. Clone the repo

```bash
git clone https://github.com/satvik2509/CGN_3D_TF213.git
cd CGN_3D_TF213
```

## 2. Set up the environment

```bash
conda env create -f env_setup/environment-tf213.yml
conda activate cgn_3d
pip install -r env_setup/requirements-tf213.txt
```

If you're on an A100 (or another GPU needing its own build), use the
A100-specific setup instead:

```bash
bash env_setup/setup_a100.sh
pip install -r env_setup/requirements-a100.txt
```

## 3. Recompile the custom TensorFlow ops for your GPU

The PointNet++ ops are compiled CUDA kernels tied to your specific GPU
architecture -- this step must be run on the machine you'll actually train/
infer on, every time you move to new hardware:

```bash
bash env_setup/recompile_ops.sh
```

## 4. Download the Contact-GraspNet checkpoint

```bash
mkdir -p checkpoints
# place/download your checkpoint directory here, e.g.:
# checkpoints/scene_test_2048_bs3_hor_sigma_001/
```

## 5. Run the inference script

Basic run on a `.ply` point cloud:

```bash
python 3d_inference_new.py \
  --ply path/to/your_scan.ply \
  --ckpt_dir checkpoints/scene_test_2048_bs3_hor_sigma_001
```

This will:
- Detect the table plane (or fall back to a virtual floor at the object's
  lowest point if no real table is found)
- Segment objects via DBSCAN
- Predict grasps per object with Contact-GraspNet
- Snap the second antipodal contact point to the object's real surface
- Rotate any floor-colliding grasps about the baseline axis until clear
- Export an interactive 3D visualization (`.html`) and save the final
  rotation + translation matrices (`.npz`)

## 6. Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--ply` | *(required)* | Path to the input `.ply` point cloud |
| `--ckpt_dir` | `checkpoints/scene_test_2048_bs3_hor_sigma_001` | Path to the CGN checkpoint directory |
| `--eps` | `0.02` | DBSCAN neighborhood radius (m) for object clustering |
| `--min_points` | `30` | DBSCAN minimum points per object cluster |
| `--min_clearance` | `0.005` | Minimum allowed gripper-to-floor clearance (m) |
| `--standoff` | `0.10` | Distance (m) checked back along the approach axis for floor collisions |
| `--score_threshold` | `0.23` | Minimum grasp confidence score to display (CGN paper default) |
| `--single_contact` | *(off)* | Show only one contact point per grasp instead of both antipodal points |
| `--top_n` | `-1` (all) | Show only the top N qualifying grasps in the 3D visualization |
| `--out` | `visualisation/PLY_grasps.glb` | Output path for the exported visualization |
| `--no_table` | *(off)* | Skip table plane detection entirely (no floor collision filtering or plane overlay) |
| `--table_normal` | `None` | Manual table plane normal override, e.g. `"0,0,1"` (skips auto-detection) |
| `--table_d` | `None` | Manual table plane offset `d` override (used with `--table_normal`) |

## 7. Example commands

Show only the top 5 best grasps:
```bash
python 3d_inference_new.py \
  --ply test_data/cube.ply \
  --ckpt_dir checkpoints/scene_test_2048_bs3_hor_sigma_001 \
  --top_n 5
```

Skip table detection entirely (object-only scan, no floor collision check):
```bash
python 3d_inference_new.py \
  --ply test_data/cube.ply \
  --ckpt_dir checkpoints/scene_test_2048_bs3_hor_sigma_001 \
  --no_table
```

Supply a known table plane manually instead of auto-detecting it:
```bash
python 3d_inference_new.py \
  --ply test_data/cube.ply \
  --ckpt_dir checkpoints/scene_test_2048_bs3_hor_sigma_001 \
  --table_normal "0,0,1" \
  --table_d -0.75
```

Tighten the confidence threshold and floor clearance:
```bash
python 3d_inference_new.py \
  --ply test_data/cube.ply \
  --ckpt_dir checkpoints/scene_test_2048_bs3_hor_sigma_001 \
  --score_threshold 0.4 \
  --min_clearance 0.01
```

## 8. Output

- `visualisation/PLY_grasps.html` -- interactive 3D scene (point cloud/mesh,
  floor plane, gripper geometry, both antipodal contact points per grasp)
- `visualisation/PLY_grasps_grasp_poses.npz` -- rotation + translation
  matrices for every grasp shown, loadable via:

```python
import numpy as np
data = np.load('visualisation/PLY_grasps_grasp_poses.npz')
R = data['rotations'][0]      # first saved grasp's 3x3 rotation
t = data['translations'][0]   # first saved grasp's translation
scores = data['scores']
```

**Note:** grasp poses are in the point cloud's original world/camera frame.
If your robot controller expects poses in a different frame, apply your
camera-to-robot extrinsic calibration before sending these to hardware.