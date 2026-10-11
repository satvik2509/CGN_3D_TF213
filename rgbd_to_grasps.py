#!/usr/bin/env python3
"""Build a point cloud from RGB-D captures, then run Contact-GraspNet.

Reads RealSense-style files from a capture folder:

    rgb_XXXX.png
    depth_XXXX.npy  (or depth_XXXX.png)
    intrinsics.json

Back-projects each frame into a colored point cloud, detects the table plane,
and sends the object cloud into the 3D grasp pipeline.

Example:
    python rgbd_to_grasps.py --captures captures
    python rgbd_to_grasps.py --captures captures --index 0
"""

import argparse
import json
import os
import sys
from importlib import import_module

import cv2
import numpy as np
import open3d as o3d
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "contact_graspnet"))
sys.path.append(os.path.join(BASE_DIR, "contact_graspnet", "pointnet2", "utils"))

import config_utils
from contact_grasp_estimator import GraspEstimator

_inf = import_module("3d_inference_new")
_ORIG_EXPORT = _inf.export_3d_scene


def table_alignment_transform(table_normal, table_d, cam_x=(1.0, 0.0, 0.0)):
    """4x4 transform into a z-up table frame: table plane -> z=0, objects at z>0.

    x follows the camera's right direction (projected onto the table), so the
    scene looks as the camera saw it, just right side up. Needs the plane
    oriented so that normal . p + d > 0 above the table.
    """
    n = np.asarray(table_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    cam_x = np.asarray(cam_x, dtype=np.float64)
    x = cam_x - np.dot(cam_x, n) * n
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - n[1] * n
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, n], axis=0)  # proper rotation (det = +1)
    T[2, 3] = table_d                        # new z = n . p + d
    return T


def _apply_pts(pts, T):
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def install_upright_viewer(z_offset):
    """Make the grasp visualization right side up WITHOUT touching inference.

    Inference runs on the camera-frame cloud exactly as before. Only at the very
    end, when the pipeline hands its (camera-frame, z-shifted) data to the
    exporter, we rotate cloud + grasps + contact points into a z-up table frame.

    z_offset is the shift 3d_inference_new applies before exporting
    (-min z of the input cloud); it is needed to place the table plane correctly.
    """
    def export_upright(pc, pc_colors, pred_grasps, scores, contact_pairs, out_path, **kw):
        n, d = kw.get("table_normal"), kw.get("table_d")
        if n is None or d is None:
            return _ORIG_EXPORT(pc, pc_colors, pred_grasps, scores, contact_pairs, out_path, **kw)

        n = np.asarray(n, dtype=np.float64)
        # Plane was fit in the unshifted camera frame; exporter data is shifted by +z_offset in z.
        T = table_alignment_transform(n, float(d) - n[2] * z_offset)

        pc_u = _apply_pts(pc, T)
        grasps_u = {k: (np.einsum("ij,njk->nik", T, g) if len(g) else g)
                    for k, g in pred_grasps.items()}
        pairs_u = None
        if contact_pairs is not None:
            pairs_u = {k: (_apply_pts(v, T) if v is not None and len(v) else v)
                       for k, v in contact_pairs.items()}

        kw = dict(kw)
        kw["table_normal"], kw["table_d"] = None, None  # table is now the z=0 plane
        if kw.get("table_inlier_points") is not None:
            kw["table_inlier_points"] = _apply_pts(kw["table_inlier_points"], T)
        print("Visualization rotated into table frame (table at z=0, +z up); "
              "inference and saved poses remain in the camera frame.")
        return _ORIG_EXPORT(pc_u, pc_colors, grasps_u, scores, pairs_u, out_path, **kw)

    _inf.export_3d_scene = export_upright


def list_capture_indices(cap_dir):
    idxs = []
    for name in os.listdir(cap_dir):
        if name.startswith("rgb_") and name.endswith(".png"):
            idxs.append(int(name[4:8]))
    return sorted(idxs)


def load_intrinsics(cap_dir):
    path = os.path.join(cap_dir, "intrinsics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path) as f:
        return json.load(f)


def load_rgb(cap_dir, idx):
    path = os.path.join(cap_dir, f"rgb_{idx:04d}.png")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(cap_dir, idx):
    npy_path = os.path.join(cap_dir, f"depth_{idx:04d}.npy")
    png_path = os.path.join(cap_dir, f"depth_{idx:04d}.png")
    if os.path.exists(npy_path):
        depth = np.load(npy_path)
    else:
        depth = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing {npy_path} and {png_path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth for frame {idx} has shape {depth.shape}, expected HxW")
    return depth


def rgbd_to_pointcloud(rgb, depth_raw, intr, depth_min, depth_max):
    """Back-project aligned RGB-D into an Open3D cloud in the OpenCV camera frame."""
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    depth_scale = float(intr["depth_scale_m_per_unit"])

    z = depth_raw.astype(np.float32) * depth_scale
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    v, u = np.nonzero(valid)
    z = z[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    h, w = depth_raw.shape
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    colors = rgb[v, u].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_capture_cloud(cap_dir, idx, intr, depth_min, depth_max, voxel_size):
    rgb = load_rgb(cap_dir, idx)
    depth = load_depth(cap_dir, idx)
    pcd = rgbd_to_pointcloud(rgb, depth, intr, depth_min, depth_max)
    n_raw = len(pcd.points)
    if voxel_size and voxel_size > 0 and n_raw:
        pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Frame {idx:04d}: {n_raw} points -> {len(pcd.points)} after voxel={voxel_size}")
    return pcd


def parse_args():
    p = argparse.ArgumentParser(
        description="RGB-D captures -> point cloud -> Contact-GraspNet (with table detection)"
    )
    p.add_argument("--captures", type=str, default="captures",
                   help="Folder with rgb_XXXX.png, depth_XXXX.npy, and intrinsics.json")
    p.add_argument("--index", type=int, default=None,
                   help="Process only this capture index (default: all frames)")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/scene_test_2048_bs3_hor_sigma_001")
    p.add_argument("--depth_min", type=float, default=0.15, help="Min depth in meters")
    p.add_argument("--depth_max", type=float, default=2.00, help="Max depth in meters")
    p.add_argument("--voxel_size", type=float, default=0.004,
                   help="Voxel size (m) for the reconstructed cloud; 0 disables downsampling")
    p.add_argument("--cloud_dir", type=str, default=None,
                   help="Where to write reconstructed .ply files (default: <captures>/pointclouds)")
    p.add_argument("--out_dir", type=str, default="visualisation",
                   help="Where to write grasp visualizations")
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--min_points", type=int, default=30)
    p.add_argument("--min_clearance", type=float, default=0.005)
    p.add_argument("--standoff", type=float, default=0.10)
    p.add_argument("--score_threshold", type=float, default=0.23)
    p.add_argument("--top_n", type=int, default=-1)
    p.add_argument("--single_contact", action="store_true")
    p.add_argument("--no_upright", action="store_true",
                   help="Do not rotate the visualization into a z-up table frame "
                        "(shows the raw OpenCV camera frame, which looks upside down)")
    return p.parse_args()


def main():
    args = parse_args()
    cap_dir = args.captures
    if not os.path.isdir(cap_dir):
        raise FileNotFoundError(f"Capture folder not found: {cap_dir}")

    indices = list_capture_indices(cap_dir)
    if not indices:
        raise FileNotFoundError(f"No rgb_XXXX.png files in {cap_dir}")
    if args.index is not None:
        if args.index not in indices:
            raise ValueError(f"Frame {args.index} not in {indices}")
        indices = [args.index]

    intr = load_intrinsics(cap_dir)
    cloud_dir = args.cloud_dir or os.path.join(cap_dir, "pointclouds")
    os.makedirs(cloud_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    print("Loading Global Config...")
    global_config = config_utils.load_config(args.ckpt_dir, batch_size=1, arg_configs=[])
    print("Initializing GraspEstimator...")
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()
    saver = tf.train.Saver(save_relative_paths=True)
    grasp_estimator.load_weights(sess, saver, args.ckpt_dir, mode="test")

    for idx in indices:
        pcd = build_capture_cloud(
            cap_dir, idx, intr, args.depth_min, args.depth_max, args.voxel_size
        )
        ply_path = os.path.join(cloud_dir, f"scene_{idx:04d}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Wrote point cloud {ply_path}")

        if args.no_upright:
            _inf.export_3d_scene = _ORIG_EXPORT
        else:
            z_off = -float(np.asarray(pcd.points)[:, 2].min())
            install_upright_viewer(z_off)

        out_path = os.path.join(args.out_dir, f"captures_{idx:04d}_grasps.glb")
        _inf.process_pcd_scene(
            sess, grasp_estimator, pcd,
            eps=args.eps, min_points=args.min_points,
            min_clearance=args.min_clearance, standoff=args.standoff,
            score_threshold=args.score_threshold,
            show_both_contacts=not args.single_contact,
            out_path=out_path,
            source_name=ply_path,
            table_method="dominant",
            require_table=True,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()