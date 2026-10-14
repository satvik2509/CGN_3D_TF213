#!/usr/bin/env python3
"""Convert RealSense RGB-D captures into colored PLY point clouds.

Reads RealSense-style files from a capture folder:

    rgb_XXXX.png
    depth_XXXX.npy  (or depth_XXXX.png)
    intrinsics.json

Back-projects each frame into a colored point cloud and writes one PLY file
per frame. By default, all captures are converted into
scenes_11_10_26/pointclouds/.

Example:
    python rgbd_to_pc.py
    python rgbd_to_pc.py --captures scenes_11_10_26 --index 0
"""

import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d


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


def edge_aware_median_filter(depth_raw, depth_scale, kernel_size=3, edge_threshold_m=0.01):
    """Reduce isolated depth jitter without filling holes or smoothing depth edges."""
    if kernel_size == 0:
        return depth_raw
    if kernel_size not in (3, 5):
        raise ValueError("Depth filter kernel must be 0, 3, or 5.")

    valid = depth_raw > 0
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    median = cv2.medianBlur(depth_raw, kernel_size)
    valid_count = cv2.boxFilter(
        valid.astype(np.uint8), -1, (kernel_size, kernel_size),
        normalize=False, borderType=cv2.BORDER_REPLICATE,
    )
    depth_min = cv2.erode(
        np.where(valid, depth_raw, np.iinfo(depth_raw.dtype).max),
        kernel, borderType=cv2.BORDER_REPLICATE,
    )
    depth_max = cv2.dilate(
        np.where(valid, depth_raw, 0), kernel, borderType=cv2.BORDER_REPLICATE,
    )
    max_local_range = edge_threshold_m / depth_scale
    smooth = (
        valid
        & (valid_count >= (kernel_size * kernel_size // 2 + 1))
        & ((depth_max.astype(np.int32) - depth_min.astype(np.int32)) <= max_local_range)
    )

    filtered = depth_raw.copy()
    filtered[smooth] = median[smooth]
    return filtered


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


def flatten_detected_table(pcd, distance_threshold=0.012):
    """Project only detected table-plane inliers onto their fitted plane."""
    if len(pcd.points) < 50:
        print("Table flattening skipped: point cloud has too few points.")
        return pcd

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=2000,
    )
    if len(inliers) < 200 or len(inliers) < 0.05 * len(pcd.points):
        print(f"Table flattening skipped: largest plane has only {len(inliers)} "
              "supporting points.")
        return pcd

    normal = np.asarray(plane_model[:3], dtype=np.float64)
    d = float(plane_model[3])
    norm = np.linalg.norm(normal)
    if norm == 0:
        print("Table flattening skipped: detected plane has an invalid normal.")
        return pcd
    normal /= norm
    d /= norm

    points = np.asarray(pcd.points).copy()
    inlier_indices = np.asarray(inliers, dtype=np.intp)
    signed_distances = points[inlier_indices] @ normal + d
    points[inlier_indices] -= signed_distances[:, None] * normal
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Flattened {len(inliers)} table-plane points "
          f"({100 * len(inliers) / len(points):.1f}% of cloud).")
    return pcd


def build_capture_cloud(cap_dir, idx, intr, depth_min, depth_max, voxel_size,
                        depth_filter_kernel, flatten_table):
    rgb = load_rgb(cap_dir, idx)
    depth = load_depth(cap_dir, idx)
    depth = edge_aware_median_filter(
        depth, float(intr["depth_scale_m_per_unit"]), depth_filter_kernel
    )
    pcd = rgbd_to_pointcloud(rgb, depth, intr, depth_min, depth_max)
    n_raw = len(pcd.points)
    if flatten_table:
        pcd = flatten_detected_table(pcd)
    if voxel_size and voxel_size > 0 and n_raw:
        pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Frame {idx:04d}: {n_raw} points -> {len(pcd.points)} after voxel={voxel_size}")
    return pcd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert RGB-D captures to colored PLY point clouds"
    )
    p.add_argument("--captures", type=str, default="scenes_11_10_26",
                   help="Folder with rgb_XXXX.png, depth_XXXX.npy, and intrinsics.json")
    p.add_argument("--index", type=int, default=None,
                   help="Process only this capture index (default: all frames)")
    p.add_argument("--depth_min", type=float, default=0.15, help="Min depth in meters")
    p.add_argument("--depth_max", type=float, default=2.00, help="Max depth in meters")
    p.add_argument("--voxel_size", type=float, default=0.004,
                   help="Voxel size (m) for the reconstructed cloud; 0 disables downsampling")
    p.add_argument("--depth_filter_kernel", type=int, choices=(0, 3, 5), default=3,
                   help="Edge-aware median depth filter kernel; 0 disables it")
    p.add_argument("--no-flatten-table", action="store_true",
                   help="Keep measured depth variation on the detected table plane")
    p.add_argument("--cloud_dir", type=str, default=None,
                   help="Where to write PLYs (default: <captures>/pointclouds)")
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

    print(f"Converting {len(indices)} capture(s) to PLY in {cloud_dir}")
    for idx in indices:
        pcd = build_capture_cloud(
            cap_dir, idx, intr, args.depth_min, args.depth_max, args.voxel_size,
            args.depth_filter_kernel, not args.no_flatten_table,
        )
        ply_path = os.path.join(cloud_dir, f"scene_{idx:04d}.ply")
        if not o3d.io.write_point_cloud(ply_path, pcd):
            raise OSError(f"Failed to write point cloud: {ply_path}")
        print(f"Wrote point cloud {ply_path}")

    print(f"Converted {len(indices)} capture(s); no grasp inference was run.")


if __name__ == "__main__":
    main()