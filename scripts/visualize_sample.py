#!/usr/bin/env python3
"""Visualize preprocessed GraspNet samples and check CGN positive labels.

The script consumes only the on-disk NPZ format. It imports neither TensorFlow
nor graspnetAPI and can be run in the ``gn_data`` environment.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
import yaml


def _load_config(config_path: Path) -> Dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError("Missing Contact-GraspNet config: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config


def _load_sample(path: str, expected_points: int) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as sample:
        result = {
            key: np.asarray(sample[key]).copy()
            for key in (
                "point_cloud",
                "segmentation",
                "contact_points",
                "contact_dirs",
                "contact_approaches",
                "finger_diffs",
                "camera_pose",
            )
        }
        result["scene_id"] = np.asarray(sample["scene_id"]).copy()
        result["ann_id"] = np.asarray(sample["ann_id"]).copy()

    points = result["point_cloud"]
    contacts = result["contact_points"]
    directions = result["contact_dirs"]
    approaches = result["contact_approaches"]
    widths = result["finger_diffs"]
    if points.shape != (expected_points, 3):
        raise ValueError(
            "Expected point_cloud shape ({},3) in {}, got {}"
            .format(expected_points, path, points.shape)
        )
    if result["segmentation"].shape != (expected_points,):
        raise ValueError("Segmentation shape does not match point cloud in {}".format(path))
    if contacts.ndim != 2 or contacts.shape[1] != 3:
        raise ValueError("Expected contact_points shape [M,3] in {}".format(path))
    if directions.shape != contacts.shape or approaches.shape != contacts.shape:
        raise ValueError("Contact/direction shapes differ in {}".format(path))
    if widths.shape != (len(contacts),) or len(contacts) % 2:
        raise ValueError("Expected one width per paired contact in {}".format(path))
    if result["camera_pose"].shape != (4, 4):
        raise ValueError("Expected 4x4 camera_pose in {}".format(path))
    if not (
        np.isfinite(points).all()
        and np.isfinite(contacts).all()
        and np.isfinite(directions).all()
        and np.isfinite(approaches).all()
        and np.isfinite(widths).all()
    ):
        raise ValueError("Non-finite values in {}".format(path))
    return result


def _positive_labels(
    point_cloud: np.ndarray,
    contact_points: np.ndarray,
    contact_dirs: np.ndarray,
    max_radius: float,
    filter_z: bool,
    z_val: float,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_mask = np.ones(len(contact_points), dtype=bool)
    if filter_z:
        candidate_mask &= contact_dirs[:, 2] > z_val
    if not candidate_mask.any():
        raise ValueError("No contact labels remain after configured Z-direction filter")

    distances, _ = cKDTree(contact_points[candidate_mask]).query(point_cloud, k=1)
    positive_mask = distances * distances < max_radius * max_radius
    return positive_mask, distances


def _select_overlay_contacts(
    contacts: np.ndarray,
    approaches: np.ndarray,
    max_grasps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    num_grasps = len(contacts) // 2
    selected_count = min(num_grasps, max_grasps)
    if selected_count == num_grasps:
        selected = np.arange(num_grasps)
    else:
        selected = np.sort(rng.choice(num_grasps, size=selected_count, replace=False))
    contact_indices = np.column_stack([2 * selected, 2 * selected + 1]).reshape(-1)
    return contacts[contact_indices], approaches[contact_indices]


def _write_overlay_ply(
    path: Path,
    point_cloud: np.ndarray,
    contacts: np.ndarray,
    approaches: np.ndarray,
    arrow_length: float,
) -> None:
    import open3d as o3d

    num_line_steps = 12
    arrow_starts = contacts - arrow_length * approaches
    alphas = np.linspace(0.0, 1.0, num_line_steps, endpoint=False, dtype=np.float32)
    line_points = (
        arrow_starts[:, None, :]
        + alphas[None, :, None] * (contacts - arrow_starts)[:, None, :]
    ).reshape((-1, 3))

    vertices = np.concatenate([point_cloud, contacts, line_points], axis=0).astype(
        np.float64, copy=False
    )
    colors = np.concatenate(
        [
            np.tile(np.array([[0.72, 0.72, 0.72]]), (len(point_cloud), 1)),
            np.tile(np.array([[1.0, 0.12, 0.08]]), (len(contacts), 1)),
            np.tile(np.array([[0.0, 0.75, 1.0]]), (len(line_points), 1)),
        ],
        axis=0,
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(vertices)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=True):
        raise IOError("Open3D failed to write overlay PLY: {}".format(path))


def _write_screenshot(
    path: Path,
    point_cloud: np.ndarray,
    contacts: np.ndarray,
    approaches: np.ndarray,
    arrow_length: float,
    scene_id: int,
    ann_id: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        s=2,
        c="0.70",
        alpha=0.7,
        label="input cloud",
    )
    axis.scatter(
        contacts[:, 0],
        contacts[:, 1],
        contacts[:, 2],
        s=12,
        c="red",
        label="grasp contacts",
    )
    starts = contacts - arrow_length * approaches
    vectors = contacts - starts
    axis.quiver(
        starts[:, 0],
        starts[:, 1],
        starts[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        color="deepskyblue",
        arrow_length_ratio=0.22,
        linewidth=0.7,
        normalize=False,
    )

    combined = np.concatenate([point_cloud, contacts, starts], axis=0)
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.55, arrow_length)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("camera X (m)")
    axis.set_ylabel("camera Y (m)")
    axis.set_zlabel("camera Z (m)")
    axis.set_title(
        "Scene {:04d}, ann {:04d}: cyan arrows point along approach"
        .format(scene_id, ann_id)
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _show_open3d(
    point_cloud: np.ndarray,
    contacts: np.ndarray,
    approaches: np.ndarray,
    arrow_length: float,
) -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("--show requires a display; omit it to use saved PLY/PNG files")

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    cloud.paint_uniform_color([0.72, 0.72, 0.72])

    contact_cloud = o3d.geometry.PointCloud()
    contact_cloud.points = o3d.utility.Vector3dVector(contacts)
    contact_cloud.paint_uniform_color([1.0, 0.1, 0.05])

    starts = contacts - arrow_length * approaches
    line_points = np.stack([starts, contacts], axis=1).reshape((-1, 3))
    lines = np.arange(len(line_points), dtype=np.int32).reshape((-1, 2))
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 0.75, 1.0]]), (len(lines), 1))
    )
    o3d.visualization.draw_geometries([cloud, contact_cloud, line_set])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gn_root",
        default=os.environ.get("GN_ROOT"),
        help="GraspNet root (defaults to GN_ROOT)",
    )
    parser.add_argument("--camera", choices=("kinect", "realsense"), default="kinect")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_overlay_grasps", type=int, default=150)
    parser.add_argument("--arrow_length", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", action="store_true", help="Open an interactive Open3D window")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "contact_graspnet" / "config.yaml",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.gn_root:
        raise SystemExit("Set GN_ROOT or pass --gn_root.")
    if args.num_samples <= 0 or args.max_overlay_grasps <= 0:
        raise ValueError("--num_samples and --max_overlay_grasps must be positive")
    if args.arrow_length <= 0:
        raise ValueError("--arrow_length must be positive")

    root = Path(args.gn_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("GraspNet root does not exist: {}".format(root))
    config = _load_config(args.config)
    data_config = config["DATA"]
    num_points = int(data_config["num_point"])
    label_config = data_config["labels"]
    if int(label_config["k"]) != 1:
        raise ValueError("Visualization currently matches CGN label assignment only for k=1")
    max_radius = float(label_config["max_radius"])
    filter_z = bool(label_config["filter_z"])
    z_val = float(label_config["z_val"])

    pattern = str(root / "cgn_preprocessed" / args.camera / "scene_*" / "*.npz")
    sample_paths = sorted(glob.glob(pattern))
    if len(sample_paths) < args.num_samples:
        raise ValueError(
            "Requested {} samples, but only {} preprocessed NPZ files exist under {}"
            .format(args.num_samples, len(sample_paths), pattern)
        )

    rng = np.random.default_rng(args.seed)
    selected_indices = np.sort(
        rng.choice(len(sample_paths), size=args.num_samples, replace=False)
    )
    selected_paths = [sample_paths[index] for index in selected_indices]
    output_dir = root / "cgn_preprocessed" / "visualizations" / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in selected_paths:
        sample = _load_sample(path, num_points)
        point_cloud = sample["point_cloud"].astype(np.float32, copy=False)
        contacts = sample["contact_points"].astype(np.float32, copy=False)
        contact_dirs = sample["contact_dirs"].astype(np.float32, copy=False)
        approaches = sample["contact_approaches"].astype(np.float32, copy=False)
        scene_id = int(sample["scene_id"])
        ann_id = int(sample["ann_id"])

        positive_mask, nearest_distances = _positive_labels(
            point_cloud,
            contacts,
            contact_dirs,
            max_radius,
            filter_z,
            z_val,
        )
        display_contacts, display_approaches = _select_overlay_contacts(
            contacts,
            approaches,
            args.max_overlay_grasps,
            rng,
        )
        sample_name = "scene_{:04d}_ann_{:04d}".format(scene_id, ann_id)
        ply_path = output_dir / (sample_name + "_overlay.ply")
        png_path = output_dir / (sample_name + "_overlay.png")
        _write_overlay_ply(
            ply_path,
            point_cloud,
            display_contacts,
            display_approaches,
            args.arrow_length,
        )
        _write_screenshot(
            png_path,
            point_cloud,
            display_contacts,
            display_approaches,
            args.arrow_length,
            scene_id,
            ann_id,
        )

        visible_distances, _ = cKDTree(point_cloud).query(contacts, k=1)
        approach_starts = contacts - args.arrow_length * approaches
        start_distances, _ = cKDTree(point_cloud).query(approach_starts, k=1)
        inward_fraction = float(np.mean(start_distances > visible_distances))
        positive_count = int(positive_mask.sum())
        print(
            "scene={:04d} ann={:04d}: grasps={} positive_points={}/{} ({:.4%}) "
            "nearest_label_dist_median={:.4f} contact_to_cloud_median={:.4f}m "
            "contact_to_cloud_p90={:.4f}m approach_inward_fraction={:.2%}"
            .format(
                scene_id,
                ann_id,
                len(contacts) // 2,
                positive_count,
                len(point_cloud),
                positive_count / float(len(point_cloud)),
                float(np.median(nearest_distances)),
                float(np.median(visible_distances)),
                float(np.percentile(visible_distances, 90)),
                inward_fraction,
            ),
            flush=True,
        )
        print("  overlay PLY: {}".format(ply_path), flush=True)
        print("  screenshot: {}".format(png_path), flush=True)
        if args.show:
            _show_open3d(
                point_cloud,
                display_contacts,
                display_approaches,
                args.arrow_length,
            )


if __name__ == "__main__":
    main()
