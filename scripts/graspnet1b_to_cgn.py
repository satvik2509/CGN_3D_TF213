#!/usr/bin/env python3
"""Preprocess GraspNet-1Billion frames into Contact-GraspNet training samples.

Run this script only in the ``gn_data`` environment. It does not import
TensorFlow; its outputs are plain compressed NPZ files for the TF environment.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import tempfile
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import yaml


_WORKER_DATASET = None
_WORKER_GRASP_LABELS = None
_WORKER_COLLISION_LABELS = None
_WORKER_OPTIONS = None

_SPLITS = {
    "train": range(0, 90),
    "val": range(90, 100),
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


def _initialize_worker(
    gn_root: str,
    camera: str,
    num_points: int,
    max_grasps: Optional[int],
    max_width: float,
    seed: int,
) -> None:
    """Load all object grasp labels once in each worker process."""
    global _WORKER_DATASET
    global _WORKER_GRASP_LABELS
    global _WORKER_COLLISION_LABELS
    global _WORKER_OPTIONS

    from graspnetAPI import GraspNet

    _WORKER_DATASET = GraspNet(
        root=gn_root,
        camera=camera,
        split="custom",
        sceneIds=[],
    )
    _WORKER_GRASP_LABELS = _WORKER_DATASET.loadGraspLabels(
        objIds=list(range(88))
    )
    _WORKER_COLLISION_LABELS = {}
    _WORKER_OPTIONS = {
        "gn_root": gn_root,
        "camera": camera,
        "num_points": num_points,
        "max_grasps": max_grasps,
        "max_width": max_width,
        "seed": seed,
    }


def _pointcloud_and_labels(
    depth_path: Path,
    label_path: Path,
    intrinsics_path: Path,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    segmentation = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError("Could not read depth image: {}".format(depth_path))
    if segmentation is None:
        raise FileNotFoundError("Could not read label image: {}".format(label_path))
    if depth.ndim != 2 or segmentation.ndim != 2:
        raise ValueError(
            "Expected single-channel depth/label images, got {} and {} for {}"
            .format(depth.shape, segmentation.shape, depth_path)
        )
    if depth.shape != segmentation.shape:
        raise ValueError(
            "Depth and label image dimensions differ for {}: {} vs {}"
            .format(depth_path, depth.shape, segmentation.shape)
        )

    intrinsics = np.asarray(np.load(str(intrinsics_path)), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            "Expected 3x3 camera intrinsics in {}, got {}"
            .format(intrinsics_path, intrinsics.shape)
        )
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0:
        raise ValueError("Invalid camera intrinsics in {}".format(intrinsics_path))

    # GraspNet depth PNGs are millimeters; API loadScenePointCloud divides by 1000.
    depth_m = depth.astype(np.float32) / 1000.0
    valid_y, valid_x = np.nonzero(depth_m > 0)
    if valid_x.size == 0:
        raise ValueError("No valid depth pixels in {}".format(depth_path))

    z = depth_m[valid_y, valid_x]
    points = np.stack(
        [(valid_x - cx) * z / fx, (valid_y - cy) * z / fy, z],
        axis=1,
    ).astype(np.float32)
    labels = segmentation[valid_y, valid_x].astype(np.int32, copy=False)
    if points.shape != (len(labels), 3):
        raise AssertionError("Unfiltered point/segmentation alignment failed")

    if len(points) >= num_points:
        selected = rng.choice(len(points), size=num_points, replace=False)
    else:
        selected = rng.choice(len(points), size=num_points, replace=True)

    point_cloud = points[selected]
    point_labels = labels[selected]
    if point_cloud.shape != (num_points, 3) or point_labels.shape != (num_points,):
        raise AssertionError("Subsampled point-cloud shapes do not match config")
    return point_cloud, point_labels


def _validate_segmentation_object_ids(
    segmentation: np.ndarray, scene_object_ids: np.ndarray
) -> None:
    if segmentation.ndim != 1:
        raise ValueError(
            "Expected per-point segmentation labels, got {}".format(segmentation.shape)
        )
    # In the inspected GraspNet frames, 0 is background and each nonzero PNG
    # value is the zero-based object ID plus one.
    observed_object_ids = np.unique(segmentation[segmentation > 0]) - 1
    unknown_ids = np.setdiff1d(observed_object_ids, scene_object_ids)
    if unknown_ids.size:
        raise ValueError(
            "Raw label values do not match object_id+1 for this scene: "
            "unknown object IDs {}"
            .format(unknown_ids.tolist())
        )


def _contacts_from_grasp_group(
    grasp_group: Any,
    max_grasps: Optional[int],
    max_width: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    grasp_array = np.asarray(grasp_group.grasp_group_array, dtype=np.float32)
    if grasp_array.ndim != 2 or grasp_array.shape[1] != 17:
        raise ValueError(
            "Expected GraspGroup array shape [G,17], got {}".format(grasp_array.shape)
        )

    if len(grasp_array) == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_values = np.empty((0,), dtype=np.float32)
        return empty_points, empty_points.copy(), empty_points.copy(), empty_values, empty_values.copy(), 0

    # GraspNetAPI GraspGroup rows are [score,width,height,depth,R(9),t(3),object_id].
    rotations = grasp_array[:, 4:13].reshape((-1, 3, 3))
    centers = grasp_array[:, 13:16]
    widths = grasp_array[:, 1]

    finite = (
        np.isfinite(centers).all(axis=1)
        & np.isfinite(rotations).all(axis=(1, 2))
        & np.isfinite(widths)
        & (widths > 0)
        & (widths <= max_width)
    )
    kept = np.flatnonzero(finite)
    width_filtered_count = len(grasp_array) - len(kept)
    if max_grasps is not None and len(kept) > max_grasps:
        kept = rng.choice(kept, size=max_grasps, replace=False)

    rotations = rotations[kept]
    centers = centers[kept]
    widths = widths[kept]
    if len(kept) == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_values = np.empty((0,), dtype=np.float32)
        return empty_points, empty_points.copy(), empty_points.copy(), empty_values, empty_values.copy(), width_filtered_count

    # GraspNetAPI's batch_viewpoint_params_to_matrix builds gripper axes as
    # rotation columns: x (column 0) is approach; plot_gripper_pro_max places
    # the opening along local y (column 1), the closing axis.
    approach = rotations[:, :, 0]
    closing_axis = rotations[:, :, 1]
    axis_norm = np.linalg.norm(closing_axis, axis=1)
    approach_norm = np.linalg.norm(approach, axis=1)
    if np.any(axis_norm <= 1e-8) or np.any(approach_norm <= 1e-8):
        raise ValueError("GraspGroup contains a zero-length gripper axis")
    closing_axis = closing_axis / axis_norm[:, None]
    approach = approach / approach_norm[:, None]

    positive_contacts = centers + 0.5 * widths[:, None] * closing_axis
    negative_contacts = centers - 0.5 * widths[:, None] * closing_axis
    contact_points = np.empty((2 * len(kept), 3), dtype=np.float32)
    contact_dirs = np.empty_like(contact_points)
    contact_approaches = np.empty_like(contact_points)
    finger_diffs = np.repeat(widths, 2).astype(np.float32, copy=False)

    contact_points[0::2] = positive_contacts
    contact_points[1::2] = negative_contacts
    # Match ACRONYM's per-contact convention: each baseline points toward the
    # opposing finger, so the two contacts have opposite direction vectors.
    contact_dirs[0::2] = -closing_axis
    contact_dirs[1::2] = closing_axis
    contact_approaches[0::2] = approach
    contact_approaches[1::2] = approach

    expected_shape = (2 * len(kept), 3)
    if contact_points.shape != expected_shape or contact_dirs.shape != expected_shape:
        raise AssertionError("Converted contact arrays have unexpected shapes")
    if contact_approaches.shape != expected_shape or finger_diffs.shape != (2 * len(kept),):
        raise AssertionError("Converted direction/width shapes are inconsistent")
    if not (
        np.isfinite(contact_points).all()
        and np.isfinite(contact_dirs).all()
        and np.isfinite(contact_approaches).all()
        and np.isfinite(finger_diffs).all()
    ):
        raise ValueError("Converted GraspNet contacts contain non-finite values")

    return (
        contact_points,
        contact_dirs,
        contact_approaches,
        finger_diffs,
        np.asarray(grasp_array[kept, 0], dtype=np.float32),
        width_filtered_count,
    )


def _process_frame(task: Tuple[int, int, str]) -> Tuple[int, int, int, int]:
    scene_id, ann_id, output_path_text = task
    if _WORKER_DATASET is None or _WORKER_OPTIONS is None:
        raise RuntimeError("Preprocessing worker was not initialized")

    options = _WORKER_OPTIONS
    root = Path(options["gn_root"])
    camera = options["camera"]
    scene_name = "scene_{:04d}".format(scene_id)
    camera_dir = root / "scenes" / scene_name / camera
    output_path = Path(output_path_text)

    object_id_path = root / "scenes" / scene_name / "object_id_list.txt"
    if not object_id_path.is_file():
        raise FileNotFoundError("Missing scene object ID list: {}".format(object_id_path))
    scene_object_ids = np.loadtxt(object_id_path, dtype=np.int32, ndmin=1)
    scene_object_ids = np.asarray(scene_object_ids, dtype=np.int32).reshape(-1)

    rng = np.random.default_rng(
        np.random.SeedSequence([options["seed"], scene_id, ann_id])
    )
    point_cloud, segmentation = _pointcloud_and_labels(
        camera_dir / "depth" / "{:04d}.png".format(ann_id),
        camera_dir / "label" / "{:04d}.png".format(ann_id),
        camera_dir / "camK.npy",
        options["num_points"],
        rng,
    )
    _validate_segmentation_object_ids(segmentation, scene_object_ids)

    if scene_id not in _WORKER_COLLISION_LABELS:
        _WORKER_COLLISION_LABELS[scene_id] = _WORKER_DATASET.loadCollisionLabels(
            sceneIds=[scene_id]
        )
    grasp_group = _WORKER_DATASET.loadGrasp(
        sceneId=scene_id,
        annId=ann_id,
        format="6d",
        camera=camera,
        grasp_labels=_WORKER_GRASP_LABELS,
        collision_labels=_WORKER_COLLISION_LABELS[scene_id],
        fric_coef_thresh=0.4,
    )
    (
        contact_points,
        contact_dirs,
        contact_approaches,
        finger_diffs,
        grasp_scores,
        width_filtered_count,
    ) = _contacts_from_grasp_group(
        grasp_group,
        options["max_grasps"],
        options["max_width"],
        rng,
    )

    # Preserve the exact raw PNG labels alongside the scene's zero-based IDs;
    # their verified mapping is raw nonzero label = object ID + 1.
    camera_pose = np.eye(4, dtype=np.float32)
    if camera_pose.shape != (4, 4):
        raise AssertionError("Camera pose must be 4x4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".{}-".format(output_path.stem),
        suffix=".npz",
        dir=str(output_path.parent),
    )
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_path,
            point_cloud=point_cloud,
            segmentation=segmentation,
            contact_points=contact_points,
            contact_dirs=contact_dirs,
            contact_approaches=contact_approaches,
            finger_diffs=finger_diffs,
            grasp_scores=grasp_scores,
            camera_pose=camera_pose,
            scene_object_ids=scene_object_ids,
            scene_id=np.int32(scene_id),
            ann_id=np.int32(ann_id),
            coordinate_frame=np.asarray("opencv_camera"),
            segmentation_encoding=np.asarray("object_id_plus_one; 0=background"),
        )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return scene_id, ann_id, len(contact_points) // 2, width_filtered_count


def _load_num_points(config_path: Path) -> Tuple[int, float]:
    if not config_path.is_file():
        raise FileNotFoundError("Missing Contact-GraspNet config: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        num_points = int(config["DATA"]["num_point"])
        max_width = float(config["DATA"]["gripper_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Config must define DATA.num_point and DATA.gripper_width"
        ) from exc
    if num_points <= 0 or max_width <= 0:
        raise ValueError("DATA.num_point and DATA.gripper_width must be positive")
    return num_points, max_width


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gn_root",
        default=os.environ.get("GN_ROOT"),
        help="GraspNet-1Billion root (defaults to the GN_ROOT environment variable)",
    )
    parser.add_argument("--camera", choices=("kinect", "realsense"), default="kinect")
    parser.add_argument("--split", choices=tuple(_SPLITS), default="train")
    parser.add_argument("--ann_stride", type=int, default=4)
    parser.add_argument("--max_grasps", type=int, default=4000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "contact_graspnet" / "config.yaml",
        help="Contact-GraspNet config used to read DATA.num_point and DATA.gripper_width",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.gn_root:
        raise SystemExit("Set GN_ROOT or pass --gn_root.")
    gn_root = Path(args.gn_root).expanduser().resolve()
    if not gn_root.is_dir():
        raise FileNotFoundError("GraspNet root does not exist: {}".format(gn_root))
    if args.ann_stride <= 0:
        raise ValueError("--ann_stride must be positive")
    if args.max_grasps <= 0:
        raise ValueError("--max_grasps must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")

    num_points, max_width = _load_num_points(args.config)
    scene_ids = list(_SPLITS[args.split])
    output_root = gn_root / "cgn_preprocessed" / args.camera
    tasks: List[Tuple[int, int, str]] = []
    skipped_existing = 0
    for scene_id in scene_ids:
        scene_output_dir = output_root / "scene_{:04d}".format(scene_id)
        for ann_id in range(0, 256, args.ann_stride):
            output_path = scene_output_dir / "{:04d}.npz".format(ann_id)
            if output_path.is_file():
                skipped_existing += 1
                continue
            tasks.append((scene_id, ann_id, str(output_path)))

    print(
        "Split={} scenes={} camera={} points={} max_grasps={} workers={}; "
        "{} existing files skipped, {} frames queued."
        .format(
            args.split,
            len(scene_ids),
            args.camera,
            num_points,
            args.max_grasps,
            args.num_workers,
            skipped_existing,
            len(tasks),
        ),
        flush=True,
    )
    initargs = (
        str(gn_root),
        args.camera,
        num_points,
        args.max_grasps,
        max_width,
        args.seed,
    )

    if not tasks:
        return

    total_grasps = 0
    total_width_filtered = 0
    empty_frames = 0
    if args.num_workers == 1:
        _initialize_worker(*initargs)
        for index, task in enumerate(tasks, start=1):
            scene_id, ann_id, count, filtered = _process_frame(task)
            total_grasps += count
            total_width_filtered += filtered
            empty_frames += count == 0
            if index % 10 == 0 or index == len(tasks):
                print(
                    "Processed {}/{} frames; latest scene={:04d} ann={:04d}, "
                    "grasps={}, width-filtered={}"
                    .format(index, len(tasks), scene_id, ann_id, count, filtered),
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_initialize_worker,
            initargs=initargs,
        ) as executor:
            futures = [executor.submit(_process_frame, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                scene_id, ann_id, count, filtered = future.result()
                total_grasps += count
                total_width_filtered += filtered
                empty_frames += count == 0
                if index % 10 == 0 or index == len(tasks):
                    print(
                        "Processed {}/{} frames; latest scene={:04d} ann={:04d}, "
                        "grasps={}, width-filtered={}"
                        .format(index, len(tasks), scene_id, ann_id, count, filtered),
                        flush=True,
                    )

    print(
        "Completed: {} new frames, {} grasp pairs, {} grasps filtered for width, "
        "{} frames with no usable grasps. Existing files skipped: {}."
        .format(
            len(tasks),
            total_grasps,
            total_width_filtered,
            empty_frames,
            skipped_existing,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
