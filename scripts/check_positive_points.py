#!/usr/bin/env python3
"""Count CGN positive point labels in preprocessed GraspNet samples using NumPy."""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _read_label_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError("Missing config file: {}".format(path))

    values: Dict[str, object] = {}
    in_data = False
    in_labels = False
    wanted = {"k", "max_radius", "filter_z", "z_val"}

    with path.open("r", encoding="utf-8") as config_file:
        for line in config_file:
            if re.match(r"^DATA:\s*(?:#.*)?$", line):
                in_data = True
                in_labels = False
                continue
            if in_data and re.match(r"^[A-Z][A-Z0-9_]*:\s*(?:#.*)?$", line):
                in_data = False
                in_labels = False
            if in_data and re.match(r"^  labels:\s*(?:#.*)?$", line):
                in_labels = True
                continue
            if in_labels and re.match(r"^  \S", line):
                in_labels = False
            if not in_labels:
                continue

            match = re.match(r"^    ([a-z_]+):\s*([^#\s]+)", line)
            if not match or match.group(1) not in wanted:
                continue
            key, value = match.groups()
            if key == "filter_z":
                if value.lower() not in ("true", "false"):
                    raise ValueError("Invalid DATA.labels.filter_z in {}".format(path))
                values[key] = value.lower() == "true"
            elif key == "k":
                values[key] = int(value)
            else:
                values[key] = float(value)

    missing = wanted - values.keys()
    if missing:
        raise ValueError(
            "Missing DATA.labels values in {}: {}".format(path, sorted(missing))
        )
    if values["k"] < 1:
        raise ValueError("DATA.labels.k must be at least 1")
    if values["max_radius"] < 0:
        raise ValueError("DATA.labels.max_radius cannot be negative")
    return values


def _positive_count_k1(
    points: np.ndarray, contacts: np.ndarray, radius: float
) -> int:
    """Exact k=1 threshold count using radius-sized spatial buckets."""
    if radius == 0:
        return 0

    cell_size = radius
    contact_cells = np.floor(
        contacts.astype(np.float64) / cell_size
    ).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for index, cell in enumerate(contact_cells):
        buckets.setdefault(tuple(int(axis) for axis in cell), []).append(index)

    point_cells = np.floor(points.astype(np.float64) / cell_size).astype(np.int64)
    radius_squared = np.float32(radius) * np.float32(radius)
    positive_count = 0
    for point, cell in zip(points, point_cells):
        base = tuple(int(axis) for axis in cell)
        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(
                        buckets.get(
                            (base[0] + dx, base[1] + dy, base[2] + dz), ()
                        )
                    )
        if not candidates:
            continue
        differences = contacts[candidates] - point
        squared_distances = np.einsum(
            "ij,ij->i", differences, differences, optimize=True
        )
        if float(np.min(squared_distances)) < radius_squared:
            positive_count += 1
    return positive_count


def _positive_count(
    points: np.ndarray, contacts: np.ndarray, k: int, radius: float
) -> int:
    if contacts.ndim != 2 or contacts.shape[1] != 3 or len(contacts) < k:
        raise ValueError(
            "Expected at least k={} contact points, got shape {}".format(k, contacts.shape)
        )
    if k == 1:
        return _positive_count_k1(points, contacts, radius)

    radius_squared = np.float32(radius) * np.float32(radius)
    positive_count = 0
    chunk_size = 32
    for start in range(0, len(points), chunk_size):
        point_chunk = points[start : start + chunk_size]
        differences = point_chunk[:, None, :] - contacts[None, :, :]
        squared_distances = np.einsum(
            "ijk,ijk->ij", differences, differences, optimize=True
        )
        nearest_k = np.partition(squared_distances, k - 1, axis=1)[:, :k]
        positive_count += int(
            np.count_nonzero(np.mean(nearest_k, axis=1) < radius_squared)
        )
    return positive_count


def _sample_files(camera_root: Path) -> Iterable[Path]:
    for scene_dir in sorted(camera_root.glob("scene_*")):
        if scene_dir.is_dir():
            yield from sorted(scene_dir.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        default=os.path.join(
            os.environ.get("GN_ROOT", os.path.expanduser("~/datasets/graspnet_1b")),
            "cgn_preprocessed",
        ),
        help="Preprocessed root containing <camera>/scene_XXXX/*.npz",
    )
    parser.add_argument("--camera", choices=("kinect", "realsense"), default="kinect")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "contact_graspnet"
            / "config_graspnet1b.yaml"
        ),
        help="Config whose DATA.labels settings are used",
    )
    parser.add_argument(
        "--low_positive_threshold",
        type=int,
        default=10,
        help="Report frames with fewer than this many positive points",
    )
    args = parser.parse_args()

    if args.low_positive_threshold < 0:
        parser.error("--low_positive_threshold cannot be negative")

    config = _read_label_config(Path(args.config))
    camera_root = Path(args.data_path).expanduser() / args.camera
    files = list(_sample_files(camera_root))
    if not files:
        raise FileNotFoundError("No preprocessed NPZs found under {}".format(camera_root))

    if config["filter_z"]:
        print(
            "NOTE: config DATA.labels.filter_z=true, but compute_labels computes "
            "contact_point_batch_cam before applying its Z mask to "
            "contact_point_batch_mesh. The mask therefore does not affect the "
            "nearest-neighbor positive labels; this scan mirrors the current TF code."
        )
    print(
        "Label settings: k={}, max_radius={} m, configured filter_z={}, z_val={}"
        .format(
            config["k"],
            config["max_radius"],
            config["filter_z"],
            config["z_val"],
        )
    )

    counts: List[int] = []
    low_positive_frames: List[Tuple[int, int, int]] = []
    for path in files:
        with np.load(str(path), allow_pickle=False) as sample:
            points = np.asarray(sample["point_cloud"], dtype=np.float32)
            contacts = np.asarray(sample["contact_points"], dtype=np.float32)
            scene_id = int(sample["scene_id"])
            ann_id = int(sample["ann_id"])
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Invalid point_cloud shape {} in {}".format(points.shape, path))
        if not np.isfinite(points).all() or not np.isfinite(contacts).all():
            raise ValueError("Non-finite point/contact coordinates in {}".format(path))

        positive_count = _positive_count(
            points,
            contacts,
            int(config["k"]),
            float(config["max_radius"]),
        )
        counts.append(positive_count)
        if positive_count < args.low_positive_threshold:
            low_positive_frames.append((scene_id, ann_id, positive_count))

    counts_array = np.asarray(counts, dtype=np.int64)
    print("Frames scanned: {}".format(len(counts_array)))
    print(
        "Positive points per frame: min={}, median={}, max={}".format(
            int(counts_array.min()),
            float(np.median(counts_array)),
            int(counts_array.max()),
        )
    )
    print(
        "Frames with fewer than {} positives: {}; zero-positive frames: {}"
        .format(
            args.low_positive_threshold,
            len(low_positive_frames),
            int(np.count_nonzero(counts_array == 0)),
        )
    )
    print(
        "Frames with fewer than {} positive points (scene_id, ann_id, count):"
        .format(args.low_positive_threshold)
    )
    for scene_id, ann_id, count in low_positive_frames:
        print("  {}, {}, {}".format(scene_id, ann_id, count))
    if not low_positive_frames:
        print("  none")


if __name__ == "__main__":
    main()
