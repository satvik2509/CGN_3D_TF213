"""Reader for Contact-GraspNet samples preprocessed from GraspNet-1Billion."""

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data import augment_point_cloud


class GraspNet1BReader:
    """Load preprocessed camera-frame samples without rendering or graspnetAPI."""

    outputs_centered_opencv = True

    def __init__(
        self,
        preprocessed_root: str,
        camera: str,
        batch_size: int,
        num_point: int,
        num_validation_scenes: int,
        scene_limit: Optional[int] = None,
        pc_augm_config: Optional[Dict[str, object]] = None,
        depth_augm_config: Optional[Dict[str, object]] = None,
    ) -> None:
        if camera not in ("kinect", "realsense"):
            raise ValueError("camera must be 'kinect' or 'realsense'")
        if batch_size <= 0 or num_point <= 0:
            raise ValueError("batch_size and num_point must be positive")
        if num_validation_scenes < 1 or num_validation_scenes > 10:
            raise ValueError(
                "num_validation_scenes must be between 1 and 10 for scenes 90-99"
            )
        if scene_limit is not None and scene_limit <= 0:
            raise ValueError("scene_limit must be positive when provided")

        self._preprocessed_root = Path(preprocessed_root).expanduser().resolve()
        self._camera_root = self._preprocessed_root / camera
        self._batch_size = batch_size
        self._num_point = num_point
        self._pc_augm_config = pc_augm_config or {}
        self._depth_augm_config = depth_augm_config or {}

        train_scene_ids = list(range(0, 90))
        if scene_limit is not None:
            train_scene_ids = train_scene_ids[:scene_limit]
        validation_scene_ids = list(range(90, 90 + num_validation_scenes))

        self._train_files = self._collect_files(train_scene_ids)
        self._validation_files = self._collect_files(validation_scene_ids)
        if not self._train_files:
            raise FileNotFoundError(
                "No preprocessed training NPZs found under {}".format(self._camera_root)
            )
        if not self._validation_files:
            raise FileNotFoundError(
                "No preprocessed validation NPZs found for scenes {}".format(
                    validation_scene_ids
                )
            )

        self._files = self._train_files + self._validation_files
        self._num_train_samples = len(self._train_files)
        self._num_test_samples = len(self._validation_files)
        self.contact_infos = self._load_contact_infos(self._files)

    def _collect_files(self, scene_ids: List[int]) -> List[Path]:
        files: List[Path] = []
        for scene_id in scene_ids:
            scene_dir = self._camera_root / "scene_{:04d}".format(scene_id)
            files.extend(
                Path(path)
                for path in sorted(glob.glob(str(scene_dir / "*.npz")))
            )
        return files

    def _load_contact_infos(self, files: List[Path]) -> List[Dict[str, np.ndarray]]:
        contact_infos: List[Dict[str, np.ndarray]] = []
        for path in files:
            with np.load(str(path), allow_pickle=False) as sample:
                scene_id = int(sample["scene_id"])
                ann_id = int(sample["ann_id"])
                contact_points = np.asarray(sample["contact_points"], dtype=np.float32)
                contact_dirs = np.asarray(sample["contact_dirs"], dtype=np.float32)
                contact_approaches = np.asarray(
                    sample["contact_approaches"], dtype=np.float32
                )
                finger_diffs = np.asarray(sample["finger_diffs"], dtype=np.float32)

            if contact_points.ndim != 2 or contact_points.shape[1] != 3:
                raise ValueError("Invalid contact_points shape in {}".format(path))
            if contact_dirs.shape != contact_points.shape:
                raise ValueError("Invalid contact_dirs shape in {}".format(path))
            if contact_approaches.shape != contact_points.shape:
                raise ValueError("Invalid contact_approaches shape in {}".format(path))
            if finger_diffs.shape != (len(contact_points),):
                raise ValueError("Invalid finger_diffs shape in {}".format(path))
            if len(contact_points) == 0:
                raise ValueError(
                    "No usable contacts in {}; filter empty frames before training"
                    .format(path)
                )
            if not (
                np.isfinite(contact_points).all()
                and np.isfinite(contact_dirs).all()
                and np.isfinite(contact_approaches).all()
                and np.isfinite(finger_diffs).all()
            ):
                raise ValueError("Non-finite contact labels in {}".format(path))
            if (
                np.any(np.linalg.norm(contact_dirs, axis=1) <= 1e-8)
                or np.any(np.linalg.norm(contact_approaches, axis=1) <= 1e-8)
            ):
                raise ValueError("Zero-length direction label in {}".format(path))

            expected_name = "scene_{:04d}".format(scene_id)
            if path.parent.name != expected_name:
                raise ValueError(
                    "Stored scene_id {} does not match file path {}"
                    .format(scene_id, path)
                )
            contact_infos.append(
                {
                    "scene_id": np.asarray(scene_id, dtype=np.int32),
                    "ann_id": np.asarray(ann_id, dtype=np.int32),
                    "contact_points": contact_points,
                    "contact_dirs": contact_dirs,
                    "contact_approaches": contact_approaches,
                    "finger_diffs": finger_diffs,
                }
            )
        return contact_infos

    def _augment_depth_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        augmented = point_cloud.copy()
        gaussian_kernel = int(self._depth_augm_config.get("gaussian_kernel", 0))
        if gaussian_kernel > 0:
            raise ValueError(
                "Depth Gaussian blur requires the organized depth image; "
                "GraspNet NPZ samples contain sampled camera-frame points only"
            )

        sigma = float(self._depth_augm_config.get("sigma", 0.0))
        if sigma > 0:
            clip = float(self._depth_augm_config["clip"])
            depth = augmented[:, 2].copy()
            noise = np.clip(
                sigma * np.random.randn(len(depth)),
                -clip,
                clip,
            )
            noisy_depth = depth + noise
            if np.any(noisy_depth <= 0):
                raise ValueError("Depth augmentation produced non-positive depth")
            augmented *= (noisy_depth / depth)[:, None]
        return augmented

    def get_scene_batch(
        self, scene_idx: Optional[int] = None, return_segmap: bool = False, save: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if return_segmap or save:
            raise NotImplementedError(
                "GraspNet1BReader returns preprocessed samples and does not render/save scenes"
            )
        if scene_idx is None:
            scene_idx = int(np.random.randint(0, self._num_train_samples))
        if scene_idx < 0 or scene_idx >= len(self._files):
            raise IndexError("Sample index {} is out of range".format(scene_idx))

        path = self._files[scene_idx]
        with np.load(str(path), allow_pickle=False) as sample:
            point_cloud = np.asarray(sample["point_cloud"], dtype=np.float32)
        if point_cloud.shape != (self._num_point, 3):
            raise ValueError(
                "Expected point_cloud shape ({},3) in {}, got {}"
                .format(self._num_point, path, point_cloud.shape)
            )
        if not np.isfinite(point_cloud).all():
            raise ValueError("Non-finite point cloud in {}".format(path))

        batch_data = np.empty(
            (self._batch_size, self._num_point, 3), dtype=np.float32
        )
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)
        for batch_index in range(self._batch_size):
            view = self._augment_depth_point_cloud(point_cloud)
            view = augment_point_cloud(view, self._pc_augm_config)
            if view.shape != (self._num_point, 3):
                raise ValueError(
                    "Point-cloud augmentation changed the expected shape in {}: {}"
                    .format(path, view.shape)
                )

            # Labels remain in the uncentered OpenCV camera frame. The pose
            # maps that frame into the centered frame consumed by the network.
            pc_mean = np.mean(view, axis=0)
            batch_data[batch_index] = view - pc_mean
            cam_poses[batch_index] = np.eye(4, dtype=np.float32)
            cam_poses[batch_index, :3, 3] = -pc_mean

        if batch_data.shape != (self._batch_size, self._num_point, 3):
            raise AssertionError("GraspNet batch shape does not match the model")
        if cam_poses.shape != (self._batch_size, 4, 4):
            raise AssertionError("GraspNet camera-pose batch shape is invalid")
        return batch_data, cam_poses, scene_idx
