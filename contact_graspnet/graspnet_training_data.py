import glob
import hashlib
import json
import os
import tempfile

import numpy as np
import trimesh
from scipy.io import loadmat


_CACHE_VERSION = 2
_TABLE_TOP_Z = 0.3
_NUM_VIEWS = 300
_NUM_ANGLES = 12


def _generate_views(num_views):
    indices = np.arange(num_views, dtype=np.float32)
    z = (2 * indices + 1) / num_views - 1
    x = np.sqrt(1 - z**2) * np.cos(2 * indices * np.pi * ((np.sqrt(5) - 1) / 2))
    y = np.sqrt(1 - z**2) * np.sin(2 * indices * np.pi * ((np.sqrt(5) - 1) / 2))
    return np.stack([x, y, z], axis=1)


def _load_grasp_candidates(label_path, friction_threshold, max_gripper_width):
    with np.load(label_path, allow_pickle=False) as label:
        points = np.asarray(label["points"], dtype=np.float32)
        offsets = label["offsets"]
        scores = label["scores"]

        if offsets.ndim != 5 or offsets.shape[-1] != 3 or scores.shape != offsets.shape[:-1]:
            raise ValueError(f"Unexpected GraspNet grasp-label shapes in {label_path}")
        if offsets.shape[1:4] != (_NUM_VIEWS, _NUM_ANGLES, 4):
            raise ValueError(f"Unexpected GraspNet grasp-label dimensions in {label_path}")
        if points.shape != (offsets.shape[0], 3):
            raise ValueError(f"Grasp point count does not match offsets in {label_path}")

        flattened_offsets = offsets.reshape(-1, 3)
        widths = flattened_offsets[:, 2]
        valid = (
            np.isfinite(scores.reshape(-1))
            & (scores.reshape(-1) > 0)
            & (scores.reshape(-1) <= friction_threshold)
            & np.isfinite(widths)
            & (widths > 0)
            & (widths <= max_gripper_width)
        )
        finite_points = np.all(np.isfinite(points), axis=1)
        valid &= finite_points.repeat(_NUM_VIEWS * _NUM_ANGLES * 4)
        flat_indices = np.flatnonzero(valid)
        if flat_indices.size == 0:
            return {
                "flat_indices": np.empty(0, dtype=np.int64),
                "num_label_values": scores.size,
                "points": np.empty((0, 3), dtype=np.float32),
                "directions": np.empty((0, 3), dtype=np.float32),
                "approaches": np.empty((0, 3), dtype=np.float32),
                "widths": np.empty(0, dtype=np.float32),
            }

        point_indices = flat_indices // (_NUM_VIEWS * _NUM_ANGLES * 4)
        view_indices = (flat_indices // (_NUM_ANGLES * 4)) % _NUM_VIEWS
        angles = flattened_offsets[flat_indices, 0]
        widths = flattened_offsets[flat_indices, 2]

    approaches = -_generate_views(_NUM_VIEWS)[view_indices]
    axis_y = np.stack([-approaches[:, 1], approaches[:, 0], np.zeros(len(approaches))], axis=1)
    vertical = np.linalg.norm(axis_y, axis=1) == 0
    axis_y[vertical] = [0, 1, 0]
    axis_y /= np.linalg.norm(axis_y, axis=1, keepdims=True)
    axis_z = np.cross(approaches, axis_y)
    directions = (
        axis_y * np.cos(angles)[:, np.newaxis]
        + axis_z * np.sin(angles)[:, np.newaxis]
    )

    return {
        "flat_indices": flat_indices.astype(np.int64, copy=False),
        "num_label_values": scores.size,
        "points": points[point_indices],
        "directions": directions.astype(np.float32, copy=False),
        "approaches": approaches.astype(np.float32, copy=False),
        "widths": widths.astype(np.float32, copy=False),
    }


def _load_mesh_mean(mesh_path):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a non-empty mesh from {mesh_path}")
    return np.mean(np.asarray(mesh.vertices), axis=0).astype(np.float32)


def _scene_specs(dataset_root, scene_limit):
    scene_dirs = sorted(glob.glob(os.path.join(dataset_root, "train", "scene_[0-9][0-9][0-9][0-9]")))
    if scene_limit is not None:
        scene_dirs = scene_dirs[:scene_limit]
    if not scene_dirs:
        raise FileNotFoundError(f"No GraspNet training scenes found under {dataset_root}/train")

    specs = []
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        camera_dir = os.path.join(scene_dir, "realsense")
        meta_path = os.path.join(camera_dir, "meta", "0000.mat")
        camera_poses_path = os.path.join(camera_dir, "camera_poses.npy")
        collision_path = os.path.join(dataset_root, "collision_label", scene_name, "collision_labels.npz")

        for required_path in (meta_path, camera_poses_path, collision_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(f"Missing required GraspNet training file: {required_path}")

        metadata = loadmat(meta_path)
        object_ids = np.asarray(metadata["cls_indexes"]).reshape(-1).astype(np.int32) - 1
        object_poses = np.asarray(metadata["poses"], dtype=np.float32)
        if object_poses.shape != (3, 4, len(object_ids)):
            raise ValueError(f"Unexpected object-pose shape in {meta_path}: {object_poses.shape}")

        camera_poses = np.load(camera_poses_path, mmap_mode="r")
        if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4) or len(camera_poses) == 0:
            raise ValueError(f"Unexpected camera-pose shape in {camera_poses_path}: {camera_poses.shape}")

        camera_pose = np.asarray(camera_poses[0], dtype=np.float32)
        scene_objects = []
        for object_index, object_id in enumerate(object_ids):
            model_dir = os.path.join(dataset_root, "models", f"{object_id:03d}")
            simplified_mesh = os.path.join(model_dir, "nontextured_simplified.ply")
            mesh_path = simplified_mesh if os.path.isfile(simplified_mesh) else os.path.join(model_dir, "nontextured.ply")
            label_path = os.path.join(dataset_root, "grasp_label", f"{object_id:03d}_labels.npz")
            if not os.path.isfile(mesh_path):
                raise FileNotFoundError(f"Missing GraspNet object mesh: {mesh_path}")
            if not os.path.isfile(label_path):
                raise FileNotFoundError(f"Missing GraspNet object grasp labels: {label_path}")

            camera_object_pose = np.eye(4, dtype=np.float32)
            camera_object_pose[:3, :4] = object_poses[:, :, object_index]
            world_object_pose = camera_pose @ camera_object_pose
            scene_objects.append({
                "object_id": int(object_id),
                "object_index": object_index,
                "label_path": label_path,
                "mesh_path": mesh_path,
                "world_object_pose": world_object_pose,
            })

        specs.append({
            "scene_id": int(scene_name[-4:]),
            "collision_path": collision_path,
            "scene_objects": scene_objects,
            "fingerprint_paths": [meta_path, camera_poses_path, collision_path]
            + [item["label_path"] for item in scene_objects]
            + [item["mesh_path"] for item in scene_objects],
        })

    return specs


def _fingerprint(
    specs, dataset_root, friction_threshold, max_gripper_width, max_contacts_per_scene, random_seed
):
    source_files = {os.path.realpath(path) for spec in specs for path in spec["fingerprint_paths"]}
    source_stats = []
    for path in sorted(source_files):
        stat = os.stat(path)
        source_stats.append((path, stat.st_size, stat.st_mtime_ns))
    payload = {
        "cache_version": _CACHE_VERSION,
        "dataset_root": os.path.realpath(dataset_root),
        "source_stats": source_stats,
        "friction_threshold": friction_threshold,
        "max_gripper_width": max_gripper_width,
        "max_contacts_per_scene": max_contacts_per_scene,
        "random_seed": random_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _scene_cache_path(cache_dir, scene_id):
    return os.path.join(cache_dir, f"scene_{scene_id:04d}.npz")


def _load_scene_cache(path):
    with np.load(path, allow_pickle=False) as cached:
        return {
            "scene_id": int(cached["scene_id"]),
            "contact_points": np.asarray(cached["contact_points"], dtype=np.float32),
            "contact_dirs": np.asarray(cached["contact_dirs"], dtype=np.float32),
            "contact_approaches": np.asarray(cached["contact_approaches"], dtype=np.float32),
            "finger_diffs": np.asarray(cached["finger_diffs"], dtype=np.float32),
            "obj_paths": np.asarray(cached["obj_paths"]).astype(str).tolist(),
            "obj_transforms": np.asarray(cached["obj_transforms"], dtype=np.float32),
            "obj_scales": np.asarray(cached["obj_scales"], dtype=np.float32),
        }


def _write_scene_cache(path, contact_info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".scene-cache-", suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(
            temp_path,
            scene_id=np.int32(contact_info["scene_id"]),
            contact_points=contact_info["contact_points"],
            contact_dirs=contact_info["contact_dirs"],
            contact_approaches=contact_info["contact_approaches"],
            finger_diffs=contact_info["finger_diffs"],
            obj_paths=np.asarray(contact_info["obj_paths"]),
            obj_transforms=contact_info["obj_transforms"],
            obj_scales=contact_info["obj_scales"],
        )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _build_scene_contact_info(
    spec,
    dataset_root,
    grasp_candidates,
    mesh_means,
    max_contacts_per_scene,
    friction_threshold,
    max_gripper_width,
    random_seed,
):
    scene_objects = spec["scene_objects"]
    per_object_budget = max(1, max_contacts_per_scene // len(scene_objects))
    rng = np.random.default_rng(random_seed + spec["scene_id"])
    contact_points = []
    contact_dirs = []
    contact_approaches = []
    finger_diffs = []
    obj_paths = []
    obj_transforms = []
    obj_scales = []

    with np.load(spec["collision_path"], allow_pickle=False) as collisions:
        for scene_object in scene_objects:
            object_id = scene_object["object_id"]
            object_index = scene_object["object_index"]
            label_path = scene_object["label_path"]
            if label_path not in grasp_candidates:
                grasp_candidates[label_path] = _load_grasp_candidates(
                    label_path, friction_threshold, max_gripper_width
                )
            candidate = grasp_candidates[label_path]
            collision_key = f"arr_{object_index}"
            if collision_key not in collisions:
                raise ValueError(f"Missing {collision_key} in {spec['collision_path']}")
            collision = collisions[collision_key]
            if collision.size != candidate["num_label_values"]:
                raise ValueError(f"Collision labels are incompatible with grasp labels in {spec['collision_path']}")

            if candidate["flat_indices"].size:
                flattened_collision = collision.reshape(-1)
                if candidate["flat_indices"][-1] >= flattened_collision.size:
                    raise ValueError(f"Collision labels are too small for {label_path}")
                selectable = np.flatnonzero(~flattened_collision[candidate["flat_indices"]])
            else:
                selectable = np.empty(0, dtype=np.int64)

            if selectable.size:
                selected = selectable
                if selected.size > per_object_budget:
                    selected = rng.choice(selected, size=per_object_budget, replace=False)

                world_pose = scene_object["world_object_pose"]
                rotation = world_pose[:3, :3]
                translation = world_pose[:3, 3] + np.array([0, 0, _TABLE_TOP_Z], dtype=np.float32)
                contact_points.append(candidate["points"][selected] @ rotation.T + translation)
                contact_dirs.append(candidate["directions"][selected] @ rotation.T)
                contact_approaches.append(candidate["approaches"][selected] @ rotation.T)
                finger_diffs.append(candidate["widths"][selected])

            mesh_path = scene_object["mesh_path"]
            if mesh_path not in mesh_means:
                mesh_means[mesh_path] = _load_mesh_mean(mesh_path)
            renderer_pose = scene_object["world_object_pose"].copy()
            renderer_pose[:3, 3] += renderer_pose[:3, :3] @ mesh_means[mesh_path]
            renderer_pose[:3, 3] += np.array([0, 0, _TABLE_TOP_Z], dtype=np.float32)
            obj_paths.append(os.path.relpath(mesh_path, dataset_root))
            obj_transforms.append(renderer_pose)
            obj_scales.append(1.0)

    if not contact_points:
        raise ValueError(f"No non-colliding GraspNet grasps survived filtering for scene {spec['scene_id']}")

    all_contact_points    = np.concatenate(contact_points).astype(np.float32, copy=False)
    all_contact_dirs      = np.concatenate(contact_dirs).astype(np.float32, copy=False)
    all_contact_approaches= np.concatenate(contact_approaches).astype(np.float32, copy=False)
    all_finger_diffs      = np.concatenate(finger_diffs).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Remove corrupt or degenerate labels before writing the scene cache.
    # ------------------------------------------------------------------
    finite_mask = (
        np.all(np.isfinite(all_contact_points), axis=1)
        & np.all(np.isfinite(all_contact_dirs), axis=1)
        & np.all(np.isfinite(all_contact_approaches), axis=1)
        & np.isfinite(all_finger_diffs)
        # Zero directions do not provide a usable grasp orientation.
        & (np.linalg.norm(all_contact_dirs, axis=1) > 1e-8)
        & (np.linalg.norm(all_contact_approaches, axis=1) > 1e-8)
    )
    n_before = len(all_contact_points)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"  Scene {spec['scene_id']:04d}: dropped {n_bad}/{n_before} "
              f"non-finite or zero-norm contact grasps before caching.")
    if finite_mask.sum() == 0:
        raise ValueError(
            f"All {n_before} contact grasps for scene {spec['scene_id']} were "
            "non-finite after world-frame transform.  Check the grasp labels for corruption."
        )

    return {
        "scene_id": spec["scene_id"],
        "contact_points":    all_contact_points[finite_mask],
        "contact_dirs":      all_contact_dirs[finite_mask],
        "contact_approaches":all_contact_approaches[finite_mask],
        "finger_diffs":      all_finger_diffs[finite_mask],
        "obj_paths": obj_paths,
        "obj_transforms": np.asarray(obj_transforms, dtype=np.float32),
        "obj_scales": np.asarray(obj_scales, dtype=np.float32),
    }


def load_graspnet_contact_infos(
    dataset_root,
    cache_dir,
    num_validation_scenes=5,
    max_contacts_per_scene=8000,
    friction_threshold=0.4,
    max_gripper_width=0.08,
    random_seed=42,
    scene_limit=None,
):
    specs = _scene_specs(dataset_root, scene_limit)
    if num_validation_scenes < 1 or num_validation_scenes >= len(specs):
        raise ValueError(
            f"Validation scene count must be between 1 and {len(specs) - 1}; got {num_validation_scenes}"
        )
    if max_contacts_per_scene < 1:
        raise ValueError("max_contacts_per_scene must be positive")

    fingerprint = _fingerprint(
        specs, dataset_root, friction_threshold, max_gripper_width, max_contacts_per_scene, random_seed
    )
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest_matches = False
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest_matches = json.load(manifest_file).get("fingerprint") == fingerprint

    contact_infos = []
    if manifest_matches:
        for spec in specs:
            cache_path = _scene_cache_path(cache_dir, spec["scene_id"])
            if not os.path.isfile(cache_path):
                contact_infos = []
                break
            contact_infos.append(_load_scene_cache(cache_path))

    if not contact_infos:
        os.makedirs(cache_dir, exist_ok=True)
        grasp_candidates = {}
        mesh_means = {}
        for index, spec in enumerate(specs):
            print(f"Preparing GraspNet scene {spec['scene_id']:04d} ({index + 1}/{len(specs)})")
            contact_info = _build_scene_contact_info(
                spec,
                dataset_root,
                grasp_candidates,
                mesh_means,
                max_contacts_per_scene,
                friction_threshold,
                max_gripper_width,
                random_seed,
            )
            _write_scene_cache(_scene_cache_path(cache_dir, spec["scene_id"]), contact_info)
            contact_infos.append(contact_info)

        temp_manifest = manifest_path + ".tmp"
        with open(temp_manifest, "w", encoding="utf-8") as manifest_file:
            json.dump({"fingerprint": fingerprint, "scene_count": len(specs)}, manifest_file)
        os.replace(temp_manifest, manifest_path)

    scene_order = np.random.default_rng(random_seed).permutation(len(contact_infos))
    validation_indices = set(scene_order[:num_validation_scenes].tolist())
    train_infos = [info for index, info in enumerate(contact_infos) if index not in validation_indices]
    validation_infos = [info for index, info in enumerate(contact_infos) if index in validation_indices]
    return train_infos + validation_infos
