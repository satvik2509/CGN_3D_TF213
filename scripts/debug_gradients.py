#!/usr/bin/env python3
"""Diagnose Contact-GraspNet loss gradients on the first training batch.

Run in the existing ``cgn_3d`` environment. This script performs no optimizer
step and writes no checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
import yaml

tf.disable_eager_execution()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_DIR = REPO_ROOT / "contact_graspnet"
sys.path.insert(0, str(CONTACT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pointnet2" / "models"))
sys.path.insert(0, str(REPO_ROOT / "pointnet2" / "utils"))

from contact_grasp_estimator import GraspEstimator
from data import PointCloudReader, center_pc_convert_cam, load_scene_contacts
from graspnet1b_data import GraspNet1BReader
from tf_train_ops import load_labels_and_losses


def _load_config(config_path: Path, data_path: str) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["DATA"]["data_path"] = data_path
    config["OPTIMIZER"]["batch_size"] = 3
    config["DATA"]["num_test_scenes"] = 1
    if config["DATA"].get("dataset_type") == "graspnet":
        config["MODEL"]["pred_contact_base"] = True
        config["MODEL"]["pred_contact_approach"] = True
        config["MODEL"]["pred_grasps_adds"] = False
        config["MODEL"]["pred_grasps_adds_gt2pred"] = False
    return config


def _make_first_batch(
    config: Dict[str, object], dataset_type: str, data_path: str
) -> Tuple[object, List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, int]:
    data_config = config["DATA"]
    batch_size = int(config["OPTIMIZER"]["batch_size"])
    if dataset_type == "graspnet":
        reader = GraspNet1BReader(
            data_path,
            camera=data_config.get("camera", "kinect"),
            batch_size=batch_size,
            num_point=data_config["num_point"],
            num_validation_scenes=1,
            scene_limit=2,
            pc_augm_config=data_config["pc_augm"],
            depth_augm_config=data_config["depth_augm"],
        )
        contacts = reader.contact_infos
        cloud, poses, scene_index = reader.get_scene_batch(scene_idx=0)
    else:
        contacts = load_scene_contacts(
            data_path,
            scene_contacts_path=data_config["scene_contacts_path"],
        )
        if len(contacts) < 2:
            raise ValueError("ACRONYM control needs at least two contact scenes")
        num_test = min(int(data_config.get("num_test_scenes", 1)), len(contacts) - 1)
        reader = PointCloudReader(
            root_folder=data_path,
            batch_size=batch_size,
            estimate_normals=data_config["input_normals"],
            raw_num_points=data_config["num_point"],
            use_uniform_quaternions=data_config["use_uniform_quaternions"],
            scene_obj_scales=[item["obj_scales"] for item in contacts],
            scene_obj_paths=[item["obj_paths"] for item in contacts],
            scene_obj_transforms=[item["obj_transforms"] for item in contacts],
            num_train_samples=len(contacts) - num_test,
            num_test_samples=num_test,
            use_farthest_point=data_config["use_farthest_point"],
            intrinsics=data_config["intrinsics"],
            elevation=data_config["view_sphere"]["elevation"],
            distance_range=data_config["view_sphere"]["distance_range"],
            pc_augm_config=data_config["pc_augm"],
            depth_augm_config=data_config["depth_augm"],
        )
        cloud, poses, scene_index = reader.get_scene_batch(scene_idx=0)
        poses, cloud = center_pc_convert_cam(poses, cloud)

    if cloud.shape != (
        batch_size,
        int(data_config["num_point"]),
        6 if data_config["input_normals"] else 3,
    ):
        raise ValueError("Unexpected point-cloud batch shape {}".format(cloud.shape))
    return reader, contacts, cloud.astype(np.float32), poses.astype(np.float32), scene_index


def _finite_stats(name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    finite = np.isfinite(array)
    if finite.any():
        finite_values = array[finite]
        print(
            "{}: shape={} finite={}/{} min={:.8g} max={:.8g} mean={:.8g}".format(
                name,
                array.shape,
                int(finite.sum()),
                array.size,
                float(finite_values.min()),
                float(finite_values.max()),
                float(finite_values.mean()),
            )
        )
    else:
        print(
            "{}: shape={} finite=0/{} min=NA max=NA mean=NA".format(
                name, array.shape, array.size
            )
        )


def _gradient_values(value: object) -> object:
    if isinstance(value, tf.IndexedSlices):
        return value.values
    return value


def _make_synthetic_cloud(
    batch_size: int, num_points: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = rng.uniform(-0.5, 0.5, (batch_size, num_points, 3)).astype(np.float32)
    cloud -= cloud.mean(axis=1, keepdims=True)
    return cloud


def _collect_layer_intermediates(graph: tf.Graph) -> List[tf.Tensor]:
    relevant_types = {
        "GroupPoint",
        "GatherPoint",
        "Conv2D",
        "BiasAdd",
        "FusedBatchNorm",
        "FusedBatchNormV3",
        "Relu",
        "Max",
        "Squeeze",
        "ConcatV2",
    }
    tensors = []
    for operation in graph.get_operations():
        if not operation.name.startswith(("layer1/", "layer2/")):
            continue
        is_batch_norm_output = (
            "/bn/" in operation.name
            and operation.type in {
                "FusedBatchNorm",
                "FusedBatchNormV3",
                "Identity",
                "If",
                "StatelessIf",
            }
            and (
                operation.type != "Identity"
                or operation.name.rsplit("/", 1)[-1] == "Identity"
            )
        )
        if operation.type not in relevant_types and not is_batch_norm_output:
            continue
        if operation.outputs:
            tensors.append(operation.outputs[0])
    return tensors


def _build_custom_op_diagnostics(point_cloud: np.ndarray) -> Dict[str, object]:
    from tf_grouping import group_point, query_ball_point
    from tf_interpolate import three_interpolate, three_nn
    from tf_sampling import gather_point

    batch_size, num_points, channels = point_cloud.shape
    if num_points < 128:
        raise ValueError("At least 128 points are required for custom-op probes")

    rng = np.random.default_rng(1949)
    points = tf.constant(point_cloud, dtype=tf.float32, name="custom_op_probe_points")

    gather_indices = tf.constant(
        np.tile(np.arange(min(1024, num_points), dtype=np.int32)[None, :], (batch_size, 1)),
        name="custom_op_gather_indices",
    )
    gathered = gather_point(points, gather_indices)
    gather_upstream = tf.constant(
        rng.standard_normal(gathered.shape.as_list()).astype(np.float32),
        name="custom_op_gather_upstream",
    )
    gather_gradient = tf.gradients(gathered, points, grad_ys=gather_upstream)[0]

    dataset_xyz = points[:, :num_points, :3]
    query_xyz = points[:, :128, :3]
    group_indices, _ = query_ball_point(0.12, 8, dataset_xyz, query_xyz)
    grouped = group_point(points, group_indices)
    group_upstream = tf.constant(
        rng.standard_normal(grouped.shape.as_list()).astype(np.float32),
        name="custom_op_group_upstream",
    )
    group_gradient = tf.gradients(grouped, points, grad_ys=group_upstream)[0]

    query_xyz = points[:, 64:128, :3]
    known_xyz = points[:, :64, :3]
    distances, neighbor_indices = three_nn(query_xyz, known_xyz)
    interp_features = points[:, :64, :]
    raw_weights = tf.math.reciprocal(tf.maximum(distances, 1e-6))
    weights = raw_weights / tf.reduce_sum(raw_weights, axis=2, keepdims=True)
    interpolated = three_interpolate(interp_features, neighbor_indices, weights)
    interp_upstream = tf.constant(
        rng.standard_normal(interpolated.shape.as_list()).astype(np.float32),
        name="custom_op_interpolate_upstream",
    )
    interpolate_gradient = tf.gradients(
        interpolated, interp_features, grad_ys=interp_upstream
    )[0]

    # These ops are discrete index/distance selectors and intentionally have no
    # registered coordinate gradient in the PointNet++ wrappers.
    query_ball_coordinate_gradient = tf.gradients(
        tf.reduce_sum(tf.cast(group_indices, tf.float32)), dataset_xyz
    )[0]
    three_nn_coordinate_gradient = tf.gradients(
        tf.reduce_sum(distances), query_xyz
    )[0]
    return {
        "group indices": group_indices,
        "group output": grouped,
        "group gradient": group_gradient,
        "gather output": gathered,
        "gather gradient": gather_gradient,
        "three_nn distances": distances,
        "three_nn indices": neighbor_indices,
        "three_nn coordinate gradient": three_nn_coordinate_gradient,
        "query-ball coordinate gradient": query_ball_coordinate_gradient,
        "three-interpolate output": interpolated,
        "three-interpolate gradient": interpolate_gradient,
    }


def run(args: argparse.Namespace) -> None:
    np.random.seed(2026)
    config_path = Path(args.config).expanduser().resolve()
    data_path = str(Path(args.data_path).expanduser().resolve())
    config = _load_config(config_path, data_path)
    reader, contact_infos, cloud, poses, sample_index = _make_first_batch(
        config, args.dataset_type, data_path
    )
    if args.mode == "synthetic":
        cloud = _make_synthetic_cloud(
            int(config["OPTIMIZER"]["batch_size"]),
            int(config["DATA"]["num_point"]),
            args.seed,
        )

    contact_info = contact_infos[sample_index]
    scene_id = int(contact_info["scene_id"]) if "scene_id" in contact_info else sample_index
    ann_id = int(contact_info["ann_id"]) if "ann_id" in contact_info else -1
    print(
        "dataset={} first_batch_index={} scene_id={} ann_id={}".format(
            args.dataset_type, sample_index, scene_id, ann_id
        )
    )
    print("mode={} tf_seed={}".format(args.mode, args.seed))
    print("data_rng_seed=2026")
    _finite_stats("input_point_cloud", cloud)
    _finite_stats("input_camera_poses", poses)
    for key in ("contact_points", "contact_dirs", "contact_approaches", "finger_diffs"):
        if key in contact_info:
            _finite_stats("target_" + key, contact_info[key])

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        estimator = GraspEstimator(config)
        model_ops = estimator.build_network()
        labels_and_losses = load_labels_and_losses(
            estimator, contact_infos, config, train=True
        )
        ops = dict(model_ops)
        ops.update(labels_and_losses)

        loss_names = (
            "offset_loss",
            "dir_loss",
            "approach_loss",
            "bin_ce_loss",
            "adds_loss",
            "adds_gt2pred_loss",
            "loss",
        )
        trainable_variables = tf.trainable_variables()
        global_variables = tf.global_variables()
        restore_placeholders = [
            tf.placeholder(
                variable.dtype.base_dtype,
                shape=variable.shape,
                name="restore_initial_value_{}".format(index),
            )
            for index, variable in enumerate(global_variables)
        ]
        restore_ops = [
            tf.assign(variable, placeholder)
            for variable, placeholder in zip(global_variables, restore_placeholders)
        ]
        restore_initial_state = tf.group(*restore_ops)
        input_tensor = estimator.placeholders["pointclouds_pl"]
        total_gradients = tf.gradients(ops["loss"], trainable_variables)
        total_present = [
            (variable, gradient)
            for variable, gradient in zip(trainable_variables, total_gradients)
            if gradient is not None
        ]
        total_gradient_fetches = [gradient for _, gradient in total_present]
        bisection_tensors = _collect_layer_intermediates(tf.get_default_graph())
        bisection_gradients = (
            tf.gradients(ops["loss"], bisection_tensors)
            if args.mode == "bisection"
            else []
        )
        gradient_fetches: Dict[str, Dict[str, object]] = {}
        custom_op_fetches: Dict[str, object] = {}
        if args.mode == "full":
            for loss_name in loss_names:
                gradients = tf.gradients(ops[loss_name], trainable_variables)
                input_gradient = tf.gradients(ops[loss_name], input_tensor)[0]
                present = [
                    (variable, gradient)
                    for variable, gradient in zip(trainable_variables, gradients)
                    if gradient is not None
                ]
                gradient_fetches[loss_name] = {
                    "variables": [variable for variable, _ in present],
                    "gradients": [gradient for _, gradient in present],
                    "input_gradient": input_gradient,
                }

            custom_op_fetches = _build_custom_op_diagnostics(cloud)
        prediction_fetches = {
            "offset": ops["grasp_offset_pred"],
            "direction": ops["grasp_dir_pred"],
            "approach": ops["approach_dir_pred"],
            "scores": ops["binary_seg_pred"],
        }
        prediction_finiteness = {
            name: tf.reduce_all(tf.math.is_finite(tensor))
            for name, tensor in prediction_fetches.items()
        }
        input_feed = {
            estimator.placeholders["pointclouds_pl"]: cloud,
            estimator.placeholders["cam_poses_pl"]: poses,
            estimator.placeholders["scene_idx_pl"]: sample_index,
            estimator.placeholders["is_training_pl"]: True,
        }
        fetches: Dict[str, object] = {
            "predictions": prediction_fetches,
            "prediction_finiteness": prediction_finiteness,
            "losses": {name: ops[name] for name in loss_names},
            "custom_ops": {
                name: value
                for name, value in custom_op_fetches.items()
                if value is not None
            },
            "variable_gradients": {
                name: values["gradients"]
                for name, values in gradient_fetches.items()
            },
            "input_gradients": {
                name: values["input_gradient"]
                for name, values in gradient_fetches.items()
                if values["input_gradient"] is not None
            },
        }

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True
        with tf.Session(config=session_config) as session:
            initialize = tf.global_variables_initializer()
            session.run(initialize)
            session.run(ops["iterator"].initializer)
            if args.mode == "determinism":
                repeat_summaries = []
                first_predictions = None
                predictions_bitwise_identical = True
                initial_values = session.run(global_variables)
                restore_feed = dict(zip(restore_placeholders, initial_values))
                repeat_fetches = {
                    "predictions": prediction_fetches,
                    "gradients": total_gradient_fetches,
                }
                for repeat in range(args.repeats):
                    session.run(restore_initial_state, feed_dict=restore_feed)
                    session.run(ops["iterator"].initializer)
                    repeat_result = session.run(
                        repeat_fetches, feed_dict=input_feed
                    )
                    prediction_finite = all(
                        np.isfinite(value).all()
                        for value in repeat_result["predictions"].values()
                    )
                    nonfinite_names = [
                        variable.op.name
                        for (variable, _), gradient_value in zip(
                            total_present, repeat_result["gradients"]
                        )
                        if not np.isfinite(_gradient_values(gradient_value)).all()
                    ]
                    first_bad = nonfinite_names[0] if nonfinite_names else "none"
                    print(
                        "repeat={} nonfinite_grad_variables={}/{} "
                        "predictions_finite={} first_affected_variable={}".format(
                            repeat + 1,
                            len(nonfinite_names),
                            len(trainable_variables),
                            prediction_finite,
                            first_bad,
                        )
                    )
                    repeat_summaries.append((len(nonfinite_names), first_bad, prediction_finite))
                    if first_predictions is None:
                        first_predictions = repeat_result["predictions"]
                    else:
                        predictions_bitwise_identical &= all(
                            np.array_equal(first_predictions[name], value)
                            for name, value in repeat_result["predictions"].items()
                        )
                print(
                    "repeat_statuses_identical={}".format(
                        len(set(repeat_summaries)) == 1
                    )
                )
                print("same_initial_global_variables_restored_each_repeat=True")
                print(
                    "prediction_tensors_bitwise_identical={}".format(
                        predictions_bitwise_identical
                    )
                )
                return

            if args.mode == "synthetic":
                result = session.run(
                    {
                        "predictions": prediction_fetches,
                        "gradients": total_gradient_fetches,
                        "loss": ops["loss"],
                    },
                    feed_dict=input_feed,
                )
                nonfinite_names = [
                    variable.op.name
                    for (variable, _), gradient_value in zip(
                        total_present, result["gradients"]
                    )
                    if not np.isfinite(_gradient_values(gradient_value)).all()
                ]
                print(
                    "synthetic_control loss_finite={} nonfinite_grad_variables={}/{} "
                    "first_affected_variable={} predictions_finite={}".format(
                        np.isfinite(result["loss"]),
                        len(nonfinite_names),
                        len(trainable_variables),
                        nonfinite_names[0] if nonfinite_names else "none",
                        all(
                            np.isfinite(value).all()
                            for value in result["predictions"].values()
                        ),
                    )
                )
                return

            if args.mode == "bisection":
                bisection_fetches = {
                    "gradients": [
                        gradient for gradient in bisection_gradients
                        if gradient is not None
                    ],
                    "variable_gradients": total_gradient_fetches,
                    "prediction_finiteness": prediction_finiteness,
                    "loss": ops["loss"],
                }
                result = session.run(bisection_fetches, feed_dict=input_feed)
                present_gradients = iter(result["gradients"])
                entries = []
                for tensor, gradient in zip(bisection_tensors, bisection_gradients):
                    gradient_value = (
                        next(present_gradients) if gradient is not None else None
                    )
                    finite = (
                        bool(
                            np.isfinite(_gradient_values(gradient_value)).all()
                        )
                        if gradient is not None
                        else None
                    )
                    entries.append((tensor, finite))
                    print(
                        "intermediate tensor={} op_type={} gradient_finite={}".format(
                            tensor.name, tensor.op.type,
                            finite if finite is not None else "undefined",
                        )
                    )
                closest_bad_index = next(
                    (
                        index
                        for index in range(len(entries) - 1, -1, -1)
                        if entries[index][1] is False
                    ),
                    None,
                )
                closest_bad = (
                    entries[closest_bad_index]
                    if closest_bad_index is not None else None
                )
                last_good = next(
                    (
                        entries[index]
                        for index in range(
                            (closest_bad_index - 1)
                            if closest_bad_index is not None else len(entries) - 1,
                            -1,
                            -1,
                        )
                        if entries[index][1] is True
                    ),
                    None,
                )
                print("bisection_loss={}".format(result["loss"]))
                bad_variables = [
                    variable.op.name
                    for (variable, _), value in zip(
                        total_present, result["variable_gradients"]
                    )
                    if not np.isfinite(_gradient_values(value)).all()
                ]
                print(
                    "bisection_nonfinite_variable_gradients={}/{} first={}".format(
                        len(bad_variables),
                        len(trainable_variables),
                        bad_variables[0] if bad_variables else "none",
                    )
                )
                print(
                    "bisection_predictions_finite={}".format(
                        all(result["prediction_finiteness"].values())
                    )
                )
                print(
                    "closest_output_nonfinite_gradient={}".format(
                        "{} ({})".format(
                            closest_bad[0].name, closest_bad[0].op.type
                        ) if closest_bad else "none"
                    )
                )
                print(
                    "last_finite_gradient_toward_input={}".format(
                        "{} ({})".format(last_good[0].name, last_good[0].op.type)
                        if last_good else "none"
                    )
                )
                return

            results = session.run(fetches, feed_dict=input_feed)
            print("forward_pass is_training=True (no optimizer step)")
            for name, value in results["predictions"].items():
                _finite_stats("prediction_" + name, value)
                print(
                    "prediction_{}_all_finite={}".format(
                        name, results["prediction_finiteness"][name]
                    )
                )

            for loss_name in loss_names:
                print("loss_gradient_term={}".format(loss_name))
                gradient_info = gradient_fetches[loss_name]
                _finite_stats(
                    "  loss_value",
                    np.asarray(results["losses"][loss_name]),
                )
                present_variables = gradient_info["variables"]
                present_values = results["variable_gradients"][loss_name]
                if present_variables:
                    nonfinite_names = []
                    for variable, gradient_value in zip(
                        present_variables, present_values
                    ):
                        values = _gradient_values(gradient_value)
                        if not np.isfinite(values).all():
                            nonfinite_names.append(variable.op.name)
                    print(
                        "  nonfinite_variable_gradients={}/{}".format(
                            len(nonfinite_names), len(trainable_variables)
                        )
                    )
                    print(
                        "  first_nonfinite_variable={}".format(
                            nonfinite_names[0] if nonfinite_names else "none"
                        )
                    )
                    print(
                        "  last_nonfinite_variable={}".format(
                            nonfinite_names[-1] if nonfinite_names else "none"
                        )
                    )
                else:
                    print(
                        "  nonfinite_variable_gradients=0/{} (no connected variables)"
                        .format(len(trainable_variables))
                    )

                input_gradient = gradient_info["input_gradient"]
                if input_gradient is None:
                    print("  input_pointcloud_gradient=undefined")
                else:
                    input_gradient_value = results["input_gradients"][loss_name]
                    _finite_stats(
                        "  input_pointcloud_gradient", input_gradient_value
                    )

            custom_values = results["custom_ops"]
            print("custom_pointnet2_operator_probes (finite random upstream)")
            for name in custom_op_fetches:
                value = custom_values.get(name)
                if value is None:
                    print("  {}=undefined_by_registered_gradient".format(name))
                elif "gradient" in name:
                    _finite_stats("  " + name, value)
                else:
                    _finite_stats("  " + name, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_type",
        choices=("graspnet", "acronym"),
        default="graspnet",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(
            os.environ.get("GN_ROOT", os.path.expanduser("~/datasets/graspnet_1b")),
            "cgn_preprocessed",
        ),
        help="GraspNet preprocessed root, or ACRONYM data root with scene_contacts",
    )
    parser.add_argument(
        "--config",
        default=str(CONTACT_DIR / "config_graspnet1b.yaml"),
        help="YAML config (use config.yaml for the ACRONYM control)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "determinism", "synthetic", "bisection"),
        default="full",
        help="Diagnostic to run; full preserves the original per-loss report",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
