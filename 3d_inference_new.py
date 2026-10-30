import os
import sys
from dataclasses import dataclass
import numpy as np
import open3d as o3d
import argparse

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'contact_graspnet'))
sys.path.append(os.path.join(BASE_DIR, 'contact_graspnet', 'pointnet2', 'utils'))

import config_utils
from contact_grasp_estimator import GraspEstimator

import plotly.graph_objects as go


# ----------------------------------------------------------------------------
# Gripper geometry
# ----------------------------------------------------------------------------
_GRIPPER_OPEN_STL_PATH = os.path.join(BASE_DIR, 'gripper_models', '2_finger_open.stl')
_GRIPPER_ASSET_SCALE = 0.001
_GRIPPER_CONTACT_ANCHOR_MM = np.array(
    [-66.17023086547852, 129.4645538330078, 23.87684440612793],
    dtype=np.float64,
)
_GRIPPER_OPPOSITE_CONTACT_MM = np.array(
    [23.380385398864746, 129.88703536987305, 23.87684440612793],
    dtype=np.float64,
)
_GRIPPER_FIXED_OPENING_M = 0.09
_GRIPPER_ASSET_OPENING_M = float(
    np.linalg.norm(_GRIPPER_OPPOSITE_CONTACT_MM - _GRIPPER_CONTACT_ANCHOR_MM)
    * _GRIPPER_ASSET_SCALE
)
_CGN_GRIPPER_DEPTH_M = 0.1034
_GRIPPER_SOURCE_TO_CGN = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
_GRIPPER_DISPLAY_POINT_LIMIT = 2500


@dataclass(frozen=True)
class GripperModel:
    collision_points_local: np.ndarray
    display_points_local: np.ndarray
    mesh_vertices_local: np.ndarray
    mesh_faces: np.ndarray


def load_gripper_model():
    """Load and validate the open, maximum-width custom gripper model."""
    if not os.path.isfile(_GRIPPER_OPEN_STL_PATH):
        raise FileNotFoundError(
            f"Required custom gripper asset is missing: {_GRIPPER_OPEN_STL_PATH}"
        )

    import trimesh

    mesh = trimesh.load(_GRIPPER_OPEN_STL_PATH, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(
            f"Custom gripper STL contains no triangle mesh: {_GRIPPER_OPEN_STL_PATH}"
        )

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_triangles = np.asarray(mesh.triangles, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    if (mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3
            or mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3
            or len(mesh_faces) == 0
            or not np.all(np.isfinite(mesh_vertices))
            or not np.all(np.isfinite(mesh_triangles))
            or mesh_faces.min() < 0 or mesh_faces.max() >= len(mesh_vertices)):
        raise ValueError("Custom gripper STL contains malformed or non-finite geometry.")

    for contact_point, expected_normal in (
        (_GRIPPER_CONTACT_ANCHOR_MM, np.array([1.0, 0.0, 0.0])),
        (_GRIPPER_OPPOSITE_CONTACT_MM, np.array([-1.0, 0.0, 0.0])),
    ):
        closest_points = trimesh.triangles.closest_point(
            mesh_triangles,
            np.broadcast_to(contact_point, (len(mesh_triangles), 3)),
        )
        distances = np.linalg.norm(closest_points - contact_point, axis=1)
        face_index = int(np.argmin(distances))
        face_normal = np.asarray(mesh.face_normals[face_index], dtype=np.float64)
        if (distances[face_index] > 0.1
                or not np.all(np.isfinite(face_normal))
                or np.dot(face_normal, expected_normal) < 0.95):
            raise ValueError(
                "Configured maximum-opening contact point does not match its "
                f"expected fingertip face in {_GRIPPER_OPEN_STL_PATH}."
            )

    determinant = float(np.linalg.det(_GRIPPER_SOURCE_TO_CGN))
    if (not np.allclose(
            _GRIPPER_SOURCE_TO_CGN.T @ _GRIPPER_SOURCE_TO_CGN,
            np.eye(3),
            atol=1e-8,
        ) or not np.isclose(determinant, 1.0, atol=1e-8)):
        raise ValueError("Custom gripper axis mapping must be a right-handed rotation.")

    opening_vector_local = (
        _GRIPPER_SOURCE_TO_CGN
        @ (_GRIPPER_OPPOSITE_CONTACT_MM - _GRIPPER_CONTACT_ANCHOR_MM)
        * _GRIPPER_ASSET_SCALE
    )
    if (not np.isclose(
            _GRIPPER_ASSET_OPENING_M,
            _GRIPPER_FIXED_OPENING_M,
            atol=0.001,
            rtol=0.0,
        ) or np.linalg.norm(opening_vector_local[1:]) > 0.001):
        raise ValueError(
            "Measured open-STL jaw contact separation "
            f"({_GRIPPER_ASSET_OPENING_M * 1000:.2f} mm) does not match the "
            f"configured maximum opening ({_GRIPPER_FIXED_OPENING_M * 1000:.1f} mm) "
            "along the CGN closing axis."
        )

    source_hull = mesh.convex_hull
    if len(source_hull.vertices) < 4:
        raise ValueError("Custom gripper STL does not enclose a 3D collision volume.")

    anchor_m = _GRIPPER_CONTACT_ANCHOR_MM * _GRIPPER_ASSET_SCALE

    def to_grasp_local(source_points):
        centered = (np.asarray(source_points, dtype=np.float64) * _GRIPPER_ASSET_SCALE
                    - anchor_m)
        return (_GRIPPER_SOURCE_TO_CGN @ centered.T).T

    display_count = min(len(mesh_vertices), _GRIPPER_DISPLAY_POINT_LIMIT)
    if display_count < len(mesh_vertices):
        rng = np.random.default_rng(0)
        display_indices = np.sort(
            rng.choice(len(mesh_vertices), display_count, replace=False)
        )
        display_points = mesh_vertices[display_indices]
    else:
        display_points = mesh_vertices

    print(
        "Loaded custom gripper: "
        f"{len(source_hull.vertices)} STL hull vertices for table checks, "
        f"{display_count} STL points for Plotly, "
        f"{_GRIPPER_ASSET_OPENING_M * 1000:.2f} mm measured maximum opening."
    )
    print(
        "Table clearance and Mayavi use the supplied open STL; predicted grasp "
        "widths are not used to articulate or reject this maximum-opening model."
    )
    return GripperModel(
        collision_points_local=to_grasp_local(np.asarray(source_hull.vertices)),
        display_points_local=to_grasp_local(display_points),
        mesh_vertices_local=to_grasp_local(mesh_vertices),
        mesh_faces=mesh_faces,
    )


def _world_gripper_points(gripper_points_local, grasp_rotation, contact_point,
                          approach_offset=0.0):
    point = np.asarray(contact_point, dtype=np.float64)
    rotation = np.asarray(grasp_rotation, dtype=np.float64)
    if point.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Gripper transform requires a 3-vector contact and 3x3 rotation.")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(rotation)):
        raise ValueError("Gripper transform contains non-finite values.")
    translation = point - approach_offset * rotation[:, 2]
    return (rotation @ gripper_points_local.T).T + translation


def _minimum_gripper_plane_clearance(gripper_model, grasp_rotation, contact_point,
                                     table_normal, table_d, standoff=0.0):
    points_final = _world_gripper_points(
        gripper_model.collision_points_local, grasp_rotation, contact_point
    )
    final_clearance = float((points_final @ table_normal + table_d).min())
    if standoff <= 0:
        return final_clearance
    points_standoff = _world_gripper_points(
        gripper_model.collision_points_local, grasp_rotation, contact_point,
        approach_offset=standoff,
    )
    standoff_clearance = float((points_standoff @ table_normal + table_d).min())
    return min(final_clearance, standoff_clearance)


def get_gripper_geometry():
    # SET THESE TO YOUR CUSTOM GRIPPER'S REAL DIMENSIONS
    w = 0.08    # max opening width (meters), center-to-center of finger pads
    d = 0.1034  # finger length: grasp-frame origin to fingertip (meters)

    # Convention: x-axis = baseline/closing direction, z-axis = approach direction
    lines = [
        [[-w/2, 0, 0], [w/2, 0, 0]],       # base
        [[-w/2, 0, 0], [-w/2, 0, d]],      # left finger
        [[w/2, 0, 0], [w/2, 0, d]],        # right finger
        [[0, 0, 0], [0, 0, -0.05]]         # arm mount
    ]
    return lines
    


def gripper_control_points_array():
    """ Flattened (N,3) array of every point used in gripper geometry, for collision checks. """
    lines = get_gripper_geometry()
    return np.array([pt for line in lines for pt in line], dtype=np.float64)


def transform_lines(lines, pose):
    transformed_lines = []
    for line in lines:
        new_line = []
        for pt in line:
            pt_hom = np.array([pt[0], pt[1], pt[2], 1.0])
            pt_trans = pose @ pt_hom
            new_line.append(pt_trans[:3])
        transformed_lines.append(new_line)
    return transformed_lines


def compute_contact_pair(g, contact_pt, width):
    """
    contact_pt is one antipodal contact (already on the object surface, as
    returned by CGN). The second contact is derived from the grasp pose's
    baseline (closing) axis and predicted gripper width.
    """
    baseline_dir = g[:3, 0]
    baseline_dir = baseline_dir / (np.linalg.norm(baseline_dir) + 1e-8)
    c1 = contact_pt
    c2 = contact_pt + width * baseline_dir
    return c1, c2


# ----------------------------------------------------------------------------
# Table plane detection
# ----------------------------------------------------------------------------
def get_table_plane(pcd, object_points, distance_threshold=0.015, ransac_n=3, num_iterations=1000):
    """
    Returns a normalized plane (normal, d) such that normal . point + d >= 0
    for points ABOVE the table (where objects sit).
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=ransac_n, num_iterations=num_iterations)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    normal, d = normal / norm_len, d / norm_len

    dists = object_points @ normal + d
    if np.mean(dists) < 0:
        normal, d = -normal, -d

    print(f"Table plane: normal={normal}, d={d:.4f}, inliers={len(inliers)}")
    return normal, d, inliers

def detect_table_robust(pcd, min_object_cluster_points=200, cluster_eps=0.02,
                         remainder_min_points=200, distance_threshold=0.01,
                         ransac_n=3, num_iterations=1000):
    """
    Finds the main object as the largest dense cluster in the RAW cloud first,
    then fits a table plane only to whatever points remain outside it. This
    guarantees the plane can never be fit through the object itself -- unlike
    plane-first detection, which has no way to tell an object's flat face from
    an actual table.

    Returns (None, None) with an explanatory message on every foreseeable
    failure (sparse cloud, degenerate RANSAC, empty clusters, etc.) so the
    caller always receives a clean signal to fall back gracefully.
    """
    points = np.asarray(pcd.points)
    if len(points) < max(remainder_min_points, ransac_n + 1):
        print(f"detect_table_robust: cloud has only {len(points)} points -- "
              "too sparse to attempt cluster-first table detection.")
        return None, None

    try:
        labels = np.array(pcd.cluster_dbscan(eps=cluster_eps,
                                              min_points=min_object_cluster_points,
                                              print_progress=False))
    except Exception as exc:
        print(f"detect_table_robust: DBSCAN failed ({exc}); skipping.")
        return None, None

    if labels.size == 0 or labels.max() < 0:
        print("detect_table_robust: no dense cluster found (all noise). "
              "Skipping cluster-first table detection.")
        return None, None

    counts = np.bincount(labels[labels >= 0])
    main_cluster_label = np.argmax(counts)
    object_mask = labels == main_cluster_label
    remainder_mask = ~object_mask

    remainder_points = points[remainder_mask]
    if len(remainder_points) < remainder_min_points:
        print(f"detect_table_robust: only {len(remainder_points)} points remain outside "
              f"the main object cluster (need >= {remainder_min_points}). "
              "Not enough to fit a reliable table plane -- skipping.")
        return None, None

    remainder_pcd = pcd.select_by_index(np.where(remainder_mask)[0])
    try:
        plane_model, inliers = remainder_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
    except Exception as exc:
        print(f"detect_table_robust: RANSAC plane fit failed ({exc}); skipping.")
        return None, None

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-8:
        print("detect_table_robust: RANSAC returned a degenerate (zero-norm) normal -- "
              "likely too few/coplanar remainder points. Skipping.")
        return None, None
    normal, d = normal / norm_len, d / norm_len

    obj_pts = points[object_mask]
    if len(obj_pts) > 0:
        obj_dists = obj_pts @ normal + d
        if np.mean(obj_dists) < 0:
            normal, d = -normal, -d

    print(f"Table plane fit from {len(inliers)}/{len(remainder_points)} points "
          f"outside the main object cluster ({object_mask.sum()} object points excluded from fitting)")
    return normal, d



def _orient_table_toward_camera(normal, d):
    """Flip the plane so the camera origin (0,0,0) lies on the above-table side."""
    if d < 0:
        return -normal, -d
    return normal, d


def detect_table_dominant_plane(pcd, distance_threshold=0.012, ransac_n=3,
                                 num_iterations=2000, voxel_size=0.006,
                                 min_inliers=800, min_inlier_ratio=0.06,
                                 max_planes=3):
    """
    Fit the table as the dominant RANSAC plane in an RGB-D cloud.

    This is the right prior for a RealSense tabletop capture: the table is
    usually the largest planar surface. Cluster-first detection often fails
    here because the object is physically connected to the table in the
    depth image, so they form one DBSCAN blob.

    Tries up to `max_planes` successive planes on a downsampled cloud and
    keeps the one with the most inliers that still has objects sitting on
    the camera-facing side.
    """
    points = np.asarray(pcd.points)
    if len(points) < max(50, ransac_n):
        print("Not enough points for dominant-plane table detection.")
        return None, None

    work = pcd.voxel_down_sample(voxel_size) if voxel_size and voxel_size > 0 else pcd
    remaining = work
    best = None

    for plane_i in range(max_planes):
        remaining_pts = np.asarray(remaining.points)
        if len(remaining_pts) < max(50, ransac_n):
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
        except Exception as exc:
            print(f"  RANSAC plane {plane_i}: fit failed ({exc}); stopping search.")
            break

        n_inliers = len(inliers)
        if n_inliers == 0:
            print(f"  RANSAC plane {plane_i}: returned 0 inliers; stopping search.")
            break

        inlier_ratio = n_inliers / float(len(remaining_pts))
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            print(f"  RANSAC plane {plane_i}: degenerate zero-norm normal; skipping candidate.")
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal, d = normal / norm_len, d / norm_len
        normal, d = _orient_table_toward_camera(normal, d)

        full_dists = points @ normal + d
        n_full_inliers = int(np.sum(np.abs(full_dists) < distance_threshold))
        above_frac = float(np.mean(full_dists > 0.01))

        # Objects should sit on the camera side of a table, not a back wall
        # that splits the cloud 50/50 with almost nothing "above" it.
        objects_ok = 0.08 <= above_frac <= 0.95
        enough = (n_inliers >= min_inliers) or (inlier_ratio >= min_inlier_ratio)
        score = n_full_inliers * (1.0 if objects_ok else 0.15)

        print(f"  RANSAC plane {plane_i}: inliers={n_inliers}/{len(remaining_pts)} "
              f"({inlier_ratio*100:.1f}%), full-cloud inliers={n_full_inliers}, "
              f"above-table={above_frac*100:.1f}%, score={score:.0f}")

        if enough:
            candidate = dict(normal=normal, d=d, score=score, n_full_inliers=n_full_inliers)
            if best is None or candidate['score'] > best['score']:
                best = candidate

        remaining = remaining.select_by_index(inliers, invert=True)


    if best is None:
        print("Dominant-plane RANSAC did not find a usable table.")
        return None, None

    print(f"Table plane (dominant RANSAC): normal={best['normal']}, d={best['d']:.4f}, "
          f"full-cloud inliers={best['n_full_inliers']}")
    return best['normal'], best['d']

# ----------------------------------------------------------------------------
# Object segmentation via DBSCAN (no RGB/SAM needed -- purely geometric)
# ----------------------------------------------------------------------------
def segment_objects_3d(pcd_no_table, eps=0.02, min_points=30):
    """
    eps: max distance (m) between points to be considered part of the same object
    min_points: minimum points for a cluster to count as a real object, not noise
    """
    labels = np.array(pcd_no_table.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    pc_segments = {}
    points = np.asarray(pcd_no_table.points)
    for label in np.unique(labels):
        if label == -1:
            continue  # noise
        pc_segments[int(label)] = points[labels == label]
    print(f"DBSCAN found {len(pc_segments)} object cluster(s) (eps={eps}, min_points={min_points})")
    for obj_id, pc in pc_segments.items():
        print(f"  Object {obj_id}: {len(pc)} points")
    return pc_segments


def validate_floor_plane(table_normal, table_d, table_points=None, is_virtual=False,
                          distance_threshold=0.01, min_inlier_frac=0.85, min_points=50):
    """
    Confirms the floor plane about to be used for collision detection is
    actually trustworthy, BEFORE any rotation-correction runs against it.

    For a real, detected plane (table_points given): checks it has enough
    supporting points and that most of them genuinely lie flat against it
    (low residual) -- a plane fit to noise or a mixed/bad region would fail
    this and should not be trusted for real collision decisions.

    For a virtual/synthetic floor (is_virtual=True, e.g. z = object's own
    z_min): there are no real points to validate against.  A synthetic floor
    is NOT a measured surface, so it is treated as NOT solid: collision
    filtering is disabled for the run, and a clear warning is printed.

    Returns True/False.  Only a REAL, well-supported, well-fitting plane
    returns True.  Virtual floors always return False.
    """
    if is_virtual:
        print(
            f"Floor plane check: VIRTUAL floor (normal={table_normal}, d={table_d:.4f}) -- "
            "this is a synthetic fallback, NOT measured from real points. "
            "Collision filtering will be DISABLED to avoid false rejections against an "
            "unmeasured surface."
        )
        return False  # synthetic floor must never drive collision decisions

    if table_points is None or len(table_points) < min_points:
        n = 0 if table_points is None else len(table_points)
        print(f"Floor plane check: FAILED -- only {n} supporting points "
              f"(need >= {min_points}). Floor is NOT solid enough to trust for collision "
              f"detection; disabling collision filtering for this run.")
        return False

    dists = np.abs(table_points @ table_normal + table_d)
    inlier_frac = float(np.mean(dists < distance_threshold))
    if inlier_frac < min_inlier_frac:
        print(f"Floor plane check: FAILED -- only {inlier_frac*100:.1f}% of the "
              f"{len(table_points)} supporting points actually lie flat against the fitted "
              f"plane (need >= {min_inlier_frac*100:.0f}%). This plane looks unreliable "
              f"(likely fit across a mix of surfaces); disabling collision filtering.")
        return False

    print(f"Floor plane check: PASSED -- {len(table_points)} points, "
          f"{inlier_frac*100:.1f}% within {distance_threshold*1000:.0f}mm of the fitted plane. "
          f"Floor is solid; collision detection will run against normal={table_normal}, "
          f"d={table_d:.4f}.")
    return True


# ----------------------------------------------------------------------------
# Table-collision + approach-path filtering
# ----------------------------------------------------------------------------
def filter_table_colliding_grasps(pred_grasps_cam, scores, contact_pts, gripper_openings,
                                   table_normal, table_d, min_clearance=0.005, standoff=0.10,
                                   contact_pairs=None, gripper_model=None):
    """
    Rejects grasps whose geometry at the final pose or approach standoff falls
    below the requested clearance from the table plane.

    When gripper_model is supplied, its STL-derived convex hull is checked;
    support against a plane is exact for the hull of the mesh vertices.
    """
    control_points = gripper_control_points_array()

    filtered_grasps, filtered_scores, filtered_contacts, filtered_widths = {}, {}, {}, {}
    filtered_pairs = {} if contact_pairs is not None else None

    for obj_id in pred_grasps_cam:
        grasps = pred_grasps_cam[obj_id]
        obj_scores = scores[obj_id]
        obj_contacts = contact_pts.get(obj_id, None)
        obj_widths = gripper_openings.get(obj_id, None) if gripper_openings is not None else None
        obj_pairs = contact_pairs.get(obj_id, None) if contact_pairs is not None else None
        if gripper_model is not None:
            if np.asarray(obj_scores).shape != (len(grasps),):
                raise ValueError(f"Object {obj_id}: grasp scores are not aligned with grasp poses.")
            if obj_contacts is None or np.asarray(obj_contacts).shape != (len(grasps), 3):
                raise ValueError(
                    f"Object {obj_id}: custom gripper collision checks require one "
                    "contact point per grasp."
                )
            if not np.all(np.isfinite(obj_contacts)):
                raise ValueError(f"Object {obj_id}: grasp contact points contain non-finite values.")
            if obj_widths is None or np.asarray(obj_widths).shape != (len(grasps),):
                raise ValueError(
                    f"Object {obj_id}: predicted openings are not aligned with grasp poses."
                )
            if not np.all(np.isfinite(obj_widths)):
                raise ValueError(
                    f"Object {obj_id}: predicted gripper openings contain non-finite values."
                )

        keep_idx = []
        for i, g in enumerate(grasps):
            R, t = g[:3, :3], g[:3, 3]

            if gripper_model is not None:
                clearance = _minimum_gripper_plane_clearance(
                    gripper_model, R, obj_contacts[i], table_normal, table_d,
                    standoff=standoff,
                )
                if clearance < min_clearance:
                    continue
            else:
                pts_world = (R @ control_points.T).T + t
                if (pts_world @ table_normal + table_d).min() < min_clearance:
                    continue

                approach_dir = R[:, 2]
                pts_standoff = (R @ control_points.T).T + (t - standoff * approach_dir)
                if (pts_standoff @ table_normal + table_d).min() < min_clearance:
                    continue

            if not np.isfinite(obj_scores[i]):
                continue

            keep_idx.append(i)

        keep_idx = np.array(keep_idx, dtype=int)
        n_kept = len(keep_idx)

        filtered_grasps[obj_id] = grasps[keep_idx] if n_kept else np.zeros((0, 4, 4))
        filtered_scores[obj_id] = obj_scores[keep_idx] if n_kept else np.zeros((0,))
        if obj_contacts is not None:
            filtered_contacts[obj_id] = obj_contacts[keep_idx] if n_kept else np.zeros((0, 3))
        if obj_widths is not None:
            filtered_widths[obj_id] = obj_widths[keep_idx] if n_kept else np.zeros((0,))
        if filtered_pairs is not None and obj_pairs is not None:
            filtered_pairs[obj_id] = obj_pairs[keep_idx] if n_kept else np.zeros((0, 2, 3))

        print(f"Object {obj_id}: {len(grasps)} grasps -> {n_kept} survive table-collision filtering")

    return filtered_grasps, filtered_scores, filtered_contacts, filtered_widths, filtered_pairs


def _rotation_about_local_x(theta):
    """ Rotation matrix about the LOCAL x-axis (baseline/closing direction). """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])


def resolve_floor_collisions_by_rotation(pred_grasps_cam, table_normal, table_d,
                                          min_clearance=0.005, standoff=0.10,
                                          max_angle_deg=180.0, angle_step_deg=0.5,
                                          contact_pts=None, gripper_openings=None,
                                          gripper_model=None):
    """
    For any grasp whose gripper geometry collides with the floor plane, tilts
    the gripper by rotating about its OWN local x-axis (the baseline/closing
    direction) until it clears the floor -- instead of discarding the grasp.
    Grasps that can't be resolved within max_angle_deg are left unchanged for
    the final clearance filter to reject.
    """
    control_points = gripper_control_points_array()
    angles = np.arange(angle_step_deg, max_angle_deg + 1e-6, angle_step_deg)

    def min_clearance_for(R, t, contact_point):
        if gripper_model is not None:
            return _minimum_gripper_plane_clearance(
                gripper_model, R, contact_point, table_normal, table_d,
                standoff=standoff,
            )
        pts_world = (R @ control_points.T).T + t
        c_final = (pts_world @ table_normal + table_d).min()
        approach_dir = R[:, 2]
        pts_standoff = (R @ control_points.T).T + (t - standoff * approach_dir)
        c_standoff = (pts_standoff @ table_normal + table_d).min()
        return min(c_final, c_standoff)

    adjusted_grasps = {}
    for obj_id, grasps in pred_grasps_cam.items():
        grasps_out = grasps.copy() if len(grasps) else grasps
        obj_contacts = contact_pts.get(obj_id) if contact_pts is not None else None
        obj_widths = gripper_openings.get(obj_id) if gripper_openings is not None else None
        if gripper_model is not None:
            if obj_contacts is None or np.asarray(obj_contacts).shape != (len(grasps), 3):
                raise ValueError(
                    f"Object {obj_id}: custom gripper collision resolution requires "
                    "one contact point per grasp."
                )
            if not np.all(np.isfinite(obj_contacts)):
                raise ValueError(f"Object {obj_id}: grasp contact points contain non-finite values.")
            if obj_widths is None or np.asarray(obj_widths).shape != (len(grasps),):
                raise ValueError(
                    f"Object {obj_id}: custom gripper rotation resolution requires one "
                    "predicted opening per grasp."
                )
            if not np.all(np.isfinite(obj_widths)):
                raise ValueError(
                    f"Object {obj_id}: predicted gripper openings contain non-finite values."
                )

        n_ok, n_adjusted, n_unresolved = 0, 0, 0

        for i, g in enumerate(grasps):
            R, t = g[:3, :3], g[:3, 3]
            contact_point = obj_contacts[i] if obj_contacts is not None else None

            if min_clearance_for(R, t, contact_point) >= min_clearance:
                n_ok += 1
                continue

            resolved = False
            for deg in angles:
                for sign in (1, -1):
                    R_test = R @ _rotation_about_local_x(np.radians(sign * deg))
                    if min_clearance_for(R_test, t, contact_point) >= min_clearance:
                        grasps_out[i, :3, :3] = R_test
                        if gripper_model is not None:
                            grasps_out[i, :3, 3] = (
                                contact_point
                                + 0.5 * obj_widths[i] * R_test[:, 0]
                                - _CGN_GRIPPER_DEPTH_M * R_test[:, 2]
                            )
                        resolved = True
                        n_adjusted += 1
                        break
                if resolved:
                    break

            if not resolved:
                n_unresolved += 1

        adjusted_grasps[obj_id] = grasps_out
        print(f"Object {obj_id}: {n_ok} clear, {n_adjusted} tilted to clear the floor, "
              f"{n_unresolved} unresolved within {max_angle_deg} deg (pending final clearance filter)")

    return adjusted_grasps


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def snap_contact_pairs(pred_grasps_cam, contact_pts, gripper_openings, pc_segments,
                       tol=0.004, search_multiplier=1.6):
    """
    For each grasp, finds the second contact point by intersecting the gripper's
    closing axis with the object's own point cloud.
    """
    contact_pairs = {}
    for obj_id, grasps in pred_grasps_cam.items():
        obj_pc = pc_segments.get(obj_id, None)
        obj_contacts = contact_pts.get(obj_id, None)
        obj_widths = gripper_openings.get(obj_id, None) if gripper_openings is not None else None
        pairs = []

        for i, g in enumerate(grasps):
            c1 = obj_contacts[i] if obj_contacts is not None and len(obj_contacts) > i else g[:3, 3]
            baseline_dir = g[:3, 0]
            baseline_dir = baseline_dir / (np.linalg.norm(baseline_dir) + 1e-8)
            predicted_width = obj_widths[i] if obj_widths is not None else 0.08

            c2 = c1 + predicted_width * baseline_dir

            if obj_pc is not None and len(obj_pc) > 0:
                rel = obj_pc - c1
                t = rel @ baseline_dir
                perp = rel - np.outer(t, baseline_dir)
                perp_dist = np.linalg.norm(perp, axis=1)

                search_max = predicted_width * search_multiplier
                candidates = (t > 0.001) & (t < search_max) & (perp_dist < tol)
                if np.any(candidates):
                    best_idx = np.argmax(t[candidates])
                    c2 = c1 + t[candidates][best_idx] * baseline_dir

            pairs.append((c1, c2))

        contact_pairs[obj_id] = np.array(pairs) if len(pairs) else np.zeros((0, 2, 3))
    return contact_pairs


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def export_3d_scene(pc, pc_colors, pred_grasps_cam, scores, contact_pairs,
                     out_path, table_normal=None, table_d=None, show_both_contacts=True,
                     score_threshold=0.23, min_clearance=0.005, standoff=0.10,
                     vis_floor_normal=None, vis_floor_d=None,
                     table_inlier_points=None,
                     require_all_points_above_z0=True,
                     top_n=-1, show_plane=False, camera_location=None,
                     gripper_contact_pts=None, gripper_model=None):
    print(f"Exporting 3D scene to {out_path}...")

    if len(pc) == 0:
        print("Warning: point cloud is empty, skipping export.")
        return

    fig = go.Figure()

    mesh_added = False
    try:
        vertices, triangles, vertex_colors = _reconstruct_scene_mesh(
            pc, pc_colors, use_bpa=True, camera_location=camera_location
        )

        if len(triangles) >= 10:
            if vertex_colors is not None and len(vertex_colors) == len(vertices):
                vc255 = (vertex_colors * 255).astype(int)
                vertex_color_strs = [f'rgb({r},{g},{b})' for r, g, b in vc255]
            else:
                vertex_color_strs = ['rgb(120,180,255)'] * len(vertices)

            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                vertexcolor=vertex_color_strs,
                opacity=1.0,
                flatshading=False,
                name='Object Mesh'
            ))
            mesh_added = True
            print(f"Mesh rendered: {len(vertices)} vertices, {len(triangles)} faces.")
    except Exception as e:
        print(f"Mesh reconstruction failed ({e}) -- falling back to point cloud.")

    if not mesh_added:
        if len(pc) > 20000:
            idx = np.random.choice(len(pc), 20000, replace=False)
            pc_vis = pc[idx]
            colors_vis = pc_colors[idx] if pc_colors is not None else None
        else:
            pc_vis = pc
            colors_vis = pc_colors

        marker_dict = dict(size=1.5, opacity=0.8)
        if colors_vis is not None and len(colors_vis) == len(pc_vis):
            cv = colors_vis * 255 if np.max(colors_vis) <= 1.0 else colors_vis
            marker_dict['color'] = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in cv]
        else:
            marker_dict['color'] = 'blue'

        fig.add_trace(go.Scatter3d(
            x=pc_vis[:, 0], y=pc_vis[:, 1], z=pc_vis[:, 2],
            mode='markers', marker=marker_dict, name='Full Point Cloud'
        ))

    if show_plane:
        cx, cy = (pc[:, 0].min() + pc[:, 0].max()) / 2, (pc[:, 1].min() + pc[:, 1].max()) / 2
        half_size = max(pc[:, 0].ptp(), pc[:, 1].ptp(), 0.2) * 4
        x_range = np.linspace(cx - half_size, cx + half_size, 2)
        y_range = np.linspace(cy - half_size, cy + half_size, 2)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz, opacity=1.0, showscale=False,
            colorscale=[[0, 'saddlebrown'], [1, 'saddlebrown']], name='z=0 Plane'))

        if table_normal is not None and table_d is not None:
            extent_source = table_inlier_points if table_inlier_points is not None else pc
            x_range = np.linspace(extent_source[:, 0].min(), extent_source[:, 0].max(), 2)
            y_range = np.linspace(extent_source[:, 1].min(), extent_source[:, 1].max(), 2)
            xx, yy = np.meshgrid(x_range, y_range)
            nz = table_normal[2] if abs(table_normal[2]) > 1e-6 else 1e-6
            zz = (-table_d - table_normal[0]*xx - table_normal[1]*yy) / nz
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz, opacity=1.0, showscale=False,
                colorscale=[[0, 'saddlebrown'], [1, 'saddlebrown']], name='Table Plane'))

    gripper_lines = get_gripper_geometry()
    ctrl_pts = gripper_control_points_array()
    if gripper_contact_pts is None:
        gripper_contact_pts = {}

    for obj_id in pred_grasps_cam.keys():
        grasps = pred_grasps_cam[obj_id]
        if len(grasps) == 0:
            continue

        obj_scores = scores[obj_id]
        obj_pairs = contact_pairs.get(obj_id, None) if contact_pairs is not None else None
        obj_gripper_contacts = gripper_contact_pts.get(obj_id)
        if gripper_model is not None and (
            obj_gripper_contacts is None
            or np.asarray(obj_gripper_contacts).shape != (len(grasps), 3)
        ):
            raise ValueError(
                f"Object {obj_id}: custom gripper visualization requires one contact "
                "point per grasp."
            )
        sorted_indices = np.argsort(obj_scores)[::-1]

        n_total = len(grasps)
        n_above_score = 0
        n_floor_clear = 0
        n_above_z0 = 0
        n_shown = 0

        for idx in sorted_indices:
            if obj_scores[idx] < score_threshold:
                continue
            n_above_score += 1

            g = grasps[idx]
            R_g, t_g = g[:3, :3], g[:3, 3]
            if gripper_model is not None:
                pts_world = _world_gripper_points(
                    gripper_model.collision_points_local, R_g,
                    obj_gripper_contacts[idx],
                )
            else:
                pts_world = (R_g @ ctrl_pts.T).T + t_g

            if vis_floor_normal is not None and vis_floor_d is not None:
                if (pts_world @ vis_floor_normal + vis_floor_d).min() < min_clearance:
                    continue
                if gripper_model is not None:
                    pts_standoff = _world_gripper_points(
                        gripper_model.collision_points_local, R_g,
                        obj_gripper_contacts[idx], approach_offset=standoff,
                    )
                else:
                    approach_dir = R_g[:, 2]
                    pts_standoff = (R_g @ ctrl_pts.T).T + (t_g - standoff * approach_dir)
                if (pts_standoff @ vis_floor_normal + vis_floor_d).min() < min_clearance:
                    continue
            n_floor_clear += 1

            if require_all_points_above_z0 and pts_world[:, 2].min() < 0.0:
                continue
            n_above_z0 += 1

            if top_n is not None and top_n > 0 and n_shown >= top_n:
                continue  # already have enough -- keep counting funnel stats above, just stop drawing

            color = 'red' if n_shown == 0 else 'green'

            if gripper_model is not None:
                gripper_display = _world_gripper_points(
                    gripper_model.display_points_local, R_g,
                    obj_gripper_contacts[idx],
                )
                fig.add_trace(go.Scatter3d(
                    x=gripper_display[:, 0], y=gripper_display[:, 1], z=gripper_display[:, 2],
                    mode='markers',
                    marker=dict(color='orange', size=2, opacity=0.85),
                    name='Custom gripper',
                    showlegend=n_shown == 0,
                    hoverinfo='skip',
                ))
            else:
                trans_lines = transform_lines(gripper_lines, g)
                for line in trans_lines:
                    fig.add_trace(go.Scatter3d(
                        x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], z=[line[0][2], line[1][2]],
                        mode='lines', line=dict(color=color, width=5), showlegend=False, hoverinfo='skip'
                    ))

            if obj_pairs is not None and len(obj_pairs) > idx:
                c1, c2 = obj_pairs[idx][0], obj_pairs[idx][1]

                points_to_plot = [(c1, 'A'), (c2, 'B')] if show_both_contacts else [(c1, 'A')]
                for cp, tag in points_to_plot:
                    fig.add_trace(go.Scatter3d(
                        x=[cp[0]], y=[cp[1]], z=[cp[2]],
                        mode='markers',
                        marker=dict(color='cyan' if n_shown == 0 else 'yellow',
                                    size=7 if n_shown == 0 else 5, symbol='diamond'),
                        name=f'Contact {tag} - grasp {n_shown+1} obj {obj_id} (Score: {obj_scores[idx]:.4f})'
                    ))
                if show_both_contacts:
                    fig.add_trace(go.Scatter3d(
                        x=[c1[0], c2[0]], y=[c1[1], c2[1]], z=[c1[2], c2[2]],
                        mode='lines', line=dict(color='magenta', width=3), showlegend=False, hoverinfo='skip'
                    ))

            n_shown += 1

        cap_note = f" (capped to top {top_n})" if (top_n is not None and top_n > 0) else ""
        print(f"Object {obj_id}: {n_total} total -> {n_above_score} above score_threshold "
              f"-> {n_floor_clear} clear floor collision -> {n_above_z0} also fully above z=0 "
              f"-> {n_shown} drawn in this scene{cap_note}")

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title='3D Grasp Visualization (Red = Top Grasp, ALL floor-clear + z>=0 grasps shown)'
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_html = os.path.splitext(out_path)[0] + '.html'
    fig.write_html(out_html)
    print(f"Successfully saved {out_html}")


def _reconstruct_scene_mesh(pc, pc_colors, use_bpa=False, camera_location=None):
    points = np.asarray(pc, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10:
        raise ValueError(f"Mesh reconstruction needs at least 10 Nx3 points; got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Mesh reconstruction input contains non-finite points.")

    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(points)
    if pc_colors is not None:
        colors = np.asarray(pc_colors)
        if colors.shape != points.shape or not np.all(np.isfinite(colors)):
            raise ValueError("Mesh colors must be finite RGB values, one row per point.")
        if colors.size and colors.max() > 1.0:
            colors = colors / 255.0
        pcd_mesh.colors = o3d.utility.Vector3dVector(colors)

    extent = float(np.ptp(points, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mesh reconstruction input has no finite spatial extent.")
    voxel_size = max(extent / 80.0, 1e-4)
    pcd_mesh = pcd_mesh.voxel_down_sample(voxel_size)
    if len(pcd_mesh.points) < 10:
        raise RuntimeError("Too few points remain after downsampling to reconstruct a mesh.")

    pcd_mesh.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=30
        )
    )
    if camera_location is None:
        camera_location = np.zeros(3, dtype=np.float64)
    camera_location = np.asarray(camera_location, dtype=np.float64)
    if camera_location.shape != (3,) or not np.all(np.isfinite(camera_location)):
        raise ValueError("Camera location for normal orientation must be a finite 3-vector.")
    pcd_mesh.orient_normals_towards_camera_location(camera_location)
    normals = np.asarray(pcd_mesh.normals)
    if (len(normals) != len(pcd_mesh.points) or not np.all(np.isfinite(normals))
            or np.any(np.linalg.norm(normals, axis=1) < 1e-8)):
        raise RuntimeError("Normal estimation produced invalid normals.")

    if use_bpa:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_mesh,
            o3d.utility.DoubleVector(
                [voxel_size * 2, voxel_size * 4, voxel_size * 8]
            ),
        )
        method_name = "BPA"
    else:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_mesh, depth=9, scale=1.1, linear_fit=False
        )
        densities = np.asarray(densities)
        if len(densities) == len(mesh.vertices):
            low_density = densities < np.quantile(densities, 0.02)
            mesh.remove_vertices_by_mask(low_density)
            mesh.remove_unreferenced_vertices()
        mesh_vertex_cloud = o3d.geometry.PointCloud()
        mesh_vertex_cloud.points = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices)
        )
        support_distances = np.asarray(
            mesh_vertex_cloud.compute_point_cloud_distance(pcd_mesh)
        )
        if len(support_distances) == len(mesh.vertices):
            unsupported = support_distances > voxel_size * 2.5
            mesh.remove_vertices_by_mask(unsupported)
            mesh.remove_unreferenced_vertices()
        mesh = mesh.crop(pcd_mesh.get_axis_aligned_bounding_box())
        method_name = "Poisson"

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) < 10:
        raise RuntimeError(
            f"{method_name} reconstruction produced only {len(triangles)} faces; "
            "cannot display a reliable mesh. Try a denser point cloud."
        )

    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if colors is not None and len(colors) != len(vertices):
        colors = None
    return vertices, triangles, colors


def show_mayavi_scene(pc, pred_grasps_cam, scores, contact_pairs,
                      score_threshold=0.23, min_clearance=0.005, standoff=0.10,
                      vis_floor_normal=None, vis_floor_d=None,
                      require_all_points_above_z0=True, top_n=-1,
                      show_both_contacts=True, pc_colors=None, mesh=False, bpa=False,
                      camera_location=None, gripper_contact_pts=None,
                      gripper_model=None):
    """Open an interactive Mayavi view of the RGB scene and filtered grasps.

    Mayavi is imported here rather than at module import time, so the normal
    HTML-export workflow does not require a working X11/Qt display.
    """
    try:
        from mayavi import mlab
    except ImportError as exc:
        raise RuntimeError(
            "Mayavi is not installed in the active environment. Install it in "
            "the Contact-GraspNet environment, then rerun with --mayavi."
        ) from exc

    print("Opening interactive Mayavi visualization (close its window to finish)...")
    mlab.figure("Contact-GraspNet grasps", bgcolor=(0.08, 0.08, 0.08), size=(1200, 900))
    if mesh:
        try:
            vertices, triangles, vertex_colors = _reconstruct_scene_mesh(
                pc, pc_colors, use_bpa=bpa, camera_location=camera_location
            )
            if vertex_colors is None:
                mlab.triangular_mesh(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                    color=(0.47, 0.71, 1.0), opacity=1.0,
                )
            else:
                vertex_colors = np.clip(
                    np.rint(vertex_colors * 255.0), 0, 255
                ).astype(np.uint8)
                vertex_ids = np.arange(len(vertices), dtype=np.int32)
                surface = mlab.triangular_mesh(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                    scalars=vertex_ids,
                )
                lut_manager = surface.module_manager.scalar_lut_manager
                lut = lut_manager.lut
                lut.number_of_colors = len(vertices)
                lut._vtk_obj.SetTableRange(0, max(len(vertices) - 1, 1))
                lut.table = np.column_stack(
                    (vertex_colors, np.full(len(vertices), 255, dtype=np.uint8))
                ).astype(np.int32)
                lut_manager.data_range = (0.0, float(max(len(vertices) - 1, 1)))
            method_name = "BPA" if bpa else "Poisson"
            print(f"Mayavi {method_name} mesh: {len(vertices)} vertices, {len(triangles)} faces.")
        except Exception as exc:
            print(f"Mayavi mesh reconstruction/rendering failed ({exc}); "
                  "falling back to the colored point cloud.")
            mlab.clf()
            mlab.figure("Contact-GraspNet grasps", bgcolor=(0.08, 0.08, 0.08), size=(1200, 900))
            _draw_mayavi_point_cloud(mlab, pc, pc_colors)
    else:
        _draw_mayavi_point_cloud(mlab, pc, pc_colors)

    gripper_lines = get_gripper_geometry()
    ctrl_pts = gripper_control_points_array()
    if gripper_contact_pts is None:
        gripper_contact_pts = {}
    n_shown = 0
    for obj_id, grasps in pred_grasps_cam.items():
        if len(grasps) == 0:
            continue
        obj_gripper_contacts = gripper_contact_pts.get(obj_id)
        if gripper_model is not None and (
            obj_gripper_contacts is None
            or np.asarray(obj_gripper_contacts).shape != (len(grasps), 3)
        ):
            raise ValueError(
                f"Object {obj_id}: custom gripper visualization requires one contact "
                "point per grasp."
            )
        n_shown_for_object = 0
        for idx in np.argsort(scores[obj_id])[::-1]:
            if scores[obj_id][idx] < score_threshold:
                continue
            g = grasps[idx]
            if gripper_model is not None:
                pts_world = _world_gripper_points(
                    gripper_model.collision_points_local, g[:3, :3],
                    obj_gripper_contacts[idx],
                )
            else:
                pts_world = (g[:3, :3] @ ctrl_pts.T).T + g[:3, 3]
            if vis_floor_normal is not None and vis_floor_d is not None:
                if (pts_world @ vis_floor_normal + vis_floor_d).min() < min_clearance:
                    continue
                if gripper_model is not None:
                    pts_standoff = _world_gripper_points(
                        gripper_model.collision_points_local, g[:3, :3],
                        obj_gripper_contacts[idx], approach_offset=standoff,
                    )
                else:
                    approach_dir = g[:3, 2]
                    pts_standoff = (g[:3, :3] @ ctrl_pts.T).T + (g[:3, 3] - standoff * approach_dir)
                if (pts_standoff @ vis_floor_normal + vis_floor_d).min() < min_clearance:
                    continue
            if require_all_points_above_z0 and pts_world[:, 2].min() < 0.0:
                continue
            if top_n is not None and top_n > 0 and n_shown_for_object >= top_n:
                continue

            color = (1.0, 0.0, 0.0) if n_shown_for_object == 0 else (0.0, 0.9, 0.1)
            if gripper_model is not None:
                gripper_mesh_world = _world_gripper_points(
                    gripper_model.mesh_vertices_local, g[:3, :3],
                    obj_gripper_contacts[idx],
                )
                mlab.triangular_mesh(
                    gripper_mesh_world[:, 0],
                    gripper_mesh_world[:, 1],
                    gripper_mesh_world[:, 2],
                    gripper_model.mesh_faces,
                    color=(1.0, 0.55, 0.05),
                    opacity=0.9,
                )
            else:
                for line in transform_lines(gripper_lines, g):
                    line = np.asarray(line)
                    mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=color,
                                tube_radius=0.0015, tube_sides=12)

            obj_pairs = contact_pairs.get(obj_id) if contact_pairs is not None else None
            if obj_pairs is not None and len(obj_pairs) > idx:
                c1, c2 = obj_pairs[idx]
                mlab.points3d(*c1, color=(0.0, 1.0, 1.0), mode='sphere', scale_factor=0.008)
                if show_both_contacts:
                    mlab.points3d(*c2, color=(1.0, 1.0, 0.0), mode='sphere', scale_factor=0.008)
                    mlab.plot3d([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                                color=(1.0, 0.0, 1.0), tube_radius=0.0008)
            n_shown_for_object += 1
            n_shown += 1

    print(f"Mayavi drew {n_shown} grasp(s).")
    mlab.orientation_axes()
    mlab.axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    mlab.view(azimuth=180, elevation=70, distance='auto')
    mlab.show()


def _draw_mayavi_point_cloud(mlab, pc, pc_colors):
    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.75, 0.75, 0.75),
                      mode='point', scale_factor=0.001)
    else:
        colors = np.asarray(pc_colors)
        if colors.shape != (len(pc), 3):
            raise ValueError(
                f"Expected one RGB color per point ({len(pc)}, 3), got {colors.shape}."
            )
        if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1.0:
            colors = colors * 255.0
        colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
        point_ids = np.arange(len(pc), dtype=np.int32)
        cloud_plot = mlab.points3d(
            pc[:, 0], pc[:, 1], pc[:, 2], point_ids,
            mode='point', scale_factor=0.001,
        )
        lut_manager = cloud_plot.module_manager.scalar_lut_manager
        lut = lut_manager.lut
        lut.number_of_colors = len(pc)
        lut._vtk_obj.SetTableRange(0, max(len(pc) - 1, 1))
        lut.table = np.column_stack(
            (colors, np.full(len(pc), 255, dtype=np.uint8))
        ).astype(np.int32)
        lut_manager.data_range = (0.0, float(max(len(pc) - 1, 1)))


# ----------------------------------------------------------------------------
# Main 3D pipeline
# ----------------------------------------------------------------------------
def process_pcd_scene(sess, grasp_estimator, pcd, eps=0.02, min_points=30,
                       min_clearance=0.005, standoff=0.10, score_threshold=0.23,
                       show_both_contacts=True, out_path="visualisation/PLY_grasps.glb",
                       no_table=False, table_normal_override=None, table_d_override=None,
                       top_n=-1, source_name="point cloud", table_method='robust',
                       require_table=False, show_plane=False, mayavi=False, mesh=False,
                       bpa=False):
    print(f"\n--- Processing {source_name} ---")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"Loaded point cloud with {len(points)} points.")

    if len(points) == 0:
        print("Point cloud is empty!")
        return

    gripper_model = load_gripper_model()
    full_points, full_colors = points, colors
    table_inlier_points = None

    if table_normal_override is not None and table_d_override is not None:
        table_normal = table_normal_override
        table_d = table_d_override
        print(f"Using manual table plane override: normal={table_normal}, d={table_d:.4f}")
        all_dists = points @ table_normal + table_d
        table_inlier_points = points[np.abs(all_dists) < 0.015]
        # Adaptive object mask: use points clearly on the dominant (above-table) side.
        # The "above" side is defined as the side that contains more than half the cloud.
        above_frac = float(np.mean(all_dists > 0))
        sign = 1 if above_frac >= 0.5 else -1          # which sign has the bulk of the cloud?
        clearance = float(np.percentile(np.abs(all_dists), 25)) + 0.005   # adaptive gap
        clearance = max(clearance, 0.008)               # at least 8 mm above the surface
        object_mask = (sign * all_dists) > clearance
        if not object_mask.any():
            print("Warning: adaptive object mask selected 0 points above the table plane. "
                  "Falling back to all points outside the table band (|dist| > 0.005 m).")
            object_mask = np.abs(all_dists) > 0.005
        print(f"Override plane: {object_mask.sum()} points selected as object region "
              f"(adaptive clearance={clearance*1000:.1f} mm, above_frac={above_frac*100:.1f}%)")
        outlier_cloud = pcd.select_by_index(np.where(object_mask)[0])
    elif no_table:
        table_normal, table_d = None, None
        outlier_cloud = pcd
    else:
        table_normal, table_d = None, None
        methods = ['dominant', 'robust'] if table_method == 'dominant' else ['robust', 'dominant']
        for method in methods:
            if table_normal is not None:
                break
            print(f"Detecting table via '{method}' method...")
            if method == 'dominant':
                table_normal, table_d = detect_table_dominant_plane(pcd)
            else:
                table_normal, table_d = detect_table_robust(pcd)

        if table_normal is None:
            if require_table:
                print("ERROR: table plane was not detected. Aborting grasp generation.")
                return
            print("No reliable table plane found -- proceeding without collision filtering or plane overlay.")
            outlier_cloud = pcd
        else:
            all_dists = points @ table_normal + table_d
            # Adaptive object mask -- robust to walls, tilted camera, and unusual scenes.
            # The above-table side is whichever side has less than 95 % of all points
            # (a back wall behind the object would own nearly 100 %).
            above_frac = float(np.mean(all_dists > 0))
            # If the orientation looks backwards (camera is below the table),
            # flip so the smaller, denser cluster is treated as "above".
            if above_frac > 0.90:
                table_normal, table_d = -table_normal, -table_d
                all_dists = -all_dists
                above_frac = 1.0 - above_frac
                print("Flipped plane orientation (too many points on 'above' side; likely a wall).")
            # Use a data-driven clearance: 25th percentile of positive distances + small gap.
            pos_dists = all_dists[all_dists > 0]
            clearance = (float(np.percentile(pos_dists, 25)) + 0.005) if len(pos_dists) > 10 else 0.010
            clearance = max(clearance, 0.008)   # never less than 8 mm
            object_mask = all_dists > clearance
            # Supporting points are the planar band, not everything on the far
            # side of the plane (walls / distant background would fail validation).
            table_inlier_points = points[np.abs(all_dists) < 0.015]
            if not object_mask.any():
                print("Warning: adaptive object mask selected 0 points. "
                      "Falling back to all non-table-band points.")
                object_mask = all_dists > 0.005
            outlier_cloud = pcd.select_by_index(np.where(object_mask)[0])
            print(f"Total points: {len(points)}, above table plane: {object_mask.sum()} "
                  f"({100*object_mask.mean():.1f}%), table-band inliers: {len(table_inlier_points)}, "
                  f"adaptive clearance={clearance*1000:.1f} mm")

    used_virtual_floor = False
    if table_normal is None:
        if require_table:
            return
        table_normal = np.array([0.0, 0.0, 1.0])
        table_d = -points[:, 2].min()
        used_virtual_floor = True
        print(f"No real table detected -- using virtual floor plane at object z_min "
              f"({points[:, 2].min():.4f}). Collision filtering will be disabled (virtual floor).")

    floor_is_solid = validate_floor_plane(
        table_normal, table_d,
        table_points=table_inlier_points, is_virtual=used_virtual_floor,
        distance_threshold=0.015, min_inlier_frac=0.70, min_points=200
    )
    if require_table and not floor_is_solid:
        print("ERROR: detected table plane failed validation. Aborting grasp generation.")
        return

    if len(outlier_cloud.points) == 0:
        print("Nothing left for object detection after excluding the table plane!")
        return

    # ---- Outlier removal (Issue 4 fix) ------------------------------------------
    # Statistical outlier removal is skipped on small / thin-object clouds where it
    # would delete real geometric structure before DBSCAN.  For larger clouds a
    # gentler std_ratio is used and the number of removed points is always reported.
    n_before = len(outlier_cloud.points)
    outlier_cloud_pre_filter = outlier_cloud   # keep an explicit reference for recovery
    _OUTLIER_MIN_CLOUD = 500   # skip if too few points to be statistically meaningful
    if n_before >= _OUTLIER_MIN_CLOUD:
        cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        outlier_cloud = outlier_cloud.select_by_index(ind)
        n_after = len(outlier_cloud.points)
        n_removed = n_before - n_after
        print(f"Statistical outlier removal: {n_before} pts to {n_after} pts "
              f"({n_removed} removed, std_ratio=2.5).")
        if n_after == 0:
            print("Warning: outlier removal deleted ALL points! Reverting to the "
                  "pre-removal cloud (no outlier filtering for this run).")
            outlier_cloud = outlier_cloud_pre_filter   # always valid; set just above
    else:
        print(f"Skipping statistical outlier removal: cloud has only {n_before} pts "
              f"(threshold is {_OUTLIER_MIN_CLOUD}). Thin-object geometry preserved for DBSCAN.")

    pc_segments = segment_objects_3d(outlier_cloud, eps=eps, min_points=min_points)
    if len(pc_segments) == 0:
        print(f"No object clusters found after DBSCAN "
              f"(eps={eps}, min_points={min_points}, cloud size={len(outlier_cloud.points)}). "
              "Try lowering --eps or --min_points, or check that the table was correctly removed.")
        return

    object_points_for_grasping = np.asarray(outlier_cloud.points)

    print("Generating Grasps...")
    pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        sess, object_points_for_grasping, pc_segments=pc_segments, local_regions=True,
        filter_grasps=True, forward_passes=1
    )

    for obj_id, grasps in pred_grasps_cam.items():
        grasp_count = len(grasps)
        score_values = np.asarray(scores[obj_id])
        contact_values = np.asarray(contact_pts[obj_id])
        opening_values = np.asarray(gripper_openings[obj_id])
        expected_sizes = (
            (score_values, grasp_count, "scores"),
            (contact_values, grasp_count * 3, "contact points"),
            (opening_values, grasp_count, "predicted openings"),
        )
        for array, expected_size, label in expected_sizes:
            if array.size != expected_size:
                raise ValueError(
                    f"Object {obj_id}: expected {expected_size} values for {label} "
                    f"across {grasp_count} grasps, got shape {array.shape}."
                )
        scores[obj_id] = score_values.reshape(grasp_count)
        contact_pts[obj_id] = contact_values.reshape(grasp_count, 3)
        gripper_openings[obj_id] = opening_values.reshape(grasp_count)

    for obj_id in scores:
        print(f"Object {obj_id}: {np.sum(scores[obj_id] > score_threshold)} grasps above "
              f"{score_threshold} ({len(scores[obj_id])} total predicted)")

    contact_pairs = snap_contact_pairs(pred_grasps_cam, contact_pts, gripper_openings, pc_segments)

    if floor_is_solid:
        # ---- Step 1: hard filter -- remove grasps that deeply collide with the table
        # and cannot plausibly be cleared by a small rotation (clearance < -standoff/2).
        # These are intrinsically bad poses regardless of tilt correction.
        HARD_COLLISION_THRESH = -standoff / 2.0  # deeply buried; no tilt will help
        (pred_grasps_cam, scores,
         contact_pts, gripper_openings,
         contact_pairs) = filter_table_colliding_grasps(
            pred_grasps_cam, scores, contact_pts, gripper_openings,
            table_normal, table_d,
            min_clearance=HARD_COLLISION_THRESH,   # negative: only reject deeply buried grasps
            standoff=standoff,
            contact_pairs=contact_pairs,
            gripper_model=gripper_model,
        )

        # ---- Step 2: for grasps that only slightly clip the table, try tilting them clear.
        pred_grasps_cam = resolve_floor_collisions_by_rotation(
            pred_grasps_cam, table_normal, table_d,
            min_clearance=min_clearance, standoff=standoff,
            contact_pts=contact_pts, gripper_openings=gripper_openings,
            gripper_model=gripper_model,
        )
        # Rotation may fail to resolve a collision. Never save those grasps as
        # if they were safe; recheck final pose and approach clearance together.
        (pred_grasps_cam, scores,
         contact_pts, gripper_openings,
         contact_pairs) = filter_table_colliding_grasps(
            pred_grasps_cam, scores, contact_pts, gripper_openings,
            table_normal, table_d,
            min_clearance=min_clearance,
            standoff=standoff,
            contact_pairs=contact_pairs,
            gripper_model=gripper_model,
        )
        print("Applied final table-clearance validation after rotation correction.")
    else:
        print("Skipping floor-collision filtering entirely -- floor plane "
              "did not pass validation (virtual or untrustworthy), "
              "so grasp poses below are UNCHECKED against it and will be marked as such.")

    print("\n" + "=" * 70)
    print("FINAL GRASP POSES (world frame, post floor-collision correction)")
    print("=" * 70)
    grasp_pose_records = []
    for obj_id in pred_grasps_cam:
        grasps = pred_grasps_cam[obj_id]
        obj_scores = scores[obj_id]
        if len(grasps) == 0:
            continue
        sorted_idx = np.argsort(obj_scores)[::-1]
        top3_rank = 0
        for idx in sorted_idx:
            if top3_rank >= 3:
                break
            if obj_scores[idx] < score_threshold:
                continue
            R, t = grasps[idx][:3, :3], grasps[idx][:3, 3]
            print(f"\nObject {obj_id}, grasp rank {top3_rank+1}:")
            print(f"  Rotation R =\n{R}")
            print(f"  Translation t = {t}")
            grasp_pose_records.append({
                'obj_id': obj_id,
                'rank': top3_rank,
                'R': R.copy(),
                't': t.copy(),
                'collision_checked': floor_is_solid,
            })
            top3_rank += 1
    print("=" * 70 + "\n")

    if grasp_pose_records:
        poses_out_path = os.path.splitext(out_path)[0] + "_grasp_poses.npz"
        os.makedirs(os.path.dirname(poses_out_path) or ".", exist_ok=True)
        np.savez(poses_out_path,
                 obj_ids=np.array([r['obj_id'] for r in grasp_pose_records]),
                 ranks=np.array([r['rank'] for r in grasp_pose_records]),
                 rotations=np.stack([r['R'] for r in grasp_pose_records]),
                 translations=np.stack([r['t'] for r in grasp_pose_records]),
                 table_collision_checked=np.array(
                     [r['collision_checked'] for r in grasp_pose_records], dtype=bool
                 ),
                 minimum_table_clearance_m=np.array(min_clearance, dtype=np.float64),
                 gripper_width_checked=np.array(False, dtype=bool),
                 predicted_gripper_width_checked=np.array(False, dtype=bool),
                 fixed_gripper_opening_m=np.array(
                     _GRIPPER_FIXED_OPENING_M, dtype=np.float64
                 ),
                 gripper_asset_measured_opening_m=np.array(
                     _GRIPPER_ASSET_OPENING_M, dtype=np.float64
                 ),
                 collision_gripper_model=np.array(os.path.basename(_GRIPPER_OPEN_STL_PATH)),
                 gripper_asset_scale_m_per_unit=np.array(
                     _GRIPPER_ASSET_SCALE, dtype=np.float64
                 ),
                 gripper_contact_anchor_source_mm=_GRIPPER_CONTACT_ANCHOR_MM.copy(),
                 gripper_opposite_contact_source_mm=_GRIPPER_OPPOSITE_CONTACT_MM.copy(),
                 gripper_source_to_cgn=_GRIPPER_SOURCE_TO_CGN.copy())
        print(f"Saved {len(grasp_pose_records)} grasp pose(s) (R, t) to {poses_out_path}; "
              f"open STL table collision checked: {floor_is_solid}; "
              "predicted jaw widths were not checked.")

    z_offset = -full_points[:, 2].min()
    shift = np.array([0.0, 0.0, z_offset])
    full_points = full_points + shift
    if table_inlier_points is not None:
        table_inlier_points = table_inlier_points + shift
    for obj_id in pred_grasps_cam:
        if len(pred_grasps_cam[obj_id]):
            pred_grasps_cam[obj_id] = pred_grasps_cam[obj_id].copy()
            pred_grasps_cam[obj_id][:, :3, 3] += shift
        if len(contact_pts.get(obj_id, [])):
            contact_pts[obj_id] = contact_pts[obj_id] + shift
        if contact_pairs.get(obj_id) is not None and len(contact_pairs[obj_id]):
            contact_pairs[obj_id] = contact_pairs[obj_id] + shift
    print(f"Shifted visualization by {z_offset:.4f} m along z so object rests at z=0 (grasp generation used original depth).")

    vis_floor_normal = np.array([0.0, 0.0, 1.0])
    vis_floor_d = 0.0
    export_3d_scene(full_points, full_colors, pred_grasps_cam, scores, contact_pairs,
                     out_path,
                     table_normal=None if (used_virtual_floor or not floor_is_solid) else table_normal,
                     table_d=None if (used_virtual_floor or not floor_is_solid) else table_d,
                     show_both_contacts=show_both_contacts, score_threshold=score_threshold,
                     min_clearance=min_clearance, standoff=standoff,
                     vis_floor_normal=vis_floor_normal, vis_floor_d=vis_floor_d,
                     table_inlier_points=table_inlier_points,
                     require_all_points_above_z0=True,
                     top_n=top_n, show_plane=show_plane,
                     camera_location=shift,
                     gripper_contact_pts=contact_pts, gripper_model=gripper_model)
    if mayavi:
        show_mayavi_scene(full_points, pred_grasps_cam, scores, contact_pairs,
                          score_threshold=score_threshold, min_clearance=min_clearance,
                          standoff=standoff, vis_floor_normal=vis_floor_normal,
                          vis_floor_d=vis_floor_d, require_all_points_above_z0=True,
                          top_n=top_n, show_both_contacts=show_both_contacts,
                          pc_colors=full_colors, mesh=mesh, bpa=bpa,
                          camera_location=shift,
                          gripper_contact_pts=contact_pts, gripper_model=gripper_model)


def process_ply_scene(sess, grasp_estimator, ply_path, eps=0.02, min_points=30,
                       min_clearance=0.005, standoff=0.10, score_threshold=0.23,
                       show_both_contacts=True, out_path="visualisation/PLY_grasps.glb",
                       no_table=False, table_normal_override=None, table_d_override=None,
                       top_n=-1, table_method='robust', require_table=False,
                       show_plane=False, mayavi=False, mesh=False, bpa=False):
    pcd = o3d.io.read_point_cloud(ply_path)
    process_pcd_scene(sess, grasp_estimator, pcd, eps=eps, min_points=min_points,
                      min_clearance=min_clearance, standoff=standoff,
                      score_threshold=score_threshold,
                      show_both_contacts=show_both_contacts, out_path=out_path,
                      no_table=no_table, table_normal_override=table_normal_override,
                      table_d_override=table_d_override, top_n=top_n,
                      source_name=ply_path, table_method=table_method,
                      require_table=require_table, show_plane=show_plane, mayavi=mayavi,
                      mesh=mesh, bpa=bpa)


def main():
    parser = argparse.ArgumentParser(description="3D point-cloud-based Contact-GraspNet inference "
                                                  "(table-plane detection + DBSCAN segmentation + "
                                                  "table-collision filtering)")
    parser.add_argument('--ply', type=str, required=True, help='Path to an input .ply point cloud file')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/scene_test_2048_bs3_hor_sigma_001',
                         help='Path to CGN checkpoint directory')

    parser.add_argument('--eps', type=float, default=0.02,
                         help='DBSCAN neighborhood radius (m) for object clustering')
    parser.add_argument('--min_points', type=int, default=30,
                         help='DBSCAN minimum points per object cluster')

    parser.add_argument('--min_clearance', type=float, default=0.005,
                         help='Minimum allowed gripper-to-table clearance (m)')
    parser.add_argument('--standoff', type=float, default=0.10,
                         help='Distance (m) to check back along the approach axis for table collisions')

    parser.add_argument('--score_threshold', type=float, default=0.23,
                         help='Minimum grasp confidence score to display (CGN paper default: 0.23)')
    parser.add_argument('--single_contact', action='store_true',
                         help='Show only one contact point per grasp instead of both antipodal points')
    parser.add_argument('--top_n', type=int, default=-1,
                         help='Show only the top N qualifying grasps (by score) in the 3D visualization. '
                              'Default -1 (or 0) shows ALL grasps that clear the floor and z>=0 checks.')

    parser.add_argument('--out', type=str, default='visualisation/PLY_grasps.glb',
                         help='Output path for the exported 3D visualization (.html will be written alongside)')

    parser.add_argument('--no_table', action='store_true',
                         help='Skip table plane detection entirely (no collision filtering or plane overlay)')
    parser.add_argument('--table_normal', type=str, default=None,
                         help='Manual table plane normal override, e.g. "0,0,1" (skips auto-detection)')
    parser.add_argument('--table_d', type=float, default=None,
                         help='Manual table plane offset d override (used with --table_normal)')

    parser.add_argument('--show_plane', action='store_true',
                         help='Draw the z=0 floor plane and detected table plane in the 3D visualization '
                              '(hidden by default)')
    parser.add_argument('--mayavi', action='store_true',
                        help='Also open an interactive Mayavi window (requires X11 forwarding on a server)')
    parser.add_argument('--mesh', action='store_true',
                        help='Display a mesh instead of points in Mayavi (Screened Poisson by default); requires --mayavi')
    parser.add_argument('--bpa', action='store_true',
                        help='Use Ball Pivoting instead of Screened Poisson for the Mayavi mesh; requires --mesh')

    parser.add_argument('--table_method', type=str, default='dominant',
                         choices=['dominant', 'robust'],
                         help='Table detection strategy: "dominant" (default) fits the largest RANSAC plane '
                              '-- best for typical tabletop RGB-D captures; '
                              '"robust" finds the main object cluster first then fits the remainder '
                              '-- better when the object is large and clearly separated from the table.')
    parser.add_argument('--require_table', action='store_true',
                         help='Abort grasp generation if no valid table plane can be detected or validated. '
                              'By default the script continues without collision filtering in that case.')

    args = parser.parse_args()

    if args.mesh and not args.mayavi:
        parser.error("--mesh requires --mayavi.")
    if args.bpa and not args.mesh:
        parser.error("--bpa requires --mesh.")

    # ---- Manual plane override validation ----------------------------------------
    # Both or neither must be supplied; one without the other is always a user error.
    if (args.table_normal is None) != (args.table_d is None):
        parser.error(
            "--table_normal and --table_d must always be used together. "
            "You supplied only one of the two; please re-run with both or neither."
        )

    table_normal_override = None
    if args.table_normal is not None:
        try:
            raw_normal = np.array(
                [float(v.strip()) for v in args.table_normal.split(',')],
                dtype=np.float64,
            )
        except ValueError:
            parser.error("--table_normal must contain exactly three comma-separated finite numbers.")
        if raw_normal.shape != (3,) or not np.all(np.isfinite(raw_normal)):
            parser.error("--table_normal must contain exactly three comma-separated finite numbers.")
        norm_len = np.linalg.norm(raw_normal)
        if not np.isfinite(norm_len) or norm_len < 1e-6:
            parser.error(
                f"--table_normal {args.table_normal!r} has zero, near-zero, or non-finite magnitude. "
                "Please supply a finite non-zero normal."
            )
        table_normal_override = raw_normal / norm_len
        if not np.isfinite(args.table_d):
            parser.error("--table_d must be finite.")

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
    grasp_estimator.load_weights(sess, saver, args.ckpt_dir, mode='test')

    process_ply_scene(sess, grasp_estimator, args.ply,
                       eps=args.eps, min_points=args.min_points,
                       min_clearance=args.min_clearance, standoff=args.standoff,
                       score_threshold=args.score_threshold,
                       show_both_contacts=not args.single_contact,
                       out_path=args.out,
                       no_table=args.no_table,
                       table_normal_override=table_normal_override,
                       table_d_override=args.table_d,
                       top_n=args.top_n,
                       table_method=args.table_method,
                       require_table=args.require_table,
                       show_plane=args.show_plane,
                       mayavi=args.mayavi,
                       mesh=args.mesh,
                       bpa=args.bpa)


if __name__ == '__main__':
    main()
