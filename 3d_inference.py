import os
import sys
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
    """
    points = np.asarray(pcd.points)
    labels = np.array(pcd.cluster_dbscan(eps=cluster_eps, min_points=min_object_cluster_points,
                                          print_progress=False))
    if labels.max() < 0:
        print("Could not find a dense object cluster; skipping table detection.")
        return None, None

    counts = np.bincount(labels[labels >= 0])
    main_cluster_label = np.argmax(counts)
    object_mask = labels == main_cluster_label
    remainder_mask = ~object_mask

    remainder_points = points[remainder_mask]
    if len(remainder_points) < remainder_min_points:
        print(f"Only {len(remainder_points)} points remain outside the main object cluster -- "
              f"not enough to reliably fit a table plane. Table detection skipped.")
        return None, None

    remainder_pcd = pcd.select_by_index(np.where(remainder_mask)[0])
    plane_model, inliers = remainder_pcd.segment_plane(distance_threshold=distance_threshold,
                                                        ransac_n=ransac_n, num_iterations=num_iterations)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal, d = normal / np.linalg.norm(normal), d / np.linalg.norm(normal)

    obj_dists = points[object_mask] @ normal + d
    if np.mean(obj_dists) < 0:
        normal, d = -normal, -d

    print(f"Table plane fit from {len(inliers)}/{len(remainder_points)} points "
          f"outside the main object cluster ({object_mask.sum()} object points excluded from fitting)")
    return normal, d

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


# ----------------------------------------------------------------------------
# Table-collision + approach-path filtering
# ----------------------------------------------------------------------------
def filter_table_colliding_grasps(pred_grasps_cam, scores, contact_pts, gripper_openings,
                                   table_normal, table_d, min_clearance=0.005, standoff=0.10,
                                   contact_pairs=None):
    """
    Rejects any grasp where the gripper's own geometry -- at the final grasp
    pose, OR while retreating 'standoff' meters back along its approach axis --
    would come closer than min_clearance to the table plane.

    min_clearance: minimum allowed distance (m) from any gripper point to the
                   table plane. A small positive margin (not 0) leaves room for
                   pose/sensor noise.
    standoff: how far back (m) along the approach axis to also check, as a
              cheap approximation of the straight-line approach/retreat path.
    contact_pairs: optional dict {obj_id: (N,2,3)} from snap_contact_pairs;
                   filtered in parallel with grasps so indices stay aligned.
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

        keep_idx = []
        for i, g in enumerate(grasps):
            R, t = g[:3, :3], g[:3, 3]

            # Gripper geometry at the final grasp pose
            pts_world = (R @ control_points.T).T + t
            if (pts_world @ table_normal + table_d).min() < min_clearance:
                continue

            # Gripper geometry retreated along its own approach axis (local z)
            approach_dir = R[:, 2]
            pts_standoff = (R @ control_points.T).T + (t - standoff * approach_dir)
            if (pts_standoff @ table_normal + table_d).min() < min_clearance:
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


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def snap_contact_pairs(pred_grasps_cam, contact_pts, gripper_openings, pc_segments,
                       tol=0.004, search_multiplier=1.6):
    """
    For each grasp, finds the second contact point by intersecting the gripper's
    closing axis with the object's own point cloud -- i.e. finds where that ray
    actually exits the real surface -- instead of trusting CGN's predicted width
    directly. Falls back to the predicted-width point only if no real surface
    point is found along the ray (e.g. sparse/noisy region).
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

            c2 = c1 + predicted_width * baseline_dir  # fallback if no surface point found

            if obj_pc is not None and len(obj_pc) > 0:
                rel = obj_pc - c1
                t = rel @ baseline_dir                          # distance along the ray
                perp = rel - np.outer(t, baseline_dir)
                perp_dist = np.linalg.norm(perp, axis=1)         # distance FROM the ray

                search_max = predicted_width * search_multiplier
                candidates = (t > 0.001) & (t < search_max) & (perp_dist < tol)
                if np.any(candidates):
                    best_idx = np.argmax(t[candidates])  # farthest hit within tolerance = far surface
                    c2 = c1 + t[candidates][best_idx] * baseline_dir

            pairs.append((c1, c2))

        contact_pairs[obj_id] = np.array(pairs) if len(pairs) else np.zeros((0, 2, 3))
    return contact_pairs


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def export_3d_scene(pc, pc_colors, pred_grasps_cam, scores, contact_pairs,
                     out_path, table_normal=None, table_d=None, show_both_contacts=True,
                     score_threshold=0.23, top_n=3, table_inlier_points=None):
    print(f"Exporting 3D scene to {out_path}...")

    if len(pc) == 0:
        print("Warning: point cloud is empty, skipping export.")
        return

    fig = go.Figure()

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

    # Draw the table as a SOLID plane, sized to where the table points actually are
    # (not the whole cloud's bounding box -- avoids the plane looking misplaced/floating)
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

    for obj_id in pred_grasps_cam.keys():
        grasps = pred_grasps_cam[obj_id]
        if len(grasps) == 0:
            continue

        obj_scores = scores[obj_id]
        obj_pairs = contact_pairs.get(obj_id, None) if contact_pairs is not None else None
        sorted_indices = np.argsort(obj_scores)[::-1]

        n_shown = 0
        for idx in sorted_indices:
            if obj_scores[idx] < score_threshold:
                continue
            if n_shown >= top_n:
                break

            g = grasps[idx]
            color = 'red' if n_shown == 0 else 'green'

            trans_lines = transform_lines(gripper_lines, g)
            for line in trans_lines:
                fig.add_trace(go.Scatter3d(
                    x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], z=[line[0][2], line[1][2]],
                    mode='lines', line=dict(color=color, width=5), showlegend=False, hoverinfo='skip'
                ))

            if obj_pairs is not None and len(obj_pairs) > idx:
                # c1, c2 come directly from snap_contact_pairs -- real geometry, no predicted-width guess
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

            print(f"Added grasp {n_shown+1} for object {obj_id} (score {obj_scores[idx]:.4f})")
            n_shown += 1

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title='3D Grasp Visualization (Red = Top Grasp, table plane shown, collision-filtered)'
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_html = out_path.replace('.glb', '.html')
    fig.write_html(out_html)
    print(f"Successfully saved {out_html}")


# ----------------------------------------------------------------------------
# Main 3D pipeline
# ----------------------------------------------------------------------------
def process_ply_scene(sess, grasp_estimator, ply_path, eps=0.02, min_points=30,
                       min_clearance=0.005, standoff=0.10, score_threshold=0.23,
                       show_both_contacts=True, out_path="visualisation/PLY_grasps.glb",
                       no_table=False, table_normal_override=None, table_d_override=None):
    print(f"\n--- Processing {ply_path} ---")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"Loaded point cloud with {len(points)} points.")

    if len(points) == 0:
        print("Point cloud is empty!")
        return

    # Keep the FULL original cloud untouched -- this is what gets visualized
    full_points, full_colors = points, colors
    table_inlier_points = None

    if table_normal_override is not None and table_d_override is not None:
        # Manual override: trust the user-supplied plane, skip all automatic detection
        table_normal = table_normal_override
        table_d = table_d_override
        print(f"Using manual table plane override: normal={table_normal}, d={table_d:.4f}")
        outlier_cloud = pcd  # no points to remove -- let DBSCAN handle object separation
    elif no_table:
        table_normal, table_d = None, None
        outlier_cloud = pcd
    else:
        table_normal, table_d = detect_table_robust(pcd)
        if table_normal is None:
            print("No reliable table plane found -- proceeding without collision filtering or plane overlay.")
            outlier_cloud = pcd
        else:
            # Remove points near the table so DBSCAN sees only object points
            all_dists = points @ table_normal + table_d
            object_mask = all_dists > 0.01  # keep points clearly above the plane
            table_inlier_points = points[~object_mask]
            outlier_cloud = pcd.select_by_index(np.where(object_mask)[0])
            print(f"Total points: {len(points)}, above table plane: {object_mask.sum()} "
                  f"({100*object_mask.mean():.1f}%)")

    if len(outlier_cloud.points) == 0:
        print("Nothing left for object detection after excluding the table plane!")
        return
    cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    outlier_cloud = outlier_cloud.select_by_index(ind)

    pc_segments = segment_objects_3d(outlier_cloud, eps=eps, min_points=min_points)
    if len(pc_segments) == 0:
        print("No object clusters found after DBSCAN!")
        return

    # predict_scene_grasps needs the non-table object cloud
    object_points_for_grasping = np.asarray(outlier_cloud.points)

    print("Generating Grasps...")
    pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        sess, object_points_for_grasping, pc_segments=pc_segments, local_regions=True,
        filter_grasps=True, forward_passes=1
    )

    for obj_id in scores:
        print(f"Object {obj_id}: {np.sum(scores[obj_id] > score_threshold)} grasps above "
              f"{score_threshold} ({len(scores[obj_id])} total predicted)")

    # Snap c2 to the real object surface geometry instead of trusting predicted width
    contact_pairs = snap_contact_pairs(pred_grasps_cam, contact_pts, gripper_openings, pc_segments)

    if table_normal is not None:
        pred_grasps_cam, scores, contact_pts, gripper_openings, contact_pairs = \
            filter_table_colliding_grasps(
                pred_grasps_cam, scores, contact_pts, gripper_openings,
                table_normal, table_d, min_clearance=min_clearance, standoff=standoff,
                contact_pairs=contact_pairs
            )

    # Visualization uses the FULL untouched cloud -- table plane drawn as a solid overlay,
    # grasps + geometry-snapped contact pairs marked on top
    export_3d_scene(full_points, full_colors, pred_grasps_cam, scores, contact_pairs,
                     out_path, table_normal=table_normal, table_d=table_d,
                     show_both_contacts=show_both_contacts, score_threshold=score_threshold,
                     table_inlier_points=table_inlier_points)


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

    parser.add_argument('--out', type=str, default='visualisation/PLY_grasps.glb',
                         help='Output path for the exported 3D visualization (.html will be written alongside)')

    parser.add_argument('--no_table', action='store_true',
                         help='Skip table plane detection entirely (no collision filtering or plane overlay)')
    parser.add_argument('--table_normal', type=str, default=None,
                         help='Manual table plane normal override, e.g. "0,0,1" (skips auto-detection)')
    parser.add_argument('--table_d', type=float, default=None,
                         help='Manual table plane offset d override (used with --table_normal)')

    args = parser.parse_args()

    # Parse manual table override if provided
    table_normal_override = None
    if args.table_normal is not None and args.table_d is not None:
        table_normal_override = np.array([float(v) for v in args.table_normal.split(',')])
        table_normal_override = table_normal_override / np.linalg.norm(table_normal_override)

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
                       table_d_override=args.table_d)


if __name__ == '__main__':
    main()