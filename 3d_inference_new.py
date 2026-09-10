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
    z_min): there are no real points to check against -- it's flat and
    solid by construction, so it always passes, but this is reported
    explicitly so it's never confused with a genuinely measured surface.

    Returns True/False and prints a clear, unambiguous verdict either way.
    """
    if is_virtual:
        print(f"Floor plane check: VIRTUAL floor (normal={table_normal}, d={table_d:.4f}) -- "
              f"flat and solid by construction, not measured from real points.")
        return True

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
                                   contact_pairs=None):
    """
    Rejects any grasp where the gripper's own geometry -- at the final grasp
    pose, OR while retreating 'standoff' meters back along its approach axis --
    would come closer than min_clearance to the table plane.
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

            pts_world = (R @ control_points.T).T + t
            if (pts_world @ table_normal + table_d).min() < min_clearance:
                continue

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


def _rotation_about_local_x(theta):
    """ Rotation matrix about the LOCAL x-axis (baseline/closing direction). """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])


def resolve_floor_collisions_by_rotation(pred_grasps_cam, table_normal, table_d,
                                          min_clearance=0.005, standoff=0.10,
                                          max_angle_deg=180.0, angle_step_deg=0.5):
    """
    For any grasp whose gripper geometry collides with the floor plane, tilts
    the gripper by rotating about its OWN local x-axis (the baseline/closing
    direction) until it clears the floor -- instead of discarding the grasp.
    Grasps that can't be resolved within max_angle_deg are left as-is.
    """
    control_points = gripper_control_points_array()
    angles = np.arange(angle_step_deg, max_angle_deg + 1e-6, angle_step_deg)

    def min_clearance_for(R, t):
        pts_world = (R @ control_points.T).T + t
        c_final = (pts_world @ table_normal + table_d).min()
        approach_dir = R[:, 2]
        pts_standoff = (R @ control_points.T).T + (t - standoff * approach_dir)
        c_standoff = (pts_standoff @ table_normal + table_d).min()
        return min(c_final, c_standoff)

    adjusted_grasps = {}
    for obj_id, grasps in pred_grasps_cam.items():
        grasps_out = grasps.copy() if len(grasps) else grasps
        n_ok, n_adjusted, n_unresolved = 0, 0, 0

        for i, g in enumerate(grasps):
            R, t = g[:3, :3], g[:3, 3]

            if min_clearance_for(R, t) >= min_clearance:
                n_ok += 1
                continue

            resolved = False
            for deg in angles:
                for sign in (1, -1):
                    R_test = R @ _rotation_about_local_x(np.radians(sign * deg))
                    if min_clearance_for(R_test, t) >= min_clearance:
                        grasps_out[i, :3, :3] = R_test
                        resolved = True
                        n_adjusted += 1
                        break
                if resolved:
                    break

            if not resolved:
                n_unresolved += 1

        adjusted_grasps[obj_id] = grasps_out
        print(f"Object {obj_id}: {n_ok} clear, {n_adjusted} tilted to clear the floor, "
              f"{n_unresolved} unresolved within {max_angle_deg} deg (left unchanged, none discarded)")

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
                     top_n=-1):
    print(f"Exporting 3D scene to {out_path}...")

    if len(pc) == 0:
        print("Warning: point cloud is empty, skipping export.")
        return

    fig = go.Figure()

    mesh_added = False
    try:
        pc_vis = pc
        colors_vis = pc_colors

        pcd_mesh = o3d.geometry.PointCloud()
        pcd_mesh.points = o3d.utility.Vector3dVector(pc_vis)
        if colors_vis is not None and len(colors_vis) == len(pc_vis):
            cv = colors_vis if np.max(colors_vis) <= 1.0 else colors_vis / 255.0
            pcd_mesh.colors = o3d.utility.Vector3dVector(cv)

        voxel_size = (np.asarray(pcd_mesh.points).ptp(axis=0).max()) / 80
        pcd_mesh = pcd_mesh.voxel_down_sample(voxel_size=max(voxel_size, 1e-4))
        pcd_mesh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30))
        pcd_mesh.orient_normals_consistent_tangent_plane(k=15)

        pts_ds = np.asarray(pcd_mesh.points)
        radii = [voxel_size * 2, voxel_size * 4, voxel_size * 8]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_mesh, o3d.utility.DoubleVector(radii)
        )

        triangles = np.asarray(mesh.triangles)
        vertices  = np.asarray(mesh.vertices)
        vtx_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

        if len(triangles) >= 10:
            if vtx_colors is not None and len(vtx_colors) == len(vertices):
                vc255 = (vtx_colors * 255).astype(int)
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
        else:
            print(f"BPA produced only {len(triangles)} face(s) -- falling back to point cloud.")
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

    for obj_id in pred_grasps_cam.keys():
        grasps = pred_grasps_cam[obj_id]
        if len(grasps) == 0:
            continue

        obj_scores = scores[obj_id]
        obj_pairs = contact_pairs.get(obj_id, None) if contact_pairs is not None else None
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
            pts_world = (R_g @ ctrl_pts.T).T + t_g

            if vis_floor_normal is not None and vis_floor_d is not None:
                if (pts_world @ vis_floor_normal + vis_floor_d).min() < min_clearance:
                    continue
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
                       no_table=False, table_normal_override=None, table_d_override=None,
                       top_n=-1):
    print(f"\n--- Processing {ply_path} ---")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"Loaded point cloud with {len(points)} points.")

    if len(points) == 0:
        print("Point cloud is empty!")
        return

    full_points, full_colors = points, colors
    table_inlier_points = None

    if table_normal_override is not None and table_d_override is not None:
        table_normal = table_normal_override
        table_d = table_d_override
        print(f"Using manual table plane override: normal={table_normal}, d={table_d:.4f}")
        outlier_cloud = pcd
    elif no_table:
        table_normal, table_d = None, None
        outlier_cloud = pcd
    else:
        table_normal, table_d = detect_table_robust(pcd)
        if table_normal is None:
            print("No reliable table plane found -- proceeding without collision filtering or plane overlay.")
            outlier_cloud = pcd
        else:
            all_dists = points @ table_normal + table_d
            object_mask = all_dists > 0.01
            table_inlier_points = points[~object_mask]
            outlier_cloud = pcd.select_by_index(np.where(object_mask)[0])
            print(f"Total points: {len(points)}, above table plane: {object_mask.sum()} "
                  f"({100*object_mask.mean():.1f}%)")

    used_virtual_floor = False
    if table_normal is None:
        table_normal = np.array([0.0, 0.0, 1.0])
        table_d = -points[:, 2].min()
        used_virtual_floor = True
        print(f"No real table detected -- using virtual floor plane at object z_min "
              f"({points[:, 2].min():.4f}) for gripper-collision filtering.")

    floor_is_solid = validate_floor_plane(
        table_normal, table_d,
        table_points=table_inlier_points, is_virtual=used_virtual_floor
    )

    if len(outlier_cloud.points) == 0:
        print("Nothing left for object detection after excluding the table plane!")
        return
    cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    outlier_cloud = outlier_cloud.select_by_index(ind)

    pc_segments = segment_objects_3d(outlier_cloud, eps=eps, min_points=min_points)
    if len(pc_segments) == 0:
        print("No object clusters found after DBSCAN!")
        return

    object_points_for_grasping = np.asarray(outlier_cloud.points)

    print("Generating Grasps...")
    pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        sess, object_points_for_grasping, pc_segments=pc_segments, local_regions=True,
        filter_grasps=True, forward_passes=1
    )

    for obj_id in scores:
        print(f"Object {obj_id}: {np.sum(scores[obj_id] > score_threshold)} grasps above "
              f"{score_threshold} ({len(scores[obj_id])} total predicted)")

    contact_pairs = snap_contact_pairs(pred_grasps_cam, contact_pts, gripper_openings, pc_segments)

    if floor_is_solid:
        pred_grasps_cam = resolve_floor_collisions_by_rotation(
            pred_grasps_cam, table_normal, table_d,
            min_clearance=min_clearance, standoff=standoff
        )
    else:
        print("Skipping floor-collision rotation correction entirely -- floor plane "
              "did not pass validation, so grasp poses below are UNCHECKED against it.")

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
            grasp_pose_records.append({'obj_id': obj_id, 'rank': top3_rank, 'R': R.copy(), 't': t.copy()})
            top3_rank += 1
    print("=" * 70 + "\n")

    if grasp_pose_records:
        poses_out_path = os.path.splitext(out_path)[0] + "_grasp_poses.npz"
        os.makedirs(os.path.dirname(poses_out_path), exist_ok=True)
        np.savez(poses_out_path,
                 obj_ids=np.array([r['obj_id'] for r in grasp_pose_records]),
                 ranks=np.array([r['rank'] for r in grasp_pose_records]),
                 rotations=np.stack([r['R'] for r in grasp_pose_records]),
                 translations=np.stack([r['t'] for r in grasp_pose_records]))
        print(f"Saved {len(grasp_pose_records)} grasp pose(s) (R, t) to {poses_out_path}")

    z_offset = -full_points[:, 2].min()
    shift = np.array([0.0, 0.0, z_offset])
    full_points = full_points + shift
    if table_inlier_points is not None:
        table_inlier_points = table_inlier_points + shift
    for obj_id in pred_grasps_cam:
        if len(pred_grasps_cam[obj_id]):
            pred_grasps_cam[obj_id] = pred_grasps_cam[obj_id].copy()
            pred_grasps_cam[obj_id][:, :3, 3] += shift
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
                     top_n=top_n)


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

    args = parser.parse_args()

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
                       table_d_override=args.table_d,
                       top_n=args.top_n)


if __name__ == '__main__':
    main()