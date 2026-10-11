"""Multi-frame RGB-D TSDF reconstruction.

Loads RGB-D frames and camera poses, applies saved segmentation masks, 
filters depth noise, integrates into a TSDF volume, and exports the 
reconstructed mesh and point cloud.
"""

import os
import sys
import numpy as np
import open3d as o3d
import cv2

# =============================================================================
#  SETTINGS
# =============================================================================

# AUTOMATION BLOCK START
if len(sys.argv) > 1:
    SAVE_DIR = sys.argv[1]
    MASK_DIR = SAVE_DIR
    OUT_MESH = os.path.join(SAVE_DIR, "reconstructed_mesh.ply")
    OUT_PCD  = os.path.join(SAVE_DIR, "reconstructed_pointcloud.ply")
else:
    SAVE_DIR   = "Captured_Samples/engine_new"
    MASK_DIR   = SAVE_DIR
    OUT_MESH = "Captured_Samples/engine_new/reconstructed_mesh.ply"
    OUT_PCD  = "Captured_Samples/engine_new/reconstructed_pointcloud.ply"

if len(sys.argv) > 2:
    IMG_PREFIX = sys.argv[2]
else:
    IMG_PREFIX = "TOY"

if len(sys.argv) > 3:
    END_IDX = int(sys.argv[3])
else:
    END_IDX = 42
START_IDX = 1
# AUTOMATION BLOCK END

VOXEL_LENGTH = 0.0003  # TSDF voxel size, metres (0.3 mm)
SDF_TRUNC    = 0.0012  # TSDF truncation distance (~4x voxel_length)
DEPTH_TRUNC_M = 0.45   # Max depth range, metres

# Boundary-noise controls
MASK_ERODE_PX         = 0      # Additional erosion to trim residual RGB/depth misalignment.
FLYING_PIXEL_KERNEL   = 3      # Neighbourhood size (px) for detecting depth discontinuities.
FLYING_PIXEL_THRESH_M = 0.010  # Drops edge pixels exceeding this local depth range to prevent 
                               # radial base spikes. Correctly rejects background leaks without 
                               # over-filtering valid curved surfaces.
DEPTH_MIN_M = 0.10

# Hole-filling controls
FILL_BASE_HOLE   = False
HOLE_SIZE_M      = 1e6   # Low values preserve deliberate, real-world holes.

# Optional Trimesh base slicer
# Slices the mesh 2mm from the bottom to create a flat, printable base.
USE_TRIMESH_SLICER = True

SKIP_CLEANUP = False    # True = skip small-disconnected-component removal on the mesh

# Soft ratio-based filter prevents deletion of real, thinly-connected geometry 
# (like top caps) while safely removing stray debris.
KEEP_LARGEST_MESH_COMPONENT_ONLY = False
MIN_COMPONENT_RATIO = 0.05   # Keep components at least this fraction of the largest cluster.


T_cam2flange = np.array([
    [ 0.999998, -0.001814, -0.001229, -0.03193],
    [ 0.001805,  0.999974, -0.007038, -0.095128],
    [ 0.001241,  0.007036,  0.999974,  0.068082],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Load / mask
# ═══════════════════════════════════════════════════════════════

def load_capture(save_dir, prefix, idx, mask_dir):
    tag = f"{idx:04d}"
    color_path = os.path.join(save_dir, f"{prefix}_{tag}.png")
    depth_path = os.path.join(save_dir, f"{prefix}_depth_raw_{tag}.png")
    pose_path  = os.path.join(save_dir, f"pose_{tag}.txt")
    mask_path  = os.path.join(mask_dir, f"mask_{tag}.png")

    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        return None
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    T_fb  = np.loadtxt(pose_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
    if color is None or depth is None:
        return None
    return color, depth, T_fb, mask


def apply_mask_to_depth(depth_raw, mask, erode_px=0):
    """Zeros depth outside the mask. Trims residual boundary noise if erode_px > 0."""
    if mask is None:
        return depth_raw
    if mask.shape != depth_raw.shape:
        mask = cv2.resize(mask, (depth_raw.shape[1], depth_raw.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    out = depth_raw.copy()
    out[mask < 128] = 0
    return out


def remove_flying_pixels(depth_raw, depth_scale, kernel_size=5, thresh_m=0.010):
    """Zeros out depth pixels sitting on edge discontinuities (flying pixels) 
    before TSDF integration, preventing thin radial artifacts."""
    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = depth_m > 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    depth_for_max = np.where(valid, depth_m, 0.0).astype(np.float32)
    local_max = cv2.dilate(depth_for_max, kernel)

    depth_for_min = np.where(valid, depth_m, 1e6).astype(np.float32)
    local_min = cv2.erode(depth_for_min, kernel)

    edge_pixels = valid & ((local_max - local_min) > thresh_m)

    out = depth_raw.copy()
    out[edge_pixels] = 0
    return out


# ═══════════════════════════════════════════════════════════════
# TSDF fusion
# ═══════════════════════════════════════════════════════════════

def run_tsdf_fusion(save_dir, prefix, mask_dir, intr, depth_scale,
                    start_idx, end_idx,
                    voxel_length, sdf_trunc, depth_trunc_m,
                    mask_erode_px, flying_pixel_kernel, flying_pixel_thresh_m):
    """Integrates frames into a TSDF volume and extracts a mesh via marching cubes."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = voxel_length,
        sdf_trunc    = sdf_trunc,
        color_type   = o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.get("width", 640), intr.get("height", 480),
        intr["fx"], intr["fy"], intr["cx"], intr["cy"])

    n_integrated = 0
    for idx in range(start_idx, end_idx + 1):
        data = load_capture(save_dir, prefix, idx, mask_dir)
        if data is None:
            continue
        color, depth, T_fb, mask = data
        if mask is None:
            print(f"  [WARN] Frame {idx:04d} missing mask. Skipping.")
            continue

        masked_depth = apply_mask_to_depth(depth, mask, erode_px=mask_erode_px)

        masked_depth = remove_flying_pixels(
            masked_depth, depth_scale,
            kernel_size=flying_pixel_kernel, thresh_m=flying_pixel_thresh_m)
        masked_depth_m = masked_depth.astype(np.float32) * depth_scale
        masked_depth[(masked_depth_m < DEPTH_MIN_M) | (masked_depth_m > DEPTH_TRUNC_M)] = 0

        if np.count_nonzero(masked_depth) < 30:
            print(f"  [WARN] Frame {idx:04d} — too few masked depth pixels. Skipping.")
            continue

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
        o3d_depth = o3d.geometry.Image(masked_depth.astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale = 1.0 / depth_scale,
            depth_trunc = depth_trunc_m,
            convert_rgb_to_intensity = False,
        )

        T_base_cam = T_fb @ T_cam2flange

        # Crucial: Open3D integrate() requires the world-to-camera extrinsic.
        extrinsic = np.linalg.inv(T_base_cam)

        volume.integrate(rgbd, o3d_intr, extrinsic)
        n_integrated += 1
        print(f"  [OK] Integrated frame {idx:04d}")

    print(f"\nIntegrated {n_integrated} frame(s) into the TSDF volume.")
    if n_integrated == 0:
        return None, None

    print("Extracting mesh via marching cubes...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print("Extracting point cloud from the same volume (for comparison)...")
    pcd = volume.extract_point_cloud()

    return mesh, pcd


def keep_largest_pointcloud_cluster(pcd, eps=0.005, min_points=30):
    """Filters raw TSDF point clouds via DBSCAN, keeping only the largest 3D cluster."""
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                         print_progress=False))
    if len(labels) == 0 or labels.max() < 0:
        return pcd
    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    valid_indices = np.where(labels == largest_cluster_idx)[0]
    return pcd.select_by_index(valid_indices)


def clean_mesh(mesh, keep_largest_only=KEEP_LARGEST_MESH_COMPONENT_ONLY,
               min_triangles_ratio=MIN_COMPONENT_RATIO):
    """Removes disconnected debris. Uses a triangle ratio threshold to 
    safely preserve thin-but-valid multi-piece geometry."""
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        return mesh

    largest_cluster_size = cluster_n_triangles.max()
    if keep_largest_only:
        threshold = largest_cluster_size - 1
    else:
        threshold = largest_cluster_size * min_triangles_ratio

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
    n_components_before = len(cluster_n_triangles)
    n_components_kept = int(np.sum(cluster_n_triangles >= threshold))
    print(f"  [CLEAN] {n_components_before} connected component(s) found, "
          f"keeping {n_components_kept} "
          f"(threshold={threshold:.0f} triangles, "
          f"largest={largest_cluster_size} triangles)")

    mesh_clean = o3d.geometry.TriangleMesh(mesh)
    mesh_clean.remove_triangles_by_mask(triangles_to_remove)
    mesh_clean.remove_unreferenced_vertices()
    mesh_clean.remove_degenerate_triangles()
    mesh_clean.remove_duplicated_triangles()
    mesh_clean.remove_duplicated_vertices()
    return mesh_clean


def fill_mesh_holes(mesh, hole_size=1e6):
    """Caps open boundary loops for downstream watertight applications."""
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    t_filled = t_mesh.fill_holes(hole_size=hole_size)
    filled = t_filled.to_legacy()
    filled.compute_vertex_normals()
    return filled


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    depth_scale = 0.001
    intr_path = os.path.join(SAVE_DIR, "intrinsics.npy")
    if not os.path.exists(intr_path):
        print(f"[ERROR] Could not find {intr_path}")
        return
    intr = np.load(intr_path, allow_pickle=True).item()

    print(f"Starting TSDF fusion (frames {START_IDX} to {END_IDX})...")
    print(f"  dir          = {SAVE_DIR}")
    print(f"  mask_dir     = {MASK_DIR}")
    print(f"  voxel_length = {VOXEL_LENGTH*1000:.2f}mm, "
          f"sdf_trunc = {SDF_TRUNC*1000:.2f}mm")
    print(f"  mask_erode_px= {MASK_ERODE_PX}px, "
          f"flying_pixel_thresh = {FLYING_PIXEL_THRESH_M*1000:.1f}mm\n")

    mesh, pcd = run_tsdf_fusion(
        SAVE_DIR, IMG_PREFIX, MASK_DIR, intr, depth_scale,
        START_IDX, END_IDX,
        VOXEL_LENGTH, SDF_TRUNC, DEPTH_TRUNC_M,
        MASK_ERODE_PX, FLYING_PIXEL_KERNEL, FLYING_PIXEL_THRESH_M)

    if mesh is None:
        print("[ERROR] No frames integrated. Check mask/depth/pose paths.")
        return

    print(f"\nRaw mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")

    if not SKIP_CLEANUP:
        print("Removing small disconnected components from mesh...")
        mesh = clean_mesh(mesh)
        print(f"Cleaned mesh: {len(mesh.vertices)} vertices, "
              f"{len(mesh.triangles)} triangles")

        print("Removing small disconnected components from point cloud...")
        pcd_before = len(pcd.points)
        pcd = keep_largest_pointcloud_cluster(pcd)
        print(f"Cleaned point cloud: {pcd_before} -> {len(pcd.points)} points")

    if FILL_BASE_HOLE:
        print("Filling open boundary (unseen base)...")
        mesh = fill_mesh_holes(mesh, hole_size=HOLE_SIZE_M)
        print(f"Filled mesh: {len(mesh.vertices)} vertices, "
              f"{len(mesh.triangles)} triangles")

    OUT_MESH_OBJ = OUT_MESH.replace(".ply", ".obj")

    o3d.io.write_triangle_mesh(OUT_MESH, mesh)
    o3d.io.write_triangle_mesh(OUT_MESH_OBJ, mesh)
    o3d.io.write_point_cloud(OUT_PCD, pcd)

    print(f"\n[SUCCESS] Saved mesh        → {OUT_MESH}")
    print(f"[SUCCESS] Saved mesh (OBJ)  → {OUT_MESH_OBJ}")
    print(f"[SUCCESS] Saved point cloud → {OUT_PCD}")

    if USE_TRIMESH_SLICER:
        print("\n[TRIMESH] Generating flat-capped version for 3D printing...")
        try:
            import trimesh

            mesh_tri = trimesh.load(OUT_MESH)

            if len(mesh_tri.vertices) > 0:
                z_min = mesh_tri.bounds[0][2]

                slice_height = z_min + 0.002
                plane_origin = [0, 0, slice_height]
                plane_normal = [0, 0, 1] 

                sliced_mesh = mesh_tri.slice_plane(plane_origin, plane_normal, cap=True)

                out_trimesh = OUT_MESH.replace(".ply", "_flat_capped.ply")
                sliced_mesh.export(out_trimesh)
                print(f"[SUCCESS] Saved flat-capped mesh  → {out_trimesh}")
            else:
                print("[WARN] Mesh is empty, skipping Trimesh slice.")

        except ImportError:
            print("[WARN] 'trimesh' library not found. Skipping flat-cap generation.")
            print("        Install it via: pip install trimesh shapely rtree")

    print("\n[TEXTURE] Baking vertex colors into a UV texture for RoboDK...")
    try:
        import trimesh
        mesh_tri = trimesh.load(OUT_MESH)  

        if hasattr(mesh_tri.visual, "kind") and mesh_tri.visual.kind == "vertex":
            mesh_tri.visual = mesh_tri.visual.to_texture()

        textured_obj_path = OUT_MESH.replace(".ply", "_textured.obj")
        mesh_tri.export(textured_obj_path)  
        print(f"[SUCCESS] Saved textured OBJ (+MTL+PNG) → {textured_obj_path}")
        print("          NOTE: keep the .obj/.mtl/.png together in this folder, "
              "the .mtl references the texture by relative filename.")
    except Exception as e:
        print(f"[WARN] Texture baking failed ({e}). RoboDK import will show a flat/grey mesh instead.")

    print("\n[DONE] Open files in MeshLab, Bambu Studio, or Blender to inspect.")

if __name__ == "__main__":
    main()
