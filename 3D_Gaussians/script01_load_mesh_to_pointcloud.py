#!/usr/bin/env python3
"""
script01_load_mesh_to_pointcloud.py

Load the TSDF-reconstructed mesh for scene0581_00, sample or extract a point cloud
from it, and save the resulting points + colors to disk as a .npz file. Later scripts
will take these points as the fixed positions for initializing 3D Gaussians.
"""

import os
import argparse
import numpy as np
import open3d as o3d
# open3d is a lightweight library to load PLY meshes and deal with point clouds

def main(args):
    # 1. Build paths
    # ----------------
    # Path to the PLY mesh inside the ScanNet folder structure.
    #mesh_path = os.path.join(args.scans_folder, "scene0581_00_vh_clean.ply")
    mesh_path = os.path.join(args.scans_folder, "scene0581_00_vh_clean_2.ply")
    # Where to save the point-cloud output (as an .npz of vertices + colors).
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)
    out_npz = os.path.join(out_folder, "scene0581_00_initial_pointcloud.npz")

    # 2. Load the TSDF mesh
    # ----------------------
    print(f"Loading mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        raise RuntimeError(f"Failed to load vertices from {mesh_path}")

    # 3. Check whether the mesh already has per-vertex colors
    # -------------------------------------------------------
    # Some TSDF meshes shipped by ScanNet already embed per‐vertex RGB; if so, we keep them.
    # Otherwise, we assign a uniform gray (0.5,0.5,0.5) to each vertex so that downstream steps still have a “color” array to optimize.
    if mesh.has_vertex_colors():
        print("Mesh has vertex colors. We will use those.")
    else:
        print("Mesh has no vertex colors. Assigning default gray color to all vertices.")
        # e.g., uniform gray = [0.5, 0.5, 0.5]
        gray = np.tile(np.array([[0.5, 0.5, 0.5]]), (np.asarray(mesh.vertices).shape[0], 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(gray)

    # 4. Convert mesh to point cloud
    # -------------------------------
    # Option A: Take the mesh vertices directly as points
    #vertices = np.asarray(mesh.vertices)            # (N, 3)
    #colors   = np.asarray(mesh.vertex_colors)       # (N, 3)

    # Option B (alternative): uniformly sample points on the mesh surface:
    # Uncomment if you prefer sampling rather than using raw vertices.
    num_samples = 3 * np.asarray(mesh.vertices).shape[0] # NOTE our new sampling point clouds contain 3 times more points

    pcd = mesh.sample_points_poisson_disk(number_of_points=num_samples)
    vertices = np.asarray(pcd.points)
    colors   = np.asarray(pcd.colors)

    # 5. Save vertices + colors to disk
    # ----------------------------------
    print(f"Saving point cloud with {vertices.shape[0]} points to: {out_npz}")
    np.savez_compressed(out_npz, points=vertices, colors=colors)

    # 6. Visualize the resulting point cloud
    # --------------------------------------
    print("Visualizing point cloud...")
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(vertices)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_vis], window_name='Initial Point Cloud', width=800, height=600)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Load TSDF mesh and convert to point cloud."
    )
    parser.add_argument(
        "--scans_folder",
        type=str,
        default="scans/scene0581_00",
        help="Path to the folder containing all ScanNet files for scene0581_00"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/initial_pointcloud",
        help="Folder in which to save the extracted point cloud"
    )
    args = parser.parse_args()
    main(args)
