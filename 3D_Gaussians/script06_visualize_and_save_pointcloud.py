#!/usr/bin/env python3
"""
script06_visualize_and_save_pointcloud.py

Load the optimized Gaussians (positions + colors) from the final .npz, 
convert them into a 3D point cloud with RGB values (6D per point), 
save this point cloud to disk (as a compressed .npz and optionally as a PLY), 
and visualize it using Open3D.
"""

import os
import argparse
import numpy as np
import open3d as o3d

def main(args):
    # -------------------------------------------------------------------------
    # 1. Build paths and ensure output directories exist
    # -------------------------------------------------------------------------
    # Path to the optimized Gaussians .npz produced by script05
    optimized_npz = os.path.join(args.optimized_gaussians_folder,
                                 "scene0581_00_optimized_gaussians.npz")
    if not os.path.isfile(optimized_npz):
        raise FileNotFoundError(f"Cannot find optimized Gaussians file: {optimized_npz}")

    # Output folder where we'll save:
    #   • a compressed .npz containing the final point cloud (positions+colors)
    #   • optionally a PLY file for visualization / downstream use
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)
    out_pointcloud_npz = os.path.join(out_folder, "scene0581_00_final_pointcloud.npz")
    out_pointcloud_ply = os.path.join(out_folder, "scene0581_00_final_pointcloud.ply")

    # -------------------------------------------------------------------------
    # 2. Load optimized Gaussians: positions (N×3) + colors (N×3)
    # -------------------------------------------------------------------------
    print(f"Loading optimized Gaussians from: {optimized_npz}")
    data = np.load(optimized_npz)
    positions = data["positions"]      # shape = (N, 3), dtype=float32
    colors = data["colors"]            # shape = (N, 3), dtype=float32, values in [0,1]

    num_points = positions.shape[0]
    print(f"  • Loaded {num_points} Gaussians → will create a point cloud with {num_points} points.")

    # -------------------------------------------------------------------------
    # 3. Save final point cloud as a compressed .npz (positions + colors)
    # -------------------------------------------------------------------------
    # We store:
    #   • "points": (N, 3) float32
    #   • "colors": (N, 3) float32
    print(f"Saving final point cloud to: {out_pointcloud_npz}")
    np.savez_compressed(out_pointcloud_npz,
                        points=positions.astype(np.float32),
                        colors=colors.astype(np.float32))

    # -------------------------------------------------------------------------
    # 4. (Optional) Save the point cloud as a PLY file for compatibility
    # -------------------------------------------------------------------------
    if args.save_ply:
        print(f"Converting to Open3D PointCloud and saving PLY to: {out_pointcloud_ply}")
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        # Set the points (convert from Nx3 NumPy array to Vector3dVector)
        pcd.points = o3d.utility.Vector3dVector(positions)
        # Set the colors (also as Vector3dVector)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Write to disk as a binary PLY
        o3d.io.write_point_cloud(out_pointcloud_ply, pcd, write_ascii=False)
        print("PLY file saved.")

    # -------------------------------------------------------------------------
    # 5. Visualize the final colored point cloud in an Open3D window
    # -------------------------------------------------------------------------
    if args.visualize:
        print("Visualizing final point cloud...")
        # Instantiate an Open3D PointCloud with the same positions + colors
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(positions)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        # Set a window name and size for clarity
        o3d.visualization.draw_geometries(
            [pcd_vis],
            window_name="Final Scene Point Cloud (RGB)",
            width=1024,
            height=768
        )

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6: Convert optimized Gaussians into a colored 3D point cloud, "
                    "save to disk, and visualize."
    )
    parser.add_argument(
        "--optimized_gaussians_folder",
        type=str,
        default="data/optimized_gaussians",
        help="Folder containing scene0581_00_optimized_gaussians.npz"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/final_pointcloud",
        help="Folder in which to save the final point cloud files"
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="If set, also write the final point cloud as a .ply file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, launch an Open3D visualization window of the final point cloud"
    )
    args = parser.parse_args()
    main(args)
