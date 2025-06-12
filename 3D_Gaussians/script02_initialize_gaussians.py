#!/usr/bin/env python3
"""
script02_initialize_gaussians.py

Load the point cloud saved by script01 (scene0581_00_initial_pointcloud.npz),
create one 3D Gaussian per point with:
  - fixed position = point coordinate
  - initial diagonal covariance = small isotropic value
  - initial color = mesh vertex color

Then save these arrays to disk as scene0581_00_initial_gaussians.npz.
"""

import os
import argparse
import numpy as np

def main(args):
    # 1. Build paths
    # ----------------
    # Path to the .npz containing raw points+colors (output of script01).
    input_npz = os.path.join(args.input_folder, "scene0581_00_initial_pointcloud.npz")
    # Folder and filename where we’ll dump the Gaussian parameters.
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)
    output_npz = os.path.join(out_folder, "scene0581_00_initial_gaussians.npz")

    # 2. Load the raw point cloud
    # ----------------------------
    print(f"Loading point cloud from: {input_npz}")
    data = np.load(input_npz)
    points = data["points"]   # shape: (N, 3), dtype=float32 or float64
    colors = data["colors"]   # shape: (N, 3), float in [0,1]

    num_points = points.shape[0]
    print(f"  • Loaded {num_points} points.")

    # 3. Initialize covariance diagonals (σ_x², σ_y², σ_z²) for each Gaussian
    # -----------------------------------------------------------------------
    # We choose a small isotropic variance, e.g. (σ = 0.01 m)^2 = 1e-4.
    # You can adjust this value based on the scale of your TSDF mesh.
    initial_sigma = args.initial_sigma  # in meters
    initial_var = initial_sigma ** 2    # variance

    # Create an array of shape (N, 3), where each row = [σ², σ², σ²]
    cov_diagonals = np.tile(np.array([initial_var, initial_var, initial_var], dtype=np.float32),
                            (num_points, 1))

    # 4. (Optional) Initialize mean color for each Gaussian
    # -----------------------------------------------------
    # We’ll simply copy the mesh‐vertex color as the starting color.
    # If your downstream optimizer expects [0..255], you can multiply by 255 here.
    # In this script, we keep them as floats in [0,1].
    gaussian_colors = colors.astype(np.float32)

    # 5. “Fix” positions = the original points
    # -----------------------------------------
    # We do not intend to optimize positions, so we simply store them as-is.
    gaussian_positions = points.astype(np.float32)

    # 6. Save everything to disk
    # --------------------------
    # We save:
    #   - positions:       (N,3) float32
    #   - cov_diagonals:   (N,3) float32
    #   - colors:          (N,3) float32
    print(f"Saving initial Gaussians ({num_points} of them) to: {output_npz}")
    np.savez_compressed(
        output_npz,
        positions=gaussian_positions,
        cov_diagonals=cov_diagonals,
        colors=gaussian_colors
    )

    print("Finished initializing Gaussians.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Initialize one 3D Gaussian per point, "
                    "with fixed position, small diagonal covariance, and mesh color."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/initial_pointcloud",
        help="Folder containing scene0581_00_initial_pointcloud.npz"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/initial_gaussians",
        help="Folder in which to save scene0581_00_initial_gaussians.npz"
    )
    parser.add_argument(
        "--initial_sigma",
        type=float,
        default=0.01,
        help="Initial standard deviation (in meters) for each Gaussian’s diagonal covariance"
    )
    args = parser.parse_args()
    main(args)
