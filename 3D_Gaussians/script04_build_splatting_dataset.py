#!/usr/bin/env python3
"""
script04_build_splatting_dataset.py

Prepare and bundle all data needed for optimizing 3D Gaussians:
  • Load initial Gaussian parameters (positions, covariances, colors)
  • Load camera intrinsics & per‐frame extrinsics
  • Collect paths to each RGB color image
  • Save a single “dataset_info.npz” that ties together:
      - gaussian file path
      - camera_params file path
      - list of image file paths (ordered by frame index)
      - intrinsics array
      - extrinsics array

Later, your optimizer can load “dataset_info.npz” to know where to find:
  – The initial Gaussians
  – The camera intrinsics/extrinsics
  – The target color images to compare against
"""

import os
import argparse
import numpy as np

def main(args):
    # 1. Build paths to existing data
    # --------------------------------
    # Where we stored initial Gaussians from script02:
    gauss_npz = os.path.join(args.initial_gaussians_folder, "scene0581_00_initial_gaussians.npz")
    # Where we stored camera intrinsics/extrinsics from script03:
    cam_params_npz = os.path.join(args.camera_params_folder, "scene0581_00_camera_params.npz")
    # The folder containing extracted color PNGs (one per frame)
    color_folder = os.path.join(args.rgbd_folder, "color")
    # Output folder & filename for our “dataset_info”
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)
    out_npz = os.path.join(out_folder, "scene0581_00_dataset_info.npz")

    # 2. Verify required files exist
    # -------------------------------
    if not os.path.isfile(gauss_npz):
        raise FileNotFoundError(f"Could not find initial Gaussians file: {gauss_npz}")
    if not os.path.isfile(cam_params_npz):
        raise FileNotFoundError(f"Could not find camera params file: {cam_params_npz}")
    if not os.path.isdir(color_folder):
        raise FileNotFoundError(f"Could not find color folder: {color_folder}")

    # 3. Load camera intrinsics & extrinsics
    # ---------------------------------------
    print(f"Loading camera parameters from: {cam_params_npz}")
    cam_data = np.load(cam_params_npz)
    # intrinsics_color: shape (num_frames, 4, 4)
    intrinsics_all = cam_data["intrinsics_color"]  
    # extrinsics_per_frame: shape (num_frames, 4, 4)
    extrinsics_all = cam_data["extrinsics_per_frame"]
    print(f"Loaded intrinsics array with shape {intrinsics_all.shape}")
    print(f"Loaded extrinsics array with shape {extrinsics_all.shape}")

    # 4. Enumerate and sort all color image paths
    # -------------------------------------------
    # We assume each frame’s PNG is named with zero‐padded index, e.g. "00000000.png", "00000001.png", …
    all_files = sorted([
        f for f in os.listdir(color_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ])
    if len(all_files) == 0:
        raise RuntimeError(f"No image files found in {color_folder}")
    num_images = len(all_files)
    print(f"Found {num_images} color images in {color_folder} (e.g. {all_files[0]})")

    # Build a full list of file‐paths, in sorted (frame) order
    image_paths = [os.path.join(color_folder, fname) for fname in all_files]

    # 5. Sanity‐check: num_images should match camera params’ num_frames
    # -------------------------------------------------------------------
    if num_images != intrinsics_all.shape[0] or num_images != extrinsics_all.shape[0]:
        raise ValueError(
            f"Number of images ({num_images}) does not match camera params ({intrinsics_all.shape[0]} frames)."
        )

    # 6. Bundle everything into a single .npz
    # ---------------------------------------
    # We will store:
    #   • "gaussians_path"   : string (path to initial_gaussians .npz)
    #   • "camera_params_path": string (path to camera_params .npz)
    #   • "image_paths"      : array of dtype object or fixed‐length strings (num_frames,)
    #   • "intrinsics"       : (num_frames, 4, 4) float32
    #   • "extrinsics"       : (num_frames, 4, 4) float32

    # To save a list of strings in .npz, convert to numpy array of dtype=object or dtype='U...'
    # We use dtype='U256' to allow up to 256‐char file paths.
    # To stash a list of file paths into a single .npz, we convert the Python list of strings into a NumPy array of Unicode strings (up to 256 chars).
    # If your paths are longer than 256 characters, bump up "U256" accordingly
    image_paths_arr = np.array(image_paths, dtype="U256")

    print(f"Saving dataset info to: {out_npz}")
    np.savez_compressed(
        out_npz,
        gaussians_path=np.array([gauss_npz], dtype="U256"),
        camera_params_path=np.array([cam_params_npz], dtype="U256"),
        image_paths=image_paths_arr,
        intrinsics=intrinsics_all.astype(np.float32),
        extrinsics=extrinsics_all.astype(np.float32)
    )

    print("Dataset info saved. You can now load this .npz in your optimizer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Build a single dataset_info.npz that bundles "
                    "Gaussians, camera params, and image file paths."
    )
    parser.add_argument(
        "--initial_gaussians_folder",
        type=str,
        default="data/initial_gaussians",
        help="Folder containing scene0581_00_initial_gaussians.npz"
    )
    parser.add_argument(
        "--camera_params_folder",
        type=str,
        default="data/camera_parameters",
        help="Folder containing scene0581_00_camera_params.npz"
    )
    parser.add_argument(
        "--rgbd_folder",
        type=str,
        default="rgbd-scene0581_00",
        help="Folder containing the extracted RGB‐D data (subfolder 'color/')"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/dataset_info",
        help="Folder in which to save scene0581_00_dataset_info.npz"
    )
    args = parser.parse_args()
    main(args)
