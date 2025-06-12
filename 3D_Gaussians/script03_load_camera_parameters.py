#!/usr/bin/env python3
"""
script03_load_camera_parameters.py

Read all color‐camera intrinsics + extrinsics (per frame) from the extracted RGB‐D folder
(`rgbd-scene0581_00/`) and save them together into a single .npz file. Later, the splatting
optimizer will load these intrinsics/extrinsics to render each 3D Gaussian into the correct
2D image plane.

Expected folder structure for `--rgbd_folder`:
    rgbd-scene0581_00/
      ├─ color/                (1496 PNGs, not used here)
      ├─ depth/                (1496 PNGs, not used here)
      ├─ intrinsic/
      │    ├─ intrinsic_color.txt    (4×4 matrix)
      │    ├─ intrinsic_depth.txt    (4×4, not used now)
      │    ├─ extrinsic_color.txt    (4×4, color‐cam → world, if present)
      │    └─ extrinsic_depth.txt    (4×4, not used now)
      └─ pose/
           ├─ 00000000.txt    (4×4, world ← camera at frame 0)
           ├─ 00000001.txt    (4×4, world ← camera at frame 1)
           └─ … (one file per color frame, total 1496)

This script will:
  1. Load “intrinsic_color.txt” (the 4×4 color‐camera intrinsics)
  2. Load “extrinsic_color.txt” (if present; otherwise skip)
  3. Load every 4×4 “pose/XXXXXX.txt” as the per‐frame camera‐to‐world extrinsic
     (we assume each “pose/NNNNNN.txt” matches exactly the color‐frame index)
  4. Stack them into arrays of shape (1496, 4, 4)
  5. Save:
       • intrinsics_color      → shape (1, 4, 4) or broadcast to (1496, 4, 4)
       • extrinsic_color_static → shape (1, 4, 4)  (if present)
       • extrinsics_per_frame   → shape (1496, 4, 4)
     into “scene0581_00_camera_params.npz”
"""

import os
import argparse
import numpy as np

def main(args):
    # 1. Build paths
    # ----------------
    rgbd_folder = args.rgbd_folder.rstrip("/")  # e.g. "rgbd-scene0581_00"
    intrinsic_dir = os.path.join(rgbd_folder, "intrinsic")
    pose_dir = os.path.join(rgbd_folder, "pose")

    # Output folder and filename
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)
    out_npz = os.path.join(out_folder, "scene0581_00_camera_params.npz")

    # 2. Load the static color‐camera intrinsic matrix
    # -------------------------------------------------
    # intrinsic_color.txt is assumed to be a 4×4 matrix (homogeneous intrinsics).
    intr_color_path = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    if not os.path.isfile(intr_color_path):
        raise FileNotFoundError(f"Cannot find: {intr_color_path}")
    K_color = np.loadtxt(intr_color_path)  # shape (4,4)
    if K_color.shape != (4, 4):
        raise ValueError(f"Expected 4×4 in {intr_color_path}, got {K_color.shape}")
    print(f"Loaded static color intrinsics from {intr_color_path}")

    # 3. (Optional) Load any static “extrinsic_color.txt”
    # ----------------------------------------------------
    # Some pipelines store a 4×4 color‐camera → world (or world → camera) transform here.
    # Later in blending or coordinate unification, we may need that static mapping. Including it here means our optimizer can optionally use it
    extr_color_path = os.path.join(intrinsic_dir, "extrinsic_color.txt")
    if os.path.isfile(extr_color_path):
        E_color_static = np.loadtxt(extr_color_path)
        if E_color_static.shape != (4, 4):
            raise ValueError(f"Expected 4×4 in {extr_color_path}, got {E_color_static.shape}")
        print(f"Loaded static color extrinsic from {extr_color_path}")
    else:
        E_color_static = None
        print(f"No static 'extrinsic_color.txt' found; skipping.")

    # 4. Read all per‐frame pose files into a list
    # ---------------------------------------------
    # In ScanNet’s export, each file under “pose/” is named like "00000000.txt", "00000001.txt", ….
    # Each is a 4×4 float matrix: camera‐to‐world transform at that frame.
    pose_files = sorted(f for f in os.listdir(pose_dir) if f.endswith(".txt"))
    if len(pose_files) == 0:
        raise RuntimeError(f"No .txt files found in pose directory: {pose_dir}")
    print(f"Found {len(pose_files)} pose files under {pose_dir} (e.g. {pose_files[0]})")

    # Pre‐allocate an array of shape (num_frames, 4, 4)
    num_frames = len(pose_files)
    extrinsics_per_frame = np.zeros((num_frames, 4, 4), dtype=np.float64)

    for idx, fname in enumerate(pose_files):
        full_path = os.path.join(pose_dir, fname)
        M = np.loadtxt(full_path)
        if M.shape != (4, 4):
            raise ValueError(f"Pose file {full_path} is not 4×4 (got {M.shape})")
        extrinsics_per_frame[idx] = M

    print(f"Loaded and stacked {num_frames} per‐frame extrinsic matrices (shape = {extrinsics_per_frame.shape})")

    # 5. Broadcast the single intrinsic to all frames
    # ------------------------------------------------
    # Gaussian‐splatting will want one intrinsics matrix per frame. Since intrinsics don't change,
    # we repeat the same 4×4 K_color num_frames times.
    intrinsics_per_frame = np.tile(K_color[None, ...], (num_frames, 1, 1))  # shape (num_frames, 4, 4)

    # 6. Save everything into one .npz
    # --------------------------------
    # We store:
    #   • "intrinsics_color"      : (num_frames, 4, 4)
    #   • "extrinsics_per_frame"  : (num_frames, 4, 4)
    #   • optionally "extrinsic_color_static" : (4, 4) if it existed
    if E_color_static is not None:
        np.savez_compressed(
            out_npz,
            intrinsics_color=intrinsics_per_frame.astype(np.float32),
            extrinsics_per_frame=extrinsics_per_frame.astype(np.float32),
            extrinsic_color_static=E_color_static.astype(np.float32)
        )
        print(f"Saved intrinsics + per‐frame extrinsics + static extrinsic to {out_npz}")
    else:
        np.savez_compressed(
            out_npz,
            intrinsics_color=intrinsics_per_frame.astype(np.float32),
            extrinsics_per_frame=extrinsics_per_frame.astype(np.float32)
        )
        print(f"Saved intrinsics + per‐frame extrinsics to {out_npz}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Load color‐camera intrinsics + per‐frame extrinsics and bundle into one file."
    )
    parser.add_argument(
        "--rgbd_folder",
        type=str,
        default="rgbd-scene0581_00",
        help="Path to the extracted RGB‐D folder (contains subfolders: intrinsic/, pose/)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/camera_parameters",
        help="Where to save scene0581_00_camera_params.npz"
    )
    args = parser.parse_args()
    main(args)
