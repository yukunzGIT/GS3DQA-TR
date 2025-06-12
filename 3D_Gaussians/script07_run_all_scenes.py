#!/usr/bin/env python3
"""
script07_run_all_scenes.py

Run the six 3D Gaussian Splatting setup scripts for multiple ScanNet scenes.
For each scene in a provided list, this script:
  1. Reads the original scripts,
  2. Replaces all occurrences of "scene0581_00" with the current scene name,
  3. Writes the modified scripts to a temporary directory,
  4. Executes them in order to produce the final 3D point cloud.

Usage:
  python script07_run_all_scenes.py --scene_list_file scene_list.txt \
      --scans_root scans --rgbd_root .

Assumes:
  - The six original scripts (script01_... through script06_...) live in the same folder as this script.
  - A text file (scene_list.txt) lists one scene name per line (e.g., scene0000_00).
  - The rgbd folder for each scene is named "rgbd-<scene_name>" and located under --rgbd_root.

Outputs:
  - data/initial_pointcloud/<scene_name>_initial_pointcloud.npz
  - data/initial_gaussians/<scene_name>_initial_gaussians.npz
  - data/camera_parameters/<scene_name>_camera_params.npz
  - data/dataset_info/<scene_name>_dataset_info.npz
  - data/optimized_gaussians/<scene_name>_optimized_gaussians.npz
  - data/final_pointcloud/<scene_name>_final_pointcloud.npz and .ply
"""
import os
import argparse
import tempfile
import subprocess

SCRIPT_NAMES = [
    "script01_load_mesh_to_pointcloud.py",
    "script02_initialize_gaussians.py",
    "script03_load_camera_parameters.py",
    "script04_build_splatting_dataset.py",
    "script05_optimize_gaussians.py",
    "script06_visualize_and_save_pointcloud.py",
]


def load_script(path):
    with open(path, 'r') as f:
        return f.read()


def write_script(content, out_path):
    with open(out_path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Run all six steps for multiple ScanNet scenes"
    )
    parser.add_argument(
        "--scene_list_file", required=True,
        help="Text file with one scene name per line (e.g., scene0581_00)"
    )
    parser.add_argument(
        "--scans_root", default="scans",
        help="Root directory containing scans/<scene_name> folders"
    )
    parser.add_argument(
        "--rgbd_root", default=".",
        help="Root directory containing rgbd-<scene_name> folders"
    )
    args = parser.parse_args()

    # Load scene names
    with open(args.scene_list_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    # Determine path to original scripts (assumed alongside this file)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    script_paths = [os.path.join(base_dir, name) for name in SCRIPT_NAMES]

    for scene in scenes:
        print(f"\n=== Processing scene: {scene} ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Copy & patch scripts
            for orig_path in script_paths:
                code = load_script(orig_path)
                patched = code.replace("scene0581_00", scene)
                out_path = os.path.join(tmpdir, os.path.basename(orig_path))
                write_script(patched, out_path)

            # 2. Execute each step in order
            # Step 1: mesh → point cloud
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[0]),
                "--scans_folder", os.path.join(args.scans_root, scene),
                "--output_folder", "data/initial_pointcloud"
            ], check=True)

            # Step 2: initialize Gaussians
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[1]),
                "--input_folder", "data/initial_pointcloud",
                "--output_folder", "data/initial_gaussians"
            ], check=True)

            # Step 3: load camera parameters
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[2]),
                "--rgbd_folder", os.path.join(args.rgbd_root, f"rgbd-{scene}"),
                "--output_folder", "data/camera_parameters"
            ], check=True)

            # Step 4: build dataset info
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[3]),
                "--initial_gaussians_folder", "data/initial_gaussians",
                "--camera_params_folder", "data/camera_parameters",
                "--rgbd_folder", os.path.join(args.rgbd_root, f"rgbd-{scene}"),
                "--output_folder", "data/dataset_info"
            ], check=True)

            # Step 5: optimize Gaussians (no shuffling for determinism)
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[4]),
                "--dataset_info_folder", "data/dataset_info",
                "--output_folder", "data/optimized_gaussians"
            ], check=True)

            # Step 6: visualize & save final pointcloud (no GUI)
            subprocess.run([
                "python", os.path.join(tmpdir, SCRIPT_NAMES[5]),
                "--optimized_gaussians_folder", "data/optimized_gaussians",
                "--output_folder", "data/final_pointcloud",
                "--save_ply"
            ], check=True)

        print(f"=== Finished scene: {scene} ===")


if __name__ == "__main__":
    main()
