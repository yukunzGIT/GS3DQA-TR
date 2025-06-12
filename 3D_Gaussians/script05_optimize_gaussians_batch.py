#!/usr/bin/env python3
"""
05test.py

This script uses the gsplat library (https://pypi.org/project/gsplat) to
perform end-to-end optimization of 3D Gaussian Splatting on ScanNet’s
scene0581_00. Each Gaussian’s position is fixed; we train only its
diagonal covariance (σ²_x,σ²_y,σ²_z) and its RGB color. 

Inputs:
  • data/dataset_info/scene0581_00_dataset_info.npz
      – “gaussians_path”: path to data/initial_gaussians/...npz
      – “camera_params_path”: path to data/camera_parameters/...npz
      – “image_paths”:    (1496,) array of color‐image filepaths (strings)
      – “intrinsics”:     (1496, 4, 4) array of camera intrinsics (4×4)
      – “extrinsics”:     (1496, 4, 4) array of camera extrinsics (4×4)
  • data/initial_gaussians/scene0581_00_initial_gaussians.npz
      – “positions”:     (N, 3) float32 array
      – “cov_diagonals”: (N, 3) float32 array
      – “colors”:        (N, 3) float32 array

After training, saves:
  • data/optimized_gaussians/scene0581_00_optimized_gaussians.npz
      – “positions”:     (N, 3)
      – “cov_diagonals”: (N, 3) = exp(log_sigma2), clamped ≥ min_variance
      – “colors”:        (N, 3) ∈ [0,1]

Usage example:
  python 05test.py \
    --dataset_info_folder data/dataset_info \
    --output_folder data/optimized_gaussians \
    --lr_sigma 5e-4 \
    --lr_color 5e-4 \
    --num_epochs 10 \
    --print_every 10 \
    --save_every 5 \
    --min_variance 1e-6 \
    --near_plane 0.02 \
    --far_plane 10.0 \
    --batch_size 8 \
    --shuffle
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------------- 
# 1. Import gsplat’s rasterization backend
# ----------------------------------------------------------------------------- 
try:
    from gsplat import rasterization
except ImportError:
    raise ImportError(
        "Cannot import gsplat. Please run:\n"
        "  pip install gsplat\n"
        "Then ensure you launch this script from a Developer Prompt so that "
        "cl.exe (and CUDA) are on your PATH."
    )

# ----------------------------------------------------------------------------- 
# 2. GaussianModel: hold N fixed‐position Gaussians; train log‐covariances & colors
# ----------------------------------------------------------------------------- 
class GaussianModel(torch.nn.Module):
    """
    Wraps N Gaussians:
      - positions   : (N,3)   fixed, no gradient
      - log_sigma2  : (N,3)   trainable (ensuring exp(log_sigma2) ≥ min_variance)
      - colors      : (N,3)   trainable (clamped to [0,1])
      - quaternions : (N,4)   fixed identity (1,0,0,0)
      - opacities   : (N,)    fixed = 1.0
    """

    def __init__(self, initial_npz_path: str, min_variance: float = 1e-6):
        """
        Args:
          initial_npz_path: path to .npz containing keys:
                              “positions” (N×3),
                              “cov_diagonals” (N×3),
                              “colors” (N×3).
          min_variance:    lower bound for each diagonal σ² to avoid collapse.
        """
        super().__init__()

        data = np.load(initial_npz_path)
        pos_np = data["positions"].astype(np.float32)       # shape = (N,3)
        cov_np = data["cov_diagonals"].astype(np.float32)   # shape = (N,3)
        col_np = data["colors"].astype(np.float32)          # shape = (N,3)

        N = pos_np.shape[0]
        if cov_np.shape[0] != N or col_np.shape[0] != N:
            raise ValueError("All arrays in initial .npz must share first dim N.")

        # 2.1 Positions as a buffer (no gradient)
        self.register_buffer("positions", torch.from_numpy(pos_np))  # (N,3)

        # 2.2 log_sigma2 ensures exp(log_sigma2) ≥ min_variance
        cov_clamped = np.maximum(cov_np, min_variance)
        log_sigma2_init = np.log(cov_clamped)  # (N,3)
        self.log_sigma2 = nn.Parameter(torch.from_numpy(log_sigma2_init))  # trainable

        self.min_variance = float(min_variance)

        # 2.3 Colors ∈ [0,1], trainable
        col_clamped = np.clip(col_np, 0.0, 1.0)
        self.colors = nn.Parameter(torch.from_numpy(col_clamped))  # (N,3)

        # 2.4 Identity quaternion for each Gaussian (w=1, x=y=z=0)
        id_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        quats = id_quat.unsqueeze(0).repeat(N, 1)  # shape = (N,4)
        self.register_buffer("quaternions", quats)

        # 2.5 Opacities = 1.0 for all N
        ops = torch.ones(N, dtype=torch.float32)
        self.register_buffer("opacities", ops)

    def get_all(self):
        """
        Returns:
          positions   : (N,3)   float32   – no grad
          scales      : (N,3)   float32   – sqrt(exp(log_sigma2)) ≥ sqrt(min_variance)
          quaternions : (N,4)   float32   – identity, no grad
          opacities   : (N,)    float32   – all 1.0, no grad
          colors      : (N,3)   float32   – clamped to [0,1], requires grad
        """
        sigma2 = torch.exp(self.log_sigma2)
        sigma2 = torch.clamp(sigma2, self.min_variance, float("inf"))  # (N,3)
        scales = torch.sqrt(sigma2)  # (N,3)
        colors_clamped = torch.clamp(self.colors, 0.0, 1.0)
        return self.positions, scales, self.quaternions, self.opacities, colors_clamped

    def clamp_parameters(self):
        """
        After each optimizer.step(), enforce:
          • colors ∈ [0,1]
          • log_sigma2 ≥ log(min_variance)
        """
        with torch.no_grad():
            self.colors.data.clamp_(0.0, 1.0)
            min_log = float(np.log(self.min_variance))
            self.log_sigma2.data.clamp_(min_log, float("inf"))

    def save_to_npz(self, out_path: str):
        """
        Save final Gaussians to .npz with keys:
          “positions”     = (N×3)
          “cov_diagonals” = (N×3) = exp(log_sigma2), clamped ≥ min_variance
          “colors”        = (N×3)
        """
        pos_np = self.positions.cpu().numpy()  # (N,3)
        cov_np = torch.exp(self.log_sigma2).clamp(self.min_variance).detach().cpu().numpy()  # (N,3)
        col_np = torch.clamp(self.colors, 0.0, 1.0).cpu().detach().numpy()  # (N,3)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path,
                            positions=pos_np,
                            cov_diagonals=cov_np,
                            colors=col_np)


# ----------------------------------------------------------------------------- 
# 3. Utility: load a color image into a CUDA tensor of shape (3, H, W)
# ----------------------------------------------------------------------------- 
def load_color_image(path: str, target_size: tuple) -> torch.Tensor:
    """
    Load a color image from `path`, resize to (H,W), normalize to [0,1], and
    permute to (3, H, W). This ensures the tensor is channel-first.

    Args:
      path        : string filepath to .png or .jpg
      target_size : (H, W) tuple

    Returns:
      Tensor of shape (3, H, W), dtype=float32, on CUDA.
    """
    img = Image.open(path).convert("RGB")
    # PIL’s resize expects (width, height), so we pass (W, H) = (target_size[1], target_size[0])
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # shape = (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).cuda()  # shape = (3, H, W)
    return tensor


# ----------------------------------------------------------------------------- 
# 4. Main training loop (with “minibatch” but single‐view rasterization per sample)
# ----------------------------------------------------------------------------- 
def main(args):
    # 4.1 Verify dataset_info exists
    dataset_info_npz = os.path.join(args.dataset_info_folder,
                                    "scene0581_00_dataset_info.npz")
    if not os.path.isfile(dataset_info_npz):
        raise FileNotFoundError(f"Cannot find {dataset_info_npz}")

    os.makedirs(args.output_folder, exist_ok=True)
    final_npz = os.path.join(args.output_folder,
                             "scene0581_00_optimized_gaussians.npz")

    # 4.2 Load dataset_info
    print(f"Loading dataset_info from: {dataset_info_npz}")
    info = np.load(dataset_info_npz, allow_pickle=True)
    gaussians_path = str(info["gaussians_path"][0])       # e.g. "data/initial_gaussians/...npz"
    cam_params_path = str(info["camera_params_path"][0])   # e.g. "data/camera_parameters/...npz"
    image_paths = info["image_paths"]                      # (1496,) array of strings
    intrinsics_np = info["intrinsics"]                     # (1496,4,4)
    extrinsics_np = info["extrinsics"]                     # (1496,4,4)

    num_frames = intrinsics_np.shape[0]
    print(f"  • Initial Gaussians path: {gaussians_path}")
    print(f"  • Camera params path:   {cam_params_path}")
    print(f"  • Num frames/images:    {num_frames}")
    if num_frames != image_paths.shape[0] or num_frames != extrinsics_np.shape[0]:
        raise ValueError("Mismatch: #images, intrinsics, and extrinsics must be equal.")

    # 4.3 Convert intrinsics (4×4) → Ks (3×3), and extrinsics (C2W) → viewmats (W2C)
    Ks_np = intrinsics_np[:, :3, :3]                   # (C,3,3)
    Ks_t = torch.from_numpy(Ks_np).float().cuda()       # (C,3,3)

    extrinsics_t = torch.from_numpy(extrinsics_np).float().cuda()  # (C,4,4)
    viewmats_t = torch.inverse(extrinsics_t)                      # (C,4,4)

    # 4.4 Determine image resolution (W, H) from first image
    sample_img = Image.open(str(image_paths[0])).convert("RGB")
    W, H = sample_img.size  # (width, height)
    print(f"Detected image resolution: W={W}, H={H}")
    target_size = (H, W)

    # 4.5 Instantiate GaussianModel and move to CUDA
    print(f"Loading initial Gaussians from: {gaussians_path}")
    model = GaussianModel(gaussians_path,
                          min_variance=args.min_variance).cuda()
    num_gauss = model.positions.shape[0]
    print(f"  • Number of Gaussians: {num_gauss}")

    # 4.6 Set up optimizer: Adam for log_sigma2 and colors
    optimizer = optim.Adam(
        [
            {"params": model.log_sigma2, "lr": args.lr_sigma},
            {"params": model.colors,    "lr": args.lr_color}
        ],
        betas=(0.9, 0.99)
    )
    print(f"Optimizer: Adam with lr_sigma={args.lr_sigma}, lr_color={args.lr_color}")

    # 4.7 Main training loop (“minibatch” over frames)
    batch_size = args.batch_size
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        # (Optional) shuffle frame order
        if args.shuffle:
            order = torch.randperm(num_frames, device="cpu").tolist()
        else:
            order = list(range(num_frames))

        # Process in groups of batch_size
        for batch_start in range(0, num_frames, batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]
            B = len(batch_indices)
            # print(f"B is: {B}")

            # 4.7a Load B target images into a single tensor of shape (B, 3, H, W)
            imgs_list = []
            for idx in batch_indices:
                img = load_color_image(str(image_paths[idx]), target_size)  # (3, H, W), on CUDA
                imgs_list.append(img)
            imgs_batch = torch.stack(imgs_list, dim=0)  # (B, 3, H, W)

            optimizer.zero_grad()

            # 4.7b Retrieve current Gaussian params (shared across the batch)
            positions_t, scales_t, quats_t, ops_t, colors_t = model.get_all()
            #    positions_t: (N, 3)
            #    scales_t:    (N, 3)
            #    quats_t:     (N, 4)
            #    ops_t:       (N,)
            #    colors_t:    (N, 3)

            # We will rasterize each view in the batch one by one, collecting outputs:
            rendered_list = []

            for idx in batch_indices:
                # 4.7c Extract single-view intrinsics & extrinsics
                viewmat_i = viewmats_t[idx].unsqueeze(0)  # (1, 4, 4)
                K_i = Ks_t[idx].unsqueeze(0)               # (1, 3, 3)

                # 4.7d Rasterize Gaussians → single rendered image
                # Returns something like (rgb_tensor_of_shape_(H,W,3), depth, meta)
                single_rendered, single_depth, single_meta = rasterization(
                    means       = positions_t,  # (N, 3)
                    quats       = quats_t,      # (N, 4)
                    scales      = scales_t,     # (N, 3)
                    opacities   = ops_t,        # (N,)
                    colors      = colors_t,     # (N, 3)
                    viewmats    = viewmat_i,    # (1, 4, 4)
                    Ks          = K_i,          # (1, 3, 3)
                    width       = W,
                    height      = H,
                    near_plane  = args.near_plane,
                    far_plane   = args.far_plane,
                    render_mode = "RGB",
                    packed      = True,
                    absgrad     = False,
                    sparse_grad = False
                )
                # single_rendered[0] has shape (H, W, 3)
                rgb_hwc = single_rendered[0]  # (H, W, 3), on CUDA

                # Convert it to (3, H, W) for consistency with imgs_batch
                # DEBUG print: NOTE
                # print(f"   DEBUG (single view): rgb_hwc.shape = {tuple(rgb_hwc.shape)}")
                rgb_chw = rgb_hwc.permute(2, 0, 1).contiguous()  # (3, H, W)
                rendered_list.append(rgb_chw)

            # Stack the B per-view outputs to shape (B, 3, H, W)
            rendered_imgs = torch.stack(rendered_list, dim=0)  # (B, 3, H, W)

            # 4.7e Compute photometric loss over this mini‐batch
            # Both rendered_imgs and imgs_batch are (B, 3, H, W)
            loss = nn.functional.mse_loss(rendered_imgs, imgs_batch)
            total_loss += loss.item() * B  # sum-of‐losses

            # 4.7f Backpropagate
            loss.backward()

            # 4.7g Optimizer step
            optimizer.step()

            # 4.7h Clamp parameters: colors ∈ [0,1], log_sigma2 ≥ log(min_variance)
            model.clamp_parameters()

            # 4.7i Print per-batch progress
            batch_idx = batch_start // batch_size
            if (batch_idx + 1) % args.print_every == 0 or (batch_start + B) >= num_frames:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Batch [{batch_idx+1}/{(num_frames + batch_size - 1)//batch_size}]  "
                    f"BatchSize={batch_size:<2d}  Loss={loss.item():.6f}"
                )

        avg_loss = total_loss / num_frames
        print(f"=== Epoch {epoch+1}/{args.num_epochs} average loss: {avg_loss:.6f} ===")

        # 4.7j Save intermediate checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_folder,
                f"scene0581_00_gaussians_epoch{epoch+1}.npz"
            )
            print(f"Saving intermediate Gaussians to: {ckpt_path}")
            model.save_to_npz(ckpt_path)

    # 4.8 Save final optimized Gaussians
    print(f"Saving final optimized Gaussians to: {final_npz}")
    model.save_to_npz(final_npz)
    print("Training complete.")


# ----------------------------------------------------------------------------- 
# 5. Command-line entrypoint
# ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="05test.py: Optimize 3D Gaussian covariances & colors using gsplat."
    )
    parser.add_argument(
        "--dataset_info_folder",
        type=str,
        default="data/dataset_info",
        help="Folder containing scene0581_00_dataset_info.npz"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/optimized_gaussians",
        help="Folder to save optimized Gaussians (.npz)"
    )
    parser.add_argument(
        "--lr_sigma",
        type=float,
        default=5e-4,
        help="Learning rate for updating log_sigma2 (covariances)"
    )
    parser.add_argument(
        "--lr_color",
        type=float,
        default=5e-4,
        help="Learning rate for updating colors"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs (full passes over all 1496 frames)"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print per‐batch loss every N batches"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save intermediate Gaussians every N epochs"
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=1e-6,
        help="Lower bound on each covariance diagonal (σ² ≥ min_variance)"
    )
    parser.add_argument(
        "--near_plane",
        type=float,
        default=0.02,
        help="Near clipping plane (meters)"
    )
    parser.add_argument(
        "--far_plane",
        type=float,
        default=10.0,
        help="Far clipping plane (meters)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of frames to group per optimizer step"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle frame order each epoch"
    )
    args = parser.parse_args()
    main(args)
