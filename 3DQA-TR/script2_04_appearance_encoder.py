# script2_04_appearance_encoder.py

"""
Appearance encoder for 3DQA-TR using PointNet++ on colored point clouds.
Follows §4.3 of the 3DQA-TR Framework:
- Input: S ∈ ℝ^{N×6} (xyz + rgb)
- Backbone: PointNet++ MSG to extract FS ∈ ℝ^{M×C_app}
- Use indices I1 (initial proposals) and I2 (after NMS) from GeometryEncoder
  to select per-object appearance features: F_app = Idx(Idx(FS, I1), I2)
- Global appearance feature: avg-pool over FS

Pre-requisites:
- Pointnet2_PyTorch installed (`pip install -e Pointnet2_PyTorch`)
"""
import torch
import torch.nn as nn

# 1) PointNet++ MSG backbone for appearance features
from pointnet2.pointnet2_modules import PointnetSAModuleMSG  # 

class AppearanceEncoder(nn.Module):
    """
    Appearance Encoder (§4.3):
    - Extract per-point color features with PointNet++
    - Index-select by I1, I2 to get per-object appearance feats
    - Global appearance by average pooling
    """
    def __init__(
        self,
        num_points_feat: int = 1024,  # M
        num_objects: int = 64         # K
    ):
        super().__init__()
        # Define PointNet++ MSG: input channels = 6 (xyzrgb)
        mlps = [[6, 64, 64, 128], [6, 64, 64, 128], [6, 96, 96, 128]]
        self.backbone = PointnetSAModuleMSG(
            npoint=num_points_feat,
            radii=[0.1, 0.2, 0.4],
            nsamples=[32, 64, 128],
            mlps=mlps
        )
        # Feature dimension after MSG = sum of last dims = 128*3 = 384
        self.feat_dim = sum([m[-1] for m in mlps])
        self.num_objects = num_objects

    def forward(self,
                xyzrgb: torch.Tensor,
                I1: torch.LongTensor,
                I2: torch.LongTensor
                ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzrgb: (B, N, 6) tensor of point cloud with colors
            I1:     (B, P) initial proposal indices from GeometryEncoder
            I2:     (B, K) NMS-selected indices
        Returns:
            app_feats:    (B, K, C_app) per-object appearance features
            global_app:   (B, C_app) global appearance feature
        """
        B, N, _ = xyzrgb.shape
        # Split positions and colors
        xyz = xyzrgb[:, :, :3].permute(0, 2, 1).contiguous()    # (B, 3, N)
        rgb = xyzrgb[:, :, 3:].permute(0, 2, 1).contiguous()    # (B, 3, N)

        # Backbone MSG: new_xyz unused, new_feats = (B, C_app, M)
        new_xyz, new_feats = self.backbone(xyz, rgb)
        # Permute to (B, M, C_app)
        FS = new_feats.permute(0, 2, 1).contiguous()

        # Index-select FS by I1 (→ P proposals)
        # Expand I1 to select across feature dim
        fm1 = torch.gather(
            FS, 1,
            I1.unsqueeze(-1).expand(-1, -1, self.feat_dim)
        )  # (B, P, C_app)

        # Then select by I2 to get K objects
        app_feats = torch.gather(
            fm1, 1,
            I2.unsqueeze(-1).expand(-1, -1, self.feat_dim)
        )  # (B, K, C_app)

        # Global appearance: average-pool FS across points
        global_app = FS.mean(dim=1)  # (B, C_app)

        return app_feats, global_app

if __name__ == "__main__":
    # Quick sanity check
    B, N = 2, 100000
    dummy_xyzrgb = torch.rand(B, N, 6)
    # simulate indices: P=128, K=64
    I1 = torch.randint(0, 1024, (B, 128), dtype=torch.long)
    I2 = torch.randint(0, 128, (B, 64), dtype=torch.long)
    encoder = AppearanceEncoder()
    af, ga = encoder(dummy_xyzrgb, I1, I2)
    print("app_feats:", af.shape)   # expect (B, 64, 384)
    print("global_app:", ga.shape)  # expect (B, 384)
