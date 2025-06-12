# script2_03_geometry_encoder.py

"""
Geometry encoder for 3DQA-TR using PointNet++ and Group-Free 3D detector.
Pre-requisites:
1. Clone Group-Free-3D repository and compile PointNet++ CUDA layers:
   ```bash
   git clone https://github.com/zeliu98/Group-Free-3D.git
   cd Group-Free-3D && sh init.sh        # compiles pointnet2 CUDA
   ```
2. Install PointNet2_PyTorch:
   ```bash
   git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
   pip install -e Pointnet2_PyTorch
   ```
3. Ensure your PYTHONPATH includes the root of Group-Free-3D:
   ```bash
   export PYTHONPATH=/path/to/Group-Free-3D:$PYTHONPATH
   ```

This script implements §4.2 of the 3DQA-TR Framework ("Geometry Encoder").
"""
import math
import torch
import torch.nn as nn

# 1) PointNet++ Set Abstraction (MSG) for backbone features
from pointnet2.pointnet2_modules import PointnetSAModuleMSG  # fileciteturn2file0

# 2) Group-Free 3D detector head (import from cloned repo)
from models.detector import GroupFreeDetector

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 12-D box vectors (Eq. 1 in §4.2).
    Input: box_vec (B, K, 12)
    Output: (B, K, 12 * d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, box_vec: torch.Tensor) -> torch.Tensor:
        B, K, D = box_vec.shape  # D == 12
        # expand dims to (B, K, 12, 1)
        pe = box_vec.unsqueeze(-1)
        # frequency: 1/(10000^(2i/d_model))  shape (1,1,1,d_model/2)
        freq = torch.exp(
            -torch.arange(0, self.d_model, 2, device=box_vec.device).float()
            * (math.log(10000.0) / self.d_model)
        ).view(1,1,1,-1)
        # apply sin & cos
        pe_sin = torch.sin(pe * freq)
        pe_cos = torch.cos(pe * freq)
        # interleave and flatten to (B, K, 12, d_model)
        pe_stacked = torch.stack((pe_sin, pe_cos), dim=-1).flatten(2)
        # reshape to (B, K, 12*d_model)
        return pe_stacked.view(B, K, D * self.d_model)

class GeometryEncoder(nn.Module):
    """
    Geometry Encoder (§4.2):
    - PointNet++ backbone → point features FM  (M points)
    - Initial proposals P via I1 sampling → proposal feats FP
    - Stacked attentions & 3D NMS → top K boxes & features
    - Index-select on FM (via I1,I2) & FP → per-object geom_feats
    - Compute spatial embeddings via PositionalEncoding
    - Global geometry feature via scene box PE + avg-pool
    """
    def __init__(self,
                 num_points_feat: int = 1024,  # M points
                 d_model: int = 256,           # PE dimension
                 num_objects: int = 64,        # K
                 gf_cfg: str = "configs/GroupFree3D.yaml",
                 gf_ckpt: str = "pretrained/groupfree.pth"):
        super().__init__()
        # --- 1) PointNet++ MSG backbone ---
        mlps = [[3,64,64,128], [3,64,64,128], [3,64,96,128]]
        self.backbone = PointnetSAModuleMSG(
            npoint=num_points_feat,
            radii=[0.1,0.2,0.4],
            nsamples=[32,64,128],
            mlps=mlps
        )
        # sum of mlp output dims = 128+128+128
        self.backbone_feat_dim = sum(m[-1] for m in mlps)

        # --- 2) Load pretrained Group-Free detector ---
        # (Assumes you've cloned the repo and added to PYTHONPATH)
        self.detector = GroupFreeDetector(cfg_path=gf_cfg)
        ckpt = torch.load(gf_ckpt, map_location="cpu")
        self.detector.load_state_dict(ckpt["model_state"])
        self.detector.eval()

        # --- 3) Positional encoding ---
        self.spatial_pe = PositionalEncoding(d_model=d_model)
        self.num_objects = num_objects
        self.d_model = d_model

        # --- 4) Global geometry FFN ---
        global_in = self.backbone_feat_dim + 12 * d_model
        self.global_ffn = nn.Sequential(
            nn.Linear(global_in, self.backbone_feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, scene_range: torch.Tensor):
        """
        xyz:         (B, N, 3) input points (no color)
        scene_range:(B, 3) scan scales [x_range,y_range,z_range]
        Returns:
          geom_feats:  (B, K, C_geo) per-object geometry
          spatial_emb: (B, K, 12*d_model)
          global_geom: (B, C_geo)
        """
        B, N, _ = xyz.shape
        # 1) Backbone → FM: (B, M, C_back)
        pts = xyz.permute(0,2,1).contiguous()
        new_xyz, new_feats = self.backbone(pts, None)
        fm_xyz = new_xyz.permute(0,2,1).contiguous()
        fm =    new_feats.permute(0,2,1).contiguous()

        # 2) Detector → proposals & features
        det_in = {"points": fm_xyz, "features": fm}
        det_out, _ = self.detector(det_in)
        I1      = det_out['init_proposal_indices']  # (B, P)
        FP      = det_out['proposal_features']      # (B, P, C_det)
        I2      = det_out['nms_indices']            # (B, K)
        boxes   = det_out['pred_boxes']             # (B, K, 6)

        # 3a) Index-select FM via I1→I2 → geo_back (B,K,C_back)
        C_back = self.backbone_feat_dim
        geo_back = torch.gather(
            torch.gather(fm, 1, I1.unsqueeze(-1).expand(-1,-1,C_back)),
            1, I2.unsqueeze(-1).expand(-1,-1,C_back)
        )
        # 3b) Index-select FP via I2 → geo_ref (B,K,C_det)
        C_det = FP.size(-1)
        geo_ref = torch.gather(
            FP, 1, I2.unsqueeze(-1).expand(-1,-1,C_det)
        )
        # 3c) Concat → geom_feats (B,K,C_back+C_det)
        geom_feats = torch.cat([geo_back, geo_ref], dim=-1)

        # 4) Spatial embedding for each box
        centers = boxes[...,:3] / scene_range.view(B,1,3)
        sizes   = boxes[...,3:]  / scene_range.view(B,1,3)
        mins    = centers - sizes/2
        maxs    = centers + sizes/2
        box_vec = torch.cat([centers,sizes,mins,maxs], dim=-1)
        spatial_emb = self.spatial_pe(box_vec)

        # 5) Global geometry
        scene_min = torch.zeros_like(centers)
        scene_max = scene_range.view(B,1,3).expand_as(centers)
        g_center = (scene_max + scene_min)/2  / scene_range.view(B,1,3)
        g_size   = (scene_max - scene_min)/1 / scene_range.view(B,1,3)
        g_box    = torch.cat([g_center,g_size,scene_min,scene_max/scene_range.view(B,1,3)], dim=-1)
        g_spatial= self.spatial_pe(g_box)
        global_back = fm.mean(dim=1)
        global_geom = self.global_ffn(torch.cat([global_back, g_spatial.squeeze(1)], dim=-1))

        return geom_feats, spatial_emb, global_geom

if __name__ == "__main__":
    # sanity check
    B, N = 2, 80000
    dummy_xyz   = torch.rand(B, N, 3)
    dummy_range = torch.tensor([[2.0,2.0,2.0]]).repeat(B,1)
    enc = GeometryEncoder()
    gf, se, gg = enc(dummy_xyz, dummy_range)
    print("geom_feats:", gf.shape)
    print("spatial_emb:", se.shape)
    print("global_geom:", gg.shape)
