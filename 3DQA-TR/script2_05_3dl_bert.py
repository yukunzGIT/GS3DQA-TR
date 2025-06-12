# script2_05_3dl_bert.py

"""
3D-Linguistic BERT integration for 3DQA-TR (§4.4).
This script fuses geometry, appearance, and linguistic embeddings into a unified BERT-based model.

Pre-requisites:
- script2_03_geometry_encoder.py exports GeometryEncoder
- script2_04_appearance_encoder.py exports AppearanceEncoder
- transformers library installed (`pip install transformers`)
"""
import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AutoModel, BertTokenizer, BertConfig

# Import custom encoders
from script2_03_geometry_encoder import GeometryEncoder  # assumes it returns (geom_feats, spatial_emb, global_geom, I1, I2)
from script2_04_appearance_encoder import AppearanceEncoder

class ThreeDLBERT(nn.Module):
    """
    3D-Linguistic BERT model combining:
      - Geometry elements: per-object geometry + spatial embeddings
      - Appearance elements: per-object appearance features
      - Linguistic elements: question token embeddings

    See §4.4 in the 3DQA-TR Framework.
    """
    def __init__(
        self,
        num_classes: int,
        bert_model_name: str = 'bert-base-uncased',
        hidden_size: int = 768,
        num_objects: int = 64,
    ):
        super().__init__()
        # --- Encoders ---
        self.geom_enc = GeometryEncoder(num_objects=num_objects)
        self.app_enc  = AppearanceEncoder(num_objects=num_objects)

        # --- BERT backbone ---
        bert_model_name = 'bert-base-uncased' # NOTE
        self.bert = BertModel.from_pretrained(bert_model_name)

        #bert_model_name = 'roberta-base' # NOTE
        #self.bert = RobertaModel.from_pretrained(bert_model_name)

        #bert_model_name = 'SpanBERT/spanbert-base-cased' # NOTE
        #self.bert = AutoModel.from_pretrained(bert_model_name)


        # freeze language layers if desired:
        # for param in self.bert.parameters(): param.requires_grad = False

        # --- Element embedding layers ---
        # Geometry: concat geom_feats + spatial_emb => embed to hidden_size
        self.geo_ffn = nn.Sequential(
            nn.Linear(self.geom_enc.backbone_feat_dim + self.app_enc.feat_dim, hidden_size),
            nn.ReLU()
        )
        # Appearance: embed app_feats to hidden_size
        self.app_ffn = nn.Sequential(
            nn.Linear(self.app_enc.feat_dim, hidden_size),
            nn.ReLU()
        )

        # CLS token as learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_size))
        # SEP token embedding (reuse BERT's [SEP])
        self.sep_token_emb = self.bert.embeddings.word_embeddings(
            torch.tensor([self.bert.config.sep_token_id])
        ).unsqueeze(0)  # (1,1,hidden_size)

        # Final classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pointcloud: torch.Tensor,      # (B, N, 6) xyzrgb
        scene_range: torch.Tensor,     # (B, 3)
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ):
        B, N, _ = pointcloud.shape
        # --- Geometry encoding ---
        # Split xyz and rgb
        xyz = pointcloud[..., :3]
        rgb = pointcloud[..., 3:]
        # geometry encoder returns indices I1, I2 internally
        geom_feats, spatial_emb, global_geom, I1, I2 = self.geom_enc(xyz, scene_range)
        # geom_feats: (B, K, C_geo); spatial_emb: (B, K, C_spatial)

        # --- Appearance encoding ---
        xyzrgb = pointcloud
        app_feats, global_app = self.app_enc(xyzrgb, I1, I2)
        # app_feats: (B, K, C_app)

        K = geom_feats.size(1)
        # --- Prepare element embeddings ---
        # 1) CLS element
        cls_emb = self.cls_token.expand(B, -1, -1)  # (B,1,hidden_size)
        # 2) Geometry elements
        # concat geom_feats + spatial_emb
        geo_concat = torch.cat([geom_feats, spatial_emb], dim=-1)  # (B,K,C_geo+C_spatial)
        geo_emb = self.geo_ffn(geo_concat)  # (B,K,hidden_size)
        # prepend scene-level global geometry as first geo element if desired
        # 3) Appearance elements
        app_emb = self.app_ffn(app_feats)   # (B,K,hidden_size)
        # 4) Linguistic elements: question tokens
        # get word-piece embeddings
        token_emb = self.bert.embeddings.word_embeddings(input_ids)  # (B,L,hidden_size)
        # add position and token-type embeddings inside BERT when forwarded

        # 5) SEP element
        sep_emb = self.sep_token_emb.expand(B, -1, -1)  # (B,1,hidden_size)

        # Concatenate all elements into sequence
        # order: [CLS] + geo_emb + app_emb + [SEP] + token_emb
        seq = torch.cat([cls_emb, geo_emb, app_emb, sep_emb, token_emb], dim=1)
        # Build attention mask for new sequence
        # Mask for CLS + geo + app + SEP = all ones
        modal_mask = torch.ones(B, 1 + K + K + 1, device=seq.device, dtype=torch.long)
        # concatenate with original attention_mask
        seq_attention_mask = torch.cat([modal_mask, attention_mask], dim=1)

        # --- Feed through BERT ---
        # We bypass token_type_ids and rely on BERT to learn modalities
        bert_out = self.bert(
            inputs_embeds=seq,
            attention_mask=seq_attention_mask
        )
        # pooled output = first token ([CLS]) hidden state
        pooled = bert_out.pooler_output  # (B,hidden_size)

        # --- Classification ---
        logits = self.classifier(pooled)  # (B,num_classes)
        return logits

if __name__ == "__main__":
    # Quick instantiation test
    B, N, L = 2, 10000, 16
    num_classes = 1000
    model = ThreeDLBERT(num_classes=num_classes)
    dummy_pc = torch.rand(B, N, 6)
    dummy_range = torch.tensor([[2.0,2.0,2.0]]).repeat(B,1)
    dummy_ids = torch.randint(0, model.bert.config.vocab_size, (B, L))
    dummy_mask = torch.ones(B, L, dtype=torch.long)
    out = model(dummy_pc, dummy_range, dummy_ids, dummy_mask)
    print("Logits shape:", out.shape)  # expect (B, num_classes)
