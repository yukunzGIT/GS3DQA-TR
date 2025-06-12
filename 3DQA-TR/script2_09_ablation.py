# script2_09_ablation.py
# Optional step
"""
Ablation studies for 3DQA-TR (§5.4 Component Validation).
Runs evaluation under three settings:
 1) Q-only    : question tokens through BERT only
 2) Geo+Q     : geometry elements + question
 3) App+Q     : appearance elements + question
Optionally: AppFromScratch pretraining ablation.

Imports previous scripts and reuses ThreeDLBERT, ThreeDQA_Dataset.
Produces EM for each setting on the test split.
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from script2_02_build_3dqa_dataset import ThreeDQA_Dataset
from script2_05_3dl_bert import ThreeDLBERT

# Utility: compute exact match

def exact_match(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(refs) * 100.0

class QOnlyModel(nn.Module):
    """BERT-only baseline: ignores 3D inputs."""
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)

@torch.no_grad()
def evaluate_setting(model, loader, idx2ans, device, setting):
    preds, refs = [], []
    model.eval()
    for batch in loader:
        # unpack
        pc    = batch['pointcloud'].to(device)
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().tolist()
        # scene_range
        B = pc.size(0)
        scene_range = torch.tensor([[2.0,2.0,2.0]]).repeat(B,1).to(device)

        if setting == 'Qonly':
            logits = model(ids, mask)                # QOnlyModel: forward(args)
        else:
            # Geo+Q or App+Q
            # ThreeDLBERT.forward uses both geom & app; we monkey-patch
            # disable one modality by zeroing embeddings
            if setting == 'Geo+Q':
                # zero appearance encoder
                orig_app_forward = model.app_enc.forward
                model.app_enc.forward = lambda xyzrgb, I1, I2: (torch.zeros_like(model.app_enc.backbone(xyzrgb[:,...:3].permute(0,2,1), xyzrgb[...,3:].permute(0,2,1))[1].permute(0,2,1)), torch.zeros(xyzrgb.size(0), model.app_enc.feat_dim, device=device))
            elif setting == 'App+Q':
                # zero geometry encoder
                orig_geo_forward = model.geom_enc.forward
                model.geom_enc.forward = lambda xyz, sr: (torch.zeros((xyz.size(0), model.geom_enc.num_objects, model.geom_enc.backbone_feat_dim*2), device=device), torch.zeros((xyz.size(0), model.geom_enc.num_objects, 12*model.geom_enc.d_model), device=device), torch.zeros((xyz.size(0), model.geom_enc.backbone_feat_dim), device=device), torch.zeros((xyz.size(0),128),dtype=torch.long), torch.zeros((xyz.size(0),model.geom_enc.num_objects),dtype=torch.long))

            logits = model(pc, scene_range, ids, mask)
            # restore
            if setting == 'Geo+Q': model.app_enc.forward = orig_app_forward
            if setting == 'App+Q': model.geom_enc.forward = orig_geo_forward

        batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
        for p, g in zip(batch_preds, labels):
            preds.append(idx2ans[p]); refs.append(idx2ans[g])
    return exact_match(preds, refs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load test dataset
    test_ds = ThreeDQA_Dataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)
    # Load vocab
    with open('data/answer_vocab.json','r') as f:
        ans2idx = json.load(f)
    idx2ans = {int(v):k for k,v in ans2idx.items()}

    # Q-only model
    qonly = QOnlyModel(num_classes=test_ds.num_classes).to(device)
    qonly.bert.config.max_position_embeddings = test_ds.max_question_len
    # assume qonly is finetuned; load checkpoint if available

    # ThreeDLBERT full model
    full = ThreeDLBERT(num_classes=test_ds.num_classes).to(device)
    full.load_state_dict(torch.load('best_3dqa_tr.pth', map_location=device))

    for setting, model in [('Qonly', qonly), ('Geo+Q', full), ('App+Q', full)]:
        score = evaluate_setting(model, test_loader, idx2ans, device, setting)
        print(f"{setting} Exact Match: {score:.2f}%")
