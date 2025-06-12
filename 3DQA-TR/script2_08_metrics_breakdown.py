# script2_08_metrics_breakdown.py

"""
Per-category evaluation for 3DQA-TR (§5.1 Breakdown Analysis).
Computes Exact Match (EM) and METEOR for each question category:
  - Answer types: Y/N, Color, Number, Other
  - Spatial subtasks: aggregation, placement, spatial, viewpoint
Requires:
  - script2_02_build_3dqa_dataset.py → ThreeDQA_Dataset
  - script2_05_3dl_bert.py         → ThreeDLBERT
  - `evaluate` library installed (`pip install evaluate`)
  - data/splits/test_qa.json with 'question_type' or 'answer_type'
  - best_3dqa_tr.pth checkpoint
"""

import json
import torch
from torch.utils.data import DataLoader
from evaluate import load as load_metric  # pip install evaluate

# Custom modules
from script2_02_build_3dqa_dataset import ThreeDQA_Dataset
from script2_05_3dl_bert import ThreeDLBERT

def compute_em(preds, refs):
    """Exact Match percentage."""
    return sum(p == r for p, r in zip(preds, refs)) / len(refs) * 100.0

def compute_meteor(preds, refs, meteor):
    """METEOR score (percentage)."""
    # meteor expects lists of reference lists
    res = meteor.compute(predictions=preds, references=[[r] for r in refs])
    return res['meteor'] * 100.0

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load test entries to extract types
    with open('data/splits/test_qa.json', 'r') as f:
        test_entries = json.load(f)

    # 2) Prepare dataset & loader
    test_ds = ThreeDQA_Dataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    # 3) Load answer vocab and build idx2ans
    with open('data/answer_vocab.json', 'r') as vf:
        ans2idx = json.load(vf)
    idx2ans = {int(v): k for k, v in ans2idx.items()}

    # 4) Load model
    model = ThreeDLBERT(num_classes=test_ds.num_classes).to(device)
    model.load_state_dict(torch.load('best_3dqa_tr.pth', map_location=device))
    model.eval()

    # 5) Prepare metrics
    meteor_metric = load_metric('meteor')

    # 6) Run inference and collect preds, refs, types
    all_preds, all_refs, all_types = [], [], []
    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            pc   = batch['pointcloud'].to(device)
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().tolist()
            B = pc.size(0)

            # placeholder scene_range; replace with actual if available
            scene_range = torch.tensor([[2.0,2.0,2.0]]).repeat(B,1).to(device)

            logits = model(pc, scene_range, ids, mask)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            for p_idx, g_idx in zip(preds, labels):
                all_preds.append(idx2ans[p_idx])
                all_refs.append(idx2ans[g_idx])
                entry = test_entries[idx]
                # prefer explicit question_type, else answer_type, else Other
                qtype = entry.get('question_type',
                                  entry.get('answer_type', 'Other'))
                all_types.append(qtype)
                idx += 1

    # 7) Determine unique categories in the test set
    categories = sorted(set(all_types))

    # 8) Compute and print EM & METEOR for each category
    print("Per-Category Evaluation:")
    for cat in categories:
        # filter by category
        cat_preds = [p for p, t in zip(all_preds, all_types) if t == cat]
        cat_refs  = [r for r, t in zip(all_refs,  all_types) if t == cat]
        if not cat_preds:
            continue
        em_score = compute_em(cat_preds, cat_refs)
        meteor_score = compute_meteor(cat_preds, cat_refs, meteor_metric)
        print(f"  {cat:12s} |  EM: {em_score:5.2f}%  METEOR: {meteor_score:5.2f}%  ({len(cat_preds)} samples)")

    # 9) Overall scores
    overall_em = compute_em(all_preds, all_refs)
    overall_meteor = compute_meteor(all_preds, all_refs, meteor_metric)
    print("\nOverall:")
    print(f"  Exact Match: {overall_em:.2f}%")
    print(f"  METEOR:      {overall_meteor:.2f}%")
