# script2_07_evaluate.py

"""
Evaluation script for 3DQA-TR (§5.1 Evaluation Metrics).
Computes Exact Match (EM) and METEOR scores on the test split.

Pre-requisites:
- script2_05_3dl_bert.py exports ThreeDLBERT
- script2_02_build_3dqa_dataset.py exports ThreeDQA_Dataset
- `evaluate` library installed (`pip install evaluate`)
"""
import json
import torch
from torch.utils.data import DataLoader
from evaluate import load as load_metric  # pip install evaluate

# Custom modules
from script2_02_build_3dqa_dataset import ThreeDQA_Dataset
from script2_05_3dl_bert import ThreeDLBERT

# Hyperparameters
BATCH_SIZE = 16
MAX_POINTS = 300000
MAX_Q_LEN  = 32

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load test dataset and DataLoader
    test_ds = ThreeDQA_Dataset(split='test', max_points=MAX_POINTS, max_question_len=MAX_Q_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2) Load inverted answer vocabulary idx2ans
    with open('data/answer_vocab.json', 'r') as vf:
        ans2idx = json.load(vf)
    idx2ans = {int(v): k for k, v in ans2idx.items()}

    # 3) Instantiate model and load best checkpoint
    num_classes = test_ds.num_classes
    model = ThreeDLBERT(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_3dqa_tr.pth', map_location=device))
    model.eval()

    # 4) Metrics: Exact Match and METEOR
    em_metric = load_metric('exact_match')  # built-in metric
    meteor_metric = load_metric('meteor')

    all_preds = []
    all_refs  = []

    # 5) Inference loop
    with torch.no_grad():
        for batch in test_loader:
            pc   = batch['pointcloud'].to(device)     # (B,N,6)
            ids  = batch['input_ids'].to(device)      # (B,L)
            mask = batch['attention_mask'].to(device) # (B,L)
            labels = batch['label'].cpu().tolist()    # (B,)
            # dummy scene_range; replace with per-scene if available
            scene_range = torch.tensor([[2.0,2.0,2.0]]).repeat(pc.size(0),1).to(device)

            logits = model(pc, scene_range, ids, mask)  # (B,C)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            # Convert indices to answer strings
            for p, g in zip(preds, labels):
                pred_ans = idx2ans[p]
                gold_ans = idx2ans[g]
                all_preds.append(pred_ans)
                all_refs.append(gold_ans)

    # 6) Compute Exact Match: fraction of exact string matches
    em_count = sum(p == r for p, r in zip(all_preds, all_refs))
    em_score = em_count / len(all_refs) * 100.0  # percentage

    # 7) Compute METEOR
    # meteor expects lists of references; wrap each ref in list
    meteor_res = meteor_metric.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs]
    )
    meteor_score = meteor_res['meteor'] * 100.0  # percentage

    # 8) Report
    print(f"Evaluation Results on Test Split:")
    print(f"  Exact Match (EM): {em_score:.2f}%")
    print(f"  METEOR Score:     {meteor_score:.2f}%")

if __name__ == '__main__':
    evaluate()
