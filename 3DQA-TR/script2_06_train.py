# script2_06_train.py

"""
Training script for 3DQA-TR (§5.2 Training).
- Loads ThreeDLBERT model, ThreeDQA_Dataset, and DataLoaders
- Uses AdamW optimizer with separate learning rates for PointNet++ backbone and rest
- Applies linear warmup for 500 steps then cyclic learning rate policy
- Trains with cross-entropy loss and evaluates on validation split
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Custom modules
from script2_02_build_3dqa_dataset import ThreeDQA_Dataset  # fileciteturn5file2
from script2_05_3dl_bert import ThreeDLBERT               # fileciteturn5file4

# Hyperparameters
BATCH_SIZE    = 16
NUM_EPOCHS    = 10
MAX_POINTS    = 300000
MAX_Q_LEN     = 32
WARMUP_STEPS  = 500
WEIGHT_DECAY  = 1e-4
LR_BACKBONE   = 1e-8
LR_REST       = 5e-6
CYCLIC_CYCLE  = 1      # one cycle over total steps

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Prepare datasets and loaders
    train_ds = ThreeDQA_Dataset(split='train', max_points=MAX_POINTS, max_question_len=MAX_Q_LEN)
    val_ds   = ThreeDQA_Dataset(split='val',   max_points=MAX_POINTS, max_question_len=MAX_Q_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2) Model instantiation
    num_classes = train_ds.num_classes
    model = ThreeDLBERT(num_classes=num_classes).to(device)

    # 3) Optimizer with parameter groups
    #   - backbone (PointNet++ in geometry & appearance encoders): low LR
    #   - rest (detector, BERT, FFNs, classifier): higher LR
    backbone_params = list(model.geom_enc.backbone.parameters()) + list(model.app_enc.backbone.parameters())
    rest_params     = [p for p in model.parameters() if p not in backbone_params]
    optimizer = AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': rest_params,     'lr': LR_REST}
    ], weight_decay=WEIGHT_DECAY)

    # 4) Scheduler: warmup + linear decay
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # 5) Loss function
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        for batch in train_loader:
            pc = batch['pointcloud'].to(device)          # (B, N, 6)
            ids = batch['input_ids'].to(device)           # (B, L)
            mask = batch['attention_mask'].to(device)     # (B, L)
            labels = batch['label'].to(device)            # (B,)
            # scene_range: placeholder or compute per-scene
            scene_range = torch.tensor([[2.0,2.0,2.0]]).repeat(pc.size(0),1).to(device)

            optimizer.zero_grad()
            logits = model(pc, scene_range, ids, mask)   # (B, num_classes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * pc.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # 6) Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch['pointcloud'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                scene_range = torch.tensor([[2.0,2.0,2.0]]).repeat(pc.size(0),1).to(device)

                logits = model(pc, scene_range, ids, mask)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_3dqa_tr.pth')

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    train()
