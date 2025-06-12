import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

# script02_build_3dqa_dataset.py
# This script constructs a PyTorch Dataset for the 3DQA-TR framework.
# It loads point-cloud data and QA splits, tokenizes questions with BERT,
# builds an answer vocabulary, and maps answers to indices.

class ThreeDQA_Dataset(Dataset):
    def __init__(self, 
                 split: str,
                 pc_root: str = 'data/final_pointcloud',
                 qa_split_dir: str = 'data/splits',
                 max_points: int = 80000,
                 tokenizer_name: str = 'bert-base-uncased',
                 max_question_len: int = 32):
        """
        Args:
            split: 'train', 'val', or 'test'
            pc_root: directory with sceneXXX_final_pointcloud.npz files
            qa_split_dir: directory containing {split}_qa.json files
            max_points: max number of points to sample per scene
            tokenizer_name: HuggingFace tokenizer to use
            max_question_len: maximum token length for questions
        """
        self.split = split
        # Load QA entries for this split
        qa_path = os.path.join(qa_split_dir, f"{split}_qa.json")
        with open(qa_path, 'r') as f:
            self.qa_list = json.load(f)

        # Build answer vocabulary from training split
        if split == 'train':
            answers = [entry['answer'] for entry in self.qa_list]
            self.answer2idx = {ans: idx for idx, ans in enumerate(sorted(set(answers)))}
            # save vocab
            with open('data/answer_vocab.json', 'w') as vf:
                json.dump(self.answer2idx, vf)
        else:
            # load existing vocab
            with open('data/answer_vocab.json', 'r') as vf:
                self.answer2idx = json.load(vf)

        self.num_classes = len(self.answer2idx)

        # Initialize BERT tokenizer for question encoding
        tokenizer_name = 'bert-base-uncased' # NOTE 
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        #tokenizer_name = 'roberta-base'
        #self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

        #tokenizer_name = 'SpanBERT/spanbert-base-cased'
        #self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_question_len = max_question_len
        self.pc_root = pc_root
        self.max_points = max_points

    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self, idx):
        # Get QA entry
        entry = self.qa_list[idx]
        scene_id = entry['scene_id']  # e.g., 'scene0581_00'
        question = entry['question']
        answer = entry['answer']

        # Load and subsample point cloud
        pc_file = os.path.join(self.pc_root, f"{scene_id}_final_pointcloud.npz")
        data = np.load(pc_file)
        points = data['points']  # (N,3)
        colors = data['colors']  # (N,3)
        N = points.shape[0]
        if N > self.max_points:
            perm = np.random.permutation(N)[:self.max_points]
            points = points[perm]
            colors = colors[perm]
        # normalize colors
        colors = colors.astype(np.float32) / 255.0
        # (max_points,6)
        pc_tensor = torch.from_numpy(np.concatenate([points.astype(np.float32), colors], axis=1))

        # Tokenize question
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_question_len,
            return_tensors='pt'
        )
        # Squeeze batch dimension
        input_ids = encoded['input_ids'].squeeze(0)         # (L,)
        attention_mask = encoded['attention_mask'].squeeze(0)  # (L,)

        # Encode answer to class index
        label = self.answer2idx.get(answer, -1)

        return {
            'pointcloud': pc_tensor,           # FloatTensor[max_points,6]
            'input_ids': input_ids,            # LongTensor[max_question_len]
            'attention_mask': attention_mask,  # LongTensor[max_question_len]
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    # Example instantiation and data loader
    train_dataset = ThreeDQA_Dataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Inspect one batch
    batch = next(iter(train_loader))
    print('Pointcloud batch shape:', batch['pointcloud'].shape)
    print('Input IDs shape:', batch['input_ids'].shape)
    print('Attention mask shape:', batch['attention_mask'].shape)
    print('Label shape:', batch['label'].shape)
    print('Number of answer classes:', train_dataset.num_classes)
