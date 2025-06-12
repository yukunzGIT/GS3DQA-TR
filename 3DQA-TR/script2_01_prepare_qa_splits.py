import json
import os

# script01_prepare_qa_splits.py
# This script loads the ScanQA all_qa.json file, extracts question-answer pairs,
# splits them into train, val, and test sets (if split info exists),
# and writes separate JSON files for each split.

def load_all_qa(json_path):
    """
    Load all QA pairs from a JSON file.
    Handles multiple possible data layouts:
      1) Separate 'train', 'val', 'test' lists in the root.
      2) Single dict with 'questions', 'answers', optional 'scene_ids', 'split'.
      3) A flat list of QA dicts each containing 'question', 'answer', etc.
    Returns a flat list of QA dicts, each with at least 'question', 'answer', and 'split'.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    qa_pairs = []

    # Case 1: separate train/val/test keys
    if all(k in data for k in ('train', 'val', 'test')):
        for split in ('train', 'val', 'test'):
            for entry in data[split]:
                entry['split'] = split
                qa_pairs.append(entry)

    # Case 2: questions & answers arrays
    elif isinstance(data, dict) and 'questions' in data and 'answers' in data:
        questions = data['questions']          # list of question strings
        answers   = data.get('answers', [])    # list of answer strings
        scene_ids = data.get('scene_ids', [None] * len(questions))
        splits    = data.get('split',    [None] * len(questions))
        # Build dict per QA pair
        for q, a, sid, sp in zip(questions, answers, scene_ids, splits):
            qa_pairs.append({
                'scene_id': sid,
                'question': q,
                'answer':   a,
                'split':    sp or 'train'
            })
    # Case 3: fallback, assume a flat list of QA dicts
    else:
        for entry in data:
            # ensure 'split' field exists
            entry.setdefault('split', 'train')
            qa_pairs.append(entry)

    return qa_pairs

if __name__ == '__main__':
    # Path to the original QA JSON
    json_path = 'data/all_qa.json'

    # Load and consolidate QA pairs
    qa_list = load_all_qa(json_path)

    # Group QA by split
    splits = {'train': [], 'val': [], 'test': []}
    for qa in qa_list:
        sp = qa.get('split', 'train')
        splits.setdefault(sp, []).append(qa)

    # Print counts for verification
    for sp in ('train', 'val', 'test'):
        print(f"{sp}: {len(splits.get(sp, []))} QA pairs")

    # Save each split into its own JSON file
    os.makedirs('data/splits', exist_ok=True)
    for sp, entries in splits.items():
        out_file = os.path.join('data/splits', f"{sp}_qa.json")
        with open(out_file, 'w') as f:
            json.dump(entries, f, indent=2)
    print("Saved train/val/test QA splits under data/splits/")
