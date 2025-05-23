# scripts/01_preprocess.py

import os
import re

INPUT_DIRS = ['data/raw/frasimed/CANTEMIST-FRASIMED', 
              'data/raw/frasimed/DISTEMIST-FRASIMED',
              'data/raw/quaero/corpus']
OUTPUT_DIR = 'data/processed'
ENTITIES = {'MALADIE', 'SYMPTÔME', 'TRAITEMENT'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_ann(ann_path):
    entities = []
    with open(ann_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith("T"):
                parts = line.strip().split('\t')
                tag_info = parts[1].split()
                label = tag_info[0]
                start = int(tag_info[1])
                end = int(tag_info[-1])
                if label.upper() in ENTITIES:
                    entities.append((start, end, label.upper()))
    return entities

def to_bio(text, entities):
    tags = ['O'] * len(text)
    for start, end, label in entities:
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            if i < len(tags):
                tags[i] = f"I-{label}"
    return tags

def process_file(txt_path, ann_path, out_path):
    with open(txt_path, encoding='utf-8') as f:
        text = f.read()

    bio_tags = ['O'] * len(text)
    for start, end, label in read_ann(ann_path):
        for i in range(start, end):
            if i == start:
                bio_tags[i] = f"B-{label}"
            else:
                bio_tags[i] = f"I-{label}"

    tokens = re.findall(r'\S+', text)
    idx = 0
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for token in tokens:
            start = text.find(token, idx)
            tag = bio_tags[start] if start < len(bio_tags) else 'O'
            out_f.write(f"{token} {tag}\n")
            idx = start + len(token)
        out_f.write("\n")

count = 0
for dir_ in INPUT_DIRS:
    for fname in os.listdir(dir_):
        if fname.endswith('.txt'):
            base = fname[:-4]
            txt_path = os.path.join(dir_, f"{base}.txt")
            ann_path = os.path.join(dir_, f"{base}.ann")
            if os.path.exists(ann_path):
                out_path = os.path.join(OUTPUT_DIR, f"{base}.bio")
                process_file(txt_path, ann_path, out_path)
                count += 1

print(f"✅ Total de fichiers traités : {count}")

