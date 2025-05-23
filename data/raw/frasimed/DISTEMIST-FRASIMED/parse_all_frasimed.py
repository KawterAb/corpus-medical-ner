import spacy
import os

nlp = spacy.load("fr_core_news_sm")

def read_ann_file(ann_path):
    entities = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("T"):
                parts = line.strip().split('\t')
                entity_info = parts[1].split()
                label = entity_info[0]
                start = int(entity_info[1])
                end = int(entity_info[-1])
                entities.append((start, end, label))
    return entities

def bio_tagging(text, entities):
    doc = nlp(text)
    tags = []
    for token in doc:
        label = "O"
        for start, end, ent_label in entities:
            if token.idx >= start and token.idx < end:
                label = f"B-{ent_label}" if token.idx == start else f"I-{ent_label}"
                break
        tags.append((token.text, label))
    return tags

def write_bio_file(tags, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, label in tags:
            f.write(f"{token}\t{label}\n")

def process_all_files(folder_path):
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    ann_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ann')])

    for i in range(min(len(txt_files), len(ann_files))):
        txt_file = txt_files[i]
        ann_file = ann_files[i+1] if i+1 < len(ann_files) else ann_files[i]  # ← Décalage ici

        print(f"📝 Traitement : {txt_file} + {ann_file}")
        with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as f:
            text = f.read()
        entities = read_ann_file(os.path.join(folder_path, ann_file))
        tagged = bio_tagging(text, entities)

        output_file = os.path.splitext(txt_file)[0] + "_bio.txt"
        write_bio_file(tagged, os.path.join(folder_path, output_file))
        print(f"✅ Sortie : {output_file}")

# Modifier ce chemin selon où sont tes fichiers :
dossier = "./frasimed"
process_all_files(dossier)

