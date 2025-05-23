import spacy
import os

# Charger le modèle spaCy en français
nlp = spacy.load("fr_core_news_sm")

def read_ann_file(ann_path):
    """Lit un fichier .ann et extrait les entités"""
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
    """Associe un label BIO à chaque token du texte"""
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
    """Écrit les tokens + BIO tags dans un fichier"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, label in tags:
            f.write(f"{token}\t{label}\n")

def process_all_files(folder_path):
    """Traite tous les couples .txt/.ann du dossier"""
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for txt_file in sorted(txt_files):
        base_name = os.path.splitext(txt_file)[0]
        ann_file = base_name + ".ann"
        txt_path = os.path.join(folder_path, txt_file)
        ann_path = os.path.join(folder_path, ann_file)

        if not os.path.exists(ann_path):
            print(f"⚠️ Annotation manquante pour {txt_file}, ignoré.")
            continue

        print(f"📝 Traitement : {txt_file} + {ann_file}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        entities = read_ann_file(ann_path)
        tagged = bio_tagging(text, entities)

        output_file = os.path.join(folder_path, base_name + "_bio.txt")
        write_bio_file(tagged, output_file)
        print(f"✅ Sortie : {output_file}")

# Modifier ce chemin selon où sont tes fichiers :
dossier = "."  # ou le chemin complet vers ton dossier
process_all_files(dossier)

