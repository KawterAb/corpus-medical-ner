from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk

# Charger les données
dataset = load_from_disk("data/processed/dataset_hf")

# Préparer le tokenizer et les étiquettes
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
label_list = sorted(set(tag for ex in dataset["train"] for tag in ex["ner_tags"]))
label_to_id = {label: i for i, label in enumerate(label_list)}

# Tokenisation et alignement
def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True, padding="max_length")
    word_ids = tokenized.word_ids()

    # Alignement des étiquettes avec les tokens
    labels = []
    ner_tags = example["ner_tags"]
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)  # Ignorer les tokens spéciaux
        else:
            labels.append(label_to_id[ner_tags[word_id]])
    tokenized["labels"] = labels
    return tokenized

# Appliquer la tokenisation
encoded = dataset.map(tokenize_and_align)

# Créer le modèle
model = AutoModelForTokenClassification.from_pretrained("camembert-base", num_labels=len(label_list))

# Configuration de l'entraînement
args = TrainingArguments(
    output_dir="outputs/camembert-ner",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_dir="outputs/logs",
    logging_steps=10,
    num_train_epochs=3
)

# Entraîneur
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    tokenizer=tokenizer
)

# Lancer l'entraînement
trainer.train()

# ✅ Sauvegarder le modèle et le tokenizer
model.save_pretrained("outputs/camembert-ner")
tokenizer.save_pretrained("outputs/camembert-ner")

