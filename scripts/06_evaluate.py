from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, AutoModelForTokenClassification
import numpy as np

# Chargement
dataset = load_from_disk("data/processed/dataset_hf")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForTokenClassification.from_pretrained("outputs/camembert-ner")
trainer = Trainer(model=model)

# Prédiction
preds = trainer.predict(dataset["test"])
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = dataset["test"]["ner_tags"]

# Aplatir les listes
y_pred_flat, y_true_flat = [], []

for pred, true in zip(y_pred, y_true):
    for p, t in zip(pred, true):
        if t != -100:  # on ignore les tokens spéciaux
            y_pred_flat.append(p)
            y_true_flat.append(t)

# Affichage du rapport
print(classification_report(y_true_flat, y_pred_flat))

