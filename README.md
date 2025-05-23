# NER Médical – FRASIMED + QUAERO

Projet de traitement de corpus pour le module "Outils de traitement de corpus" (M1 TAL).

## Objectif

Constituer un corpus médical annoté pour la reconnaissance d'entités nommées (NER) à partir des corpus FRASIMED et QUAERO.  
Les entités ciblées sont : MALADIE, SYMPTÔME, TRAITEMENT.

## Structure du projet

- `data/` : 
  - `raw/` : données brutes
  - `processed/` : fichiers BIO, augmentés, puis convertis en dataset
- `scripts/` : tous les scripts de traitement, visualisation, entraînement, évaluation
- `outputs/` : figures (histogrammes, résultats)
- `models/` *(optionnel)* : pour sauvegarder un modèle fine-tuné

## Pipeline

1. `01_scrape.py` – Nettoyage et conversion en BIO (`data/processed/*.bio`)
2. `02_visualize.py` – Histogramme de la longueur des phrases
3. `03_augment.py` – Augmentation par synonymes avec `textaugment`
4. `04_create_dataset.py` – Conversion en format HuggingFace `datasets.Dataset`
5. `05_train.py` – Fine-tuning de CamemBERT sur le corpus médical
6. `06_eval.py` – Évaluation sur le test set

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
python -m textblob.download_corpora

