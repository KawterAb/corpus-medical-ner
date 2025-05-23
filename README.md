# NER Médical – FRASIMED + QUAERO

Projet réalisé dans le cadre du module "Outils de traitement de corpus" (M1 TAL).

## Objectif

Constituer un corpus médical annoté en entités nommées (MALADIE, SYMPTÔME, TRAITEMENT) à partir des corpus FRASIMED et QUAERO, et entraîner un modèle de type Transformer (CamemBERT) pour la tâche de reconnaissance d'entités nommées (NER).

## Structure du projet

- `data/` : données brutes, données au format BIO, données augmentées
- `scripts/` : scripts de traitement, visualisation, entraînement et évaluation
- `outputs/` : sorties graphiques, modèles entraînés, logs
- `requirements.txt` : bibliothèques Python nécessaires
- `README.md` : description du projet

## Pipeline du projet

1. Fusion et annotation des corpus (`01_merge.py`)
2. Visualisation des données et statistiques textuelles (`02_stats.py`)
3. Augmentation de données (`03_augment.py`)
4. Conversion vers le format HuggingFace Dataset (`04_create_dataset.py`)
5. Entraînement du modèle CamemBERT (`05_finetune.py`)
6. Évaluation du modèle (`06_evaluate.py`)

## Données et tâche

- **Tâche** : Reconnaissance d'entités nommées (NER)
- **Données utilisées** : FRASIMED et QUAERO
- **Format** : BIO (token \t étiquette)
- **Entités annotées** : MALADIE, SYMPTÔME, TRAITEMENT

## Résultats attendus

- Corpus annoté et augmenté
- Visualisations statistiques (fréquences, longueurs, etc.)
- Modèle entraîné (CamemBERT)
- Évaluation du modèle avec `classification_report` (sklearn)
- Sauvegarde des sorties dans `outputs/`

## Installation

Créer un environnement virtuel et installer les dépendances avec :

```bash
pip install -r requirements.txt



Remarques

    Tous les scripts sont fournis et bien structurés.

    Certains modules n’ont pas pu être exécutés jusqu’au bout localement à cause d’un problème d’espace disque, mais le code est complet.

    L’environnement virtuel n’est pas inclus dans le dépôt.
