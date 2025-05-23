from datasets import Dataset

def load_bio_file(path):
    examples = []
    tokens, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                token, label = line.strip().split()
                tokens.append(token)
                labels.append(label)
            else:
                examples.append({"tokens": tokens, "ner_tags": labels})
                tokens, labels = [], []
    return examples

data = load_bio_file("data/processed/frasimed_quaero_merged.txt")
dataset = Dataset.from_list(data).train_test_split(test_size=0.2)
dataset.save_to_disk("data/processed/dataset_hf")
