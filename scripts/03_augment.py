from textaugment import EDA
import os

augmenter = EDA()

def augment_sentence(sentence):
    return augmenter.synonym_replacement(sentence)

input_file = "data/processed/frasimed_bio.txt"
output_file = "data/processed/frasimed_augmented.txt"

with open(input_file, encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if line.strip():
            token, label = line.strip().split()
            f_out.write(f"{token}\t{label}\n")
        else:
            f_out.write("\n")
