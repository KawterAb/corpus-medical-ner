import os
import matplotlib.pyplot as plt
import seaborn as sns

def read_bio_data(folder):
    lengths = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        print(f"Lecture de {file}...")  # Ajout 1
        with open(path, encoding='utf-8') as f:
            words = [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
            print(f"{file} : {len(words)} tokens")  # Ajout 2
            lengths.append(len(words))
    return lengths

lengths = read_bio_data("data/processed")
print(f"Total de fichiers traités : {len(lengths)}")  # Ajout 3

sns.histplot(lengths, bins=20)
plt.title("Distribution des longueurs de textes")
plt.xlabel("Nombre de tokens")
plt.ylabel("Nombre de phrases")
plt.savefig("outputs/figures/longueurs.png")
plt.show()

