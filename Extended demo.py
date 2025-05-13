import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load model (trained on E. coli sgRNAs)
model = load_model("crispr_model.h5")
max_length = 20

# Utilities
def one_hot_encode(seq, L):
    mapping = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    arr = np.array([mapping[b] for b in seq], dtype='float32')
    if len(arr) < L:
        arr = np.vstack([arr, np.zeros((L-len(arr),4),dtype='float32')])
    return arr

def generate_sgRNAs(seq, L=20):
    return [seq[i:i+L] for i in range(len(seq)-L+1) if set(seq[i:i+L]) <= set("ACGT")]

def top_guides_for_gene(gene_name, gene_seq):
    candidates = generate_sgRNAs(gene_seq, max_length)
    encoded = np.stack([one_hot_encode(s, max_length) for s in candidates], axis=0)
    preds = model.predict(encoded, batch_size=64).flatten()
    df = pd.DataFrame({'sgRNA': candidates, 'Score': preds})
    print(f"\nTop 5 guides for {gene_name}:")
    print(df.nlargest(5, 'Score').to_string(index=False))

if __name__ == "__main__":
    # Example fragments (replace with full sequences as needed)
    lacZ_fragment = "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGAC"
    rpoB_fragment = "ATGGCCAACGCTACGATGACCGGCTGCAGGAGGCGATGTCGGCGATGC"

    top_guides_for_gene("lacZ", lacZ_fragment)
    top_guides_for_gene("rpoB", rpoB_fragment)
