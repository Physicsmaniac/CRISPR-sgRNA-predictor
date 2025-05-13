import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ─── 1. Load the trained model ─────────────────────────────────────────────────
model = load_model("crispr_model.h5")

# ─── 2. One-hot encoding utility ────────────────────────────────────────────────
def one_hot_encode(sequence, max_length):
    """Turn A/C/G/T string into (max_length×4) float32 array, padded with zeros."""
    mapping = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    arr = np.array([mapping[base] for base in sequence], dtype='float32')
    if arr.shape[0] < max_length:
        pad = np.zeros((max_length - arr.shape[0], 4), dtype='float32')
        arr = np.vstack([arr, pad])
    return arr

# ─── 3. sgRNA candidate generator ───────────────────────────────────────────────
def generate_sgRNAs(sequence, length=20):
    """
    Slide a window of size `length` along `sequence` to generate
    all possible sgRNA candidates (no 'N' bases allowed).
    """
    candidates = []
    for i in range(len(sequence) - length + 1):
        subseq = sequence[i : i + length]
        if set(subseq) <= {"A","C","G","T"}:
            candidates.append(subseq)
    return candidates

# ─── 4. Demo workflow ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example E. coli lacZ sub-sequence (replace with your full gene)
    gene_sequence = (
        "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTT"
        "ACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCA"
    )

    max_length = 20  # Must match the model’s training input length

    # 1. Generate candidates
    sgRNAs = generate_sgRNAs(gene_sequence, length=max_length)
    print(f"Generated {len(sgRNAs)} sgRNA candidates.")

    # 2. Encode them
    encoded = np.stack([one_hot_encode(s, max_length) for s in sgRNAs], axis=0)
    # encoded.shape == (num_candidates, max_length, 4)
    
    # 3. Predict scores
    preds = model.predict(encoded, batch_size=32).flatten()

    # 4. Collect & show top 10
    df = pd.DataFrame({
        'sgRNA': sgRNAs,
        'Predicted_E+S': preds
    })
    top10 = df.nlargest(10, 'Predicted_E+S')
    print("\nTop 10 sgRNA candidates:")
    print(top10.to_string(index=False))
