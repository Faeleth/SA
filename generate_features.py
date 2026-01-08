import librosa
import numpy as np
import pandas as pd
import os

DATA_DIR = "data/raw" 
rows = []
labels = os.listdir(DATA_DIR)

print(f"Extracting features from {DATA_DIR}...")

for label in sorted(labels):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    
    files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
    print(f"  {label}: {len(files)} files")
    
    for file in files:
        path = os.path.join(label_dir, file)
        try:
            y, sr = librosa.load(path, sr=16000)
            
            # Wspolczynniki MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Delty MFCC (predkosc zmian)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)
            
            # Polacz wszystkie cechy
            features = list(mfcc_mean) + list(mfcc_std) + \
                      list(mfcc_delta_mean) + list(mfcc_delta_std)
            
            rows.append([*features, label])
        except Exception as e:
            print(f"    Error processing {file}: {e}")

# Utworz dataframe z poprawnymi nazwami kolumn
n_mfcc = 13
columns = (
    [f"mfcc_mean_{i}" for i in range(n_mfcc)] +
    [f"mfcc_std_{i}" for i in range(n_mfcc)] +
    [f"mfcc_delta_mean_{i}" for i in range(n_mfcc)] +
    [f"mfcc_delta_std_{i}" for i in range(n_mfcc)] +
    ["label"]
)

df = pd.DataFrame(rows, columns=columns)
os.makedirs("data", exist_ok=True)
df.to_csv("data/features.csv", index=False)

print(f"\n✓ Extracted {len(df)} samples with {len(columns)-1} features")
print(f"Features saved to: data/features.csv")
