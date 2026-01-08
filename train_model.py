import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Wczytaj cechy
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} samples with {len(df.columns)-1} features")
print(f"Classes: {df['label'].unique()}")
print(f"Samples per class:\n{df['label'].value_counts()}\n")

X = df.drop(columns="label")
y = df["label"]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Etykiety
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Podzial danych
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

# Model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           
    min_samples_split=5,   
    min_samples_leaf=2,     
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced' 
)
model.fit(X_train, y_train)

# Ocen jakosc
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score:.2%}")
print(f"Test accuracy: {test_score:.2%}")
print(f"Generalization gap: {(train_score - test_score):.2%}\n")

# Dokladnosc dla kazdej klasy
y_pred = model.predict(X_test)
for i, label in enumerate(le.classes_):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    print(f"{label}: {class_acc:.2%}")

# Zapisz model
os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "le": le, "scaler": scaler}, "model/voice_model.pkl")
print("\n✓ Model saved to model/voice_model.pkl")
