import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"

df = pd.read_csv(DATA_DIR / "processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load(MODEL_PATH)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")
