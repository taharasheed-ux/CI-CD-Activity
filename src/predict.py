import joblib
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([[5, 6]], columns=["feature1", "feature2"])

pred = model.predict(sample)

print("Prediction:", pred[0])
