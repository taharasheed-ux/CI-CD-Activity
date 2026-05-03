import joblib
from pathlib import Path
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"

def test_prediction():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([[5, 6]], columns=["feature1", "feature2"])
    pred = model.predict(sample)
    assert pred[0] in [0,1]
