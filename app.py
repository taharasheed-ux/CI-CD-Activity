from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import pandas as pd

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    values = pd.DataFrame([data["features"]], columns=["feature1", "feature2"])
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
