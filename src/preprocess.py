from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

df = pd.read_csv(DATA_DIR / "sample.csv")
df.dropna(inplace=True)
df.to_csv(DATA_DIR / "processed.csv", index=False)

print("Preprocessing Completed")
