"""Aggregate detection results into a LaTeX table (mean +- std over seeds)."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = []
for f in glob.glob(os.path.join(ROOT, "results", "detect", "*.json")):
    r = json.load(open(f))
    for setting in ("default", "tuned"):
        if setting in r:
            s = r[setting]
            rows.append(dict(method=r["method"], seed=r["seed"], setting=setting, F1=s["F1"], P=s["P"],
                             R=s["R"], MAE=s["MAE"], h=s.get("h"), t=s.get("t"),
                             time=r["train_time_s"], mem=r["peak_MB"]))
df = pd.DataFrame(rows)
agg = df.groupby(["method", "setting"]).agg(
    n=("seed", "count"), F1=("F1", "mean"), F1s=("F1", "std"), P=("P", "mean"), R=("R", "mean"),
    MAE=("MAE", "mean"), MAEs=("MAE", "std"), time=("time", "mean"), mem=("mem", "mean"),
    h=("h", lambda x: ",".join(str(v) for v in x))).reset_index()
print(agg.to_string())
agg.to_csv(os.path.join(ROOT, "results", "detect_summary.csv"), index=False)
