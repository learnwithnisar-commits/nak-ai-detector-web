"""RECOVERY: rebuild model artifacts after the corrupted bundle incident.
Uses existing models/_feats/chunk_*.npz (30k raw 187-dim samples) + a fresh
trigram LM built on OA corpus human texts. Trains logistic regression.
"""
import json, glob, os, sys, time
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.lm import build_lm

t0 = time.time()

# --- 1. Build trigram LM on OA human texts ---
texts = []
for jf in sorted(glob.glob("data/oa_corpus/*_docs.jsonl")):
    with open(jf) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("abstract") or rec.get("text") or ""
            if len(t.split()) >= 40:
                texts.append(t)
            if len(texts) >= 8000:
                break
    if len(texts) >= 8000:
        break
print(f"LM corpus: {len(texts)} docs", flush=True)
lm = build_lm(texts)
print(f"LM built: vocab={len(lm['vocab'])} [{time.time()-t0:.0f}s]", flush=True)

# --- 2. Temp bundle so features.py can resolve the LM + schema ---
os.makedirs("models", exist_ok=True)
joblib.dump({"lm": lm}, "models/_tmp_lm_bundle.joblib")

import src.features as F
names = F.feature_names()
print(f"feature schema: {len(names)} names, lm_enabled={F._lm_enabled}", flush=True)
assert len(names) == 187, f"schema mismatch: {len(names)}"

# --- 3. Load chunks, fit scaler, train ---
Xs, ys = [], []
for p in sorted(glob.glob("models/_feats/chunk_*.npz")):
    d = np.load(p)
    Xs.append(d["X"]); ys.append(d["y"])
X = np.vstack(Xs); y = np.concatenate(ys)
print(f"chunks: X={X.shape} y={np.bincount(y)}", flush=True)
assert X.shape[1] == 187

scaler = StandardScaler().fit(X)
Xs_ = scaler.transform(X)
clf = LogisticRegression(max_iter=2000, C=1.0)
cv = cross_val_score(clf, Xs_, y, cv=3)
print(f"CV acc: {cv.mean():.4f} +/- {cv.std():.4f}", flush=True)
clf.fit(Xs_, y)

# --- 4. Save final artifacts ---
joblib.dump(clf, "models/nak_detector.joblib")
bundle = {"scaler": scaler, "feature_names": names, "winner": "logreg",
          "cv_f1": {"logreg": float(cv.mean())}, "test_f1": {}, "lm": lm}
joblib.dump(bundle, "models/feature_artifacts.joblib")
os.remove("models/_tmp_lm_bundle.joblib")
print(f"saved artifacts [{time.time()-t0:.0f}s]", flush=True)

# --- 5. Smoke test: fresh load + extract ---
import importlib
importlib.reload(F)
vec = F.extract_features("This is a simple human written sentence for testing the feature extractor pipeline.")
print(f"extract_features -> {vec.shape}, finite={np.isfinite(vec).all()}", flush=True)
p = clf.predict_proba(scaler.transform(vec.reshape(1, -1)))[0, 1]
print(f"sample AI prob: {p:.3f}", flush=True)
print("RECOVERY OK", flush=True)
