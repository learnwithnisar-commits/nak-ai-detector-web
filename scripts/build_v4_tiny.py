#!/usr/bin/env python3
"""Build NAK v4 'tiny' candidate: 10 within-register features, logistic
regression on 8-sentence windows, calibrated on v2-val.

TRAINING ONLY on v2-train (+ sentence labels from v2-train PDFs only).
v1 locked test (data/phase4_splits/test.json) is scored EXACTLY ONCE,
guarded by a flag file. Live model never touched. No deploys.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.path.expanduser("~/workspace/ai-detector"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.tiny_features import extract_tiny_features, FEATURE_NAMES
from app.scan import split_sentences

DATA = ROOT / "data"
EVAL_FLAG = DATA / "phase5_baseline" / "v4_locked_test_done.flag"

WIN, STRIDE = 8, 4


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def windows(text):
    sents = [s for s in split_sentences(text) if s.strip()]
    out = []
    i = 0
    while i < len(sents):
        out.append(" ".join(sents[i:i + WIN]))
        if i + WIN >= len(sents):
            break
        i += STRIDE
    return out


def main():
    # ---- integrity: live model untouched -----------------------------------
    import hashlib as _h
    live_md5 = _h.md5(open(ROOT / "models/nak_detector.joblib", "rb").read()).hexdigest()
    assert live_md5 == "9eeba80e03d06c7db8b35796965746c9", "LIVE MODEL CHANGED - abort"
    test_sha = sha256(DATA / "phase4_splits" / "test.json")
    assert test_sha.startswith("89f8b799"), "locked test manifest changed - abort"

    v2_recs = {json.loads(l)["doc_id"]: json.loads(l)
               for l in open(DATA / "phase2_dataset_v2.jsonl", encoding="utf-8")}
    v1_recs = {json.loads(l)["doc_id"]: json.loads(l)
               for l in open(DATA / "phase2_dataset.jsonl", encoding="utf-8")}
    train_ids = set(json.load(open(DATA / "phase4_splits_v2" / "train.json")))
    val_ids = set(json.load(open(DATA / "phase4_splits_v2" / "val.json")))
    v1_test_ids = set(json.load(open(DATA / "phase4_splits" / "test.json")))

    # quarantine re-check
    assert not (train_ids & v1_test_ids) and not (val_ids & v1_test_ids)

    # ---- sentence labels restricted to v2-train PDFs ------------------------
    pdf2id = {}
    for r in v2_recs.values():
        for p in [r["source_pdf"]] + (r.get("all_source_pdfs") or []):
            pdf2id.setdefault(p, r["doc_id"])
    labs = json.load(open(DATA / "sentence_labels_all.json"))
    train_labs = [l for l in labs if pdf2id.get(l["pdf"]) in train_ids]
    leak = [l for l in train_labs if pdf2id.get(l["pdf"]) in v1_test_ids]
    assert not leak, f"label leakage into training from v1 test: {len(leak)}"
    n_v1test_labeled = sum(1 for l in labs if pdf2id.get(l["pdf"]) in v1_test_ids)
    print(f"({n_v1test_labeled} labels belong to v1-test PDFs - excluded from training)")
    print(f"labels: {len(labs)} total, {len(train_labs)} from v2-train PDFs")

    # ---- window training set ----------------------------------------------
    X, y, groups = [], [], []
    # (a) label windows: consecutive sentences, same pdf
    by_pdf = {}
    for l in train_labs:
        by_pdf.setdefault(l["pdf"], []).append(l)
    n_lab = 0
    for pdf, seq in by_pdf.items():
        did = pdf2id[pdf]
        binlab = [1 if s["label"] == "ai" else 0 for s in seq]
        i = 0
        while i < len(seq):
            chunk = seq[i:i + WIN]
            wtext = " ".join(s["sentence_text"] for s in chunk)
            frac = sum(binlab[i:i + WIN]) / len(chunk)
            X.append(extract_tiny_features(wtext))
            y.append(1 if frac >= 0.5 else 0)
            groups.append(did)
            n_lab += 1
            if i + WIN >= len(seq):
                break
            i += STRIDE
    # (b) pure-doc windows from v2-train
    n_pure = 0
    for did in train_ids:
        r = v2_recs[did]
        s = r.get("turnitin_ai_score")
        if s not in (0, 100):
            continue
        for wtext in windows(r["text"]):
            X.append(extract_tiny_features(wtext))
            y.append(1 if s == 100 else 0)
            groups.append(did)
            n_pure += 1
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    print(f"windows: {len(X)} (label {n_lab}, pure-doc {n_pure}), "
          f"pos rate {y.mean():.3f}, groups {len(set(groups))}")

    # ---- group-by-PDF CV over C --------------------------------------------
    from sklearn.metrics import roc_auc_score
    best_c, best_auc = None, -1
    for C in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
        aucs = []
        for tr, va in GroupKFold(5).split(X, y, groups):
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, max_iter=2000))
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:, 1]
            if len(set(y[va])) > 1:
                aucs.append(roc_auc_score(y[va], p))
        m = float(np.mean(aucs)) if aucs else 0.0
        print(f"  C={C:<5} group-CV AUC={m:.4f} (n={len(aucs)})")
        if m > best_auc:
            best_auc, best_c = m, C
    print(f"best C={best_c} AUC={best_auc:.4f}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(C=best_c, max_iter=2000).fit(Xs, y)

    def win_prob(wtext):
        return float(lr.predict_proba(scaler.transform(
            [extract_tiny_features(wtext)]))[0][1])

    def doc_raw(text):
        ws = windows(text)
        return float(np.mean([win_prob(w) for w in ws])) if ws else 0.0

    # ---- calibration on v2-val (numeric docs) ------------------------------
    val_rows = []
    for did in val_ids:
        r = v2_recs[did]
        s = r.get("turnitin_ai_score")
        if not isinstance(s, (int, float)) or not r.get("text", "").strip():
            continue
        val_rows.append((did, doc_raw(r["text"]), s / 100.0))
    assert len(val_rows) >= 5, f"too few val docs: {len(val_rows)}"
    print(f"val calibration docs: {len(val_rows)}")
    raws = np.array([x[1] for x in val_rows]).reshape(-1, 1)
    tgts = np.array([x[2] for x in val_rows])

    def loo_mae(predict_fn):
        errs = []
        for i in range(len(val_rows)):
            mask = np.ones(len(val_rows), bool); mask[i] = False
            errs.append(abs(predict_fn(raws[mask], tgts[mask],
                                       raws[i].item()) - tgts[i]))
        return float(np.mean(errs))

    from scipy.optimize import curve_fit

    def _sig(x, a, b):
        return 1.0 / (1.0 + np.exp(-(a * x + b)))

    def _fit_sigmoid(r1, t_):
        try:
            (a, b), _ = curve_fit(_sig, r1, t_, p0=[10.0, -5.0], maxfev=8000)
        except Exception:
            m = float(np.clip(t_.mean(), 1e-6, 1 - 1e-6))
            a, b = 0.0, float(np.log(m / (1 - m)))
        return a, b

    def platt_predict(r_, t_, x):
        a, b = _fit_sigmoid(r_.ravel(), t_)
        return float(_sig(x, a, b))

    # isotonic needs a fitted object per fold; wrap carefully
    def iso_predict(r_, t_, x):
        iso = IsotonicRegression(out_of_bounds="clip").fit(r_.ravel(), t_)
        return float(iso.predict([x])[0])

    def iso_loo(r_, t_, x):
        return iso_predict(r_, t_, x)

    mae_platt = loo_mae(platt_predict)
    mae_iso = loo_mae(iso_loo)
    print(f"val LOO MAE: platt={mae_platt:.4f} iso={mae_iso:.4f}")
    if mae_platt <= mae_iso:
        cal_kind = "platt"
        a_fin, b_fin = _fit_sigmoid(raws.ravel(), tgts)
        cal_payload = {"a": float(a_fin), "b": float(b_fin)}

        def calibrate(x):
            return float(_sig(x, a_fin, b_fin))
    else:
        cal_kind = "iso"
        iso_final = IsotonicRegression(out_of_bounds="clip").fit(raws.ravel(), tgts)
        xs = iso_final.X_thresholds_
        cal_payload = {"x": [float(v) for v in xs],
                       "y": [float(v) for v in iso_final.predict(xs)]}

        def calibrate(x):
            return float(iso_final.predict([x])[0])
    print("calibration:", cal_kind)

    def doc_pct(text):
        return round(calibrate(doc_raw(text)) * 100, 1)

    # ---- save candidate -----------------------------------------------------
    bundle = {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "lr_coef": lr.coef_.tolist(), "lr_intercept": lr.intercept_.tolist(),
        "lr_C": best_c, "cal_kind": cal_kind, "cal_payload": cal_payload,
        "feature_names": FEATURE_NAMES, "window": WIN, "stride": STRIDE,
        "meta": {"built": "2026-10-17", "train_windows": len(X),
                 "train_groups": len(set(groups)), "pos_rate": float(y.mean()),
                 "group_cv_auc": best_auc, "val_n": len(val_rows),
                 "val_loo_mae_platt": mae_platt, "val_loo_mae_iso": mae_iso,
                 "live_md5_pre": live_md5, "test_sha_pre": test_sha},
    }
    joblib.dump(bundle, ROOT / "models" / "nak_detector_v4_tiny.joblib")
    json.dump(bundle["meta"], open(ROOT / "models" / "nak_detector_v4_tiny_meta.json", "w"), indent=1)
    print("saved models/nak_detector_v4_tiny.joblib")

    # ---- LOCKED TEST: exactly once ------------------------------------------
    if EVAL_FLAG.exists():
        print("LOCKED TEST ALREADY SCORED - refusing second run")
        return
    test_ids = json.load(open(DATA / "phase4_splits" / "test.json"))
    assert len(test_ids) == 16
    rows = []
    for did in test_ids:
        r = v1_recs[did]
        ref = r["turnitin_ai_score"]
        assert isinstance(ref, (int, float))
        pred = doc_pct(r["text"])
        err = abs(pred - ref)
        rows.append({"doc_id": did, "ref": ref, "pred": pred,
                     "abs_err": round(err, 1), "signed_err": round(pred - ref, 1),
                     "within_5": err <= 5})
    ae = np.array([x["abs_err"] for x in rows])
    se = np.array([x["signed_err"] for x in rows])
    metrics = {
        "n": 16, "mae": round(float(ae.mean()), 2),
        "median_ae": round(float(np.median(ae)), 2),
        "mean_signed": round(float(se.mean()), 2),
        "within_5_n": int(sum(x["within_5"] for x in rows)),
        "within_5_pct": round(100 * sum(x["within_5"] for x in rows) / 16, 1),
    }
    print("LOCKED TEST:", json.dumps(metrics))
    for x in sorted(rows, key=lambda z: -z["abs_err"]):
        print(f"  {x['doc_id']}: ref={x['ref']} pred={x['pred']} err={x['abs_err']}"
              f"{' WITHIN5' if x['within_5'] else ''}")

    star = json.load(open(DATA / "phase4_splits" / "star_band_pool.json"))
    in_band = total = skipped = 0
    for did in star["doc_ids"]:
        r = v1_recs.get(did)
        if not r or not (r.get("text") or "").strip():
            skipped += 1
            continue
        total += 1
        if 1 <= doc_pct(r["text"]) <= 19:
            in_band += 1
    star_res = {"in_band": in_band, "total": total, "skipped": skipped,
                "pct": round(100 * in_band / total, 1) if total else 0}
    print(f"STAR POOL: {in_band}/{total} ({star_res['pct']}%), skipped={skipped}")

    # gates
    gates = {
        "mae_lt_19_54": metrics["mae"] < 19.54,
        "within5_ge_4": metrics["within_5_n"] >= 4,
        "star_ge_70": star_res["pct"] >= 70.0,
    }
    verdict = "PASS" if all(gates.values()) else "HOLD"
    print("GATES:", gates, "->", verdict)

    out = {"metrics": metrics, "rows": rows, "star": star_res,
           "gates": gates, "verdict": verdict,
           "integrity": {"live_md5_pre": live_md5, "test_sha_pre": test_sha}}
    json.dump(out, open(DATA / "phase5_baseline" / "v4_locked_test_eval.json", "w"), indent=1)
    EVAL_FLAG.write_text("scored once 2026-10-17\n")

    # post integrity
    live_md5_post = _h.md5(open(ROOT / "models/nak_detector.joblib", "rb").read()).hexdigest()
    test_sha_post = sha256(DATA / "phase4_splits" / "test.json")
    assert live_md5_post == live_md5 and test_sha_post == test_sha
    print("post-checks OK: live model + test manifest unchanged")


if __name__ == "__main__":
    main()
