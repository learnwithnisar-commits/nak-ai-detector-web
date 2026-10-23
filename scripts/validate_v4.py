#!/usr/bin/env python3
"""Validate segment v2 on anchors (AI anchors as test; human anchors now in train)."""
import os, sys
from pathlib import Path
PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import joblib, numpy as np
from src.features import extract_features_unscaled
from app.parse import parse_document
from app.scan import SentenceScorer, split_sentences

clf = joblib.load(os.path.join(PROJ, "models/segments_v3/seg_candidate_v4.joblib"))
scaler = joblib.load(os.path.join(PROJ, "models/segments_v3/seg_scaler_v4.joblib"))

def score_text(t):
    v = np.asarray(extract_features_unscaled(t), dtype=np.float64).reshape(1,-1)
    return float(clf.predict_proba(scaler.transform(v))[0][1])

def score_doc(path):
    ext = Path(path).suffix.lower()
    scorer = SentenceScorer(score_text)
    ssum, sn = 0.0, 0
    for unit in parse_document(Path(path), ext):
        if unit["tag"]=="p" and unit["text"].strip():
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        g, s = scorer.finalized.popleft(); ssum+=s; sn+=1
    return round(ssum/sn*100,1) if sn else 0.0

U = "/home/hatch/workspace/user/files"
# AI anchors (test) + *-band (test) + theses human (test)
anchors = [
    ("CriticalReflection", f"{U}/Critical_Reflection.docx", 99),
    ("19", f"{U}/19.docx", 100),
    ("99141", "/tmp/bulk2/99141.docx", 64),
    ("MA114", f"{U}/MA114-.docx", 45),
    ("XDR", f"{PROJ}/reference/ai/XDR_Typhoid_Pakistan_PJMR_Format.docx", 64),
    ("Listeria", f"{U}/Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx", 54),
    ("Composites", f"{U}/Composites_Research_Overview2.pdf", 26),
    ("IBUS_3000", f"{U}/IBUS_3000.docx", 10),
    ("Perfume", f"{U}/unknown_20260315071136.docx", 10),
    ("Group7", "/tmp/bulk1/CVL298_PD2_Technical_Report_Group7 (1) (1).docx", 10),
    ("CVL343", "/tmp/bulk1/CVL343_Results_and_Discussion.docx", 10),
    ("pd2july", "/tmp/bulk1/pd2 12 july cvl298.docx", 10),
    ("10814", "/tmp/bulk2/10814-revised.pdf", 10),
    ("MScSynopsis", f"{U}/Hasnain_synopsis__1.docx", 10),
]
print(f"{'Anchor':18s} {'TT':>4s} {'v2':>6s}  verdict")
n=0; tot=0
for name, p, tt in anchors:
    if not os.path.exists(p):
        print(f"{name:18s} MISSING"); continue
    tot+=1
    v = score_doc(p)
    if tt==10: ok=v<20; vd="MATCH *" if ok else "MISS *"
    else: ok=abs(v-tt)<=5; vd="HIT±5" if ok else f"gap {abs(v-tt):.0f}"
    n+=ok
    print(f"{name:18s} {tt:4d} {v:6.1f}  {vd}", flush=True)
# human FPR on theses (held-out human)
import glob
th = sorted(glob.glob(f"{PROJ}/data/real_human/theses/*_prose.txt"))[:20]
vals=[]
for f in th:
    t=open(f,encoding="utf-8",errors="ignore").read()[:6000]
    vals.append(score_text(t)*100)
print(f"\ntheses FPR: mean={np.mean(vals):.1f}% max={np.max(vals):.1f}%")
print(f"\n{n}/{tot} anchors within target")
