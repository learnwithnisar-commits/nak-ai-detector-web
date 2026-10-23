#!/usr/bin/env python3
"""Evaluate v3 (nak_detector_v3_stack) vs live baseline on all anchors.

App-identical doc scoring: 8-sentence windows stride 4, sentence = mean of
covering windows, doc AI% = mean(sentence)*100.

Gates:
  a) mean|v3-TT| <= 15 on AI anchors (also report +/-5 count)
  b) human anchors: per-anchor v3 <= baseline+5 ; mean v3 <= mean baseline + 5
  c) ash_coal v3 >= 65
  d) star anchors: frac with 1<=v3<=19 >= 0.70
  e) old suite: no anchor with v3 < baseline - 10
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from app.scan import SentenceScorer, split_sentences
from app.parse import parse_document
from src.features import extract_features
from src.v3stack_model import V3StackModel

DATA = os.path.join(ROOT, "data")
ANCH = os.path.join(DATA, "anchors_oct16r2")
MANIFEST = json.load(open(os.path.join(ANCH, "anchors_oct16r2_manifest.json")))

live = joblib.load(os.path.join(ROOT, "models", "nak_detector.joblib"))
v3 = V3StackModel()


def base_window_p(t):
    f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
    return float(np.clip(live.predict_proba(f)[0][1], 0, 1))


def score_doc_baseline(text):
    scorer = SentenceScorer(base_window_p)
    ssum, sn = 0.0, 0
    for s in split_sentences(text) or [text.strip()]:
        scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        _, sc = scorer.finalized.popleft()
        ssum += sc
        sn += 1
    return round(ssum / sn * 100, 1) if sn else 0.0


def load_units(path):
    """App-faithful: parse_document units; ONLY tag=='p' units are scored.

    For .txt anchors: strip Turnitin margin line numbers (standalone digit
    lines) - an extraction artifact, not document content; then replicate
    parse_txt paragraph splitting exactly.
    """
    from app.parse import _flush_para
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        text = open(path, encoding="utf-8").read()
        text = "\n".join(ln for ln in text.split("\n")
                         if not re.match(r"^\s*\d+\s*$", ln))
        units, cur = [], []
        for line in text.split("\n"):
            if line.strip():
                cur.append(line.strip())
            elif cur:
                u = _flush_para(cur)
                cur = []
                if u:
                    units.append(u)
        if cur:
            u = _flush_para(cur)
            if u:
                units.append(u)
        return units
    return list(parse_document(Path(path), ext))


def score_doc_baseline(path):
    scorer = SentenceScorer(base_window_p)
    ssum, sn = 0.0, 0
    for unit in load_units(path):
        if unit["tag"] == "p" and unit["text"].strip():
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        _, sc = scorer.finalized.popleft()
        ssum += sc
        sn += 1
    return round(ssum / sn * 100, 1) if sn else 0.0


def score_doc_v3(path):
    units = load_units(path)
    scored_text = "\n".join(u["text"] for u in units
                            if u["tag"] == "p" and u["text"].strip())
    if not scored_text.strip():
        return {"ai_pct": 0.0, "w_report": 0.0, "sentences": 0}
    w = v3.genre_weight(scored_text)
    scorer = SentenceScorer(lambda t: v3.window_proba(t, w))
    ssum, sn = 0.0, 0
    for unit in units:
        if unit["tag"] == "p" and unit["text"].strip():
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        _, sc = scorer.finalized.popleft()
        ssum += sc
        sn += 1
    return {"ai_pct": round(ssum / sn * 100, 1) if sn else 0.0,
            "w_report": round(w, 3), "sentences": sn}


def main():
    results = {}
    # manifest anchors
    items = []
    for key, v in MANIFEST.items():
        p = os.path.join(ANCH, key + ".txt")
        if not os.path.exists(p):
            continue
        items.append((key, p, v["class"], v["tt_pct"]))
    # extra old-suite docs not in manifest
    items.append(("ash_coal", os.path.join(DATA, "real_ai", "ash_coal_essay_chatgpt.txt"), "ai", 100))
    items.append(("upload_competition", "/home/hatch/workspace/user/files/upload.docx", "star", "*"))

    for key, path, cls, tt in items:
        b = score_doc_baseline(path)
        r = score_doc_v3(path)
        if r["sentences"] < 5:
            print(f"SKIP {key}: too short", flush=True)
            continue
        results[key] = {"class": cls, "tt": tt, "baseline": b,
                        "v3": r["ai_pct"], "w_report": r["w_report"],
                        "sents": r["sentences"]}
        if len(results) % 25 == 0:
            print(f"scored {len(results)}/{len(items)}", flush=True)
    json.dump(results, open("/tmp/eval_v3stack.json", "w"), indent=1)

    ai = {k: r for k, r in results.items() if r["class"] == "ai" and isinstance(r["tt"], int)}
    hu = {k: r for k, r in results.items() if r["class"] == "human"}
    st = {k: r for k, r in results.items() if r["class"] == "star"}

    gaps = [abs(r["v3"] - r["tt"]) for r in ai.values()]
    gate_a = float(np.mean(gaps))
    at5 = sum(1 for g in gaps if g <= 5)
    print(f"\nGate A: mean|v3-TT| = {gate_a:.2f} (target <=15) | +/-5: {at5}/{len(gaps)}")

    b_base = np.mean([r["baseline"] for r in hu.values()])
    b_v3 = np.mean([r["v3"] for r in hu.values()])
    viol = [k for k, r in hu.items() if r["v3"] > r["baseline"] + 5]
    print(f"Gate B: human mean base={b_base:.2f} v3={b_v3:.2f} (target <= {b_base+5:.2f}); "
          f"violations: {len(viol)}/{len(hu)}")
    for k in sorted(viol, key=lambda k: hu[k]["v3"] - hu[k]["baseline"], reverse=True)[:12]:
        r = hu[k]
        print(f"   {k[:45]:45} base={r['baseline']:5.1f} v3={r['v3']:5.1f} w={r['w_report']}")

    ash = results.get("ash_coal", {})
    print(f"Gate C: ash_coal v3={ash.get('v3')} (target >=65)")

    inband = [k for k, r in st.items() if 1 <= r["v3"] <= 19]
    print(f"Gate D: star in-band {len(inband)}/{len(st)} = {len(inband)/max(1,len(st)):.3f} (target >=0.70)")

    old = ["con", "listeria_goat_manuscript_improved_clean_1", "xdr_1",
           "aigc_download_2027795315876175872_aigc_rate", "ash_coal", "19_1",
           "critical_reflection_1", "liu_yueru_vicky_fa1_yueru_vicky_liu_1",
           "upload_competition", "essay_junlan_li_25041323_1", "es1002_essay_1",
           "318_1", "7163_a1_ing_1"]
    regs = []
    for k in old:
        if k in results:
            r = results[k]
            if r["v3"] < r["baseline"] - 10:
                regs.append((k, r["baseline"], r["v3"]))
    print(f"Gate E: regressions>10: {len(regs)} (target 0)")
    for k, b, v in regs:
        print(f"   {k[:45]:45} base={b:5.1f} v3={v:5.1f}")

    # per-AI-anchor detail
    print("\nAI anchors (tt | base | v3 | w):")
    for k in sorted(ai, key=lambda k: ai[k]["tt"]):
        r = ai[k]
        flag = " *" if abs(r["v3"] - r["tt"]) <= 5 else ""
        print(f"  {str(r['tt']):>4} {r['baseline']:6.1f} {r['v3']:6.1f} w={r['w_report']:.2f}{flag} {k[:40]}")

    gates = {
        "a_mean_gap": round(gate_a, 2), "a_at5": f"{at5}/{len(gaps)}", "a_pass": gate_a <= 15,
        "b_mean_base": round(float(b_base), 2), "b_mean_v3": round(float(b_v3), 2),
        "b_viol": f"{len(viol)}/{len(hu)}",
        "b_pass": bool(b_v3 <= b_base + 5 and len(viol) == 0),
        "c_ash": ash.get("v3"), "c_pass": bool(ash.get("v3", 0) >= 65),
        "d_inband": f"{len(inband)}/{len(st)}", "d_pass": bool(len(inband) / max(1, len(st)) >= 0.70),
        "e_regs": len(regs), "e_pass": len(regs) == 0,
    }
    json.dump(gates, open("/tmp/gates_v3stack.json", "w"), indent=1)
    print("\nGATES:", json.dumps(gates, indent=1))
    print("PASS" if all([gates["a_pass"], gates["b_pass"], gates["c_pass"], gates["d_pass"], gates["e_pass"]]) else "HOLD")


if __name__ == "__main__":
    main()
