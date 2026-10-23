"""Turnitin-scale score calibration for NAK AI Detector.

Maps the raw document-level AI% (mean of sentence scores) onto the
Turnitin scale via a fitted monotonic lookup table.

- Table: models/score_mapping.json  {"xs": [...], "ys": [...], "anchors": [...]}
- Fitted by scripts/fit_mapping.py on real (raw%, Turnitin%) anchor pairs
  from Nisar's manuscripts, plus a (0,0) human anchor.
- Validated by scripts/validate_mapping.py: real-human gate docs must
  stay <20% after mapping (false-positive protection).

If the table is missing or invalid, calibrate_pct() is the identity —
the app never breaks because of calibration.
"""
import json
import os

_TABLE = None
_TABLE_TRIED = False


def _table_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "models", "score_mapping.json")


def _load():
    global _TABLE, _TABLE_TRIED
    if _TABLE_TRIED:
        return _TABLE
    _TABLE_TRIED = True
    try:
        with open(_table_path(), encoding="utf-8") as f:
            m = json.load(f)
        xs, ys = m["xs"], m["ys"]
        if len(xs) == len(ys) and len(xs) >= 2:
            _TABLE = (xs, ys)
    except Exception:
        _TABLE = None
    return _TABLE


def calibrate_pct(raw_pct: float) -> float:
    """Map raw document AI% onto the Turnitin scale. Identity if no table."""
    t = _load()
    if t is None:
        return raw_pct
    xs, ys = t
    # linear interpolation on the lookup grid
    if raw_pct <= xs[0]:
        return float(ys[0])
    if raw_pct >= xs[-1]:
        return float(ys[-1])
    import bisect
    i = bisect.bisect_right(xs, raw_pct) - 1
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    frac = (raw_pct - x0) / (x1 - x0) if x1 > x0 else 0.0
    return round(y0 + frac * (y1 - y0), 1)


def mapping_info() -> dict:
    t = _load()
    if t is None:
        return {"active": False}
    return {"active": True, "points": len(t[0])}
