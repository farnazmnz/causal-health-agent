"""Stage 1 evaluation (paper Sec. IV-A, Eq. 2, Table II)."""
import numpy as np
import pandas as pd

from .. import config as C
from .. import stage1_anomaly as s1


def agreement(flag_sets, min_flagged=C.MIN_FLAGGED_FOR_AGREEMENT):
    """Cross-detector agreement, Eq. (2).

    flag_sets: {(participant, feature): DataFrame(days x detectors, bool)}.
    A_{M,p,v} = fraction of M's flagged days also flagged by >=1 other detector; pairs with fewer than
    `min_flagged` flagged days are excluded; macro-averaged. Returns (agreement, mean flagged days) per detector."""
    per_det = {}
    for (_p, _v), fl in flag_sets.items():
        for m in fl.columns:
            n = int(fl[m].sum())
            if n < min_flagged:
                continue
            others = fl.drop(columns=m).any(axis=1) if fl.shape[1] > 1 else pd.Series(False, index=fl.index)
            per_det.setdefault(m, []).append(((fl[m] & others).sum() / n, n))
    return pd.DataFrame({m: {"agreement": np.mean([a for a, _ in v]), "flagged": np.mean([n for _, n in v]),
                             "n_pairs": len(v)} for m, v in per_det.items()}).T


def inject_anomalies(series, k=C.N_SYNTHETIC_ANOMALIES, magnitude=C.SYNTHETIC_MAGNITUDE, seed=C.SEED):
    """Inject k point anomalies (+/- magnitude*std) into the evaluation region (second half, so the
    neural detectors are scored fairly). Returns (modified series, injected dates)."""
    rng = np.random.default_rng(seed)
    s = series.copy()
    valid = [i for i in range(int(len(s) * C.TRAIN_FRAC) + 1, len(s)) if not np.isnan(s.iloc[i])]
    pick = sorted(rng.choice(valid, size=min(k, len(valid)), replace=False))
    sd = float(np.nanstd(series.values)) or 1.0
    for i in pick:
        s.iloc[i] += rng.choice([-1, 1]) * magnitude * sd
    return s, s.index[pick]


def synthetic_prf(frames, detectors, features=C.CORE_VARIABLES, seed=C.SEED):
    """Precision / recall / F1 of each detector on injected anomalies (mean over participant-feature pairs)."""
    rows = {m: [] for m in detectors}
    for pid, frame in frames.items():
        df = s1._prepare(frame, features)
        for f in features:
            if df[f].notna().sum() < 40:
                continue
            mod, truth = inject_anomalies(df[f], seed=seed)
            for m in detectors:
                try:
                    flagged = s1.DETECTORS[m](mod)
                except Exception:
                    continue
                pred = set(flagged.index[flagged.fillna(False).astype(bool)])
                tp = len(pred & set(truth))
                p = tp / len(pred) if pred else 0.0
                r = tp / len(truth)
                rows[m].append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
    return pd.DataFrame({m: dict(zip(("precision", "recall", "f1"), np.mean(v, axis=0)))
                         for m, v in rows.items() if v}).T


def evaluate_stage1(frames, detectors, features=C.CORE_VARIABLES):
    """Table II: agreement + mean flagged days (original series) and synthetic P/R/F1."""
    sets = {}
    for pid, frame in frames.items():
        fl = s1.run_stage1(frame, features, detectors, verbose=False)
        for f in features:
            cols = {m: fl[(m, f)] for m in detectors if (m, f) in fl.columns}
            if cols:
                sets[(pid, f)] = pd.DataFrame(cols)
    return agreement(sets).join(synthetic_prf(frames, detectors, features), how="outer")
