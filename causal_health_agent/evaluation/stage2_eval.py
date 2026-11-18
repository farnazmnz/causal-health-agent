"""Stage 2 evaluation (paper Sec. IV-B, Eq. 3, Table III): Agreement, Rolling Stability, Growing Convergence."""
import itertools

import numpy as np
import pandas as pd

from .. import config as C
from .. import stage2_causal_graph as s2


def _keys(edges):
    return {(e["source"], e["target"]) for e in edges}        # directed edge skeleton (lag ignored)


def agreement(results):
    """Eq. (3): fraction of method M's edges also recovered by at least one other method."""
    sets = {m: _keys(es) for m, es in results.items() if es is not None}
    out = {}
    for m, E in sets.items():
        others = set().union(*[s for k, s in sets.items() if k != m]) if len(sets) > 1 else set()
        out[m] = len(E & others) / len(E) if E else np.nan
    return out


def _try_keys(method, sub):
    """Edge set for a sub-window, or None when the method cannot be fit (e.g. 30 days < 36 predictors)."""
    try:
        return _keys(s2.METHODS[method](sub))
    except Exception:
        return None


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


def rolling_stability(z, method, window=C.ROLLING_WINDOW, stride=C.ROLLING_STRIDE):
    """Mean pairwise Jaccard between edge sets from overlapping windows (60 days, 20-day stride)."""
    sets = []
    for s in range(0, len(z) - window + 1, stride):
        sub = z.iloc[s:s + window]
        sub = sub.loc[:, sub.std() > 0]
        sub = (sub - sub.mean()) / sub.std()
        sets.append(_try_keys(method, sub))
    sets = [x for x in sets if x is not None]
    vals = [_jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return float(np.nanmean(vals)) if vals else np.nan


def growing_convergence(z, method, sizes=C.GROWING_WINDOWS):
    """Mean Jaccard between graphs from consecutive growing windows (first 30, 50, ..., 130 days)."""
    sets = []
    for n in sizes:
        if n > len(z):
            break
        sub = z.iloc[:n]
        sub = sub.loc[:, sub.std() > 0]
        sub = (sub - sub.mean()) / sub.std()
        sets.append(_try_keys(method, sub))
    sets = [x for x in sets if x is not None]
    vals = [_jaccard(a, b) for a, b in zip(sets[:-1], sets[1:])]
    return float(np.nanmean(vals)) if vals else np.nan


def evaluate_stage2(frames, methods=None):
    """Table III averaged over participants."""
    methods = methods or list(s2.METHODS)
    rows = []
    for pid, frame in frames.items():
        z = s2.standardize(frame)
        if len(z) < C.ROLLING_WINDOW:
            continue
        res = s2.run_stage2(frame, methods, verbose=False)
        agr = agreement(res)
        for m in methods:
            if res.get(m) is None:
                continue
            rows.append({"participant": pid, "method": m, "agreement": agr.get(m), "n_edges": len(res[m]),
                         "rolling_stability": rolling_stability(z, m), "growing_convergence": growing_convergence(z, m)})
    df = pd.DataFrame(rows)
    return df.drop(columns="participant").groupby("method").mean().round(3), df
