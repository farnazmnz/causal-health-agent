"""Stage 3 evaluation (paper Sec. IV-C, Table IV): ranking consistency under limited history.

Operational graph = first 60% of a participant's history; reference graph = full history. For every
anomaly in the operational window both graphs are ranked with RCRS; Top-1 asks whether the operational
top candidate equals the reference top candidate, Top-K whether it is in the reference top K=ceil(n/2).
Baselines (ranked on the operational graph): Random, PageRank, In-Degree, Marginal-Score.
NOTE (as in the paper): the reference is itself RCRS-derived, so this measures temporal self-consistency,
not physiological accuracy."""
import math

import networkx as nx
import numpy as np
import pandas as pd

from .. import config as C
from .. import stage2_causal_graph as s2
from .. import stage3_root_cause as s3


def _anomaly_scores(df, window=C.WINDOW):
    mu = df.rolling(window).mean().shift(1)
    sd = df.rolling(window).std().shift(1).replace(0, np.nan)
    return ((df - mu) / sd).abs()


def _pick(method, S, target, scores_row, G, rng):
    cands = sorted(n for n in S.nodes if n != target)
    if not cands:
        return None
    if method == "rcrs":
        r = s3.rank_root_causes(G, target)
        return r[0]["root"] if r else None
    if method == "random":
        return cands[rng.integers(len(cands))]
    if method == "pagerank":
        pr = nx.pagerank(S, weight=None)
        return max(cands, key=lambda n: (pr.get(n, 0), n))
    if method == "in_degree":
        return max(cands, key=lambda n: (S.in_degree(n), n))
    if method == "marginal":
        return max(cands, key=lambda n: (np.nan_to_num(scores_row.get(n, 0)), n))
    raise ValueError(method)


def evaluate_stage3(frames, events_by_pid, causal_method=C.DEFAULT_CAUSAL_METHOD, seed=C.SEED,
                    methods=("rcrs", "random", "pagerank", "in_degree", "marginal")):
    """events_by_pid: {participant: [(feature, date), ...]} from Stage 1 (see stage1_anomaly.anomaly_events)."""
    rng, hits = np.random.default_rng(seed), {m: [] for m in methods}
    for pid, frame in frames.items():
        cc = frame[C.CORE_VARIABLES].astype(float)
        cut = int(len(cc) * C.OPERATIONAL_FRAC)
        op_frame, cut_date = frame.iloc[:cut], cc.index[cut - 1]
        try:
            G_op = s3.build_graph(s2.discover(op_frame, causal_method))
            G_ref = s3.build_graph(s2.discover(frame, causal_method))
        except Exception as e:
            print(f"  [stage3-eval] {pid} skipped: {e}")
            continue
        wop = s3.max_strength(G_op)
        sc = _anomaly_scores(cc.interpolate(limit=2, limit_area="inside"))
        for feat, date in events_by_pid.get(pid, []):
            if date > cut_date:
                continue
            ref = s3.rank_root_causes(G_ref, feat)
            if not ref:
                continue
            ref_top, K = ref[0]["root"], math.ceil(len(ref) / 2)
            ref_set = {r["root"] for r in ref[:K]}
            S_op = s3.ancestor_subgraph(G_op, feat)
            for m in methods:
                pick = _pick(m, S_op, feat, sc.loc[date] if date in sc.index else {}, G_op, rng)
                hits[m].append((pick == ref_top, pick in ref_set))
    return pd.DataFrame({m: {"top1": np.mean([a for a, _ in h]), "topK": np.mean([b for _, b in h]), "n": len(h)}
                         for m, h in hits.items() if h}).T.round(3)
