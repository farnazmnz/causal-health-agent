"""
Stage 2 - personalised, directed, lag-aware causal graph (paper Sec. III-D).

Input : complete-case daily series of the nine features for ONE participant (never pooled).
Output: list of edges {source, target, lag, strength, pvalue}; `lag` = days by which the cause
        precedes the effect (lag 2 => source at t-2 explains target at t); `strength` = standardised
        effect; `pvalue` = FDR-corrected p-value where the method provides one (else None).

Methods: VARLiNGAM (default, [9]), Granger [7], PCMCI [8], DYNOTEARS [10], and Pearson as a
non-causal (undirected) baseline. No LLM is used here.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.optimize as sopt
from scipy import stats

from . import config as C
from .preprocess import complete_cases

warnings.filterwarnings("ignore")
DIRECTED = {"varlingam": True, "granger": True, "pcmci": True, "dynotears": True, "pearson": False}


def standardize(frame, variables=C.CORE_VARIABLES):
    sub = complete_cases(frame.reindex(columns=variables)) if "participant" in frame else frame.dropna()
    sub = sub.loc[:, sub.std() > 0]
    return (sub - sub.mean()) / sub.std()


def _fdr(p, alpha=C.ALPHA):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    if p.size == 0:
        return np.array([], bool), p
    rej, q, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return rej, q


def _edge(s, t, lag, w, p=None):
    return {"source": s, "target": t, "lag": int(lag), "strength": float(w), "pvalue": None if p is None else float(p)}


# --------------------------------------------------------------------- Pearson
def pearson_baseline(z):
    cols, pairs, ps, rs = list(z.columns), [], [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(z[cols[i]], z[cols[j]])
            pairs.append((cols[i], cols[j])); ps.append(p); rs.append(r)
    rej, q = _fdr(ps)
    return [_edge(a, b, 0, r, qq) for (a, b), r, ok, qq in zip(pairs, rs, rej, q)
            if ok and abs(r) >= C.PEARSON_MIN_R]


# --------------------------------------------------------------------- Granger
def granger_causality(z, max_lag=C.MAX_LAG):
    """Pairwise F-tests; lag = argmin p over 1..max_lag; Benjamini-Hochberg FDR over all pairs.
    Strength = sum of the cause's lagged coefficients in the unrestricted regression."""
    from statsmodels.tsa.stattools import grangercausalitytests
    cols, cand = list(z.columns), []
    for tgt in cols:
        for src in cols:
            if src == tgt:
                continue
            try:
                res = grangercausalitytests(z[[tgt, src]], maxlag=max_lag, verbose=False)
            except Exception:
                continue
            best = min(range(1, max_lag + 1), key=lambda l: res[l][0]["ssr_ftest"][1])
            beta = np.asarray(res[best][1][1].params)[best:2 * best].sum()   # [y lags | x lags | const]
            cand.append((src, tgt, best, res[best][0]["ssr_ftest"][1], beta))
    if not cand:
        return []
    rej, q = _fdr([c[3] for c in cand])
    return [_edge(s, t, l, b, qq) for (s, t, l, _p, b), ok, qq in zip(cand, rej, q) if ok]


# ----------------------------------------------------------------------- PCMCI
def pcmci_causal_graph(z, tau_max=C.MAX_LAG):
    from tigramite import data_processing as tp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    cols = list(z.columns)
    pc = PCMCI(dataframe=tp.DataFrame(z.values, var_names=cols), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=0.05)
    q = pc.get_corrected_pvalues(p_matrix=res["p_matrix"], fdr_method="fdr_bh", tau_min=1, tau_max=tau_max)
    return [_edge(cols[i], cols[j], tau, res["val_matrix"][i, j, tau], q[i, j, tau])
            for i in range(len(cols)) for j in range(len(cols)) if i != j
            for tau in range(1, tau_max + 1) if q[i, j, tau] < C.ALPHA]


# ------------------------------------------------------------------- VARLiNGAM
def varlingam_causal_graph(z, lags=C.MAX_LAG, threshold=C.LINGAM_THRESHOLD):
    """VAR-LiNGAM: k=0 -> instantaneous effects, k>=1 -> lag-k effects (edges kept if |coef| >= threshold)."""
    import lingam
    cols = list(z.columns)
    m = lingam.VARLiNGAM(lags=lags, random_state=C.SEED).fit(z.values)
    return [_edge(cols[j], cols[i], k, B[i, j])         # B[i, j]: effect of j(t-k) on i(t)
            for k, B in enumerate(m.adjacency_matrices_)
            for i in range(len(cols)) for j in range(len(cols))
            if i != j and abs(B[i, j]) >= threshold]


# ------------------------------------------------------------------- DYNOTEARS
def dynotears_causal_graph(z, p=C.MAX_LAG, lam=C.DYNOTEARS_LAMBDA, threshold=C.DYNOTEARS_THRESHOLD,
                           max_iter=30, h_tol=1e-8, rho_max=1e12):
    """DYNOTEARS [10]: structural VAR  X_t = X_t W + sum_k X_{t-k} A_k + e,  W acyclic (NOTEARS
    constraint), L1-penalised, solved with an augmented Lagrangian + L-BFGS-B (scipy only)."""
    cols, X = list(z.columns), z.values
    n_full, d = X.shape
    Y = X[p:]
    n = len(Y)
    lagged = [X[p - k:n_full - k] for k in range(1, p + 1)]
    m = d * d * (1 + p)                                   # parameters per sign

    def unpack(w):
        M = (w[:m] - w[m:]).reshape(1 + p, d, d)
        return M[0], M[1:]

    def fun(w):
        W, A = unpack(w)
        R = Y - Y @ W - sum(Xk @ Ak for Xk, Ak in zip(lagged, A))
        loss = 0.5 / n * (R ** 2).sum()
        E = sla.expm(W * W)
        h = np.trace(E) - d
        gW = -Y.T @ R / n + (rho * h + alpha) * 2 * E.T * W
        gA = [-Xk.T @ R / n for Xk in lagged]
        g = np.concatenate([gW.ravel()] + [a.ravel() for a in gA])
        obj = loss + 0.5 * rho * h * h + alpha * h + lam * w.sum()
        return obj, np.concatenate([g + lam, -g + lam])

    bnds = [(0, 0) if (b < d * d and (b // d) == (b % d)) else (0, None) for _ in range(2) for b in range(m)]
    w, rho, alpha, h = np.zeros(2 * m), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            sol = sopt.minimize(fun, w, jac=True, method="L-BFGS-B", bounds=bnds)
            h_new = np.trace(sla.expm(unpack(sol.x)[0] ** 2)) - d
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w, h = sol.x, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W, A = unpack(w)
    edges = [_edge(cols[i], cols[j], 0, W[i, j]) for i in range(d) for j in range(d)
             if i != j and abs(W[i, j]) >= threshold]                  # W[i, j]: i -> j
    for k, Ak in enumerate(A, start=1):
        edges += [_edge(cols[i], cols[j], k, Ak[i, j]) for i in range(d) for j in range(d)
                  if abs(Ak[i, j]) >= threshold]                       # A_k[i, j]: i(t-k) -> j(t)
    return edges


METHODS = {"varlingam": varlingam_causal_graph, "granger": granger_causality, "pcmci": pcmci_causal_graph,
           "dynotears": dynotears_causal_graph, "pearson": pearson_baseline}


def discover(frame_or_z, method=C.DEFAULT_CAUSAL_METHOD, variables=C.CORE_VARIABLES):
    """Edges for one participant. Accepts the daily frame (complete cases are taken) or an
    already-standardised DataFrame."""
    z = frame_or_z if "participant" not in frame_or_z else standardize(frame_or_z, variables)
    return METHODS[method](z)


def run_stage2(frame, methods=None, variables=C.CORE_VARIABLES, verbose=True):
    z, out = standardize(frame, variables), {}
    for name in (methods or METHODS):
        try:
            out[name] = METHODS[name](z)
        except Exception as e:
            out[name] = None
            if verbose:
                print(f"  [stage2] {name} FAILED: {e}")
    return out


def to_dataframe(edges, participant=None):
    df = pd.DataFrame(edges, columns=["source", "target", "lag", "strength", "pvalue"])
    if participant is not None:
        df.insert(0, "participant", participant)
    return df


# ---------------------------------------------------------------- visualisation
def plot_graph(edges, path, title="", variables=C.CORE_VARIABLES):
    """Circular layout, arrow = causal direction, colour = lag (0-3 days), label = standardised strength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    n = len(variables)
    pos = {v: (np.cos(2 * np.pi * i / n + np.pi / 2), np.sin(2 * np.pi * i / n + np.pi / 2))
           for i, v in enumerate(variables)}
    lag_col = {0: "#2ca02c", 1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd"}
    G = nx.MultiDiGraph(); G.add_nodes_from(variables)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#dbe9f6", edgecolors="#33658a", node_size=1500)
    nx.draw_networkx_labels(G, pos, labels={v: C.SHORT_NAMES.get(v, v) for v in variables}, ax=ax, font_size=8)
    for e in edges:
        rad = 0.15 + 0.1 * e["lag"]
        nx.draw_networkx_edges(nx.DiGraph([(e["source"], e["target"])]), pos, ax=ax, edge_color=lag_col[e["lag"]],
                               arrows=True, arrowstyle="-|>", arrowsize=16, node_size=1500,
                               connectionstyle=f"arc3,rad={rad}", width=1.5)
        (x1, y1), (x2, y2) = pos[e["source"]], pos[e["target"]]
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.05, f"{e['strength']:.2f}", fontsize=6.5, color=lag_col[e["lag"]])
    for k, c in lag_col.items():
        ax.plot([], [], color=c, label=f"lag {k}")
    ax.legend(loc="lower right", fontsize=8); ax.set_title(title); ax.axis("off")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
