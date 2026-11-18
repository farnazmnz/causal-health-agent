"""
Stage 3 - root-cause discovery and ranking (paper Sec. III-E).

For an anomalous feature `a`:
  1. extract the anomaly-specific ANCESTOR SUBGRAPH by a depth-first search over reversed causal edges;
  2. DFS-enumerate all simple paths  v0 -> ... -> a  inside it;
  3. score every path with the Root-Cause Ranking Score (Eq. 1)

        RCRS(P) = (1/L) * sum_i  w~_i / (1 + l_i),      w~_i = |w_i| / max_{e in E} |w_e|

     (w_i strength, l_i lag of edge i, E = ALL edges of the participant's graph);
  4. a candidate root's score is its best path; roots are ranked by that score.
No LLM is used. Output feeds Stage 4.
"""
import networkx as nx
import numpy as np


def build_graph(edges):
    """Directed graph; if a pair has several edges (different lags) keep the strongest."""
    G = nx.DiGraph()
    for e in edges:
        u, v = e["source"], e["target"]
        if u == v:
            continue
        if G.has_edge(u, v) and abs(G[u][v]["strength"]) >= abs(e["strength"]):
            continue
        G.add_edge(u, v, lag=int(e["lag"]), strength=float(e["strength"]))
    return G


def max_strength(G):
    return max((abs(d["strength"]) for *_, d in G.edges(data=True)), default=0.0)


def ancestors_dfs(G, target):
    """Nodes that can reach `target`: iterative DFS over reverse edges."""
    seen, stack = set(), [target]
    while stack:
        n = stack.pop()
        for p in G.predecessors(n):
            if p not in seen and p != target:
                seen.add(p)
                stack.append(p)
    return seen


def ancestor_subgraph(G, target):
    if target not in G:
        return nx.DiGraph()
    keep = ancestors_dfs(G, target) | {target}
    return G.subgraph(keep).copy()


def enumerate_paths(G, target, max_len=4):
    """All simple paths ending at target (at most `max_len` edges) inside the ancestor subgraph."""
    S = ancestor_subgraph(G, target)
    return [p for a in S.nodes if a != target for p in nx.all_simple_paths(S, a, target, cutoff=max_len)]


def rcrs(path, G, wmax=None):
    """Eq. (1)."""
    wmax = wmax or max_strength(G) or 1.0
    terms = [abs(G[u][v]["strength"]) / wmax / (1 + G[u][v]["lag"]) for u, v in zip(path[:-1], path[1:])]
    return float(np.mean(terms))


def rank_root_causes(G, target, wmax=None, max_len=4):
    """Candidate roots ranked by their best path's RCRS (descending)."""
    wmax = wmax or max_strength(G)
    best = {}
    for p in enumerate_paths(G, target, max_len):
        s = rcrs(p, G, wmax)
        if p[0] not in best or s > best[p[0]]["score"]:
            best[p[0]] = {"root": p[0], "path": p, "score": s}
    return sorted(best.values(), key=lambda r: (-r["score"], len(r["path"]), r["root"]))


def path_edges(G, path):
    return [{"source": u, "target": v, "lag": G[u][v]["lag"], "strength": G[u][v]["strength"]}
            for u, v in zip(path[:-1], path[1:])]


def select_root_cause(G, target, top_k=1, **kw):
    """Top-ranked pathway(s) for an anomalous feature, ready for Stage 4. None if no ancestors."""
    ranked = rank_root_causes(G, target, **kw)
    return [{"anomaly_node": target, "root_cause": r["root"], "chain": r["path"],
             "edges": path_edges(G, r["path"]), "rcrs": r["score"]} for r in ranked[:top_k]] or None
