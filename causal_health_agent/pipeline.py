"""End-to-end orchestration (Algorithm 1) for all participants. Stages 1-3 are fully data-driven (no LLM)."""
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import preprocess as pp
from . import stage1_anomaly as s1
from . import stage2_causal_graph as s2
from . import stage3_root_cause as s3


def run_stages_1_to_3(frames, out, detectors=(C.DEFAULT_DETECTOR,), causal_method=C.DEFAULT_CAUSAL_METHOD,
                      plots=True, verbose=True):
    """Stage 1 + Stage 2 (independent, per participant) -> Stage 3. Writes CSVs/PNGs to `out`.
    Returns (cases, graphs): cases = Stage-3 records with anomaly dates/values, ready for Stage 4."""
    os.makedirs(out, exist_ok=True)
    cases, graphs, ev_rows, edge_rows = [], {}, [], []
    for pid, frame in frames.items():
        if verbose:
            print(f"=== {pid} ===")
        flags = s1.run_stage1(frame, C.CORE_VARIABLES, list(detectors))
        flags.to_csv(os.path.join(out, f"stage1_flags_{pid}.csv"))
        det = C.DEFAULT_DETECTOR if C.DEFAULT_DETECTOR in detectors else detectors[0]
        events = s1.anomaly_events(flags, det)
        ev_rows += [{"participant": pid, "variable": f, "date": d, "is_anomaly": True} for f, d in events]

        if len(pp.complete_cases(frame)) < 40:
            print("  stage2 skipped (<40 complete days)")
            continue
        edges = s2.discover(frame, causal_method)
        edge_rows.append(s2.to_dataframe(edges, pid))
        G = s3.build_graph(edges)
        graphs[pid] = G
        if plots:
            s2.plot_graph(edges, os.path.join(out, f"graph_{pid}.png"), f"{pid} - {causal_method}")

        raw = frame[C.CORE_VARIABLES]
        for feat in sorted({f for f, _ in events}):
            sel = s3.select_root_cause(G, feat)
            if not sel:
                continue
            c = sel[0]
            dates = [d for f, d in events if f == feat]
            sub_nodes = [n for n in s3.ancestor_subgraph(G, feat).nodes if n != feat]
            c.update({
                "participant": pid,
                "anomaly_dates": [d.strftime("%Y-%m-%d") for d in dates],
                "actual_values": [None if pd.isna(raw.loc[d, feat]) else round(float(raw.loc[d, feat]), 2) for d in dates],
                "subgraph_context": {n: [None if pd.isna(raw.loc[d, n]) else round(float(raw.loc[d, n]), 3) for d in dates]
                                     for n in sub_nodes},
            })
            cases.append(c)
        if verbose:
            print(f"  {len(events)} anomalous events, {len(edges)} edges, {sum(c['participant'] == pid for c in cases)} cases")

    pd.DataFrame(ev_rows).to_csv(os.path.join(out, "stage1_anomalies.csv"), index=False)
    if edge_rows:
        pd.concat(edge_rows).to_csv(os.path.join(out, "stage2_edges.csv"), index=False)
    pd.DataFrame([{"participant": c["participant"], "anomaly_node": c["anomaly_node"], "rcrs_pick": c["root_cause"],
                   "chain": " -> ".join(c["chain"]), "rcrs": round(c["rcrs"], 4),
                   "n_dates": len(c["anomaly_dates"])} for c in cases]).to_csv(os.path.join(out, "stage3_results.csv"), index=False)
    return cases, graphs


def apply_ablation(case, ablation, G, rng):
    """Structural ablations of Table IX. Returns (case, strategy, react_flag, verify_flag)."""
    strategy, verify = "react_graph_aware", True
    if ablation == "stage2":            # no temporal causal graph: strongest same-day correlate instead
        cc = case["_complete"]
        corr = cc.corr()[case["anomaly_node"]].drop(case["anomaly_node"]).abs()
        root = corr.idxmax()
        case = dict(case, root_cause=root, chain=[root, case["anomaly_node"]],
                    edges=[{"source": root, "target": case["anomaly_node"], "lag": 0, "strength": float(corr[root])}])
    elif ablation == "stage3":          # no RCRS: random ancestor path instead of the top-ranked one
        paths = s3.enumerate_paths(G, case["anomaly_node"])
        if paths:
            p = paths[rng.integers(len(paths))]
            case = dict(case, root_cause=p[0], chain=p, edges=s3.path_edges(G, p))
    elif ablation == "stage4":          # no agentic retrieval: one-shot graph-aware query
        strategy = "graph_aware"
    elif ablation == "stage5":          # no evidence verification
        verify = False
    return case, strategy, verify


def run_llm_stages(cases, out, client, search, strategy="react_graph_aware", per_edge=False, models=None,
                   max_cases=None, judge=True, ablation=None, graphs=None, frames=None, verbose=True):
    """Stages 4-6 (+ LLM-as-judge). Checkpoints JSON/CSV under `out`; safe to re-run (resumes)."""
    from .evaluation import judge as J
    from .stage5_verification import verify_case
    from .stage6_narrative import build_payload, generate_all
    tag = f"_ablate_{ablation}" if ablation else ""
    vpath, npath, jpath = (os.path.join(out, f"stage5_verified{tag}.json"), os.path.join(out, f"stage6_narratives{tag}.json"),
                           os.path.join(out, f"stage6_judged{tag}.csv"))
    load = lambda p: json.load(open(p)) if os.path.exists(p) else {}
    verified, narratives = load(vpath), load(npath)
    rng = np.random.default_rng(C.SEED)
    rows = pd.read_csv(jpath).to_dict("records") if os.path.exists(jpath) else []
    for c in (cases[:max_cases] if max_cases else cases):
        key = f"{c['participant']}__{c['anomaly_node']}"
        if key not in verified:
            case, strat, do_verify = c, strategy, True
            if ablation:
                case = dict(c, _complete=pp.complete_cases(frames[c["participant"]]))
                case, strat, do_verify = apply_ablation(case, ablation, graphs[c["participant"]], rng)
                case.pop("_complete", None)
            if do_verify:
                verified[key] = verify_case(case, client, search, strat, per_edge)
            else:
                verified[key] = dict(case, verification={"strategy": "none", "per_edge": [], "overall": {
                    "verdict": "not verified", "mechanism": "", "sources": []}})
            json.dump(verified, open(vpath, "w"), indent=1, default=str)
            if verbose:
                print(f"[stage5] {key}: {verified[key]['verification']['overall']['verdict']}")
        if key not in narratives:
            narratives[key] = generate_all(client, build_payload(verified[key]), models)
            json.dump(narratives, open(npath, "w"), indent=1)
        if judge:
            done = {(r["case"], r["model"]) for r in rows}
            v = verified[key]
            findings = {k: v[k] for k in ("root_cause", "anomaly_node", "chain", "verification")}
            for model, text in narratives[key].items():
                if text and (key, model) not in done:
                    rows.append({"case": key, "model": model, "configuration": ablation or "full",
                                 **J.score_narrative(client, text, findings)})
                    pd.DataFrame(rows).to_csv(jpath, index=False)
    return verified, narratives, rows
