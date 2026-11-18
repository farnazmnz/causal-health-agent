"""
Causal Health Agent - command line entry point.

  # smoke test, no data and no API keys
  python run_pipeline.py --synthetic --fast

  # Stages 1-3 on real PMData (no LLM, no keys)
  python run_pipeline.py --data-root /path/to/PMData

  # + Stages 4-6 and the LLM judge (needs OPENROUTER_API_KEY and TAVILY_API_KEY in the environment)
  python run_pipeline.py --data-root /path/to/PMData --llm --max-cases 15

  # paper evaluations (Tables II-IV; slow)
  python run_pipeline.py --data-root /path/to/PMData --evaluate stage1 stage2 stage3

  # ablation of one component (Table IX)
  python run_pipeline.py --data-root /path/to/PMData --llm --ablation stage5
"""
import argparse
import os
import warnings

warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import pandas as pd

from causal_health_agent import config as C
from causal_health_agent import pipeline as P
from causal_health_agent import preprocess as pp
from causal_health_agent import stage1_anomaly as s1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--participants", nargs="*", help="e.g. p01 p02 (default: all)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--detectors", nargs="*", default=[C.DEFAULT_DETECTOR], choices=list(s1.DETECTORS))
    ap.add_argument("--fast", action="store_true", help="cheap detectors only (zscore, ar, isolation_forest)")
    ap.add_argument("--causal-method", default=C.DEFAULT_CAUSAL_METHOD)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--llm", action="store_true", help="run Stages 4-6 + judge (needs API keys)")
    ap.add_argument("--strategy", default="react_graph_aware",
                    choices=["template", "multi_hop", "llm_based", "graph_aware", "react_graph_aware"])
    ap.add_argument("--per-edge", action="store_true", help="also verify every link of the path separately")
    ap.add_argument("--max-cases", type=int); ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--ablation", choices=["stage2", "stage3", "stage4", "stage5"])
    ap.add_argument("--evaluate", nargs="*", choices=["stage1", "stage2", "stage3"])
    a = ap.parse_args()

    frames = pp.synthetic_frames() if a.synthetic else pp.load_all_participants(a.data_root)
    if a.participants:
        frames = {k: v for k, v in frames.items() if k in a.participants}
    if not frames:
        raise SystemExit("No participants loaded (use --data-root or --synthetic).")
    os.makedirs(a.out, exist_ok=True)
    summ = pp.summarize(frames); summ.to_csv(os.path.join(a.out, "participants_summary.csv"), index=False)
    print(summ.to_string(index=False))
    dets = list(s1.FAST) if a.fast else a.detectors

    cases, graphs = P.run_stages_1_to_3(frames, a.out, dets, a.causal_method, plots=not a.no_plots)

    if a.evaluate:
        from causal_health_agent.evaluation import stage1_eval, stage2_eval, stage3_eval
        if "stage1" in a.evaluate:
            t = stage1_eval.evaluate_stage1(frames, list(s1.FAST) if a.fast else list(s1.DETECTORS))
            t.to_csv(os.path.join(a.out, "table2_stage1.csv")); print("\nTable II\n", t.round(3))
        if "stage2" in a.evaluate:
            t, _ = stage2_eval.evaluate_stage2(frames)
            t.to_csv(os.path.join(a.out, "table3_stage2.csv")); print("\nTable III\n", t)
        if "stage3" in a.evaluate:
            ev = pd.read_csv(os.path.join(a.out, "stage1_anomalies.csv"), parse_dates=["date"])
            events = {p: list(zip(g.variable, g.date)) for p, g in ev.groupby("participant")}
            t = stage3_eval.evaluate_stage3(frames, events, a.causal_method)
            t.to_csv(os.path.join(a.out, "table4_stage3.csv")); print("\nTable IV\n", t)

    if a.llm:
        from causal_health_agent.llm import get_client
        from causal_health_agent.stage5_verification import TavilySearch
        client, search = get_client(), TavilySearch(cache_path=os.path.join(a.out, "tavily_cache.json"))
        _, _, rows = P.run_llm_stages(cases, a.out, client, search, a.strategy, a.per_edge, None, a.max_cases,
                                      not a.no_judge, a.ablation, graphs, frames)
        if rows:
            from causal_health_agent.evaluation.judge import summarize
            t = summarize(rows, "model"); t.to_csv(os.path.join(a.out, "table8_narratives.csv")); print("\nTable VIII\n", t)
    print(f"\nDone -> {a.out}/")


if __name__ == "__main__":
    main()
