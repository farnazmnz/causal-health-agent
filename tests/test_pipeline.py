"""Offline tests: no network, no API keys. Fake LLM/search clients exercise Stages 4-6 control flow."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from causal_health_agent import config as C
from causal_health_agent import pipeline as P
from causal_health_agent import preprocess as pp
from causal_health_agent import stage1_anomaly as s1
from causal_health_agent import stage2_causal_graph as s2
from causal_health_agent import stage3_root_cause as s3
from causal_health_agent import stage4_queries as s4
from causal_health_agent import stage5_verification as s5
from causal_health_agent import stage6_narrative as s6
from causal_health_agent.evaluation import judge, stage1_eval


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create; answers by prompt type."""
    def __init__(self, verdicts=("inconclusive", "supported")):
        self.verdicts, self.calls = list(verdicts), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature=0.0):
        text = messages[-1]["content"]
        self.calls.append(model)
        if "CLAIM:" in text:
            v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            body = json.dumps({"verdict": v, "mechanism": "Sleep affects heart rate.", "refined_query": "sleep heart rate"})
        elif "NARRATIVE:" in text:
            body = json.dumps({"scores": {d: 4 for d in C.RUBRIC_DIMENSIONS}})
        elif "Pathway:" in text:
            body = "How does sleep duration affect resting heart rate?"
        else:
            body = "Hello. " + C.DISCLAIMER
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=body))])


def fake_search(query):
    return {"results": [{"title": "T", "url": "https://www.nih.gov/x", "content": "evidence"}]}


CASE = {"anomaly_node": "resting_hr", "root_cause": "steps", "chain": ["steps", "sleep_minutes", "resting_hr"],
        "edges": [{"source": "steps", "target": "sleep_minutes", "lag": 1, "strength": 0.4},
                  {"source": "sleep_minutes", "target": "resting_hr", "lag": 0, "strength": -0.5}],
        "anomaly_dates": ["2019-12-01"], "actual_values": [60.0], "subgraph_context": {"steps": [9000]}}


def test_trusted_domains_are_16():
    assert len(set(C.TRUSTED_DOMAINS)) == 16


def test_rcrs_matches_equation_1():
    G = s3.build_graph([{"source": "a", "target": "b", "lag": 1, "strength": 0.5},
                        {"source": "b", "target": "c", "lag": 0, "strength": 1.0},
                        {"source": "a", "target": "c", "lag": 2, "strength": 0.3}])
    assert s3.rcrs(["a", "b", "c"], G) == pytest.approx((0.5 / 2 + 1.0 / 1) / 2)
    ranked = s3.rank_root_causes(G, "c")
    assert [r["root"] for r in ranked] == ["b", "a"] and ranked[1]["path"] == ["a", "b", "c"]


def test_preprocess_zero_bpm_is_missing(tmp_path):
    (tmp_path / "fitbit").mkdir()
    json.dump([{"dateTime": "2019-11-01 00:00:00", "value": {"value": 0.0}},
               {"dateTime": "2019-11-02 00:00:00", "value": {"value": 55.0}}], open(tmp_path / "fitbit" / "resting_heart_rate.json", "w"))
    s = pp.load_resting_hr(str(tmp_path))
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 55.0


def test_stage1_flags_per_feature_and_catches_spike():
    f = pp.synthetic_frames(1, 120)["p01"]
    f.loc[f.index[80], "steps"] += 500
    fl = s1.run_stage1(f, detectors=["zscore", "ar"])
    assert fl.columns.names == ["detector", "feature"] and fl.shape[1] == 18
    assert fl[("zscore", "steps")].iloc[80]
    assert ("steps", f.index[80]) in s1.anomaly_events(fl, "zscore")


def test_stage1_synthetic_eval_runs():
    t = stage1_eval.synthetic_prf(pp.synthetic_frames(1, 120), ["zscore"], ["steps", "resting_hr"])
    assert {"precision", "recall", "f1"} <= set(t.columns)


def test_stage2_methods_return_edge_schema():
    f = pp.synthetic_frames(1, 200)["p01"]
    res = s2.run_stage2(f, ["pearson", "granger", "dynotears"], verbose=False)
    for es in res.values():
        assert es is not None
        for e in es:
            assert set(e) == {"source", "target", "lag", "strength", "pvalue"} and 0 <= e["lag"] <= C.MAX_LAG


def test_stage4_strategies():
    assert s4.template_query(CASE)[0]["query"] == "Does steps affect resting hr?"
    assert len(s4.multi_hop_queries(CASE)) == 2
    q = s4.graph_aware_query(CASE)[0]["query"]
    assert "1-day delay" in q and "same day" in q and "sleep minutes" in q
    assert s4.build_queries(CASE, "llm_based", FakeClient())[0]["query"].startswith("How does")
    with pytest.raises(ValueError):
        s4.build_queries(CASE, "nope")


def test_stage5_react_refines_until_decisive():
    client = FakeClient(verdicts=("inconclusive", "supported"))
    v = s5.verify(client, fake_search, "x", "seed", react=True)
    assert v["verdict"] == "supported" and len(v["steps"]) == 2
    one = s5.verify(FakeClient(verdicts=("inconclusive",)), fake_search, "x", "seed", react=False)
    assert one["verdict"] == "inconclusive" and len(one["steps"]) == 1


def test_stage5_multi_hop_and_per_edge():
    c = FakeClient(verdicts=("supported",))
    out = s5.verify_case(CASE, c, fake_search, "multi_hop")
    assert len(out["verification"]["per_edge"]) == 2 and out["verification"]["overall"]["verdict"] == "supported"
    out = s5.verify_case(CASE, FakeClient(verdicts=("supported",)), fake_search, "graph_aware", per_edge=True)
    assert len(out["verification"]["per_edge"]) == 2


def test_stage6_payload_and_narratives():
    v = s5.verify_case(CASE, FakeClient(verdicts=("supported",)), fake_search)
    payload = s6.build_payload(v)
    assert payload["overall_verdict"] == "supported" and payload["sources"][0]["url"].startswith("https://")
    out = s6.generate_all(FakeClient(), payload, {"A": "m1", "B": "m2"})
    assert set(out) == {"A", "B"} and all(C.DISCLAIMER in t for t in out.values())
    assert C.DISCLAIMER in s6.NARRATIVE_SYSTEM_PROMPT


def test_full_llm_stages_and_ablations(tmp_path):
    frames = pp.synthetic_frames(1, 150)
    cases, graphs = P.run_stages_1_to_3(frames, str(tmp_path), ("zscore",), plots=False, verbose=False)
    assert cases
    for abl in (None, "stage2", "stage3", "stage4", "stage5"):
        v, n, rows = P.run_llm_stages(cases, str(tmp_path), FakeClient(verdicts=("supported",)), fake_search,
                                      models={"A": "m"}, max_cases=2, ablation=abl, graphs=graphs, frames=frames, verbose=False)
        assert len(v) == 2 and len(rows) == 2
        if abl == "stage5":
            assert list(v.values())[0]["verification"]["overall"]["verdict"] == "not verified"
    t = judge.summarize(rows, "model")
    assert t.loc["A", "safety"] == 4
