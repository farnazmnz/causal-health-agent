"""
Stage 4 - graph-to-query transformation (paper Sec. III-F).

Turns a ranked causal path into medical-verification queries. Five strategies share the same input:
    template           endpoint-only fixed question (non-LLM baseline)
    multi_hop          one query per edge, so middle nodes are verified too
    llm_based          an LLM (GPT-5.6) rewrites the whole path as a natural-language medical query
    graph_aware        deterministic query that keeps the full path AND its lags
    react_graph_aware  graph_aware seed + iterative Thought-Action-Observation refinement
                       (the refinement loop lives in stage5_verification.verify)
A `case` is a Stage-3 record: {anomaly_node, root_cause, chain, edges:[{source,target,lag,strength}], ...}.
"""
from . import config as C
from .llm import chat

STRATEGIES = ("template", "multi_hop", "llm_based", "graph_aware", "react_graph_aware")


def _h(name):
    return name.replace("_", " ")


def _delay(lag):
    return "on the same day" if lag == 0 else f"with a {lag}-day delay"


def template_query(case):
    return [{"query": f"Does {_h(case['root_cause'])} affect {_h(case['anomaly_node'])}?"}]


def multi_hop_queries(case):
    return [{"query": f"Does {_h(e['source'])} affect {_h(e['target'])}?", "edge": e} for e in case["edges"]]


def graph_aware_query(case):
    steps = "; ".join(f"{_h(e['source'])} affects {_h(e['target'])} {_delay(e['lag'])}" for e in case["edges"])
    chain = " -> ".join(_h(n) for n in case["chain"])
    return [{"query": f"Explain the physiological mechanism connecting {chain} ({steps})"}]


def llm_based_query(case, client, model=C.QUERY_MODEL):
    prompt = ("Convert this causal pathway found in a person's smartwatch data into ONE concise medical "
              "search query for retrieving scientific evidence. Return only the query.\n"
              f"Pathway: {' -> '.join(_h(n) for n in case['chain'])}\n"
              + "\n".join(f"- {_h(e['source'])} -> {_h(e['target'])} ({_delay(e['lag'])})" for e in case["edges"]))
    return [{"query": chat(client, model, [{"role": "user", "content": prompt}]).strip().strip('"')}]


def build_queries(case, strategy="react_graph_aware", client=None):
    """Returns a list of {"query": str, ["edge": dict]}."""
    if strategy == "template":
        return template_query(case)
    if strategy == "multi_hop":
        return multi_hop_queries(case)
    if strategy == "llm_based":
        return llm_based_query(case, client)
    if strategy in ("graph_aware", "react_graph_aware"):
        return graph_aware_query(case)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
