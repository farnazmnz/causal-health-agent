"""
Stage 5 - medical evidence verification (paper Sec. III-G).

Each query is sent to the Tavily Search API restricted to 16 trusted medical/scientific domains; an LLM
(RAG) judges the retrieved passages and returns a structured verdict
    supported | contradicted | inconclusive   + a short evidence-grounded mechanism summary.
The LLM may use ONLY the retrieved results. For `react_graph_aware` the judge may propose a refined query
and the loop repeats (Thought -> Action: search -> Observation) until a decisive verdict or max steps.
"""
import hashlib
import json
import os

from . import config as C
from .llm import chat, parse_json
from .stage4_queries import build_queries

SYSTEM_PROMPT = """You are a biomedical evidence researcher for wearable-health data.
You are given a CAUSAL CLAIM and search results retrieved from trusted medical sources.
ANTI-HALLUCINATION RULES (non-negotiable):
1. Do NOT invent any fact, mechanism or citation; use ONLY the provided search results.
2. If the results do not address the claim, say so (verdict "inconclusive").
3. Never output URLs or DOIs that are not in the results."""

ASSESS_PROMPT = """CLAIM: {claim}

SEARCH RESULTS:
{results}

Based ONLY on these results, is the claim supported, contradicted, or not sufficiently addressed?
Reply as JSON: {{"verdict": "supported|contradicted|inconclusive", "mechanism": "<1-3 sentences grounded in the results>", "refined_query": "<better search query>" or null}}"""


class TavilySearch:
    """Tavily search limited to TRUSTED_DOMAINS, with an optional on-disk cache."""

    def __init__(self, api_key=None, cache_path=None, domains=C.TRUSTED_DOMAINS, max_results=C.TAVILY_MAX_RESULTS):
        from tavily import TavilyClient
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("Set the TAVILY_API_KEY environment variable (see .env.example).")
        self.client, self.domains, self.max_results, self.cache_path = TavilyClient(api_key=key), domains, max_results, cache_path
        self.cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}

    def __call__(self, query):
        k = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        if k not in self.cache:
            try:
                self.cache[k] = self.client.search(query=query, max_results=self.max_results, include_domains=self.domains)
            except Exception as e:
                print(f"    [tavily error] {e}")
                return {"results": []}
            if self.cache_path:
                json.dump(self.cache, open(self.cache_path, "w"), default=str)
        return self.cache[k]


def _compact(results, n=400):
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:n]} for r in results]


def assess(client, claim, results, model=C.QUERY_MODEL):
    prompt = ASSESS_PROMPT.format(claim=claim, results=json.dumps(_compact(results), indent=1))
    raw = chat(client, model, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    p = parse_json(raw)
    if p.get("verdict") not in ("supported", "contradicted", "inconclusive"):
        p["verdict"] = "inconclusive"
    return p


def verify(client, search, claim, seed_query, react=True, max_steps=C.REACT_MAX_STEPS):
    """One claim -> {verdict, mechanism, sources, steps}. `react=False` = single retrieve+assess."""
    query, steps, last, results = seed_query, [], {}, []
    for i in range(max_steps if react else 1):
        results = search(query).get("results", [])
        last = assess(client, claim, results)
        steps.append({"step": i, "query": query, "n_results": len(results), "verdict": last["verdict"]})
        if last["verdict"] != "inconclusive" or not react or not last.get("refined_query"):
            break
        query = last["refined_query"]
    return {"verdict": last.get("verdict", "inconclusive"), "mechanism": last.get("mechanism", ""),
            "sources": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results[:3]], "steps": steps}


def _aggregate(per_edge):
    v = [e["verdict"] for e in per_edge]
    return "supported" if v and all(x == "supported" for x in v) else \
           "contradicted" if "contradicted" in v else "inconclusive"


def verify_case(case, client, search, strategy="react_graph_aware", per_edge=False):
    """Verify one Stage-3 case. Returns the case enriched with `verification`:
        overall: verdict/mechanism/sources for the whole pathway
        per_edge: (multi_hop, or when per_edge=True) a verdict per link so middle nodes are checked too."""
    queries = build_queries(case, strategy, client)
    react = strategy == "react_graph_aware"
    out = dict(case)
    if strategy == "multi_hop":
        edges = [dict(q["edge"], **verify(client, search, f"{q['edge']['source']} affects {q['edge']['target']}",
                                          q["query"], react=False)) for q in queries]
        out["verification"] = {"strategy": strategy, "per_edge": edges, "overall": {"verdict": _aggregate(edges),
                               "mechanism": "; ".join(e["mechanism"] for e in edges if e["mechanism"]),
                               "sources": [s for e in edges for s in e["sources"]][:3]}}
        return out
    claim = f"{case['root_cause']} affects {case['anomaly_node']} via {' -> '.join(case['chain'])}"
    overall = verify(client, search, claim, queries[0]["query"], react=react)
    edges = []
    if per_edge:
        for e in case["edges"]:
            edges.append(dict(e, **verify(client, search, f"{e['source']} affects {e['target']}",
                                          f"Does {e['source'].replace('_', ' ')} affect {e['target'].replace('_', ' ')}?", react=False)))
    out["verification"] = {"strategy": strategy, "per_edge": edges, "overall": overall}
    return out
