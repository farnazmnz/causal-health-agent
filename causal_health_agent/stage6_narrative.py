"""
Stage 6 - graph-grounded health narrative generation (paper Sec. III-H).

Input : structured case = flagged anomaly (dates, values), descriptive subgraph values, verified pathway
        with per-link/overall verdicts and real sources.
Output: one personalised narrative per LLM. Subgraph values are context only and may not be presented
        as causal. The prompt forbids unsupported facts, causal claims, mechanisms and citations.
All models go through the same OpenRouter interface.
"""
import json

from . import config as C
from .llm import chat

NARRATIVE_SYSTEM_PROMPT = f"""You are a health-insight communicator writing a short, warm, personalised
"health story" for a wearable-device user from ALREADY-VERIFIED findings.

You receive JSON with:
- "anomaly": variable, dates and values that were flagged.
- "subgraph_context": raw values of OTHER variables on the same dates. Background only: you MAY mention
  them descriptively, but must NOT present them as causal.
- "verified_chain", "per_edge_verification", "overall_verdict", "overall_mechanism", "sources".

STRUCTURE
1. Identify the anomalous feature with its date(s) and value(s).
2. Walk through the causal pathway in order and state each link's evidence status honestly
   (supported / contradicted / inconclusive); never present inconclusive links as established.
3. Cite only source domains present in the input - NEVER invent a source.
4. End with ONE brief practical suggestion grounded only in the verified findings.
5. Close with exactly: "{C.DISCLAIMER}"

RULES: no diagnosis, no overstated certainty, no causal claim that is not in the verification input,
no unsupported facts, mechanisms or citations."""


def build_payload(vcase):
    v = vcase["verification"]
    return {
        "anomaly": {"variable": vcase["anomaly_node"], "dates": vcase["anomaly_dates"], "values": vcase["actual_values"]},
        "subgraph_context": vcase.get("subgraph_context", {}),
        "verified_chain": vcase["chain"],
        "per_edge_verification": v.get("per_edge") or [
            {"source": e["source"], "target": e["target"], "lag": e["lag"], "verdict": "not separately verified"}
            for e in vcase["edges"]],
        "overall_verdict": v["overall"]["verdict"],
        "overall_mechanism": v["overall"]["mechanism"],
        "sources": v["overall"]["sources"],
    }


def generate_narrative(client, model, payload):
    return chat(client, model, [{"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                                {"role": "user", "content": json.dumps(payload, default=str)}], temperature=0.4)


def generate_all(client, payload, models=None):
    """{display name: narrative or None} for every model (a failing model never stops the run)."""
    out = {}
    for name, slug in (models or C.NARRATIVE_MODELS).items():
        try:
            out[name] = generate_narrative(client, slug, payload)
        except Exception as e:
            print(f"  [stage6] {name} failed: {e}")
            out[name] = None
    return out
