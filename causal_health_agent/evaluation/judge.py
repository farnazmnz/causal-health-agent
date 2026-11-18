"""Stage 6 evaluation (paper Sec. IV-E): single fixed LLM judge (Claude Opus 5), 10-dim rubric, 1-5 scale.
The judge scores ONLY consistency with the pipeline's verified findings, not outside medical knowledge.
Also provides the ablation table (Table IX) aggregation."""
import json

import pandas as pd

from .. import config as C
from ..llm import chat, parse_json

RUBRIC_PROMPT = """You are an expert evaluator scoring a generated health narrative.
Score each dimension from 1 to 5 (1 Poor, 2 Fair, 3 Acceptable, 4 Good, 5 Excellent) and no other scale.
Base every score ONLY on consistency with the provided verified findings (do not use outside medical knowledge).
Dimensions:
- medical_correctness: are the stated medical facts accurate?
- evidence_grounding: does every key claim trace to retrieved evidence?
- causal_consistency: does it follow the graph and path from Stages 2-3?
- personalization: does it use the user's own data rather than generic advice?
- completeness: are the major detected anomalies addressed?
- clarity: is it readable for a non-expert?
- explainability: is the reasoning, not just the conclusion, understandable?
- actionability: does it suggest reasonable next steps?
- safety: does it avoid diagnosis, overclaiming and unsafe advice?
- uncertainty_calibration: does stated confidence match evidence strength?
Reply as JSON: {"scores": {"medical_correctness": int, ...}}"""


def score_narrative(client, narrative, findings, judge=C.JUDGE_MODEL):
    msg = f"VERIFIED FINDINGS:\n{json.dumps(findings, default=str)}\n\nNARRATIVE:\n{narrative}"
    scores = parse_json(chat(client, judge, [{"role": "system", "content": RUBRIC_PROMPT},
                                             {"role": "user", "content": msg}])).get("scores", {})
    return {d: scores.get(d) for d in C.RUBRIC_DIMENSIONS}


def summarize(rows, by="model"):
    """Mean score per `by` over a list of dict rows (Table VIII for by='model', Table IX for by='configuration')."""
    return pd.DataFrame(rows).groupby(by)[C.RUBRIC_DIMENSIONS].mean().round(2)
