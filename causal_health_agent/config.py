"""Central configuration. Values follow the paper (Section III) unless marked ASSUMPTION."""

# ---- Data (Sec. III-A) ------------------------------------------------------
CORE_VARIABLES = [
    "resting_hr", "sleep_minutes", "deep_sleep_minutes", "sleep_score",
    "hrv_proxy_restlessness", "steps", "very_active_minutes",
    "moderately_active_minutes", "sedentary_minutes",
]
SHORT_NAMES = {
    "resting_hr": "RHR", "sleep_minutes": "Sleep", "deep_sleep_minutes": "Deep",
    "sleep_score": "SleepScr", "hrv_proxy_restlessness": "HRV*", "steps": "Steps",
    "very_active_minutes": "VeryAct", "moderately_active_minutes": "ModAct",
    "sedentary_minutes": "Sedent",
}

# ---- Stage 1 (Sec. III-C) ---------------------------------------------------
WINDOW = 14             # Z-score, Isolation Forest, AR, ARIMA, Chronos
SARIMA_WINDOW = 30      # captures weekly seasonality
TRAIN_FRAC = 0.5        # neural detectors: train on first 50%, score the rest
DEFAULT_DETECTOR = "sarima"
Z_THRESHOLD = 2.5       # ASSUMPTION: not stated in the paper
IF_CONTAMINATION = 0.10 # ASSUMPTION
N_SYNTHETIC_ANOMALIES = 6   # Sec. IV-A
SYNTHETIC_MAGNITUDE = 4.0   # ASSUMPTION: spike size in standard deviations
MIN_FLAGGED_FOR_AGREEMENT = 3

# ---- Stage 2 (Sec. III-D) ---------------------------------------------------
MAX_LAG = 3             # edges carry lags 0-3 days
ALPHA = 0.05
DEFAULT_CAUSAL_METHOD = "varlingam"
LINGAM_THRESHOLD = 0.15     # ASSUMPTION: |coef| cut-off for VARLiNGAM edges
DYNOTEARS_LAMBDA = 0.1      # ASSUMPTION
DYNOTEARS_THRESHOLD = 0.3   # ASSUMPTION
PEARSON_MIN_R = 0.3         # ASSUMPTION
ROLLING_WINDOW, ROLLING_STRIDE = 60, 20            # Rolling Stability
GROWING_WINDOWS = (30, 50, 70, 90, 110, 130)       # Growing Convergence

# ---- Stage 3 (Sec. III-E, IV-C) --------------------------------------------
OPERATIONAL_FRAC = 0.6  # operational graph = first 60% of history

# ---- Stage 5: 16 trusted sources (footnote 1) --------------------------------
TRUSTED_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nih.gov", "cdc.gov", "who.int",
    "mayoclinic.org", "my.clevelandclinic.org", "health.harvard.edu",
    "hopkinsmedicine.org", "heart.org", "sleepfoundation.org", "nhlbi.nih.gov",
    "nature.com", "bmj.com", "webmd.com", "medlineplus.gov",
]
TAVILY_MAX_RESULTS = 6
REACT_MAX_STEPS = 4     # ASSUMPTION

# ---- LLMs, all through OpenRouter (Sec. III-G/H, IV-E) ----------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QUERY_MODEL = "openai/gpt-5.6-sol"      # LLM-based queries + evidence assessment
JUDGE_MODEL = "anthropic/claude-opus-5" # single fixed judge
# ASSUMPTION: check these slugs on openrouter.ai/models (the paper names the models, not slugs).
NARRATIVE_MODELS = {
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "GPT-5.6": "openai/gpt-5.6-sol",
    "Gemini 3.7 Flash": "google/gemini-3.7-flash",
    "Qwen3.7-Max": "qwen/qwen3.7-max",
    "DeepSeek-V4": "deepseek/deepseek-v4-pro-0813",
}
RUBRIC_DIMENSIONS = [
    "medical_correctness", "evidence_grounding", "causal_consistency", "personalization",
    "completeness", "clarity", "explainability", "actionability", "safety",
    "uncertainty_calibration",
]
DISCLAIMER = "These are patterns from your own data, not medical diagnoses."
SEED = 0
