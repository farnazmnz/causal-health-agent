"""
PMData preprocessing -> one daily DataFrame per participant (Sec. III-A).

Expected raw layout (PMData release):
    <root>/pXX/fitbit/{steps,calories,*_minutes,resting_heart_rate,sleep}.json, sleep_score.csv
    <root>/pXX/googledocs/reporting.csv     (lifestyle, optional)
    <root>/participant-overview.xlsx        (demographics, optional)

The four data-quality issues reported in the paper are handled here:
  1. zero-bpm resting heart rate is a missing-data sentinel      -> NaN
  2. incorrect / unparseable timestamps                          -> parsed defensively, bad rows dropped;
                                                                    sleep-score rows are dated through
                                                                    their sleep-log id (dateOfSleep)
  3. duplicate rows (lifestyle records; also minute-level files) -> dropped
  4. inconsistent whitespace in the demographic record           -> stripped
Missing days are flagged, never imputed.
"""
import json
import os

import numpy as np
import pandas as pd

from .config import CORE_VARIABLES


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _to_date(s):
    """Parse mixed timestamps defensively; unparseable -> NaT (dropped later)."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    return out.dt.tz_localize(None).dt.normalize()


def _daily(pdir, name, how):
    data = _read_json(os.path.join(pdir, "fitbit", f"{name}.json"))
    if not data:
        return pd.Series(dtype=float, name=name)
    df = pd.DataFrame(data).drop_duplicates()                   # (3)
    df["date"] = pd.to_datetime(df["dateTime"], errors="coerce").dt.normalize()   # (2)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.groupby("date")["value"].agg(how if how != "sum" else lambda x: x.sum(min_count=1))


def load_resting_hr(pdir):
    data = _read_json(os.path.join(pdir, "fitbit", "resting_heart_rate.json"))
    if not data:
        return pd.Series(dtype=float, name="resting_hr")
    rows = [(r.get("dateTime"), (r["value"].get("value") if isinstance(r.get("value"), dict) else r.get("value")))
            for r in data]
    df = pd.DataFrame(rows, columns=["dateTime", "value"]).drop_duplicates()
    df["date"] = pd.to_datetime(df["dateTime"], errors="coerce").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["value"] <= 0, "value"] = np.nan                  # (1) 0 bpm = "not computed"
    return df.dropna(subset=["date"]).groupby("date")["value"].mean().rename("resting_hr")


def load_sleep(pdir):
    """Returns (daily sleep/deep-sleep minutes, {logId: dateOfSleep})."""
    cols = ["sleep_minutes", "deep_sleep_minutes"]
    data = _read_json(os.path.join(pdir, "fitbit", "sleep.json"))
    if not data:
        return pd.DataFrame(columns=cols), {}
    rows, log_dates = [], {}
    for r in data:
        deep = ((r.get("levels") or {}).get("summary") or {}).get("deep", {}).get("minutes", np.nan)
        rows.append((r.get("logId"), r.get("dateOfSleep"), r.get("minutesAsleep", np.nan), deep))
        log_dates[r.get("logId")] = r.get("dateOfSleep")
    df = pd.DataFrame(rows, columns=["logId", "date", *cols]).drop_duplicates("logId")   # (3)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date"]).groupby("date")[cols].sum(min_count=1), log_dates


def load_sleep_score(pdir, log_dates):
    cols = ["sleep_score", "hrv_proxy_restlessness"]
    path = os.path.join(pdir, "fitbit", "sleep_score.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path).drop_duplicates()
    by_log = pd.to_datetime(df["sleep_log_entry_id"].map(log_dates), errors="coerce")
    df["date"] = by_log.fillna(_to_date(df["timestamp"])).dt.normalize()      # (2)
    df = df.rename(columns={"overall_score": "sleep_score", "restlessness": "hrv_proxy_restlessness"})
    return df.dropna(subset=["date"]).groupby("date")[cols].mean()


def build_daily_dataframe(pdir, participant_id):
    sleep, log_dates = load_sleep(pdir)
    frame = pd.DataFrame({
        "steps": _daily(pdir, "steps", "sum"),
        "very_active_minutes": _daily(pdir, "very_active_minutes", "sum"),
        "moderately_active_minutes": _daily(pdir, "moderately_active_minutes", "sum"),
        "sedentary_minutes": _daily(pdir, "sedentary_minutes", "sum"),
        "resting_hr": load_resting_hr(pdir),
    })
    frame = frame.join(sleep, how="outer").join(load_sleep_score(pdir, log_dates), how="outer").sort_index()
    frame.index.name = "date"
    frame = frame.reindex(columns=CORE_VARIABLES)
    frame["is_missing_day"] = frame.isna().any(axis=1)
    frame["participant"] = participant_id
    return frame


def list_participants(root):
    return sorted(d for d in os.listdir(root) if d.startswith("p") and os.path.isdir(os.path.join(root, d)))


def load_all_participants(root):
    frames = {}
    for pid in list_participants(root):
        try:
            frames[pid] = build_daily_dataframe(os.path.join(root, pid), pid)
        except Exception as e:      # report, never silently drop
            print(f"[preprocess] {pid} FAILED: {e}")
    return frames


def complete_cases(frame):
    """Complete-case daily series of the nine core features (input to Stage 2)."""
    return frame[CORE_VARIABLES].astype(float).dropna()


def summarize(frames):
    return pd.DataFrame([{"participant": p, "total_days": len(f),
                          "complete_9var_days": len(complete_cases(f))} for p, f in frames.items()])


def load_demographics(root):
    path = os.path.join(root, "participant-overview.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, header=None)
    df.columns = df.iloc[0].astype(str).str.strip()
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(lambda col: col.astype(str).str.strip() if col.dtype == object else col)   # (4)
    return df.loc[:, ~df.columns.duplicated()]


def synthetic_frames(n_participants=3, days=150, seed=0):
    """Fake VAR(1) data with known lagged structure + injected anomalies (smoke tests)."""
    rng = np.random.default_rng(seed)
    v, frames = CORE_VARIABLES, {}
    for k in range(n_participants):
        A = np.zeros((9, 9)); np.fill_diagonal(A, 0.3)
        A[v.index("sleep_minutes"), v.index("steps")] = 0.4
        A[v.index("resting_hr"), v.index("sleep_minutes")] = -0.4
        A[v.index("sleep_score"), v.index("sleep_minutes")] = 0.5
        x = np.zeros((days, 9))
        for t in range(1, days):
            x[t] = A @ x[t - 1] + rng.normal(size=9)
        for t in rng.choice(np.arange(20, days), 8, replace=False):
            x[t] += rng.normal(0, 5, size=9)
        f = pd.DataFrame(x * 10 + 50, index=pd.date_range("2019-11-01", periods=days, name="date"), columns=v)
        f["is_missing_day"] = False
        f["participant"] = f"p{k + 1:02d}"
        frames[f"p{k + 1:02d}"] = f
    return frames
