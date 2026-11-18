"""
Stage 1 - personalised anomaly detection (paper Sec. III-C).

Input : daily series of one participant (DataFrame, columns = features).
Output: boolean flags per day AND per feature (MultiIndex columns: detector, feature).

Ten detectors in four families
    statistical    zscore, ar (Forecast-AR), arima, sarima
    distance       isolation_forest
    forecasting    chronos (pretrained, optional)
    reconstruction autoencoder, lstm (forecaster), lstm_ae, tranad (simplified)

Classical detectors score day t from the preceding window only (refit every day):
14 days (Z-score, IForest, AR, ARIMA, Chronos) and 30 days (SARIMA, weekly seasonality).
Neural detectors train once on the first 50% of the history and score the rest.
SARIMA is the pipeline default. No LLM is used here.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ classical
def zscore_detector(s, window=C.WINDOW, thr=C.Z_THRESHOLD):
    mu = s.rolling(window).mean().shift(1)
    sd = s.rolling(window).std().shift(1).replace(0, np.nan)
    return ((s - mu) / sd).abs() > thr


def ar_detector(s, window=C.WINDOW, thr=C.Z_THRESHOLD):
    """AR(1) refit on the trailing window; flag large one-step forecast residuals."""
    x, flags = s.values, np.zeros(len(s), bool)
    for t in range(window, len(x)):
        w = x[t - window:t]
        if np.isnan(w).any() or np.isnan(x[t]) or w[:-1].std() == 0:
            continue
        slope, icpt = np.polyfit(w[:-1], w[1:], 1)
        sd = (w[1:] - (slope * w[:-1] + icpt)).std()
        if sd > 0:
            flags[t] = abs(x[t] - (slope * w[-1] + icpt)) / sd > thr
    return pd.Series(flags, index=s.index)


def arima_detector(s, window=C.WINDOW, thr=C.Z_THRESHOLD, seasonal=False):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    w_len = C.SARIMA_WINDOW if seasonal else window
    x, flags = s.values, np.zeros(len(s), bool)
    for t in range(w_len, len(x)):
        w = x[t - w_len:t]
        if np.isnan(w).any() or np.isnan(x[t]) or np.std(w) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if seasonal:
                    m = SARIMAX(w, order=(1, 0, 0), seasonal_order=(1, 0, 0, 7)).fit(disp=False)
                else:
                    m = ARIMA(w, order=(1, 0, 1)).fit()
            sd = np.std(m.resid)
            if sd > 0:
                flags[t] = abs(x[t] - float(np.asarray(m.forecast(1))[0])) / sd > thr
        except Exception:
            continue
    return pd.Series(flags, index=s.index)


def isolation_forest_detector(s, window=C.WINDOW):
    x, flags = s.values.reshape(-1, 1), np.zeros(len(s), bool)
    for t in range(window, len(x)):
        w = x[t - window:t]
        if np.isnan(w).any() or np.isnan(x[t]).any():
            continue
        m = IsolationForest(n_estimators=100, contamination=C.IF_CONTAMINATION, random_state=C.SEED).fit(w)
        flags[t] = m.predict(x[t:t + 1])[0] == -1
    return pd.Series(flags, index=s.index)


def chronos_detector(s, window=C.WINDOW, thr=C.Z_THRESHOLD):
    """Zero-shot Chronos forecast; needs `pip install chronos-forecasting torch` (+ model download)."""
    import torch
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
    x, flags = s.values, np.zeros(len(s), bool)
    for t in range(window, len(x)):
        ctx = x[t - window:t]
        if np.isnan(ctx).any() or np.isnan(x[t]):
            continue
        smp = pipe.predict(torch.tensor(ctx, dtype=torch.float32), 1, num_samples=20)[0].numpy()
        if smp.std() > 0:
            flags[t] = abs(x[t] - smp.mean()) / smp.std() > thr
    return pd.Series(flags, index=s.index)


# --------------------------------------------------------------------- neural
SEQ_LEN, HIDDEN, LATENT, D_MODEL, N_HEAD, EPOCHS, LR = 7, 16, 4, 16, 2, 80, 1e-2


def _windows(X, L):
    return np.stack([X[i - L:i] for i in range(L, len(X) + 1)])


def _fit(model, data, loss_fn):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    data = torch.tensor(data)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model, data).backward()
        opt.step()
    model.eval()
    return model


def _neural_flags(s, build):
    """Train once on the first TRAIN_FRAC of the history (standardised with TRAIN statistics) and flag
    only the remaining days. The detector is fitted on the first 80% of that training segment; the
    threshold (95th percentile of anomaly scores) is calibrated on the held-out last 20%, because
    scores on data the network was fitted to are too optimistic and would flag most of the test period."""
    import torch
    torch.manual_seed(C.SEED)
    x = s.interpolate(limit_area="inside").values.astype("float64")
    split = int(len(x) * C.TRAIN_FRAC)
    fit_end = int(split * 0.8)
    flags = pd.Series(False, index=s.index)
    if fit_end <= SEQ_LEN + 5 or np.isnan(x[:split]).all():
        return flags
    mu, sd = np.nanmean(x[:split]), np.nanstd(x[:split]) or 1.0
    X = np.nan_to_num((x - mu) / sd).reshape(-1, 1).astype("float32")
    scorer = build(X[:fit_end], SEQ_LEN)
    sc = scorer(X)
    calib = sc[fit_end:split]
    thr = np.nanpercentile(calib[~np.isnan(calib)], 95)
    flags.iloc[split:] = sc[split:] > thr
    return flags


def lstm_detector(s):
    """LSTM forecaster: predict day t from the previous SEQ_LEN days; score = squared error."""
    import torch, torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__(); self.rnn = nn.LSTM(1, HIDDEN, batch_first=True); self.out = nn.Linear(HIDDEN, 1)
        def forward(self, x):
            h, _ = self.rnn(x); return self.out(h[:, -1])

    def build(Xtr, L):
        m = _fit(M(), _windows(Xtr, L + 1), lambda mm, w: ((mm(w[:, :-1]) - w[:, -1]) ** 2).mean())
        def scorer(X):
            W = torch.tensor(_windows(X, L + 1))
            with torch.no_grad():
                err = ((m(W[:, :-1]) - W[:, -1]) ** 2).mean(1).numpy()
            return np.concatenate([np.full(L, np.nan), err])
        return scorer
    return _neural_flags(s, build)


def autoencoder_detector(s):
    """Dense AE over short windows of the series; score = reconstruction error of the window ending at t."""
    import torch, torch.nn as nn

    def build(Xtr, L):
        m = nn.Sequential(nn.Linear(L, 8), nn.ReLU(), nn.Linear(8, LATENT), nn.ReLU(),
                          nn.Linear(LATENT, 8), nn.ReLU(), nn.Linear(8, L))
        _fit(m, _windows(Xtr, L)[:, :, 0], lambda mm, w: ((mm(w) - w) ** 2).mean())
        def scorer(X):
            W = torch.tensor(_windows(X, L)[:, :, 0])
            with torch.no_grad():
                err = ((m(W) - W) ** 2).mean(1).numpy()
            return np.concatenate([np.full(L - 1, np.nan), err])
        return scorer
    return _neural_flags(s, build)


def lstm_ae_detector(s):
    import torch, torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(1, HIDDEN, batch_first=True)
            self.dec = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
            self.out = nn.Linear(HIDDEN, 1)
        def forward(self, x):
            _, (h, _) = self.enc(x)
            y, _ = self.dec(h[-1].unsqueeze(1).repeat(1, x.shape[1], 1))
            return self.out(y)

    def build(Xtr, L):
        m = _fit(M(), _windows(Xtr, L), lambda mm, w: ((mm(w) - w) ** 2).mean())
        def scorer(X):
            W = torch.tensor(_windows(X, L))
            with torch.no_grad():
                err = ((m(W) - W) ** 2).mean((1, 2)).numpy()
            return np.concatenate([np.full(L - 1, np.nan), err])
        return scorer
    return _neural_flags(s, build)


def tranad_detector(s):
    """Simplified TranAD: transformer encoder reconstructs a window; a second pass is conditioned on
    the first-pass error (focus score). Score = second-pass reconstruction error."""
    import torch, torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(2, D_MODEL)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                D_MODEL, N_HEAD, dim_feedforward=32, dropout=0.0, batch_first=True), 1)
            self.out = nn.Linear(D_MODEL, 1)
        def _pass(self, x, focus):
            return self.out(self.enc(self.inp(torch.cat([x, focus], -1))))
        def forward(self, x):
            o1 = self._pass(x, torch.zeros_like(x))
            return o1, self._pass(x, (o1 - x) ** 2)

    def build(Xtr, L):
        def loss(mm, w):
            o1, o2 = mm(w)
            return 0.5 * ((o1 - w) ** 2).mean() + 0.5 * ((o2 - w) ** 2).mean()
        m = _fit(M(), _windows(Xtr, L), loss)
        def scorer(X):
            W = torch.tensor(_windows(X, L))
            with torch.no_grad():
                err = ((m(W)[1] - W) ** 2).mean((1, 2)).numpy()
            return np.concatenate([np.full(L - 1, np.nan), err])
        return scorer
    return _neural_flags(s, build)


DETECTORS = {
    "zscore": zscore_detector,
    "sarima": lambda s: arima_detector(s, seasonal=True),
    "ar": ar_detector,
    "arima": lambda s: arima_detector(s, seasonal=False),
    "chronos": chronos_detector,
    "lstm": lstm_detector,
    "isolation_forest": isolation_forest_detector,
    "lstm_ae": lstm_ae_detector,
    "autoencoder": autoencoder_detector,
    "tranad": tranad_detector,
}
FAST = ("zscore", "ar", "isolation_forest")   # cheap subset for smoke tests


def _prepare(frame, features):
    df = frame[features].astype(float)
    return df.interpolate(limit=2, limit_area="inside")   # bridge tiny gaps only; long gaps stay NaN


def run_stage1(frame, features=C.CORE_VARIABLES, detectors=None, verbose=True):
    """Returns DataFrame (days x (detector, feature)) of boolean anomaly flags."""
    df, out = _prepare(frame, features), {}
    for name in (detectors or [C.DEFAULT_DETECTOR]):
        fn = DETECTORS[name]
        for f in features:
            try:
                out[(name, f)] = fn(df[f]).reindex(df.index).fillna(False).astype(bool)
            except Exception as e:
                if verbose:
                    print(f"  [stage1] {name}/{f} FAILED: {e}")
    res = pd.DataFrame(out)
    res.columns = pd.MultiIndex.from_tuples(res.columns, names=["detector", "feature"])
    return res


def anomaly_events(flags, detector=C.DEFAULT_DETECTOR):
    """Flat list of (feature, date) anomalous events from the chosen detector (Algorithm 1, lines 1-5)."""
    sub = flags[detector]
    return [(f, d) for f in sub.columns for d in sub.index[sub[f]]]
