"""
AERIS - Aerospace Equipment & Reliability Intelligence System
==============================================================

The AI model behind the Spacecraft Health Monitoring System (SHMS).

AERIS is trained on the NASA SMAP & MSL telemetry anomaly dataset and works in
two stages, both from the CBSE AI syllabus (Class XI & XII):

  Stage 1 - Forecaster (Linear Regression)
      Learns how *normal* telemetry behaves by predicting the next reading from
      the previous readings. Where the real reading is far away from the
      forecast, something unusual is happening.
      Evaluated with MSE, RMSE, MAE and MAPE.

  Stage 2 - Anomaly Classifier (best of Logistic Regression / Decision Tree /
      Random Forest, chosen with Group K-Fold cross validation)
      Looks at statistical features of the signal + the forecast errors and
      decides, for every time step, whether it is an anomaly.
      Evaluated with Accuracy, Precision, Recall and F1 score on spacecraft
      channels it has never seen (train-test split by channel).

Usage
-----
    python aeris.py             # trains AERIS once (skips if already trained)
    python aeris.py --retrain   # forces a brand new training run

After training, the platform simply does:

    from aeris import AERIS
    model = AERIS.load()
    result = model.analyze(signal, chan_id="A-1")
"""

import argparse
import ast
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- #
# Paths & settings
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
TRAIN_DIR = DATASET_DIR / "data" / "train"
TEST_DIR = DATASET_DIR / "data" / "test"
LABELS_CSV = DATASET_DIR / "labeled_anomalies.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "aeris.joblib"
METRICS_PATH = MODEL_DIR / "aeris_metrics.json"
REPORT_DIR = MODEL_DIR / "reports"

MODEL_NAME = "AERIS"
MODEL_FULL_NAME = "Aerospace Equipment & Reliability Intelligence System"
MODEL_VERSION = "2.0"

# Hyperparameters (values we choose before training - Class XII, Model Lifecycle)
CONFIG = {
    "lags": 30,               # past readings the forecaster looks at
    "short_window": 50,       # rolling window for short-term statistics
    "long_window": 250,       # rolling window for long-term statistics
    "very_long_window": 1000, # rolling window for the overall trend
    "holdout_fraction": 0.2,  # 20% of channels kept aside as the test set
    "cv_folds": 5,            # K in K-fold cross validation
    "cv_stride": 4,           # use every 4th row while comparing models (speed)
    "train_stride": 2,        # use every 2nd row for the final model
    "synthetic_channels": 72, # augmentation signals added to training only
    "brain_lags": 60,         # readings each sensor's mini brain looks at
    "brain_smoothing": 105,   # smoothing span for the mini brain's surprise
    "expert_folds": 5,        # cross-fitted expert models (see train())
    "min_event_length": 5,    # ignore anomaly blips shorter than this
    "merge_gap": 20,          # join anomaly events closer than this
    "random_state": 42,
}

FEATURE_NAMES = [
    "value", "roll_mean", "roll_std", "roll_min", "roll_max", "zscore",
    "diff", "abs_diff_mean", "long_mean_dev", "long_std", "file_median_dev",
    "forecast_error", "error_smooth", "error_smooth_long", "error_max",
    "value_rarity", "mean_rarity", "std_rarity", "very_long_mean_dev",
    "very_long_std", "robust_zscore",
    # compared with the sensor's own "normal behaviour" baseline
    "base_rarity", "base_mean_rarity", "base_out_of_range", "base_median_dev",
    "base_std_ratio", "base_std_exceeded",
]


def log(msg=""):
    """Plain ASCII printing so it works in every Windows terminal."""
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Signal processing helpers (shared by training and analysis)
# --------------------------------------------------------------------------- #
def clean_signal(x):
    """Turn any 1-D input into a clean float array (NaNs filled)."""
    s = pd.Series(np.asarray(x, dtype=float).ravel())
    s = s.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    return s.fillna(0.0).to_numpy()


def scale_signal(x):
    """Min-max scale a signal to [-1, 1] (same scaling NASA used)."""
    x = clean_signal(x)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def make_lag_matrix(x, lags):
    """Rows of [x(t-lags) ... x(t-1)] with target x(t)."""
    if len(x) <= lags:
        return np.empty((0, lags)), np.empty(0)
    windows = sliding_window_view(x, lags + 1)
    return windows[:, :-1], windows[:, -1]


def forecast_signal(regressor, x, lags):
    """One-step-ahead forecast for every point (first `lags` points = itself)."""
    forecast = x.copy()
    X, _ = make_lag_matrix(x, lags)
    if len(X):
        forecast[lags:] = regressor.predict(X)
    return forecast


def rarity(values, bins=40):
    """How rare each value is inside this file (higher = rarer).

    Uses a histogram: -log(fraction of readings that fall in the same bin).
    """
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    return -np.log((counts[idx] + 1) / len(values))


def default_baseline(raw):
    """When we don't know a sensor's normal history, use the first 30% of the file."""
    return raw[:max(CONFIG["lags"], int(len(raw) * 0.3))]


def baseline_features(raw, base, w):
    """Compare readings with the sensor's normal behaviour (its baseline).

    NASA gives a 'normal' training recording for every channel; for any other
    file the first 30% of the file is used as the normal reference.
    """
    base = clean_signal(base)
    if len(base) < 2:
        base = raw
    lo, hi = base.min(), base.max()
    span = hi - lo + 1e-6
    edges = np.linspace(lo, hi, 41)
    counts, _ = np.histogram(base, bins=edges)

    def base_rarity(v):
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, 39)
        r = -np.log((counts[idx] + 1) / len(base))
        outside = (v < lo - 0.01 * span) | (v > hi + 0.01 * span)
        r[outside] = np.log(len(base) + 1)   # never seen in normal operation
        return r

    s = pd.Series(raw)
    mean = s.rolling(w, min_periods=1).mean().to_numpy()
    std = s.rolling(w, min_periods=1).std().fillna(0.0).to_numpy()
    base_std = pd.Series(base).rolling(w, min_periods=1).std().fillna(0.0).to_numpy()
    return {
        "base_rarity": base_rarity(raw),
        "base_mean_rarity": base_rarity(mean),
        "base_out_of_range": np.maximum(0.0, np.maximum(lo - raw, raw - hi)) / span,
        "base_median_dev": np.abs(raw - np.median(base)) / span,
        "base_std_ratio": np.clip(std / (np.median(base_std) + 1e-6), 0, 1000),
        "base_std_exceeded": (std > base_std.max() * 1.2 + 1e-9).astype(float),
    }


def build_features(x, forecast, raw, baseline, cfg=CONFIG):
    """Statistical features for every time step (past-looking windows).

    x        - signal scaled to [-1, 1]
    forecast - forecaster output for x
    raw      - the original (unscaled) readings
    baseline - normal-behaviour readings of the same sensor (same units as raw)
    """
    w, lw, vlw = cfg["short_window"], cfg["long_window"], cfg["very_long_window"]
    s = pd.Series(x)
    err = pd.Series(np.abs(x - forecast))
    roll = s.rolling(w, min_periods=1)
    mean, std = roll.mean(), roll.std().fillna(0.0)
    diff = s.diff().fillna(0.0)
    feats = pd.DataFrame({
        "value": s,
        "roll_mean": mean,
        "roll_std": std,
        "roll_min": roll.min(),
        "roll_max": roll.max(),
        "zscore": ((s - mean) / (std + 1e-6)).clip(-50, 50),
        "diff": diff,
        "abs_diff_mean": diff.abs().rolling(w, min_periods=1).mean(),
        "long_mean_dev": s - s.rolling(lw, min_periods=1).mean(),
        "long_std": s.rolling(lw, min_periods=1).std().fillna(0.0),
        "file_median_dev": s - s.median(),
        "forecast_error": err,
        "error_smooth": err.rolling(w, min_periods=1).mean(),
        "error_smooth_long": err.rolling(lw, min_periods=1).mean(),
        "error_max": err.rolling(w, min_periods=1).max(),
        "value_rarity": rarity(x),
        "mean_rarity": rarity(mean),
        "std_rarity": rarity(std),
        "very_long_mean_dev": s - s.rolling(vlw, min_periods=1).mean(),
        "very_long_std": s.rolling(vlw, min_periods=1).std().fillna(0.0),
        "robust_zscore": (s - s.median()).abs() / ((s - s.median()).abs().median() + 1e-3),
        **baseline_features(clean_signal(raw), baseline, w),
    })
    return feats[FEATURE_NAMES].to_numpy(dtype=np.float32)


def find_events(flags):
    """List of (start, end) index pairs (inclusive) where flags are True."""
    padded = np.concatenate([[0], flags.astype(np.int8), [0]])
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def postprocess(prob, threshold, smooth, cfg=CONFIG):
    """Smooth probabilities, apply threshold, merge close events, drop blips."""
    p = np.asarray(prob, dtype=float)
    if smooth > 1:
        p = pd.Series(p).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    events = find_events(p >= threshold)

    merged = []
    for start, end in events:
        if merged and start - merged[-1][1] <= cfg["merge_gap"]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    events = [(s, e) for s, e in merged if e - s + 1 >= cfg["min_event_length"]]

    flags = np.zeros(len(p), dtype=bool)
    for s, e in events:
        flags[s:e + 1] = True
    return p, flags, events


def labels_to_array(sequences, n):
    y = np.zeros(n, dtype=np.int8)
    for start, end in sequences:
        y[max(0, start):min(n, end + 1)] = 1
    return y


def classification_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def regression_metrics(actual, predicted):
    """MSE, RMSE, MAE and MAPE of the forecaster.

    MAPE divides by the actual value, so readings very close to zero would make
    it explode; we shift the [-1, 1] signal to [1, 3] before computing it.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if len(actual) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0}
    mse = mean_squared_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 2.0))) * 100
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(mape),
    }


def event_detection(true_sequences, flags):
    """Event-level view: real events caught, and alarms that were real.

    events_detected - real anomaly events that got at least one alarm
    alarms_correct  - AERIS alarms that overlap a real anomaly event
    """
    hit = sum(1 for s, e in true_sequences if flags[max(0, s):e + 1].any())
    truth = labels_to_array(true_sequences, len(flags))
    alarms = find_events(np.asarray(flags, dtype=bool))
    correct = sum(1 for s, e in alarms if truth[s:e + 1].any())
    return {"events_total": len(true_sequences), "events_detected": hit,
            "alarms_total": len(alarms), "alarms_correct": correct}


def event_scores(counts):
    """Event recall / precision / F1 from event_detection() counts."""
    recall = counts["events_detected"] / counts["events_total"] if counts["events_total"] else 0.0
    precision = counts["alarms_correct"] / counts["alarms_total"] if counts["alarms_total"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"event_recall": recall, "event_precision": precision, "event_f1": f1}


# --------------------------------------------------------------------------- #
# Sensor mini brains (one small neural network per NASA sensor)
# --------------------------------------------------------------------------- #
EXPERT_FEATURE_NAMES = FEATURE_NAMES + ["brain_surprise", "brain_surprise_max", "brain_surprise_mean"]


def train_brain(normal, cfg=CONFIG):
    """Teach one sensor's mini brain how that sensor normally behaves.

    It is a small neural network (MLP) that predicts the next reading from the
    previous ones, trained only on NASA's normal recording of that sensor.
    It never sees the anomaly labels.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    lags, span = cfg["brain_lags"], cfg["brain_smoothing"]
    X, y = make_lag_matrix(clean_signal(normal), lags)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                           random_state=cfg["random_state"]).fit(X, y)
    err = pd.Series(np.abs(y - net.predict(X))).ewm(span=span).mean().to_numpy()[span:]
    return {"net": net, "mean": float(err.mean()), "std": float(err.std() + 1e-6)}


def brain_surprise(brain, raw, cfg=CONFIG):
    """How surprised the mini brain is by every reading, in units of its normal error."""
    lags, span = cfg["brain_lags"], cfg["brain_smoothing"]
    raw = clean_signal(raw)
    z = np.zeros(len(raw))
    X, y = make_lag_matrix(raw, lags)
    if len(X):
        err = pd.Series(np.abs(y - brain["net"].predict(X))).ewm(span=span).mean().to_numpy()
        z[lags:] = (err - brain["mean"]) / brain["std"]
    z[:lags + 30] = 0.0   # not enough history yet at the very start
    return np.clip(z, -5, 100)


def expert_features(features, z):
    s = pd.Series(z)
    return np.column_stack([features, z, s.rolling(250, min_periods=1).max(),
                            s.rolling(50, center=True, min_periods=1).mean()]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The trained model (used by the platform)
# --------------------------------------------------------------------------- #
class AERIS:
    """Trained AERIS model: load it once, then call analyze() on any signal."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.regressor = bundle["regressor"]
        self.classifier = bundle["classifier"]
        self.cfg = bundle["config"]
        self.threshold = bundle["threshold"]
        self.smooth = bundle["smooth"]
        self.labels = bundle["labels"]
        self.metrics = bundle["metrics"]
        self.brains = bundle["brains"]
        self.experts = bundle["experts"]
        self.expert_fold = bundle["expert_fold"]
        self._fingerprints = None

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained AERIS model not found at {path}. Run: python aeris.py")
        bundle = joblib.load(path)
        if bundle.get("feature_names") != FEATURE_NAMES or "experts" not in bundle:
            raise RuntimeError(
                "The saved AERIS model is from an older version. Run: python aeris.py --retrain")
        return cls(bundle)

    def find_channel(self, name):
        """Match a file name like 'A-1.npy' or 'a-1_test.csv' to a NASA channel id."""
        if not name:
            return None
        stem = Path(str(name)).stem.upper().replace("_", "-").replace(" ", "")
        if stem in self.labels:
            return stem
        for chan in sorted(self.labels, key=len, reverse=True):
            if stem.startswith(chan + "-") or stem == chan:
                return chan
        return None

    def identify(self, values):
        """Recognise a NASA file by its readings, even if it was renamed.

        Returns (channel, part): part is "test" (has NASA expert labels) or
        "train" (the channel's normal recording, which has no labels).
        """
        if self._fingerprints is None:
            self._fingerprints = {}
            for part, folder in (("test", TEST_DIR), ("train", TRAIN_DIR)):
                for f in folder.glob("*.npy") if folder.exists() else []:
                    if f.stem in self.labels:
                        ref = np.load(f)[:, 0]
                        self._fingerprints.setdefault(len(ref), []).append((f.stem, part, ref[:500]))
        values = np.asarray(values, dtype=float).ravel()
        for chan, part, ref in self._fingerprints.get(len(values), []):
            head = values[:len(ref)]
            if np.all(np.isfinite(head)) and np.allclose(head, ref, atol=1e-4, rtol=0):
                return chan, part
        return None, None

    def baseline_for(self, chan_id):
        """NASA's normal-behaviour recording for a known channel (or None)."""
        if not chan_id:
            return None
        path = TRAIN_DIR / f"{chan_id}.npy"
        if not path.exists():
            return None
        return np.load(path)[:, 0]

    @property
    def info(self):
        b = self.bundle
        return {
            "name": MODEL_NAME, "full_name": MODEL_FULL_NAME,
            "version": b["version"], "trained_at": b["trained_at"],
            "classifier": b["classifier_name"], "forecaster": "Linear Regression",
            "threshold": self.threshold, "smooth": self.smooth,
            "train_channels": b["train_channels"],
            "holdout_channels": b["holdout_channels"],
        }

    def ground_truth(self, chan_id, n):
        """NASA expert labels for a known dataset channel (or None)."""
        if not chan_id or chan_id not in self.labels:
            return None, []
        seqs = self.labels[chan_id]["sequences"]
        return labels_to_array(seqs, n), seqs

    def analyze(self, signal, chan_id=None, baseline=None, labelled=True):
        """Run AERIS on one telemetry signal (1-D array of readings).

        chan_id  - NASA channel id (e.g. "A-1") if known: enables expert-label
                   evaluation and the channel's own normal baseline
        baseline - optional normal-behaviour readings of the same sensor

        Returns a dict with:
          series  - numpy arrays for plotting (scaled, forecast, error, probability, flags)
          summary - JSON-friendly result (status, health score, events, metrics)
        """
        raw = clean_signal(signal)
        if len(raw) < 2:
            raise ValueError("The telemetry signal needs at least 2 readings.")
        if baseline is None:
            baseline = self.baseline_for(chan_id)
        baseline_source = "nasa_normal_history" if baseline is not None else "first_30_percent"
        if baseline is None:
            baseline = default_baseline(raw)
        x = scale_signal(raw)
        lags = self.cfg["lags"]
        forecast = forecast_signal(self.regressor, x, lags)
        features = build_features(x, forecast, raw, baseline, self.cfg)

        # Known NASA sensor: use its own mini brain + the expert model that was
        # trained WITHOUT this sensor's answers (so the evaluation is a fair test).
        expert = chan_id in self.brains and chan_id in self.expert_fold and baseline_source == "nasa_normal_history"
        if expert:
            z = brain_surprise(self.brains[chan_id], raw, self.cfg)
            clf = self.experts[self.expert_fold[chan_id]]
            prob_raw = clf.predict_proba(expert_features(features, z))[:, 1]
        else:
            prob_raw = self.classifier.predict_proba(features)[:, 1]
        prob, flags, events = postprocess(prob_raw, self.threshold, self.smooth, self.cfg)

        n = len(x)
        anomaly_ratio = float(flags.mean())
        peak = float(prob.max())
        event_list = [{
            "start": int(s), "end": int(e), "length": int(e - s + 1),
            "peak_probability": float(prob[s:e + 1].max()),
        } for s, e in events]

        if not events:
            status = "HEALTHY"
        elif anomaly_ratio >= 0.10 or peak >= 0.90:
            status = "DANGER"
        else:
            status = "WARNING"
        # Simple, explainable health score: lose points for time spent in an
        # anomaly and for how confident the strongest alarm is.
        health = 100.0 - anomaly_ratio * 300.0 - (peak * 30.0 if events else 0.0)
        health = float(np.clip(health, 0.0, 100.0))

        start = min(lags, n)
        summary = {
            "chan_id": chan_id,
            "mode": "expert" if expert else "general",
            "baseline_source": baseline_source,
            "num_readings": n,
            "status": status,
            "health_score": round(health, 1),
            "anomaly_ratio": anomaly_ratio,
            "anomalous_readings": int(flags.sum()),
            "peak_probability": peak,
            "mean_probability": float(prob.mean()),
            "num_events": len(events),
            "events": event_list,
            "signal_stats": {
                "min": float(raw.min()), "max": float(raw.max()),
                "mean": float(raw.mean()), "std": float(raw.std()),
            },
            "forecast_metrics": regression_metrics(x[start:], forecast[start:]),
            "evaluation": None,
            "model_validation": {
                "channels": len(self.bundle["holdout_channels"]),
                **self.metrics["expert_validation" if expert else "holdout"],
            },
        }

        y_true, seqs = self.ground_truth(chan_id, n) if labelled else (None, [])
        if y_true is not None:
            ev = classification_metrics(y_true, flags)
            ev.update(event_detection(seqs, flags))
            ev.update(event_scores(ev))
            ev["brier_mse"] = float(np.mean((prob - y_true) ** 2))
            # in expert mode the model used never saw this sensor's answers
            ev["seen_in_training"] = (not expert) and chan_id in self.bundle["train_channels"]
            ev["true_events"] = [[int(s), int(e)] for s, e in seqs]
            summary["evaluation"] = ev

        series = {
            "scaled": x, "forecast": forecast, "error": np.abs(x - forecast),
            "probability": prob, "flags": flags,
            "truth": y_true,
        }
        return {"series": series, "summary": summary}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def load_labels():
    df = pd.read_csv(LABELS_CSV)
    labels = {}
    for _, row in df.iterrows():
        entry = labels.setdefault(row["chan_id"], {
            "spacecraft": row["spacecraft"], "sequences": [], "classes": []})
        entry["sequences"] += [list(map(int, s)) for s in ast.literal_eval(row["anomaly_sequences"])]
        entry["classes"] += [c.strip() for c in row["class"].strip("[]").split(",")]
    return labels


def synthetic_channels(n, rs, length=3000):
    """Data augmentation: extra sensor-like signals so AERIS also understands
    smooth periodic data (e.g. temperature cycles), not only NASA's step-like
    channels. Half are healthy, half get an injected fault (spike, level shift
    or frozen sensor) with a matching label. Used for training only."""
    rng = np.random.default_rng(rs)
    t = np.arange(length)
    out = []
    kinds = 6
    for i in range(n):
        kind = i % kinds
        if kind == 0:    # orbit-like cycle
            x = np.sin(2 * np.pi * t / rng.uniform(80, 600))
        elif kind == 1:  # two cycles mixed
            x = np.sin(2 * np.pi * t / rng.uniform(100, 400)) + 0.5 * np.sin(2 * np.pi * t / rng.uniform(20, 90))
        elif kind == 2:  # slow drifting random walk
            x = pd.Series(np.cumsum(rng.normal(0, 1, length))).rolling(25, min_periods=1).mean().to_numpy()
        elif kind == 3:  # periodic on/off steps (heaters, valves)
            x = np.sign(np.sin(2 * np.pi * t / rng.uniform(150, 700)))
        elif kind == 4:  # steady sensor with measurement noise (e.g. pressure)
            x = rng.normal(0, 1, length)
        else:            # noisy sensor with a slow drift
            x = rng.normal(0, 1, length) + np.linspace(0, rng.uniform(-2, 2), length)
        x = x + rng.normal(0, rng.uniform(0.0, 0.15) * (np.std(x) + 1e-6), length)
        y = np.zeros(length, dtype=np.int8)
        if (i // kinds) % 2 == 1:   # every signal type appears both healthy and faulty
            start = int(rng.integers(length // 3, length - 300))
            size = int(rng.integers(30, 250))
            fault, amp = rng.integers(3), np.ptp(x) + 1e-6
            if fault == 0:
                x[start:start + size] += rng.choice([-1, 1]) * amp * rng.uniform(0.8, 2.0)
            elif fault == 1:
                x[start:] += rng.choice([-1, 1]) * amp * rng.uniform(0.6, 1.5)
                size = min(size, length - start)
            else:
                x[start:start + size] = x[start] + rng.normal(0, 1e-4, size) + amp * 0.5
            y[start:start + size] = 1
        out.append((x, y))
    return out


def candidate_models(rs):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50, class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=rs),
    }


def stack(channels, data, stride):
    X = np.concatenate([data[c]["X"][::stride] for c in channels])
    y = np.concatenate([data[c]["y"][::stride] for c in channels])
    return X, y


def tune_postprocessing(oof_prob, data, channels):
    """Pick smoothing window + threshold that give the best F1 (on CV data only)."""
    best = (-1.0, 0.5, 1)
    y_all = np.concatenate([data[c]["y"] for c in channels])
    for smooth in (1, 25, 75, 151):
        smoothed = {c: pd.Series(oof_prob[c]).rolling(smooth, center=True, min_periods=1)
                    .mean().to_numpy() for c in channels}
        for threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
            pred = np.concatenate([postprocess(smoothed[c], threshold, 1)[1] for c in channels])
            f1 = f1_score(y_all, pred, zero_division=0)
            if f1 > best[0]:
                best = (f1, float(threshold), smooth)
    return best


def save_charts(comparison, cm, importances):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not installed - skipping report charts.")
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    purple, ink, muted = "#7C3AED", "#1F1B2E", "#6B6880"

    def style(ax, title):
        ax.set_title(title, loc="left", fontsize=12, color=ink, pad=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#D4D2DC")
        ax.tick_params(colors=muted)

    # 1. Model comparison (cross-validated F1)
    names = list(comparison)
    f1s = [comparison[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(names, f1s, color=purple, height=0.5)
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=ink, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cross-validated F1 score", color=muted)
    style(ax, "AERIS - model comparison (5-fold Group CV)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix on the hold-out channels
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(grid, cmap="Purples")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > grid.max() / 2 else ink, fontsize=11)
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    ax.set_xlabel("Predicted", color=muted)
    ax.set_ylabel("Actual", color=muted)
    style(ax, "Confusion matrix (unseen channels)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. Feature importance (only tree models have it)
    if importances is not None:
        order = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(FEATURE_NAMES)[order], importances[order], color=purple, height=0.6)
        ax.set_xlabel("Importance", color=muted)
        style(ax, "What AERIS looks at most")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)


def train():
    t0 = time.time()
    cfg, rs = CONFIG, CONFIG["random_state"]
    log("=" * 64)
    log(f" Training {MODEL_NAME} - {MODEL_FULL_NAME}")
    log("=" * 64)

    if not TRAIN_DIR.exists() or not TEST_DIR.exists() or not LABELS_CSV.exists():
        sys.exit(f"Dataset not found. Expected {TRAIN_DIR}, {TEST_DIR} and {LABELS_CSV}")

    # ---- Step 1: data collection ------------------------------------------
    labels = load_labels()
    log(f"[1/6] Loaded labels for {len(labels)} channels "
        f"({sum(len(v['sequences']) for v in labels.values())} anomaly events)")

    # ---- Step 2: forecaster on normal (training) telemetry ----------------
    X_parts, y_parts = [], []
    for f in sorted(TRAIN_DIR.glob("*.npy")):
        x = scale_signal(np.load(f)[:, 0])
        X, y = make_lag_matrix(x, cfg["lags"])
        X_parts.append(X)
        y_parts.append(y)
    X_reg, y_reg = np.concatenate(X_parts), np.concatenate(y_parts)
    regressor = LinearRegression().fit(X_reg, y_reg)
    train_reg = regression_metrics(y_reg, regressor.predict(X_reg))
    log(f"[2/6] Forecaster (Linear Regression) trained on {len(y_reg):,} readings "
        f"- train RMSE {train_reg['rmse']:.4f}")

    # ---- Step 3: features for every labelled test channel -----------------
    data = {}
    for f in sorted(TEST_DIR.glob("*.npy")):
        chan = f.stem
        if chan not in labels:
            continue  # a channel without expert labels cannot be used
        raw = np.load(f)[:, 0]
        normal = TRAIN_DIR / f.name
        baseline = np.load(normal)[:, 0] if normal.exists() else default_baseline(raw)
        x = scale_signal(raw)
        forecast = forecast_signal(regressor, x, cfg["lags"])
        data[chan] = {
            "x": x, "forecast": forecast,
            "X": build_features(x, forecast, raw, baseline, cfg),
            "y": labels_to_array(labels[chan]["sequences"], len(x)),
        }
    channels = sorted(data)
    total = sum(len(d["y"]) for d in data.values())
    pos = sum(int(d["y"].sum()) for d in data.values())
    log(f"[3/6] Built {len(FEATURE_NAMES)} features for {len(channels)} channels "
        f"({total:,} readings, {pos / total:.1%} anomalous)")

    # ---- Step 4: train-test split by channel + cross validation -----------
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["holdout_fraction"], random_state=rs)
    tr_idx, ho_idx = next(splitter.split(channels, groups=channels))
    train_ch = [channels[i] for i in tr_idx]
    holdout_ch = [channels[i] for i in ho_idx]
    for i, (raw, y) in enumerate(synthetic_channels(cfg["synthetic_channels"], rs)):
        x = scale_signal(raw)
        forecast = forecast_signal(regressor, x, cfg["lags"])
        name = f"SYN-{i + 1}"
        data[name] = {"x": x, "forecast": forecast,
                      "X": build_features(x, forecast, raw, default_baseline(raw), cfg), "y": y}
        train_ch.append(name)
    log(f"[4/6] Split: {len(train_ch)} training channels "
        f"(incl. {cfg['synthetic_channels']} synthetic), {len(holdout_ch)} unseen NASA test channels")
    log(f"      Comparing models with {cfg['cv_folds']}-fold Group cross validation...")

    gkf = GroupKFold(n_splits=cfg["cv_folds"])
    comparison, oof_store = {}, {}
    for name, proto in candidate_models(rs).items():
        tm = time.time()
        oof = {}
        for fit_idx, val_idx in gkf.split(train_ch, groups=train_ch):
            fit_ch = [train_ch[i] for i in fit_idx]
            X_fit, y_fit = stack(fit_ch, data, cfg["cv_stride"])
            model = clone(proto).fit(X_fit, y_fit)
            for i in val_idx:
                c = train_ch[i]
                oof[c] = model.predict_proba(data[c]["X"])[:, 1]
        y_all = np.concatenate([data[c]["y"] for c in train_ch])
        pred_all = np.concatenate([oof[c] >= 0.5 for c in train_ch])
        comparison[name] = classification_metrics(y_all, pred_all)
        oof_store[name] = oof
        m = comparison[name]
        log(f"      - {name:<20} F1 {m['f1']:.3f}  Precision {m['precision']:.3f}  "
            f"Recall {m['recall']:.3f}  ({time.time() - tm:.0f}s)")

    best_name = max(comparison, key=lambda n: comparison[n]["f1"])
    cv_f1, threshold, smooth = tune_postprocessing(oof_store[best_name], data, train_ch)
    log(f"      Best model: {best_name}. Tuned threshold={threshold}, smoothing={smooth} "
        f"(CV F1 {cv_f1:.3f})")

    # ---- Step 5: final model + evaluation on unseen channels --------------
    X_tr, y_tr = stack(train_ch, data, cfg["train_stride"])
    classifier = clone(candidate_models(rs)[best_name]).fit(X_tr, y_tr)
    log(f"[5/6] Final {best_name} trained on {len(y_tr):,} rows. Testing on unseen channels...")

    y_true, y_pred, probs = [], [], []
    seqs_hit = {"events_total": 0, "events_detected": 0, "alarms_total": 0, "alarms_correct": 0}
    actual_fc, pred_fc = [], []
    for c in holdout_ch:
        d = data[c]
        p, flags, _ = postprocess(classifier.predict_proba(d["X"])[:, 1], threshold, smooth)
        y_true.append(d["y"])
        y_pred.append(flags)
        probs.append(p)
        ev = event_detection(labels[c]["sequences"], flags)
        seqs_hit = {k: seqs_hit[k] + ev[k] for k in seqs_hit}
        actual_fc.append(d["x"][cfg["lags"]:])
        pred_fc.append(d["forecast"][cfg["lags"]:])
    y_true, y_pred, probs = map(np.concatenate, (y_true, y_pred, probs))
    holdout = classification_metrics(y_true, y_pred)
    holdout.update(seqs_hit)
    holdout.update(event_scores(seqs_hit))
    holdout["brier_mse"] = float(np.mean((probs - y_true) ** 2))
    holdout["forecast"] = regression_metrics(np.concatenate(actual_fc), np.concatenate(pred_fc))

    # ---- Model lifecycle: Build -> Test -> Retrain -> Deploy ---------------
    # The hold-out scores above are the honest ones. Now that the model is
    # validated, retrain it on ALL channels so the deployed AERIS learns from
    # every anomaly NASA labelled (standard practice before deployment).
    production_ch = train_ch + holdout_ch
    X_all, y_all = stack(production_ch, data, cfg["train_stride"])
    classifier = clone(candidate_models(rs)[best_name]).fit(X_all, y_all)
    nasa_ch = sorted(c for c in production_ch if not c.startswith("SYN-"))
    log(f"      Retrained {best_name} on all {len(nasa_ch)} NASA channels "
        f"({len(y_all):,} rows) for deployment.")

    # ---- Expert mode: a mini brain for every NASA sensor --------------------
    # Each mini brain learns only its sensor's NORMAL recording (no labels).
    # Its "surprise" becomes 3 extra clues for an expert Random Forest. The expert
    # is cross-fitted: 5 copies, each trained without one fifth of the sensors,
    # and every sensor is always judged by the copy that never saw its answers.
    tb = time.time()
    brains = {}
    for c in nasa_ch:
        normal = TRAIN_DIR / f"{c}.npy"
        if normal.exists():
            brains[c] = train_brain(np.load(normal)[:, 0], cfg)
            raw = np.load(TEST_DIR / f"{c}.npy")[:, 0]
            data[c]["Xz"] = expert_features(data[c]["X"], brain_surprise(brains[c], raw, cfg))
    expert_ch = sorted(brains)
    log(f"      Trained {len(brains)} sensor mini brains (neural networks) in {time.time() - tb:.0f}s.")

    experts, expert_fold = [], {}
    oof_flags, y_true_all = {}, []
    for k, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=cfg["expert_folds"]).split(expert_ch, groups=expert_ch)):
        fit_ch = [expert_ch[i] for i in fit_idx]
        Xf = np.concatenate([data[c]["Xz"][::cfg["train_stride"]] for c in fit_ch])
        yf = np.concatenate([data[c]["y"][::cfg["train_stride"]] for c in fit_ch])
        expert = clone(candidate_models(rs)["Random Forest"]).fit(Xf, yf)
        experts.append(expert)
        for i in val_idx:
            c = expert_ch[i]
            expert_fold[c] = k
            oof_flags[c] = postprocess(expert.predict_proba(data[c]["Xz"])[:, 1], threshold, smooth)[1]
    counts = {"events_total": 0, "events_detected": 0, "alarms_total": 0, "alarms_correct": 0}
    for c in expert_ch:
        ev = event_detection(labels[c]["sequences"], oof_flags[c])
        counts = {key: counts[key] + ev[key] for key in counts}
    expert_val = classification_metrics(np.concatenate([data[c]["y"] for c in expert_ch]),
                                        np.concatenate([oof_flags[c] for c in expert_ch]))
    expert_val.update(counts)
    expert_val.update(event_scores(counts))
    expert_val["channels"] = len(expert_ch)
    log(f"      Expert mode on all {len(expert_ch)} NASA sensors (each judged by a model that never saw it): "
        f"F1 {expert_val['f1']:.3f}, {counts['events_detected']} of {counts['events_total']} events.")

    # ---- Step 6: save ------------------------------------------------------
    importances = getattr(classifier, "feature_importances_", None)
    metrics = {
        "model": MODEL_NAME, "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "classifier": best_name, "forecaster": "Linear Regression",
        "threshold": threshold, "smooth": smooth,
        "config": cfg,
        "forecaster_train": train_reg,
        "cross_validation": comparison,
        "cross_validation_tuned_f1": float(cv_f1),
        "holdout": holdout,
        "expert_validation": expert_val,
        "feature_importance": (dict(zip(FEATURE_NAMES, map(float, importances)))
                               if importances is not None else None),
        "train_channels": nasa_ch, "holdout_channels": holdout_ch,
        "training_seconds": round(time.time() - t0, 1),
    }
    bundle = {
        "version": MODEL_VERSION, "trained_at": metrics["trained_at"],
        "regressor": regressor, "classifier": classifier, "classifier_name": best_name,
        "config": cfg, "threshold": threshold, "smooth": smooth,
        "feature_names": FEATURE_NAMES, "labels": labels, "metrics": metrics,
        "train_channels": nasa_ch, "holdout_channels": holdout_ch,
        "brains": brains, "experts": experts, "expert_fold": expert_fold,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    save_charts(comparison, holdout["confusion_matrix"], importances)

    fm = holdout["forecast"]
    log(f"[6/6] Saved model to {MODEL_PATH.relative_to(BASE_DIR)}")
    log("-" * 64)
    log(" AERIS results on spacecraft channels it has NEVER seen")
    log("-" * 64)
    log(f"  Accuracy  : {holdout['accuracy']:.3f}")
    log(f"  Precision : {holdout['precision']:.3f}")
    log(f"  Recall    : {holdout['recall']:.3f}")
    log(f"  F1 score  : {holdout['f1']:.3f}")
    log(f"  Events    : {holdout['events_detected']} of {holdout['events_total']} anomaly events detected, "
        f"{holdout['alarms_correct']} of {holdout['alarms_total']} alarms were real "
        f"(event F1 {holdout['event_f1']:.3f})")
    log(f"  Forecaster: MSE {fm['mse']:.5f} | RMSE {fm['rmse']:.4f} | "
        f"MAE {fm['mae']:.4f} | MAPE {fm['mape']:.2f}%")
    ev = expert_val
    log(f"  Expert mode (NASA sensors, {ev['channels']} channels): Precision {ev['precision']:.3f} | "
        f"Recall {ev['recall']:.3f} | F1 {ev['f1']:.3f} | events {ev['events_detected']}/{ev['events_total']}")
    log("-" * 64)
    log(f" Done in {metrics['training_seconds']:.0f}s. AERIS is trained and ready.")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    parser.add_argument("--retrain", action="store_true", help="train again even if a model exists")
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.retrain:
        model = AERIS.load()
        h = model.metrics["holdout"]
        log(f"{MODEL_NAME} is already trained ({model.info['trained_at']}, "
            f"{model.info['classifier']}).")
        log(f"  F1 {h['f1']:.3f} | Precision {h['precision']:.3f} | Recall {h['recall']:.3f}")
        log("  Use 'python aeris.py --retrain' to train it again.")
        return
    train()


if __name__ == "__main__":
    main()
