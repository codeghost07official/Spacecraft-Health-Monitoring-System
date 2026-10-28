"""
Spacecraft Health Monitoring System (SHMS)
==========================================

Flask web platform around the AERIS model.

    python app.py      ->  open http://127.0.0.1:5000

Settings come from the .env file (see .env.example):
    DATABASE_URL       Neon PostgreSQL connection string
    GEMINI_API_KEY     Google Gemini API key (reports + chatbot)
    FLASK_SECRET_KEY   any random text (keeps users logged in)
    ADMIN_USERNAME_1 / ADMIN_PASSWORD_1, ADMIN_USERNAME_2 / ADMIN_PASSWORD_2
"""

import base64
import hmac
import html
import io
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from markupsafe import Markup

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from aeris import AERIS, MODEL_FULL_NAME, MODEL_PATH, TEST_DIR  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SQLITE_PATH = BASE_DIR / "instance" / "shms_local.db"

ALLOWED_EXTENSIONS = {".npy", ".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json"}
MAX_SENSORS = 6
MAX_ROWS = 200_000
MIN_ROWS = 20

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex()
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads


def env(name):
    return (os.getenv(name) or "").strip()


# --------------------------------------------------------------------------- #
# Users (pre-created in .env, no registration)
# --------------------------------------------------------------------------- #
def configured_users():
    users = {}
    for i in (1, 2, 3):
        name, password = env(f"ADMIN_USERNAME_{i}"), env(f"ADMIN_PASSWORD_{i}")
        if name and password:
            users[name] = password
    return users


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please log in again."}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------- #
# Database: Neon PostgreSQL (falls back to a local SQLite file if not set)
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    health_score REAL NOT NULL,
    sensors INTEGER NOT NULL,
    readings INTEGER NOT NULL,
    summary TEXT NOT NULL,
    charts TEXT NOT NULL,
    report TEXT NOT NULL,
    short_summary TEXT NOT NULL,
    report_source TEXT NOT NULL,
    csv_path TEXT
)
"""


class Database:
    def __init__(self, url):
        self.url = url
        self.kind = "sqlite"
        self.error = None
        if url:
            try:
                import psycopg2
                self._pg = psycopg2
                self.kind = "neon"
                self.run(SCHEMA)
                return
            except Exception as exc:  # keep the platform usable for the demo
                self.kind = "sqlite"
                self.error = str(exc).strip().splitlines()[0][:200]
                print(f"[SHMS] Could not connect to Neon ({self.error}). Using local SQLite instead.")
        SQLITE_PATH.parent.mkdir(exist_ok=True)
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.execute(SCHEMA)

    @property
    def label(self):
        return "Neon PostgreSQL" if self.kind == "neon" else "Local storage (Neon not connected)"

    def run(self, sql, params=(), fetch=None):
        if self.kind == "neon":
            sql = sql.replace("?", "%s")
            conn = self._pg.connect(self.url, connect_timeout=15)
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(sql, params)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall() if fetch else []
            finally:
                conn.close()
        else:
            with sqlite3.connect(SQLITE_PATH) as conn:
                cur = conn.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall() if fetch else []
        rows = [dict(zip(cols, r)) for r in rows]
        if fetch == "one":
            return rows[0] if rows else None
        return rows

    # -- analyses ----------------------------------------------------------
    def save(self, rec):
        cols = list(rec)
        self.run(f"INSERT INTO analyses ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                 tuple(rec[c] for c in cols))

    def get(self, analysis_id, user):
        return self.run("SELECT * FROM analyses WHERE id = ? AND username = ?",
                        (analysis_id, user), fetch="one")

    def list(self, user, limit=500):
        return self.run(
            "SELECT id, filename, created_at, status, health_score, sensors, readings, short_summary "
            "FROM analyses WHERE username = ? ORDER BY created_at DESC LIMIT ?", (user, limit), fetch="all")

    def delete(self, analysis_id, user):
        row = self.run("SELECT csv_path FROM analyses WHERE id = ? AND username = ?",
                       (analysis_id, user), fetch="one")
        if row:
            self.run("DELETE FROM analyses WHERE id = ? AND username = ?", (analysis_id, user))
            remove_file(row["csv_path"])
        return bool(row)

    def delete_all(self, user):
        rows = self.run("SELECT csv_path FROM analyses WHERE username = ?", (user,), fetch="all")
        self.run("DELETE FROM analyses WHERE username = ?", (user,))
        for r in rows:
            remove_file(r["csv_path"])
        return len(rows)


def remove_file(name):
    if name:
        path = UPLOAD_DIR / Path(name).name
        if path.exists():
            path.unlink()


db = Database(env("DATABASE_URL"))


# --------------------------------------------------------------------------- #
# AERIS model (trained once with: python aeris.py)
# --------------------------------------------------------------------------- #
try:
    model = AERIS.load()
    MODEL_ERROR = None
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)
    print(f"[SHMS] {MODEL_ERROR}")


# --------------------------------------------------------------------------- #
# File -> CSV -> sensor signals
# --------------------------------------------------------------------------- #
class TelemetryError(ValueError):
    """A problem with the uploaded file that we can explain to the user."""


def file_to_dataframe(path, ext):
    try:
        if ext == ".npy":
            arr = np.load(path, allow_pickle=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            names = ["telemetry"] + [f"command_{i}" for i in range(1, arr.shape[1])]
            return pd.DataFrame(arr, columns=names)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if ext == ".json":
            return pd.read_json(path)
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise TelemetryError(f"We could not read this file ({exc.__class__.__name__}). "
                             "Please upload a .npy, .csv, .xlsx, .xls, .json or .txt telemetry file.")


SKIP_NAME = re.compile(r"(^|_|\s)(time|timestamp|date|datetime|index|idx|id|unnamed.*)($|_|\s)", re.I)


def pick_sensors(df):
    """Choose the sensor columns to analyse (numeric, not time/index/on-off flags)."""
    numeric = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().mean() >= 0.8:
            numeric[str(col)] = values
    if not numeric:
        raise TelemetryError("No numeric sensor readings were found in this file.")

    chosen = []
    for name, values in numeric.items():
        v = values.dropna()
        if SKIP_NAME.search(name):
            continue
        if v.nunique() <= 1:
            continue
        if set(np.unique(v)).issubset({0, 1}):      # on/off command flags
            continue
        if v.is_monotonic_increasing and np.allclose(np.diff(v.to_numpy()), 1):  # a counter
            continue
        chosen.append(name)
    if not chosen:  # fall back to the first numeric column
        chosen = [next(iter(numeric))]
    return {name: numeric[name].to_numpy(dtype=float) for name in chosen[:MAX_SENSORS]}


# --------------------------------------------------------------------------- #
# Charts (Matplotlib, styled to match the dark UI)
# --------------------------------------------------------------------------- #
C = {
    "bg": "#0B0B12", "panel": "#12121C", "grid": "#23232F", "text": "#ECEAF4",
    "muted": "#8E8BA3", "purple": "#A78BFA", "purple_deep": "#7C3AED",
    "danger": "#F43F5E", "truth": "#F5B54A",
}


def new_figure(rows, height):
    fig = Figure(figsize=(11, height), dpi=110, facecolor=C["bg"])
    axes = fig.subplots(rows, 1, sharex=True) if rows > 1 else [fig.subplots()]
    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["grid"])
        ax.tick_params(colors=C["muted"], labelsize=9)
    return fig, axes


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def title(ax, text):
    ax.set_title(text, loc="left", color=C["text"], fontsize=11.5, pad=10, fontweight="bold")


def shade(ax, events, color, alpha):
    for e in events:
        ax.axvspan(e[0], e[1], color=color, alpha=alpha, linewidth=0)


def sensor_chart(name, raw, result, threshold):
    s, summary = result["series"], result["summary"]
    n = len(raw)
    t = np.arange(n)
    events = [(e["start"], e["end"]) for e in summary["events"]]
    truth = (summary["evaluation"] or {}).get("true_events", [])

    fig, (ax1, ax2, ax3) = new_figure(3, 8.2)
    fig.subplots_adjust(hspace=0.45)

    # 1. readings with detected anomalies
    shade(ax1, events, C["danger"], 0.22)
    ax1.plot(t, raw, color=C["purple"], linewidth=1.0)
    if truth:
        lo, hi = ax1.get_ylim()
        for a, b in truth:
            ax1.plot([a, b], [hi, hi], color=C["truth"], linewidth=5, solid_capstyle="butt")
    title(ax1, f"{name}: sensor readings (red zones = anomalies found by AERIS)")
    ax1.set_ylabel("Reading", color=C["muted"], fontsize=9)
    handles = [Line2D([], [], color=C["purple"], label="Reading"),
               Patch(color=C["danger"], alpha=0.35, label="AERIS anomaly")]
    if truth:
        handles.append(Line2D([], [], color=C["truth"], linewidth=5, label="NASA-labelled anomaly"))
    leg = ax1.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, ncol=3,
                     bbox_to_anchor=(0, 1.0))
    for txt in leg.get_texts():
        txt.set_color(C["muted"])

    # 2. anomaly probability
    ax2.fill_between(t, s["probability"], color=C["purple_deep"], alpha=0.35, linewidth=0)
    ax2.plot(t, s["probability"], color=C["purple"], linewidth=1.0)
    ax2.axhline(threshold, color=C["danger"], linewidth=1.0, linestyle="--")
    ax2.text(n * 0.995, threshold + 0.03, f"alarm line ({threshold:.2f})", color=C["danger"],
             fontsize=8.5, ha="right", va="bottom")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Risk", color=C["muted"], fontsize=9)
    title(ax2, "AERIS risk score (above the dashed line = alarm)")

    # 3. forecast error
    ax3.plot(t, s["error"], color=C["muted"], linewidth=0.8)
    shade(ax3, events, C["danger"], 0.18)
    ax3.set_ylabel("Surprise", color=C["muted"], fontsize=9)
    ax3.set_xlabel("Time step (reading number)", color=C["muted"], fontsize=9)
    title(ax3, "Forecast surprise: how far each reading was from what AERIS expected")
    return fig_to_b64(fig)


def overview_chart(sensor_rows):
    names = [r["name"] for r in sensor_rows]
    scores = [r["health_score"] for r in sensor_rows]
    colors = [C["danger"] if r["status"] == "DANGER" else
              C["truth"] if r["status"] == "WARNING" else C["purple"] for r in sensor_rows]
    fig, (ax,) = new_figure(1, 0.6 + 0.55 * len(names))
    y = np.arange(len(names))[::-1]
    ax.barh(y, [100] * len(names), color=C["panel"], height=0.55)
    ax.barh(y, scores, color=colors, height=0.55)
    for yi, sc in zip(y, scores):
        ax.text(min(sc, 100) + 1.5, yi, f"{sc:.0f}/100", va="center", color=C["text"], fontsize=9.5)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 112)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    title(ax, "Health score of every sensor (100 = perfectly healthy)")
    return fig_to_b64(fig)


def confusion_chart(cm, name):
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig = Figure(figsize=(4.6, 4.0), dpi=110, facecolor=C["bg"])
    ax = fig.subplots()
    ax.set_facecolor(C["bg"])
    ax.imshow(grid, cmap="Purples", vmin=0)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="#ffffff" if v > grid.max() / 2 else "#1B1530")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    ax.set_xlabel("AERIS said", color=C["muted"])
    ax.set_ylabel("NASA experts said", color=C["muted"])
    ax.tick_params(colors=C["muted"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    title(ax, f"Confusion matrix: {name}")
    return fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# Analysis pipeline
# --------------------------------------------------------------------------- #
STATUS_RANK = {"HEALTHY": 0, "WARNING": 1, "DANGER": 2}


def run_analysis(src_path, original_name):
    if model is None:
        raise TelemetryError("The AERIS model is not trained yet. Run: python aeris.py")
    ext = Path(original_name).suffix.lower()
    df = file_to_dataframe(src_path, ext)
    if len(df) < MIN_ROWS:
        raise TelemetryError(f"This file has only {len(df)} rows. AERIS needs at least {MIN_ROWS} readings.")

    # every file is converted to CSV first; the analysis reads the CSV
    analysis_id = uuid.uuid4().hex
    csv_name = f"{analysis_id}.csv"
    df.to_csv(UPLOAD_DIR / csv_name, index=False)
    df = pd.read_csv(UPLOAD_DIR / csv_name)

    truncated = len(df) > MAX_ROWS
    if truncated:
        df = df.tail(MAX_ROWS).reset_index(drop=True)
    sensors = pick_sensors(df)
    chan_id, nasa_part = None, None

    sensor_rows, charts = [], []
    for i, (name, values) in enumerate(sensors.items()):
        # NASA files are recognised from their readings (so renamed files work);
        # the file name is only a fallback.
        cid, part = model.identify(values)
        if not cid and i == 0:
            cid = model.find_channel(original_name)
            part = "test" if cid else None
        if cid:
            chan_id, nasa_part = chan_id or cid, nasa_part or part
            if name.isdigit():
                name = "telemetry"
        # A channel's normal recording has no expert labels and is what the
        # expert model treats as "normal", so it is checked like any other file.
        result = model.analyze(values, chan_id=cid if part == "test" else None)
        result["summary"]["chan_id"] = cid
        sm = result["summary"]
        sm["name"] = name
        sm["nasa_part"] = part
        sensor_rows.append(sm)
        charts.append({
            "title": f"Sensor: {name}",
            "caption": (f"{name}: {sm['num_readings']:,} readings, {sm['num_events']} anomaly event(s), "
                        f"{sm['anomaly_ratio']:.1%} of readings flagged. Status {sm['status']}."),
            "img": sensor_chart(name, values, result, model.threshold),
        })
        if sm["evaluation"]:
            charts.append({
                "title": f"Accuracy check: {name}",
                "caption": "Compares AERIS's answers with NASA's expert labels for this channel.",
                "img": confusion_chart(sm["evaluation"]["confusion_matrix"], name),
                "small": True,
            })
    if len(sensor_rows) > 1:
        charts.insert(0, {
            "title": "All sensors at a glance",
            "caption": "; ".join(f"{r['name']} {r['health_score']:.0f}/100" for r in sensor_rows),
            "img": overview_chart(sensor_rows),
        })

    worst = max(sensor_rows, key=lambda r: STATUS_RANK[r["status"]])
    overall = {
        "filename": original_name,
        "file_type": ext.lstrip(".").upper(),
        "chan_id": chan_id,
        "nasa_part": nasa_part,
        "status": worst["status"],
        "health_score": round(min(r["health_score"] for r in sensor_rows), 1),
        "num_sensors": len(sensor_rows),
        "num_readings": int(len(df)),
        "num_columns": int(df.shape[1]),
        "truncated": truncated,
        "total_events": sum(r["num_events"] for r in sensor_rows),
        "sensors": sensor_rows,
        "model": model.info,
        "model_validation": sensor_rows[0]["model_validation"],
        "analysed_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    for c in ("train_channels", "holdout_channels"):
        overall["model"].pop(c, None)
    return analysis_id, csv_name, overall, charts


# --------------------------------------------------------------------------- #
# Gemini: reports + chatbot
# --------------------------------------------------------------------------- #
# Tried in order. Measured on 28 Oct 2026: 3.6 Flash answered a report in ~6-14 s,
# the "lite" models in 2-5 s, while the newest 3.7 / 3.8 Flash were overloaded and
# timed out. "flash-lite-latest" always points to Google's current lite model.
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-lite-latest"]
GEMINI_TIMEOUT_S = 20      # max wait for one Gemini answer
GEMINI_TOTAL_S = 50        # max time spent on Gemini for one request (all models)
_gemini_client = None


def gemini_client():
    """One shared client with a hard timeout and only short retries.

    Without these, the SDK waits forever for a stuck request and retries a busy
    or rate-limited request up to 5 times with up to 60 s pauses. Instead of
    retrying a slow model we move on to the next one.
    """
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        from google.genai import types
        _gemini_client = genai.Client(
            api_key=env("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_S * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _gemini_client


def thinking_for(name, types):
    """Keep Gemini's 'thinking' short: it is not needed to write a report."""
    if name.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    if name.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="LOW")
    return None


def gemini_generate(contents, system, json_mode=False, temperature=0.5):
    """Call Gemini (tries a few model names). Returns text or raises."""
    if not env("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    from google.genai import types
    client = gemini_client()
    started = time.time()
    last = None
    models = list(GEMINI_MODELS)
    tried = set()
    while models:
        name = models.pop(0)
        if name in tried:
            continue
        tried.add(name)
        if time.time() - started > GEMINI_TOTAL_S - 5:
            break
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=thinking_for(name, types),
        )
        t0 = time.time()
        try:
            try:
                resp = client.models.generate_content(model=name, contents=contents, config=config)
            except Exception as exc:
                if "thinking" not in str(exc).lower():
                    raise
                # this model does not accept our thinking setting: use its default
                config.thinking_config = None
                resp = client.models.generate_content(model=name, contents=contents, config=config)
            if resp.text:
                print(f"[SHMS] Gemini {name} answered in {time.time() - t0:.1f}s")
                return resp.text
            last = RuntimeError("empty answer")
        except BaseException as exc:  # never let a Gemini problem stop the server
            if isinstance(exc, KeyboardInterrupt):
                raise
            last = exc
            msg = str(exc)
            print(f"[SHMS] Gemini {name} failed after {time.time() - t0:.1f}s: {msg[:200]}")
            # a wrong or blocked API key fails the same way for every model
            if "API key" in msg or "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg:
                break
            # a retired model: Google names its replacement, so try that next
            suggested = re.findall(r"models/([\w.\-]+)", msg)
            models[:0] = [m for m in suggested if m != name and m not in tried]
    raise RuntimeError(f"Gemini did not answer ({str(last)[:200]})")


def compact_for_ai(overall):
    """A small version of the analysis to send to Gemini."""
    sensors = []
    for s in overall["sensors"]:
        item = {k: s[k] for k in ("name", "status", "health_score", "num_readings", "anomalous_readings",
                                  "num_events", "peak_probability", "signal_stats", "forecast_metrics",
                                  "baseline_source")}
        item["aeris_mode"] = ("expert: this NASA sensor's own neural-network mini brain + expert model"
                              if s.get("mode") == "expert" else "general model")
        item["anomaly_percent"] = round(s["anomaly_ratio"] * 100, 2)
        item["events"] = s["events"][:8]
        if s["evaluation"]:
            ev = s["evaluation"]
            item["evaluation_vs_nasa_labels"] = {k: ev[k] for k in (
                "accuracy", "precision", "recall", "f1", "events_total", "events_detected",
                "alarms_total", "alarms_correct", "brier_mse", "seen_in_training")}
        sensors.append(item)
    v = overall["model_validation"]
    return {
        "file": overall["filename"], "nasa_channel": overall["chan_id"],
        "overall_status": overall["status"], "overall_health_score": overall["health_score"],
        "readings": overall["num_readings"], "sensors_analysed": overall["num_sensors"],
        "sensors": sensors,
        "aeris_validated_scores_on_unseen_nasa_channels": {
            k: round(v[k], 3) for k in ("accuracy", "precision", "recall", "f1", "event_recall")},
        "charts_shown_to_user": [
            "Per sensor: (1) the readings over time with red zones where AERIS found anomalies"
            " (and yellow bars on top for NASA-labelled anomalies when known), (2) the AERIS risk"
            " score with a dashed alarm line, (3) forecast surprise = gap between expected and real reading.",
            "If several sensors: a health score bar for each sensor.",
            "If NASA labels are known: a confusion matrix comparing AERIS with the experts.",
        ],
    }


REPORT_SYSTEM = (
    "You are AERIS Report Writer inside the Spacecraft Health Monitoring System. "
    "You turn a machine-learning telemetry analysis into a report that anyone can understand, "
    "even a child or someone with no science background. Use short sentences and everyday words, "
    "explain any technical term in brackets the first time, never invent numbers that are not in "
    "the data, and be honest about uncertainty. Status meanings: HEALTHY = no anomalies found, "
    "WARNING = some unusual behaviour, DANGER = strong or long-lasting anomalies."
)

REPORT_PROMPT = """Here is the analysis result (JSON):

{data}

Write a JSON object with exactly two keys:
"report": a Markdown report with these sections (use '## ' headings):
  ## Verdict  (one or two sentences: is the spacecraft safe or in danger?)
  ## What we checked
  ## What AERIS found  (explain events, where they happen, how strong)
  ## Sensor by sensor  (bullet list, one bullet per sensor)
  ## Understanding the charts  (explain each chart the user sees in simple words)
  ## How much can we trust this?  (explain the evaluation metrics: precision, recall, F1, MSE, RMSE in kid-friendly words with the actual values)
  ## What should the team do next?  (3-5 practical bullet points)
"summary": 2-3 plain sentences that sum up the whole analysis for a busy reader.
"""


def fallback_report(overall):
    """Built-in report used when Gemini is not available."""
    status_text = {
        "HEALTHY": "The spacecraft looks healthy. AERIS did not find anything unusual.",
        "WARNING": "The spacecraft needs attention. AERIS found some unusual behaviour that engineers should check.",
        "DANGER": "The spacecraft may be in danger. AERIS found strong or long-lasting unusual behaviour.",
    }[overall["status"]]
    lines = ["## Verdict", status_text, "",
             "## What we checked",
             f"AERIS read **{overall['num_readings']:,} readings** from **{overall['num_sensors']} sensor(s)** "
             f"in the file *{overall['filename']}* and compared every reading with how a healthy sensor normally behaves.",
             "", "## Sensor by sensor"]
    for s in overall["sensors"]:
        lines.append(f"- **{s['name']}**: {s['status']} (health {s['health_score']:.0f}/100), "
                     f"{s['num_events']} anomaly event(s), {s['anomaly_ratio']:.1%} of readings flagged.")
    lines += ["", "## How much can we trust this?"]
    ev = next((s["evaluation"] for s in overall["sensors"] if s["evaluation"]), None)
    if ev:
        lines.append(f"For this NASA channel we know the right answers. AERIS scored precision "
                     f"{ev['precision']:.2f}, recall {ev['recall']:.2f} and F1 {ev['f1']:.2f}, and caught "
                     f"{ev['events_detected']} of {ev['events_total']} real anomaly events.")
    else:
        v = overall["model_validation"]
        lines.append(f"On spacecraft channels it had never seen, AERIS caught {v['events_detected']} of "
                     f"{v['events_total']} real anomaly events (F1 {v['f1']:.2f}).")
    lines += ["", "## What should the team do next?",
              "- Look closely at the red zones in the charts.",
              "- Compare them with commands sent to the spacecraft at the same time.",
              "- Keep monitoring the sensors marked WARNING or DANGER.",
              "", "*Gemini was not available, so this short report was written by the platform itself.*"]
    summary = (f"{status_text} {overall['total_events']} anomaly event(s) across "
               f"{overall['num_sensors']} sensor(s); overall health {overall['health_score']:.0f}/100.")
    return "\n".join(lines), summary


def make_report(overall):
    try:
        text = gemini_generate(REPORT_PROMPT.format(data=json.dumps(compact_for_ai(overall), indent=1)),
                               REPORT_SYSTEM, json_mode=True, temperature=0.4)
        try:
            data = json.loads(text)
            report, summary = data.get("report", ""), data.get("summary", "")
        except json.JSONDecodeError:
            report, summary = text, ""
        if not report:
            raise RuntimeError("empty report")
        if not summary:
            summary = fallback_report(overall)[1]
        return report.strip(), summary.strip(), "gemini", None
    except Exception as exc:
        print(f"[SHMS] Using built-in report: {exc}")
        report, summary = fallback_report(overall)
        return report, summary, "built-in", str(exc)


CHAT_SYSTEM = """You are the AERIS Assistant, the friendly chatbot of the Spacecraft Health Monitoring System (SHMS).
About the platform: space organisations log in with pre-created accounts, upload a telemetry file
(.npy, .csv, .xlsx, .xls, .json, .txt), press Analyze, and the AERIS model ({full}) checks every sensor.
AERIS = Linear Regression forecaster (MSE/RMSE/MAE/MAPE) + a neural-network "mini brain" for every NASA sensor + Random Forest anomaly classifier chosen by
5-fold cross validation against Logistic Regression and Decision Tree, trained on the NASA SMAP & MSL
dataset (82 channels, 105 expert-labelled anomaly events). Results show charts, a Gemini report,
evaluation metrics (precision, recall, F1, accuracy, MSE, RMSE) and a short summary, and are saved in History
(which the user can delete). Status meanings: HEALTHY, WARNING, DANGER.
You can help with: the user's analyses and history, how the platform works, telemetry, anomaly detection,
machine learning concepts, spacecraft, aerospace and astronautics.
Style: friendly, simple words anyone can follow, short paragraphs or bullet points, use **bold** for key
facts. Never invent analysis numbers - only use the data given below. If you do not know, say so.

The user's recent analyses (newest first):
{history}
{current}"""


def chat_context(user, analysis_id=None):
    rows = db.list(user, limit=15)
    history = "\n".join(
        f"- {r['created_at'][:16].replace('T', ' ')} | {r['filename']} | {r['status']} | "
        f"health {r['health_score']:.0f}/100 | {r['short_summary']}" for r in rows) or "- (no analyses yet)"
    current = ""
    if analysis_id:
        rec = db.get(analysis_id, user)
        if rec:
            current = ("\nThe user is currently looking at this analysis:\n"
                       + json.dumps(compact_for_ai(json.loads(rec["summary"])))
                       + f"\nIts summary: {rec['short_summary']}")
    return CHAT_SYSTEM.format(full=MODEL_FULL_NAME, history=history, current=current)


# --------------------------------------------------------------------------- #
# Small Markdown -> HTML (for Gemini reports)
# --------------------------------------------------------------------------- #
def inline_md(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def markdown_to_html(md):
    out, para, list_tag = [], [], None

    def flush():
        nonlocal para, list_tag
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
            para = []
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in (md or "").splitlines():
        s = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)", s)
        bullet = re.match(r"^[-*+]\s+(.*)", s)
        number = re.match(r"^\d+[.)]\s+(.*)", s)
        if not s:
            flush()
        elif heading:
            flush()
            level = min(len(heading.group(1)) + 1, 4)
            out.append(f"<h{level}>{inline_md(heading.group(2))}</h{level}>")
        elif bullet or number:
            tag = "ul" if bullet else "ol"
            if para:
                out.append(f"<p>{inline_md(' '.join(para))}</p>")
                para = []
            if list_tag != tag:
                if list_tag:
                    out.append(f"</{list_tag}>")
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{inline_md((bullet or number).group(1))}</li>")
        else:
            if list_tag:
                out.append(f"</{list_tag}>")
                list_tag = None
            para.append(s)
    flush()
    return Markup("\n".join(out))


app.jinja_env.filters["markdown"] = markdown_to_html


@app.template_filter("pct")
def pct(value, digits=1):
    return f"{(value or 0) * 100:.{digits}f}%"


@app.template_filter("when")
def when(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return value


@app.context_processor
def globals_for_templates():
    return {
        "current_user": session.get("user"),
        "db_label": db.label,
        "db_kind": db.kind,
        "gemini_ready": bool(env("GEMINI_API_KEY")),
        "model_ready": model is not None,
        "year": datetime.now().year,
    }


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def sample_files():
    if not TEST_DIR.exists():
        return []
    return sorted((p.name for p in TEST_DIR.glob("*.npy")),
                  key=lambda n: (n.split("-")[0], int(re.sub(r"\D", "", n) or 0)))


@app.route("/")
@app.route("/about")
def about():
    metrics = model.metrics if model else None
    return render_template("about.html", metrics=metrics)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("analyze"))
    users = configured_users()
    if request.method == "POST":
        name = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        expected = users.get(name)
        if expected is not None and hmac.compare_digest(expected, password):
            session.clear()
            session["user"] = name
            session.permanent = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("analyze"))
        flash("That username or password is not right. Please try again.", "error")
    return render_template("login.html", no_users=not users)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out safely.", "success")
    return redirect(url_for("login"))


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "GET":
        return render_template("analyze.html", samples=sample_files(),
                               recent=db.list(session["user"], limit=4), model_error=MODEL_ERROR)

    upload = request.files.get("file")
    sample = request.form.get("sample", "").strip()
    tmp = None
    try:
        if upload and upload.filename:
            name = Path(upload.filename).name
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise TelemetryError(f"'{ext or 'no extension'}' files are not supported. "
                                     "Please use .npy, .csv, .xlsx, .xls, .json or .txt.")
            tmp = UPLOAD_DIR / f"incoming_{uuid.uuid4().hex}{ext}"
            upload.save(tmp)
            src = tmp
        elif sample:
            if sample not in sample_files():
                raise TelemetryError("Please choose a sample from the list.")
            name, src = sample, TEST_DIR / sample
        else:
            raise TelemetryError("Please choose a telemetry file (or a NASA sample) first.")

        analysis_id, csv_name, overall, charts = run_analysis(src, name)
        # The results page opens straight away with the built-in report; the page
        # then asks Gemini for the full report in the background (/api/.../report).
        report, short = fallback_report(overall)
        source = "pending" if env("GEMINI_API_KEY") else "built-in"
        db.save({
            "id": analysis_id, "username": session["user"], "filename": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": overall["status"], "health_score": overall["health_score"],
            "sensors": overall["num_sensors"], "readings": overall["num_readings"],
            "summary": json.dumps(overall), "charts": json.dumps(charts),
            "report": report, "short_summary": short, "report_source": source, "csv_path": csv_name,
        })
        return redirect(url_for("result", analysis_id=analysis_id))
    except TelemetryError as exc:
        flash(str(exc), "error")
    except Exception as exc:
        app.logger.exception("Analysis failed")
        flash(f"Something went wrong while analysing the file: {exc}", "error")
    finally:
        if tmp and tmp.exists():
            tmp.unlink()
    return redirect(url_for("analyze"))


@app.route("/analysis/<analysis_id>")
@login_required
def result(analysis_id):
    rec = db.get(analysis_id, session["user"])
    if not rec:
        abort(404)
    return render_template("result.html", rec=rec, a=json.loads(rec["summary"]),
                           charts=json.loads(rec["charts"]))


@app.route("/api/analysis/<analysis_id>/report", methods=["POST"])
@login_required
def gemini_report(analysis_id):
    """Ask Gemini to write the report for a saved analysis (called by the results page)."""
    rec = db.get(analysis_id, session["user"])
    if not rec:
        return jsonify({"error": "This analysis no longer exists."}), 404
    if rec["report_source"] != "gemini":
        report, short, source, error = make_report(json.loads(rec["summary"]))
        if source == "gemini":
            db.run("UPDATE analyses SET report = ?, short_summary = ?, report_source = ? WHERE id = ? AND username = ?",
                   (report, short, source, analysis_id, session["user"]))
        else:
            db.run("UPDATE analyses SET report_source = ? WHERE id = ? AND username = ?",
                   ("built-in", analysis_id, session["user"]))
            if error and "GEMINI_API_KEY" in error:
                error = "The Gemini API key is not set in the .env file."
            elif error and ("DEADLINE" in error or "timed out" in error.lower()):
                error = "Gemini took too long to answer (Google's servers are busy)."
            else:
                error = "Gemini could not be reached right now."
            return jsonify({"ok": False, "error": error})
        rec = db.get(analysis_id, session["user"])
    return jsonify({"ok": True, "report_html": str(markdown_to_html(rec["report"])),
                    "summary": rec["short_summary"]})


@app.route("/analysis/<analysis_id>/csv")
@login_required
def download_csv(analysis_id):
    rec = db.get(analysis_id, session["user"])
    if not rec or not rec["csv_path"] or not (UPLOAD_DIR / rec["csv_path"]).exists():
        abort(404)
    return send_file(UPLOAD_DIR / rec["csv_path"], as_attachment=True,
                     download_name=f"{Path(rec['filename']).stem}_converted.csv")


@app.route("/history")
@login_required
def history():
    return render_template("history.html", rows=db.list(session["user"]))


@app.route("/history/<analysis_id>/delete", methods=["POST"])
@login_required
def delete_analysis(analysis_id):
    if db.delete(analysis_id, session["user"]):
        flash("The analysis was deleted from your history.", "success")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    count = db.delete_all(session["user"])
    flash(f"Deleted {count} analyses from your history.", "success")
    return redirect(url_for("history"))


# --------------------------------------------------------------------------- #
# Chatbot API
# --------------------------------------------------------------------------- #
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()[:4000]
    if not message:
        return jsonify({"error": "Please type a question."}), 400
    contents = []
    for turn in (data.get("history") or [])[-12:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("text") or "")[:4000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    try:
        system = chat_context(session["user"], data.get("analysis_id"))
        reply = gemini_generate(contents, system, temperature=0.6)
        return jsonify({"reply": reply})
    except Exception as exc:
        msg = str(exc)
        if "GEMINI_API_KEY" in msg:
            msg = "The Gemini API key is not set yet. Add GEMINI_API_KEY to the .env file and restart the app."
        else:
            msg = f"Sorry, I could not reach Gemini right now. {msg}"
        return jsonify({"error": msg}), 503


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(_):
    return render_template("message.html", code=404, heading="Page not found",
                           text="This page does not exist or the analysis was deleted."), 404


@app.errorhandler(413)
def too_large(_):
    flash("That file is too big. The limit is 50 MB.", "error")
    return redirect(url_for("analyze"))


if __name__ == "__main__":
    print("=" * 60)
    print(" Spacecraft Health Monitoring System")
    print(f" AERIS model : {'ready' if model else 'NOT TRAINED - run: python aeris.py'}")
    print(f" Database    : {db.label}")
    print(f" Gemini      : {'ready' if env('GEMINI_API_KEY') else 'key missing (built-in reports used)'}")
    print(f" Logins      : {len(configured_users())} account(s) configured")
    print(" Open        : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
