# Spacecraft Health Monitoring System (SHMS)

An AI platform that reads spacecraft **telemetry** (sensor data sent back to Earth) and tells you, in plain
language, whether the spacecraft is **HEALTHY**, needs attention (**WARNING**) or may be in **DANGER**.

It is powered by **AERIS** (*Aerospace Equipment & Reliability Intelligence System*), a machine-learning model
trained on the NASA SMAP & MSL telemetry anomaly dataset.

> CBSE Class XII Artificial Intelligence (843) Capstone Project.

---

## Features

- **Login** with pre-created accounts (no public registration). Built for space organisations.
- **One-click analysis**: upload `.npy`, `.csv`, `.xlsx`, `.xls`, `.json` or `.txt` and press **Analyze**.
  Every file is converted to **CSV** first, and the analysis reads that CSV. You can also pick a NASA sample file.
- **Matplotlib charts**: readings with anomaly zones, the AERIS risk score with its alarm line, forecast surprise,
  a health bar per sensor, and a confusion matrix when expert labels exist.
- **Gemini report** in simple language, plus a **short summary** at the end.
- **Evaluation metrics on every prediction**: Precision, Recall, F1, Accuracy, MSE, RMSE, MAE, MAPE.
- **History** of all analyses (stored in **Neon PostgreSQL**), with search, filter, delete and delete-all.
- **Floating AERIS Assistant** (Gemini chatbot) on every page. It knows your history and the analysis you are viewing.
- **Accessibility**: text size, high contrast, easy-reading spacing, reduce motion, read-aloud reports,
  skip link, keyboard navigation, screen-reader labels and chart descriptions, and print / save as PDF.

## Project structure

```
aeris.py            AERIS model: training + analysis (python aeris.py)
app.py              Flask web app: login, upload, charts, Neon, Gemini, chatbot
templates/          HTML pages (about, login, analyze, result, history)
static/css, js      Black and purple UI, interactions
models/             Trained model (aeris.joblib), metrics JSON, training charts
Dataset/            NASA SMAP & MSL data (train/test .npy + labeled_anomalies.csv)
requirements.txt    Python dependencies
.env.example        Settings template
```

## Setup and run

```bash
venv\Scripts\activate
pip install -r requirements.txt
python aeris.py        # trains AERIS (about 1 minute); skipped if it is already trained
python app.py          # then open http://127.0.0.1:5000
```

Copy `.env.example` to `.env` and fill in:

| Variable | What it is |
|---|---|
| `DATABASE_URL` | Neon connection string (`postgresql://...?sslmode=require`) |
| `GEMINI_API_KEY` | Google Gemini API key (https://aistudio.google.com/apikey) |
| `FLASK_SECRET_KEY` | Any long random text |
| `ADMIN_USERNAME_1` / `ADMIN_PASSWORD_1` | First login |
| `ADMIN_USERNAME_2` / `ADMIN_PASSWORD_2` | Second login (optional) |

The `analyses` table is created in Neon automatically. If Neon or Gemini is not reachable, the app keeps
working: it uses local storage and a built-in report, and the footer shows what is connected.

## How AERIS works

1. **Data collection**: 82 NASA channels (SMAP satellite and MSL Curiosity rover) with 105 expert-labelled anomaly events.
   Column 0 of each `.npy` file is the telemetry value; the other columns are on/off command flags.
2. **Stage 1, Forecaster (Linear Regression)**: predicts each reading from the previous 30.
   Evaluated with **MSE, RMSE, MAE, MAPE**.
3. **Feature engineering**: 27 features per reading. Examples: rolling statistics, z-scores, forecast error,
   how rare a value is, and comparison with the sensor's own normal-behaviour recording.
4. **Stage 2, Anomaly classifier**: **Logistic Regression**, **Decision Tree** and **Random Forest** are compared using
   **5-fold Group cross validation**. The best (Random Forest) is chosen, and its alarm threshold and smoothing
   (**hyperparameters**) are tuned on cross-validation data only.
5. **Train-test split by channel**: 20% of NASA channels are hidden during training and used for the honest test scores.
6. **Data augmentation**: 72 synthetic sensor signals (cycles, noise, drifts, on/off steps, with injected spikes,
   level shifts and frozen sensors) teach AERIS that smooth or noisy signals can be healthy.
7. **Model lifecycle (Build, Test, Retrain, Deploy)**: after testing, AERIS is retrained on all NASA channels for deployment.
8. **Stage 3, Sensor mini brains (Neural Networks)**: every NASA sensor gets its own small neural network (MLP) that
   learns only that sensor's normal recording. Its "surprise" gives the expert Random Forest 3 extra clues.
   The expert is **cross-fitted**: 5 copies, each trained without one fifth of the sensors, and every sensor is
   judged by the copy that never saw its answers. NASA files are recognised by their readings, even when renamed.

### Results

Expert mode (NASA sensors): all 81 labelled sensors, each judged by a model that never saw its answers.

| Metric | Score |
|---|---|
| Anomaly events detected | 97 of 105 (92%) |
| Accuracy | 0.880 |
| Precision | 0.520 |
| Recall | 0.567 |
| F1 score | 0.542 |

General mode (any other file): 17 hidden NASA sensors, never seen during training.

| Metric | Score |
|---|---|
| Anomaly events detected | 24 of 26 (92%) |
| Accuracy | 0.781 |
| Precision | 0.237 |
| Recall | 0.470 |
| F1 score | 0.315 |
| Forecaster RMSE | 0.251 |

| Model (5-fold CV F1) | F1 |
|---|---|
| Logistic Regression | 0.389 |
| Decision Tree | 0.374 |
| **Random Forest** | **0.482** |

Per-reading scores are strict: a reading counts as correct only if it falls inside NASA's exact anomaly window.
At the event level, AERIS finds almost every real anomaly. Training charts are saved in `models/reports/`.

## Syllabus mapping (CBSE AI 843)

| Syllabus topic | Where it is used |
|---|---|
| Problem decomposition, data requirements, data collection | NASA telemetry, labels, CSV conversion |
| Modelling approach, commonly used algorithms | Linear Regression, Logistic Regression, Decision Tree, Random Forest, Neural Network (MLP) |
| Train-test split, cross validation | Channel-wise hold-out split, 5-fold Group K-Fold |
| MSE, RMSE, MAPE | Forecaster evaluation on every analysis |
| Precision, Recall, F1, confusion matrix (Class XI) | Classifier evaluation on every analysis |
| Hyperparameters, model lifecycle (build, deploy, retrain) | `CONFIG` in `aeris.py`, retrain before deployment |
| Storytelling through data | Matplotlib charts + Gemini plain-language report |

## Dataset

NASA SMAP & MSL anomaly detection dataset (Hundman et al., 2018), via
[Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl).

## License

Apache 2.0. See [LICENSE](LICENSE).
