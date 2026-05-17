# SkyBuffer — AI-Powered Flight Delay Prediction Platform

**Academic Machine Learning Project**
Predicting Indian domestic and international flight delays using supervised classification and ensemble methods.

Built with: FastAPI · React/Vite · XGBoost · scikit-learn · Supabase · Open-Meteo

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [ML Tasks Implemented](#ml-tasks-implemented)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)
6. [Features](#features)
7. [Visualization Dashboard](#visualization-dashboard)
8. [Installation and Setup](#installation-and-setup)
9. [Running the Application](#running-the-application)
10. [API Reference](#api-reference)
11. [Project Structure](#project-structure)
12. [Academic Notes](#academic-notes)

---

## Project Overview

SkyBuffer is a full-stack machine learning platform for predicting flight delays at Indian airports. Users search a route, pick a flight from a live schedule, and instantly receive a complete ML classification analysis across four models simultaneously.

Every prediction page includes:

- Delay probability percentage from XGBoost (best model)
- Estimated delay minutes derived from risk probability
- Side-by-side prediction from all four classifiers
- Classification metrics: Accuracy, Precision, Recall, F1-Score per model
- Confusion matrices for all four classifiers
- Feature importance chart (XGBoost classifier)
- Consolidated comparison table: Base Learners vs Ensemble Methods
- Final model recommendation with academic justification

---

## ML Tasks Implemented

### Task 3 — Classification Models (2 Base Learners)

**Objective:** Predict binary outcome — Delayed (DepDelay > 15 min) or On Time.  
**Target variable:** `IsDelayed` (0 = On Time, 1 = Delayed)  
**Dataset:** 479,772 real BTS 2024 flights · 23.7% delayed · Train/Test: 80/20

| Model | Type | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | Base Learner | 73.1% | 34.8% | 15.6% | 21.5% | 4.9s |
| Decision Tree | Base Learner | 74.2% | 37.1% | 12.8% | 19.1% | 0.7s |

**Note:** Base learners on imbalanced data (3.2:1 ratio) struggle to recall delays.
Ensemble methods in Task 5 correct this significantly.

**Outputs:**
- Confusion matrix with TN, FP, FN, TP values
- Classification report per model (Precision, Recall, F1 per class)

---

### Task 5 — Ensemble Methods (2 Ensemble Models)

**Objective:** Apply ensemble techniques and compare against base learners from Task 3.

| Model | Technique | Accuracy | Precision | Recall | F1-Score | Train Time | Improvement |
|---|---|---|---|---|---|---|---|
| Random Forest | Bagging | 73.1% | 40.2% | 27.9% | 33.0% | 23.3s | +11.5% F1 over LR |
| **XGBoost** | **Boosting** | **73.3%** | **42.3%** | **35.1%** | **38.4%** | **9.7s** | **+16.9% F1 over LR** |

**Outputs:**
- Feature importance bar chart for XGBoost
- Accuracy improvement comparison table vs base learners

---

### Task 6 — Model Comparison and Recommendation

Every prediction result page renders a full consolidated analysis:

**Consolidated Metrics Table (all 4 models) — BTS 2024 Dataset:**

| Model | Type | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | Base | 73.1% | 34.8% | 15.6% | 21.5% | 4.9s |
| Decision Tree | Base | 74.2% | 37.1% | 12.8% | 19.1% | 0.7s |
| Random Forest | Ensemble | 73.1% | 40.2% | 27.9% | 33.0% | 23.3s |
| **XGBoost** | **Ensemble ★** | **73.3%** | **42.3%** | **35.1%** | **38.4%** | **9.7s** |

**Why accuracy is similar across models but F1 varies:**
The dataset has 76.3% On Time flights. Accuracy is driven by the majority class.
F1-Score reveals real capability — XGBoost's 38.4% F1 vs Decision Tree's 19.1% shows a 2× improvement in detecting actual delays.

**Comparison Findings:**

| Criterion | Finding |
|---|---|
| Best Accuracy | Decision Tree — 74.2% |
| Best F1-Score | **XGBoost — 38.4%** |
| Best Precision | **XGBoost — 42.3%** |
| Best Recall | **XGBoost — 35.1%** |
| Fastest Training | Decision Tree — 0.7s |
| Overfitting Risk | Decision Tree: high; RF and XGBoost generalize better |
| Underfitting | Logistic Regression: underfits non-linear patterns |
| **Recommendation** | **XGBoost — best F1, Precision, and Recall across all metrics** |

---

## System Architecture

```
User Browser (React + Vite)         localhost:5173
         |
         |   REST API
         |
FastAPI Backend                     localhost:8000
         |
         +-- /api/predict           Runs all 4 classifiers, returns full comparison
         +-- /api/flights/search    DGCA static schedule lookup by route + date
         +-- /api/airports/search   44-airport fuzzy search
         +-- /api/history           Full prediction history (Supabase)
         +-- /api/model-comparison  All stored training metrics (classification)
         +-- /api/plots             Seaborn visualization URLs
         +-- /plots/                Static file serving for generated charts
         |
         +-- ML Models (joblib)
         |       clf_logistic.joblib
         |       clf_decision_tree.joblib
         |       clf_random_forest.joblib
         |       clf_xgboost.joblib
         |       best_classifier.joblib
         |
         +-- Live APIs
         |       Open-Meteo     Weather forecast (no API key required)
         |       OpenSky        Airport traffic (optional credentials)
         |
         +-- Supabase
                 predictions table    — stores every prediction with full metadata
                 model_metadata table — stores training run info per session
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, Recharts, Vanilla CSS |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Machine Learning | scikit-learn, XGBoost, NumPy, pandas, joblib |
| Visualization | Seaborn, Matplotlib (training), Recharts (live dashboard) |
| Database and Auth | Supabase (PostgreSQL + Supabase Auth) |
| Weather API | Open-Meteo (free, no API key required) |
| Traffic Data | OpenSky Network (optional free account) |
| Flight Schedule | Static DGCA-based timetable (40+ routes, 200+ flights) |
| Airports | 44 airports — 30 Indian, 14 international |

---

## Dataset

- **Source:** BTS (Bureau of Transportation Statistics) 2024 domestic flight data  
  Fallback: Synthetic dataset (30,000 samples) generated if CSV is absent
- **Training size:** 479,772 samples (500,000 cap, filtered for valid records)
- **Train / Test split:** 80% / 20% stratified
- **Delayed rate:** 23.7% — class imbalance handled with `class_weight` on all models
- **Threshold:** A flight is Delayed if departure delay > 15 minutes (FAA standard)

### Feature Engineering

| Feature | Description | Encoding |
|---|---|---|
| `Hour_sin`, `Hour_cos` | Departure hour | Cyclical (sine/cosine) |
| `Mon_sin`, `Mon_cos` | Month of year | Cyclical |
| `Dow_sin`, `Dow_cos` | Day of week | Cyclical |
| `IsWeekend` | Saturday or Sunday | Binary |
| `IsRushHour` | 06:00–09:00 or 16:00–19:00 | Binary |
| `Distance` | Route distance in miles | Standard scaled |
| `Visibility` | Live weather — visibility in miles | Standard scaled |
| `WindSpeed` | Live weather — wind speed in mph | Standard scaled |
| `Airline` | Carrier IATA code | Label encoded |
| `Origin` | Departure airport IATA code | Label encoded |
| `Dest` | Arrival airport IATA code | Label encoded |

### Handling Class Imbalance

The dataset is 76.3% On Time and 23.7% Delayed — a 3.2:1 ratio.

All four models use an identical `class_weight = {0: 1.0, 1: 2.0}` to ensure fair competition.
XGBoost additionally uses `scale_pos_weight = 2.0` for consistent treatment.

This setup allows XGBoost to demonstrate natural gradient-boosting superiority rather than winning due to hyperparameter advantage.

---

## Features

### 3-Step Flight Search Flow

**Step 1 — Route Selection**
Type an airport name or IATA code into a searchable autocomplete dropdown. Matches from 44 airports are shown with full names and country flags.

**Step 2 — Flight Selection**
Available scheduled flights on the selected route and date are displayed as cards, showing flight number, airline, departure/arrival times, and aircraft type.

**Step 3 — Prediction**
Clicking a flight auto-fills the form and runs all four ML classifiers instantly. Results appear on the same page.

### Per-Prediction Analysis (shown on every result)

- Hero summary: delay risk %, estimated delay minutes, best model name, key metrics
- 4 model prediction cards with per-metric progress bars (Accuracy, Precision, Recall, F1)
- Consolidated comparison table: Base Learners vs Ensemble Methods
- Confusion matrices for all 4 classifiers (TN, FP, FN, TP with axis labels)
- Feature importance for XGBoost classifier
- Final recommendation with academic reasoning
- Alternative lower-risk departure time suggestions

### Live Data Integration

- Real-time weather from Open-Meteo API (visibility, wind speed, precipitation)
- Airport traffic estimation (OpenSky live or hourly estimate fallback)
- Estimated delay in minutes derived from classification probability

### Authentication

- Email login and signup via Supabase Auth
- Persistent sessions
- All predictions saved to Supabase with full metadata

---

## Visualization Dashboard

SkyBuffer includes a dedicated **📊 Visualizations** tab with two sub-sections:

### 📡 Live Statistics (updates every 5 seconds automatically)

| Chart | Description |
|---|---|
| Summary cards | Total predictions, On Time count, Delayed count, Avg delay |
| Prediction Distribution | Pie chart of On Time vs Delayed from prediction history |
| Predictions by Airline | Grouped bar chart per carrier |
| Delay Trend | Line chart of last 20 predicted delay estimates |
| Model Metrics Radar | Radar chart of F1, Accuracy, Precision, Recall per model |
| F1 / Precision / Recall Bar | Side-by-side bar comparison of all 4 classifiers |

### 🎓 Training Analysis (Seaborn — generated by train.py)

| Plot | Description |
|---|---|
| `metrics_comparison.png` | Grouped bar chart — Accuracy, Precision, Recall, F1 per model |
| `confusion_matrix.png` | 2×2 confusion matrices for all 4 classifiers |
| `roc_curves.png` | ROC curves with AUC scores |
| `metrics_heatmap.png` | Color-coded heatmap — models × metrics |
| `precision_recall.png` | Precision vs Recall scatter plot |
| `training_time.png` | Horizontal bar chart of training times |
| `feature_importance.png` | XGBoost classifier feature importance |

Click any chart to **expand fullscreen** in a lightbox.

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Supabase account (free tier is sufficient)

### Step 1 — Clone and Install

```bash
git clone https://github.com/Saumyajeet21/SkyBuffer.git
cd SkyBuffer

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### Step 2 — Configure Environment Variables

```bash
cp .env.example .env
```

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

Predictions work fully without Supabase credentials. Only authentication and history persistence are disabled.

### Step 3 — Train the Models

Place BTS 2024 flight data at `data/flight_data_2024.csv` (optional — synthetic fallback is used if absent).

```bash
python train.py
```

Generated files:

```
models/
    clf_logistic.joblib
    clf_decision_tree.joblib
    clf_random_forest.joblib
    clf_xgboost.joblib
    best_classifier.joblib
    scaler.joblib
    label_encoders.joblib
    feature_columns.joblib
    all_metrics.joblib
    report.joblib

plots/
    metrics_comparison.png
    confusion_matrix.png
    roc_curves.png
    metrics_heatmap.png
    precision_recall.png
    training_time.png
    feature_importance.png
```

Training takes approximately **40–45 seconds** on the full 479k-row dataset.

---

## Running the Application

**Terminal 1 — Backend:**

```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Run all 4 classifiers, return full comparison |
| GET | `/api/flights/search` | Search flights by origin, destination, date |
| GET | `/api/airports/search` | Fuzzy search airports by name or IATA code |
| GET | `/api/model-comparison` | Retrieve stored classification training metrics |
| GET | `/api/model-info` | Best model name, F1, accuracy summary |
| GET | `/api/history` | Prediction history (Supabase + in-memory fallback) |
| GET | `/api/plots` | List all available seaborn visualization URLs |
| GET | `/plots/{filename}` | Serve a generated chart image |
| POST | `/api/auth/login` | Sign in via Supabase |
| POST | `/api/auth/signup` | Create account via Supabase |

### Predict — Request Body

```json
{
  "origin": "DEL",
  "dest": "BOM",
  "airline": "6E",
  "departure_date": "2026-05-20",
  "departure_hour": 9
}
```

### Predict — Response (key fields)

```json
{
  "status": "On Time",
  "prob": 33.2,
  "consensus": "On Time",
  "delay_minutes": 13.9,
  "recommended_model": "XGBoost",
  "recommendation_reason": "Boosting ensemble — iteratively corrects errors, best F1 score.",
  "model_comparison": {
    "Logistic Regression": { "prediction": "On Time", "probability": 29.4, "Accuracy": 0.7309, "F1": 0.2152 },
    "Decision Tree":       { "prediction": "On Time", "probability": 31.0, "Accuracy": 0.7420, "F1": 0.1905 },
    "Random Forest":       { "prediction": "On Time", "probability": 32.1, "Accuracy": 0.7309, "F1": 0.3297 },
    "XGBoost":             { "prediction": "On Time", "probability": 33.2, "Accuracy": 0.7331, "F1": 0.3839 }
  },
  "weather": { "visibility": 7.0, "windSpeed": 10.0, "precip": 0.0, "source": "Open-Meteo Live" },
  "congestion": { "index": 0.3, "source": "OpenSky Live" },
  "alternates": [{ "hour": 7, "prob": 28.1 }]
}
```

---

## Project Structure

```
SkyBuffer/
|
+-- backend/                          FastAPI application
|   +-- main.py                       All route handlers and ML inference
|   +-- airports_db.py                44-airport database with fuzzy search
|   +-- flight_schedule.py            Static DGCA-based schedule (40+ routes)
|
+-- frontend/                         React + Vite frontend
|   +-- index.html
|   +-- vite.config.js
|   +-- package.json
|   +-- src/
|       +-- App.jsx                   Root component and routing
|       +-- main.jsx                  React entry point
|       +-- api.js                    All API call functions
|       +-- index.css                 Global design system and CSS variables
|       +-- pages/
|       |   +-- Dashboard.jsx         3-step prediction flow + tab navigation
|       |   +-- Dashboard.css
|       |   +-- Auth.jsx              Login and signup
|       |   +-- Auth.css
|       +-- components/
|           +-- ResultCard.jsx        Full ML comparison display
|           +-- ResultCard.css
|           +-- AirportSearch.jsx     Autocomplete airport dropdown
|           +-- AirportSearch.css
|           +-- FlightSelector.jsx    Flight card grid
|           +-- FlightSelector.css
|           +-- History.jsx           Prediction history table (Supabase)
|           +-- Visualizations.jsx    Live charts + seaborn training plots
|           +-- Visualizations.css
|
+-- data/                             Input datasets (not committed)
|   +-- flight_data_2024.csv          BTS 2024 real data (download separately)
|
+-- models/                           Generated by train.py (not committed)
|   +-- clf_logistic.joblib
|   +-- clf_decision_tree.joblib
|   +-- clf_random_forest.joblib
|   +-- clf_xgboost.joblib
|   +-- best_classifier.joblib
|   +-- scaler.joblib
|   +-- label_encoders.joblib
|   +-- feature_columns.joblib
|   +-- all_metrics.joblib
|   +-- report.joblib
|
+-- plots/                            Generated by train.py (not committed)
|   +-- metrics_comparison.png
|   +-- confusion_matrix.png
|   +-- roc_curves.png
|   +-- metrics_heatmap.png
|   +-- precision_recall.png
|   +-- training_time.png
|   +-- feature_importance.png
|
+-- train.py                          Classification-only training pipeline
+-- requirements.txt                  All Python dependencies
+-- .env                              Supabase credentials (not committed)
+-- .env.example                      Template for environment variables
+-- .gitignore
+-- README.md
```

---

## Academic Notes

### Why F1-Score is the Primary Metric

The dataset has 76.3% On Time and 23.7% Delayed flights — significant class imbalance. A model that always predicts "On Time" achieves 76.3% accuracy while catching zero delays. F1-Score penalizes this by requiring a balance between Precision (correct positive predictions) and Recall (catching actual delays). All model comparisons are therefore ranked by F1-Score first.

### Why Regression was Excluded

Regression models (Linear Regression, Ridge, Random Forest Regressor, XGBoost Regressor) predicting exact delay minutes were evaluated but showed very low R² scores (~0.03) on the BTS 2024 dataset. This indicates that exact delay minutes cannot be reliably predicted from schedule and weather features alone — the binary classification problem (On Time vs Delayed) is substantially more tractable. Estimated delay minutes are now derived from classification probability × historical average delay (42 min).

### Why XGBoost is the Best Classifier

1. **Highest F1-Score:** 38.4% — best across all four classifiers
2. **Best Precision:** 42.3% — fewest false alarms
3. **Best Recall:** 35.1% — catches the most real delays
4. **Fast training:** 9.7 seconds — 2× faster than Random Forest
5. **Gradient boosting** iteratively corrects prior tree errors — naturally handles class imbalance
6. **Built-in regularization** (gamma, min_child_weight, reg_alpha, reg_lambda) prevents overfitting

### Fair Training Setup

All 4 models are trained with **identical** `class_weight = {0: 1.0, 1: 2.0}`. No model receives preferential hyperparameter tuning. XGBoost's superiority is driven entirely by algorithmic capability (gradient boosting) rather than configuration bias.

### Overfitting vs Underfitting Analysis

| Model | Behaviour | Reason |
|---|---|---|
| Logistic Regression | Underfits | Linear decision boundary cannot model non-linear delay patterns |
| Decision Tree | Overfits | High variance without ensembling; memorizes training data |
| Random Forest | Good generalization | Bagging averages many independent trees, reducing variance |
| XGBoost | Best generalization | Boosting + regularization achieves best bias-variance trade-off |

### Ensemble Improvement over Base Learners

| Ensemble Model | vs Base Learner | F1 Improvement |
|---|---|---|
| Random Forest (Bagging) | vs Logistic Regression | +11.5% F1 |
| Random Forest (Bagging) | vs Decision Tree | +13.9% F1 |
| XGBoost (Boosting) | vs Logistic Regression | +16.9% F1 |
| XGBoost (Boosting) | vs Decision Tree | +20.3% F1 |

---

*Dataset: BTS 2024 domestic flight records (479,772 flights)*  
*Weather: Open-Meteo forecast API (live)*  
*Schedule: DGCA-based static timetable (40+ routes, 200+ flights)*  
*Traffic: OpenSky Network*
