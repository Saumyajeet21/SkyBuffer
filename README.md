# SkyBuffer — AI-Powered Flight Delay Prediction Platform

**Academic Machine Learning Project**
Predicting Indian domestic and international flight delays using supervised learning and ensemble methods.

Built with: FastAPI · React/Vite · XGBoost · scikit-learn · Supabase · Open-Meteo

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [ML Tasks Implemented](#ml-tasks-implemented)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)
6. [Features](#features)
7. [Installation and Setup](#installation-and-setup)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Project Structure](#project-structure)
11. [Academic Notes](#academic-notes)

---

## Project Overview

SkyBuffer is a full-stack machine learning platform for predicting flight delays at Indian airports. Users search a route, pick a flight from a live schedule, and instantly receive a complete ML analysis across four models simultaneously.

Every prediction page includes:

- Delay probability percentage from the best model
- Side-by-side prediction from all four models
- Classification report: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for all four classifiers
- Feature importance charts for ensemble models
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
| Logistic Regression | Base Learner | 76.2% | 27.4% | 0.3% | 0.6% | 4.6s |
| Decision Tree | Base Learner | 76.0% | 40.6% | 2.6% | 5.0% | 0.9s |

**Note:** Base learners with this imbalanced dataset (3.2:1 ratio) learn to mostly predict "On Time".
Ensemble methods in Task 5 correct this significantly.

**Outputs:**
- Confusion matrix with TN, FP, FN, TP values and percentages
- Classification report per model (Precision, Recall, F1 per class)


---

### Task 5 — Ensemble Methods (2 Ensemble Models)

**Objective:** Apply ensemble techniques and compare against base learners from Task 3.

| Model | Technique | Accuracy | Precision | Recall | F1-Score | Train Time | Improvement over Base |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **Bagging** | **74.2%** | **40.5%** | **18.8%** | **25.7%** | **23.5s** | **+5x F1 over Logistic Regression** |
| XGBoost | Boosting | 75.9% | 47.5% | 17.5% | 25.5% | 6.5s | +5x F1 over Decision Tree |

**Outputs:**
- Feature importance bar charts for Random Forest and XGBoost
- Accuracy improvement comparison table vs base learners


---

### Task 6 — Model Comparison and Recommendation

Every prediction result page renders a full consolidated analysis:

**Consolidated Metrics Table (all 4 models) — BTS 2024 Dataset:**

| Model | Type | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | Base | 76.2% | 27.4% | 0.3% | 0.6% | 4.6s |
| Decision Tree | Base | 76.0% | 40.6% | 2.6% | 5.0% | 0.9s |
| **Random Forest** | **Ensemble** | **74.2%** | **40.5%** | **18.8%** | **25.7%** | **23.5s** |
| XGBoost | Ensemble | 75.9% | 47.5% | 17.5% | 25.5% | 6.5s |

**Why accuracy exceeds 76% for base learners but F1 is low:**
The dataset has 76.3% On Time flights. A model predicting "always On Time" scores 76.3% accuracy with 0% recall.
Base learners with mild class weighting (scale=1.5) fall into this trap.
Ensemble methods with tree-depth and feature randomness escape it — RF achieves 18.8% recall vs 0.3% for LR.

**Comparison Findings:**

| Criterion | Finding |
|---|---|
| Best Accuracy | XGBoost — 75.9% |
| Best F1-Score | Random Forest — 25.7% |
| Best Precision | XGBoost — 47.5% |
| Best Recall | Random Forest — 18.8% |
| Fastest Training | Decision Tree — 0.9s |
| Overfitting Risk | Decision Tree: high; RF and XGBoost generalize better |
| Underfitting | Logistic Regression: underfits non-linear delay patterns |
| **Recommendation** | **Random Forest — best F1, good recall, strong generalization** |

**Visualizations on every prediction result page:**
- 4 model prediction cards with probability bars (Accuracy, Precision, Recall, F1)
- Consolidated comparison table
- Confusion matrices for all 4 classifiers with axis labels
- Feature importance charts for Random Forest and XGBoost
- Final recommendation with 4-point justification
- Estimated delay in minutes (XGBoost Regressor)


---

## System Architecture

```
User Browser (React + Vite)    localhost:5173
         |
         |   REST API
         |
FastAPI Backend                localhost:8000
         |
         +-- /api/predict          Runs all 4 classifiers, returns full comparison
         +-- /api/flights/search   DGCA static schedule lookup by route + date
         +-- /api/airports/search  44-airport fuzzy search
         +-- /api/history          Prediction history (Supabase)
         |
         +-- ML Models (joblib)
         |       clf_logistic.joblib
         |       clf_decision_tree.joblib
         |       clf_random_forest.joblib
         |       clf_xgboost.joblib
         |
         +-- Live APIs
         |       Open-Meteo     Weather forecast (no API key required)
         |       OpenSky        Airport traffic (optional credentials)
         |
         +-- Supabase
                 predictions table
                 model_metadata table
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, Vanilla CSS |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Machine Learning | scikit-learn, XGBoost, NumPy, pandas, joblib |
| Database and Auth | Supabase (PostgreSQL + Supabase Auth) |
| Weather API | Open-Meteo (free, no API key required) |
| Traffic Data | OpenSky Network (optional free account) |
| Flight Schedule | Static DGCA-based timetable (40+ routes, 200+ flights) |
| Airports | 44 airports — 30 Indian, 14 international |

---

## Dataset

- **Source:** BTS (Bureau of Transportation Statistics) 2024 domestic flight data
  Fallback: Synthetic dataset (30,000 samples) generated if CSV is absent
- **Training size:** 500,000 samples (subsampled for speed)
- **Train / Test split:** 80% / 20%
- **Delayed rate:** 19.6% — class imbalance handled explicitly
- **Threshold:** A flight is Delayed if departure delay > 15 minutes (FAA standard)

### Feature Engineering

| Feature | Description | Encoding |
|---|---|---|
| `Hour_sin`, `Hour_cos` | Departure hour | Cyclical (sine/cosine) |
| `Mon_sin`, `Mon_cos` | Month of year | Cyclical |
| `Dow_sin`, `Dow_cos` | Day of week | Cyclical |
| `IsWeekend` | Saturday or Sunday | Binary |
| `IsRushHour` | 06:00–09:00 or 16:00–19:00 | Binary |
| `Distance` | Route distance | Standard scaled |
| `Visibility` | Live weather — visibility in miles | Standard scaled |
| `WindSpeed` | Live weather — wind speed in mph | Standard scaled |
| `Airline` | Carrier IATA code | Label encoded |
| `Origin` | Departure airport IATA code | Label encoded |
| `Dest` | Arrival airport IATA code | Label encoded |

### Handling Class Imbalance

The dataset is 80.4% On Time and 19.6% Delayed — a 4.1:1 ratio.

- Logistic Regression, Decision Tree, Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=4.1`

This increases Recall for the Delayed class at the cost of raw accuracy, which is the correct trade-off for a delay prediction system.

---

## Features

### 3-Step Flight Search Flow

**Step 1 — Route Selection**
Type an airport name or IATA code into a searchable autocomplete dropdown. Matches from 44 airports are shown with full names and country flags.

**Step 2 — Flight Selection**
Available scheduled flights on the selected route and date are displayed as cards, showing flight number, airline, departure/arrival times, and aircraft type.

**Step 3 — Prediction**
Clicking a flight auto-fills the form and runs all four ML models instantly. Results appear on the same page.

### Per-Prediction Analysis (shown on every result)

- Hero summary: delay risk %, best model name, Accuracy, F1, Precision, Recall
- 4 model prediction cards with per-metric progress bars
- Consolidated comparison table: Base Learners vs Ensemble Methods
- Confusion matrices for all 4 classifiers (TN, FP, FN, TP with percentages and axis labels)
- Feature importance for Random Forest and XGBoost
- Final recommendation with academic reasoning

### Live Data Integration

- Real-time weather from Open-Meteo API (visibility, wind speed, precipitation)
- Airport traffic estimation (OpenSky live or hourly estimate fallback)
- Alternative lower-risk departure time suggestions

### Authentication

- Email login and signup via Supabase Auth
- Persistent sessions
- Prediction history stored per user

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

Copy the example file and fill in your Supabase credentials:

```bash
cp .env.example .env
```

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

Predictions work fully without Supabase credentials. Only authentication and history are disabled.

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
    confusion_matrix.png
    feature_importance.png
```

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
| POST | `/api/predict` | Run all 4 models, return full comparison |
| GET | `/api/flights/search` | Search flights by origin, destination, date |
| GET | `/api/airports/search` | Fuzzy search airports by name or IATA code |
| GET | `/api/model-comparison` | Retrieve stored training metrics |
| GET | `/api/history` | Retrieve prediction history (auth required) |
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
  "prob": 35.0,
  "consensus": "On Time",
  "recommended_model": "XGBoost",
  "recommendation_reason": "Boosting ensemble — iteratively corrects errors, best F1 score.",
  "model_comparison": {
    "Logistic Regression": { "prediction": "On Time", "probability": 29.4, "Accuracy": 0.5702, "F1": 0.3864 },
    "Decision Tree":       { "prediction": "Delayed", "probability": 51.2, "Accuracy": 0.5999, "F1": 0.3929 },
    "Random Forest":       { "prediction": "On Time", "probability": 32.1, "Accuracy": 0.6405, "F1": 0.4090 },
    "XGBoost":             { "prediction": "On Time", "probability": 35.0, "Accuracy": 0.6484, "F1": 0.4203 }
  },
  "confusion_matrices": {
    "XGBoost": { "TN": 52189, "FP": 28130, "FN": 6868, "TP": 12813 }
  },
  "feature_importance": {
    "XGBoost": [{ "feature": "Hour_sin", "importance": 0.183 }]
  }
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
|   +-- tsconfig.json
|   +-- public/
|   |   +-- favicon.svg
|   |   +-- icons.svg
|   +-- src/
|       +-- App.jsx                   Root component and routing
|       +-- main.jsx                  React entry point
|       +-- api.js                    All API call functions
|       +-- index.css                 Global design system and CSS variables
|       +-- pages/
|       |   +-- Dashboard.jsx         3-step prediction flow
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
|           +-- History.jsx           Prediction history viewer
|
+-- data/                             Input datasets (not committed)
|   +-- flight_data_2024.csv          BTS 2024 real data (download separately)
|   +-- flight_data_2024_sample.csv
|   +-- flight_data_2024_data_dictionary.csv
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
|   +-- confusion_matrix.png
|   +-- feature_importance.png
|
+-- train.py                          Full ML training pipeline (Task 3 + Task 5)
+-- requirements.txt                  All Python dependencies (pinned versions)
+-- .env                              Supabase credentials (not committed)
+-- .env.example                      Template for environment variables
+-- .gitignore
+-- README.md
```

---

## Academic Notes

### Why F1-Score is the Primary Metric

The dataset has 80.4% On Time and 19.6% Delayed flights — significant class imbalance. If a model simply predicts "On Time" for every flight, it achieves 80% accuracy but catches zero delays. F1-Score penalizes this behaviour by requiring a balance between Precision (correct positive predictions) and Recall (catching actual delays).

### Why XGBoost is Recommended

1. Highest F1-Score (42.0%) among all four classifiers
2. Second fastest training time (2.9 seconds)
3. Boosting architecture corrects prior tree errors iteratively — better on imbalanced data
4. Built-in regularization (gamma, min_child_weight) prevents overfitting
5. XGBoost achieves the best bias-variance trade-off across all evaluation metrics

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
| Random Forest (Bagging) | vs Decision Tree | +1.6% F1 |
| XGBoost (Boosting) | vs Logistic Regression | +5.6% F1 |
| XGBoost (Boosting) | vs Decision Tree | +4.7% F1 |

---

*Dataset: BTS 2024 domestic flight records*
*Weather: Open-Meteo forecast API*
*Schedule: DGCA-based static timetable*
*Traffic: OpenSky Network*
