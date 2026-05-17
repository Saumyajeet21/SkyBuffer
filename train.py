#!/usr/bin/env python3
"""
train.py - Airline Flight Delay Prediction
==========================================
Research Paper: "Predicting Flight Delays Using Machine Learning"
MDPI Applied Sciences 2023 - DOI: 10.3390/app13148295

TASK 2 - REGRESSION : Ridge, Random Forest, XGBoost
  Metrics : MAE, MSE, RMSE, R²
  Plot    : Actual vs Predicted

TASK 3 - CLASSIFICATION : Logistic Regression, Random Forest, XGBoost
  Metrics : Precision, Recall, F1, Accuracy
  Plot    : Confusion Matrix + Classification Report
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE   = Path(__file__).parent
MODELS = BASE / "models";  MODELS.mkdir(exist_ok=True)
PLOTS  = BASE / "plots";   PLOTS.mkdir(exist_ok=True)
(BASE / "data").mkdir(exist_ok=True)

# Dataset path — CSV lives in data/
DATA_PATH = BASE / "data" / "flight_data_2024.csv"


# Indian + International airports
AIRPORTS = [
    # India
    "DEL","BOM","BLR","HYD","MAA","CCU","COK","GOI",
    "AMD","PNQ","JAI","LKO","ATQ","TRV","BBI","NAG",
    # International
    "DXB","LHR","CDG","SIN","NRT","HKG","SYD","FRA",
    "AMS","IST","DOH","KUL","BKK","ICN","JFK","LAX",
    "ORD","ATL","PEK","SFO",
]
# Indian + International airlines
AIRLINES = ["AI","6E","SG","G8","UK","IX","QP",
            "EK","EY","QR","SQ","BA","LH","AF",
            "AA","DL","UA","QF","TK","MH","CX"]
SEED = 42

# ── Supabase ───────────────────────────────────────────────────
try:
    from supabase import create_client
    _sb_url = os.getenv("SUPABASE_URL","").rstrip("/").replace("/rest/v1","")
    _sb_key = os.getenv("SUPABASE_KEY","")
    sb = create_client(_sb_url, _sb_key) if _sb_url and _sb_key else None
    SUPABASE_ON = bool(_sb_url and _sb_key)
except Exception:
    SUPABASE_ON = False
    sb = None

# ══════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════
def make_data(n=30_000):
    print(f"Generating {n:,} synthetic rows ...")
    rng = np.random.default_rng(SEED)
    hour  = rng.integers(0, 24, n)
    month = rng.integers(1, 13, n)
    dow   = rng.integers(1,  8, n)
    al    = rng.choice(AIRLINES, n)
    org   = rng.choice(AIRPORTS, n)
    dst   = rng.choice(AIRPORTS, n)
    dist  = rng.uniform(100, 2700, n)
    vis   = rng.uniform(1.0, 10.0, n)
    wind  = rng.uniform(0,   35.0, n)

    delay = rng.normal(5, 15, n)
    rush  = ((hour>=6)&(hour<=9)) | ((hour>=16)&(hour<=19))
    delay += 12 * rush + rng.normal(0,3,n)
    delay += np.where(vis < 3,  rng.uniform(8,25,n), 0)
    delay += np.where(wind>25,  rng.uniform(5,18,n), 0)
    delay += np.where(dow>=6,   rng.uniform(2, 8,n), 0)
    delay += np.where((month==7)|(month==12), 8, 0)
    bias  = {"AA":2,"DL":-3,"UA":5,"WN":1,"B6":8,"AS":-2,"NK":10,"F9":4}
    delay += np.array([bias.get(a,0) for a in al])
    delay  = np.clip(delay, -10, 180).round(1)

    return pd.DataFrame({
        "Month":month, "DayOfWeek":dow, "Hour":hour,
        "Airline":al, "Origin":org, "Dest":dst,
        "Distance":dist.round(0), "Visibility":vis.round(2),
        "WindSpeed":wind.round(1), "DepDelay":delay
    })

# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def make_features(df, les=None, scaler=None, fit=True):
    d = df.copy()
    # Cyclic time features
    d["Hour_sin"]  = np.sin(2*np.pi*d["Hour"]/24)
    d["Hour_cos"]  = np.cos(2*np.pi*d["Hour"]/24)
    d["Mon_sin"]   = np.sin(2*np.pi*d["Month"]/12)
    d["Mon_cos"]   = np.cos(2*np.pi*d["Month"]/12)
    d["Dow_sin"]   = np.sin(2*np.pi*d["DayOfWeek"]/7)
    d["Dow_cos"]   = np.cos(2*np.pi*d["DayOfWeek"]/7)
    # Binary flags
    d["IsWeekend"]  = (d["DayOfWeek"]>=6).astype(int)
    d["IsRushHour"] = (((d["Hour"]>=6)&(d["Hour"]<=9))|
                       ((d["Hour"]>=16)&(d["Hour"]<=19))).astype(int)
    # Encode categoricals
    cats = ["Airline","Origin","Dest"]
    if fit:
        les = {}
        for c in cats:
            le = LabelEncoder()
            d[c] = le.fit_transform(d[c].astype(str))
            les[c] = le
    else:
        for c in cats:
            le = les[c]
            d[c] = d[c].astype(str).apply(
                lambda x: x if x in set(le.classes_) else le.classes_[0])
            d[c] = le.transform(d[c])

    FEATS = ["Hour_sin","Hour_cos","Mon_sin","Mon_cos","Dow_sin","Dow_cos",
             "IsWeekend","IsRushHour","Airline","Origin","Dest",
             "Distance","Visibility","WindSpeed"]
    X = d[FEATS].copy()
    sc_cols = ["Distance","Visibility","WindSpeed"]
    if fit:
        scaler = StandardScaler()
        X[sc_cols] = scaler.fit_transform(X[sc_cols])
    else:
        X[sc_cols] = scaler.transform(X[sc_cols])
    return X, les, scaler, FEATS

# ══════════════════════════════════════════════════════════════
# 3. TASK 2 — REGRESSION
# ══════════════════════════════════════════════════════════════
def run_regression(Xtr, Xte, ytr, yte, feats):
    print("\n" + "="*60)
    print("TASK 2 — REGRESSION  (target: DepDelay minutes)")
    print("="*60)

    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=0.5),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                                   min_samples_leaf=5, n_jobs=-1,
                                                   random_state=SEED),
        "XGBoost":           XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, n_jobs=-1,
                                          random_state=SEED, verbosity=0),
    }


    results = {}
    print(f"\n{'Model':<22} {'MAE':>7} {'MSE':>9} {'RMSE':>7} {'R²':>7} {'Time':>6}")
    print("-"*62)
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        train_time = round(time.time() - t0, 2)
        yp   = m.predict(Xte)
        mae  = mean_absolute_error(yte, yp)
        mse  = mean_squared_error(yte, yp)
        rmse = np.sqrt(mse)
        r2   = r2_score(yte, yp)
        print(f"{name:<22} {mae:>7.2f} {mse:>9.2f} {rmse:>7.2f} {r2:>7.4f} {train_time:>5.1f}s")
        results[name] = {"model":m,"y_pred":yp,"MAE":mae,"MSE":mse,"RMSE":rmse,"R2":r2,"train_time":train_time}

    best = max(results, key=lambda k: results[k]["R2"])
    print(f"\n Best Regressor: {best}  (R²={results[best]['R2']:.4f})")

    # Plot: Actual vs Predicted (best model)
    yp_best = results[best]["y_pred"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(yte, yp_best, alpha=0.3, s=8, color="#4F8EF7", label="Predictions")
    lim = [min(yte.min(), yp_best.min()), max(yte.max(), yp_best.max())]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect fit")
    ax.set(xlabel="Actual Delay (min)", ylabel="Predicted Delay (min)",
           title=f"Actual vs Predicted — {best}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS/"actual_vs_predicted.png", dpi=150)
    plt.close()
    print("  Saved: plots/actual_vs_predicted.png")

    # Plot: Feature importance (XGBoost)
    xgb_model = results["XGBoost"]["model"]
    imp = xgb_model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh([feats[i] for i in idx], imp[idx], color="#4F8EF7")
    ax.set(xlabel="Importance", title="Feature Importance — XGBoost")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PLOTS/"feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: plots/feature_importance.png")

    return results, best

# ══════════════════════════════════════════════════════════════
# 4. TASK 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════
def run_classification(Xtr, Xte, ytr, yte):
    print("\n" + "="*60)
    print("TASK 3 — CLASSIFICATION  (target: IsDelayed 0/1)")
    print("="*60)

    # Compute class imbalance ratio for XGBoost scale_pos_weight
    n_neg = (ytr == 0).sum(); n_pos = (ytr == 1).sum()
    scale = round(n_neg / max(n_pos, 1), 2)
    print(f"  Class ratio (neg/pos): {scale}  — using class_weight='balanced'")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED, n_jobs=-1,
            class_weight="balanced", C=0.1, solver="saga"),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=10, random_state=SEED, class_weight="balanced",
            min_samples_split=20, min_samples_leaf=10, max_features="sqrt"),
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED,
            class_weight="balanced", min_samples_leaf=5, max_features="sqrt"),
        "XGBoost":             XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, n_jobs=-1, random_state=SEED, verbosity=0,
            eval_metric="logloss", scale_pos_weight=scale),
    }

    results = {}
    print(f"\n{'Model':<22} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} {'Time':>6}")
    print("-"*60)
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        train_time = round(time.time() - t0, 2)
        yp   = m.predict(Xte)
        prob = m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else None
        acc  = accuracy_score(yte, yp)
        prec = precision_score(yte, yp, zero_division=0)
        rec  = recall_score(yte, yp, zero_division=0)
        f1   = f1_score(yte, yp, zero_division=0)
        print(f"{name:<22} {acc:>7.4f} {prec:>7.4f} {rec:>8.4f} {f1:>7.4f} {train_time:>5.1f}s")
        results[name] = {"model":m,"y_pred":yp,"prob":prob,"Acc":acc,"Prec":prec,"Rec":rec,"F1":f1,"train_time":train_time}

    best = max(results, key=lambda k: results[k]["F1"])
    print(f"\n Best Classifier: {best}  (F1={results[best]['F1']:.4f})")

    # Classification Report
    print(f"\nClassification Report — {best}:")
    print(classification_report(yte, results[best]["y_pred"],
                                 target_names=["On Time","Delayed"]))

    # Confusion Matrix plot
    cm  = confusion_matrix(yte, results[best]["y_pred"])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["On Time","Delayed"],
                yticklabels=["On Time","Delayed"])
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {best}")
    plt.tight_layout()
    plt.savefig(PLOTS/"confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: plots/confusion_matrix.png")

    return results, best

# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n== AIRLINE FLIGHT DELAY -- TRAINING PIPELINE ==")


    # Load data
    if DATA_PATH.exists():
        print(f"Loading real dataset from {DATA_PATH} ...")
        df = pd.read_csv(DATA_PATH, low_memory=False)

        # Columns are lowercase in the 2024 BTS dataset
        df.columns = df.columns.str.upper()
        col_map = {
            "MONTH":"Month", "DAY_OF_WEEK":"DayOfWeek",
            "OP_UNIQUE_CARRIER":"Airline", "REPORTING_AIRLINE":"Airline",
            "ORIGIN":"Origin", "DEST":"Dest",
            "DISTANCE":"Distance", "DEP_DELAY":"DepDelay"
        }
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

        # Extract departure hour from CRS_DEP_TIME
        for col in ["CRSDEPTIME","CRS_DEP_TIME"]:
            if col in df.columns:
                df["Hour"] = (df[col] // 100).clip(0, 23)
                break
        else:
            df["Hour"] = 9  # fallback

        # Drop cancelled/diverted, remove extreme outliers
        df = df.dropna(subset=["DepDelay"])
        if "CANCELLED" in df.columns:
            df = df[df["CANCELLED"] == 0]
        df = df[df["DepDelay"].between(-60, 300)]  # remove extreme outliers

        # Synthetic weather (no historical weather in BTS data)
        rng = np.random.default_rng(SEED)
        df["Visibility"] = rng.uniform(2, 10, len(df)).round(2)
        df["WindSpeed"]  = rng.uniform(0, 30, len(df)).round(1)

        # Sample to 500k max for speed
        if len(df) > 500_000:
            df = df.sample(500_000, random_state=SEED)

        dataset_type = "real_2024"
    else:
        df = make_data(30_000)
        dataset_type = "synthetic"

    df = df.dropna(subset=["DepDelay"]).reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows | type={dataset_type}")

    # Targets
    y_reg = df["DepDelay"]
    y_clf = (df["DepDelay"] > 15).astype(int)
    print(f"Delayed rate: {y_clf.mean()*100:.1f}%")

    # Features
    X, les, scaler, feats = make_features(df, fit=True)

    # Split
    Xtr,Xte,yr_tr,yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=SEED)
    _,  _,  yc_tr,yc_te = train_test_split(X, y_clf, test_size=0.2, random_state=SEED)

    # Task 2 — Regression
    reg_res, best_reg = run_regression(Xtr, Xte, yr_tr, yr_te, feats)

    # Task 3 — Classification
    clf_res, best_clf = run_classification(Xtr, Xte, yc_tr, yc_te)

    # Save
    print("\n Saving models ...")
    # Save each classifier individually for per-prediction comparison
    clf_key_map = {
        "Logistic Regression": "logistic",
        "Decision Tree":       "decision_tree",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }
    for name, key in clf_key_map.items():
        if name in clf_res:
            joblib.dump(clf_res[name]["model"], MODELS/f"clf_{key}.joblib")

    # Keep backward-compat aliases
    joblib.dump(clf_res[best_clf]["model"], MODELS/"best_classifier.joblib")
    joblib.dump(reg_res[best_reg]["model"], MODELS/"best_model.joblib")
    joblib.dump(scaler,                     MODELS/"scaler.joblib")
    joblib.dump(les,                        MODELS/"label_encoders.joblib")
    joblib.dump(feats,                      MODELS/"feature_columns.joblib")
    joblib.dump({
        "best_reg": best_reg, "best_clf": best_clf,
        "reg_mae":  round(reg_res[best_reg]["MAE"], 3),
        "reg_r2":   round(reg_res[best_reg]["R2"],  4),
        "clf_f1":   round(clf_res[best_clf]["F1"],  4),
        "dataset":  dataset_type,
        "n_samples": len(df),
    }, MODELS/"report.joblib")

    # Build confusion matrices for all classifiers
    _, _, yc_tr2, yc_te2 = train_test_split(X, y_clf, test_size=0.2, random_state=SEED)
    cms = {}
    feat_imp = {}
    for name, res in clf_res.items():
        yp  = res["y_pred"]
        cm  = confusion_matrix(yc_te2, yp).tolist()
        cms[name] = {"TN":cm[0][0],"FP":cm[0][1],"FN":cm[1][0],"TP":cm[1][1]}
        # Feature importance for tree-based
        m = res["model"]
        if hasattr(m, "feature_importances_"):
            feat_imp[name] = [
                {"feature": f, "importance": round(float(v), 5)}
                for f, v in sorted(zip(feats, m.feature_importances_),
                                   key=lambda x: -x[1])
            ]

    # Save ALL comparison metrics
    all_metrics = {
        "regression": {
            name: {
                "MAE":           round(res["MAE"],  2),
                "MSE":           round(res["MSE"],  2),
                "RMSE":          round(res["RMSE"], 2),
                "R2":            round(res["R2"],   4),
                "Training_Time": res["train_time"],
                "Type":          "Ensemble" if name in ["Random Forest","XGBoost"] else "Base",
            } for name, res in reg_res.items()
        },
        "classification": {
            name: {
                "Accuracy":      round(res["Acc"],  4),
                "Precision":     round(res["Prec"], 4),
                "Recall":        round(res["Rec"],  4),
                "F1":            round(res["F1"],   4),
                "Training_Time": res["train_time"],
                "Type":          "Ensemble" if name in ["Random Forest","XGBoost"] else "Base",
            } for name, res in clf_res.items()
        },
        "confusion_matrices": cms,
        "feature_importance": feat_imp,
        "best_reg":    best_reg,
        "best_clf":    best_clf,
        "dataset":     dataset_type,
        "n_samples":   int(len(df)),
        "delayed_rate": round(float(y_clf.mean()), 4),
    }
    joblib.dump(all_metrics, MODELS/"all_metrics.joblib")
    print("  Saved to models/")


    # Supabase log
    if SUPABASE_ON and sb:
        try:
            r = reg_res[best_reg]; c = clf_res[best_clf]
            sb.table("model_metadata").insert({
                "model_name":   best_reg,
                "dataset_type": dataset_type,
                "n_samples":    int(len(df)),
                "mae":          float(round(r["MAE"],3)),
                "rmse":         float(round(r["RMSE"],3)),
                "r2_score":     float(round(r["R2"],4)),
                "accuracy":     float(round(c["Acc"],4)),
                "f1_score":     float(round(c["F1"],4)),
            }).execute()
            print("  Logged to Supabase ✅")
        except Exception as e:
            print(f"  Supabase log failed (check table schema): {e}")

    print("\n✅ Done! Run:  streamlit run app.py")

if __name__ == "__main__":
    main()
