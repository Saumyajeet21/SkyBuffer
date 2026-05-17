#!/usr/bin/env python3
"""
train.py - SkyBuffer: Airline Flight Delay Prediction
=====================================================
TASK 3 - CLASSIFICATION : Logistic Regression, Decision Tree (Base)
                          Random Forest, XGBoost  (Ensemble)
  Metrics : Accuracy, Precision, Recall, F1
  Plot    : Confusion Matrix, Feature Importance

TASK 2 - REGRESSION (supporting only — used for delay minutes estimate)
  Models  : Linear Regression, Ridge, Random Forest, XGBoost
  Metrics : MAE, MSE, RMSE, R²
  Plot    : Actual vs Predicted
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

DATA_PATH = BASE / "data" / "flight_data_2024.csv"

AIRPORTS = [
    "DEL","BOM","BLR","HYD","MAA","CCU","COK","GOI",
    "AMD","PNQ","JAI","LKO","ATQ","TRV","BBI","NAG",
    "DXB","LHR","CDG","SIN","NRT","HKG","SYD","FRA",
    "AMS","IST","DOH","KUL","BKK","ICN","JFK","LAX",
    "ORD","ATL","PEK","SFO",
]
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
    d["Hour_sin"]  = np.sin(2*np.pi*d["Hour"]/24)
    d["Hour_cos"]  = np.cos(2*np.pi*d["Hour"]/24)
    d["Mon_sin"]   = np.sin(2*np.pi*d["Month"]/12)
    d["Mon_cos"]   = np.cos(2*np.pi*d["Month"]/12)
    d["Dow_sin"]   = np.sin(2*np.pi*d["DayOfWeek"]/7)
    d["Dow_cos"]   = np.cos(2*np.pi*d["DayOfWeek"]/7)
    d["IsWeekend"]  = (d["DayOfWeek"]>=6).astype(int)
    d["IsRushHour"] = (((d["Hour"]>=6)&(d["Hour"]<=9))|
                       ((d["Hour"]>=16)&(d["Hour"]<=19))).astype(int)
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
            d[c] = d[c].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else 0
            )
    feat_cols = ["Hour_sin","Hour_cos","Mon_sin","Mon_cos","Dow_sin","Dow_cos",
                 "IsWeekend","IsRushHour","Distance","Visibility","WindSpeed",
                 "Airline","Origin","Dest"]
    X = d[feat_cols].values.astype(float)
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, les, scaler, feat_cols

# ══════════════════════════════════════════════════════════════
# 3. REGRESSION (Task 2 — supporting)
# ══════════════════════════════════════════════════════════════
def run_regression(Xtr, Xte, ytr, yte):
    print("\n" + "="*60)
    print("TASK 2 — REGRESSION  (target: DepDelay minutes)")
    print("="*60)
    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED,
            min_samples_leaf=5, max_features="sqrt"),
        "XGBoost":           XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=SEED, verbosity=0),
    }
    results = {}
    print(f"\n{'Model':<22} {'MAE':>7} {'MSE':>9} {'RMSE':>7} {'R2':>7} {'Time':>6}")
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
    print(f"\n Best Regressor: {best}  (R2={results[best]['R2']:.4f})")

    # Plot Actual vs Predicted
    yp_best = results[best]["y_pred"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(yte, yp_best, alpha=0.3, s=8, color="#4F8EF7", label="Predictions")
    lim = [min(yte.min(), yp_best.min()), max(yte.max(), yp_best.max())]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect fit")
    ax.set(xlabel="Actual Delay (min)", ylabel="Predicted Delay (min)",
           title=f"Actual vs Predicted — {best}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS/"actual_vs_predicted.png", dpi=120)
    plt.close()
    print("  Saved: plots/actual_vs_predicted.png")

    # Feature importance for RF
    rf = results["Random Forest"]["model"]
    fig2, ax2 = plt.subplots(figsize=(8,5))
    imp = pd.Series(rf.feature_importances_,
                    index=["Hour_sin","Hour_cos","Mon_sin","Mon_cos","Dow_sin","Dow_cos",
                           "IsWeekend","IsRushHour","Distance","Visibility","WindSpeed",
                           "Airline","Origin","Dest"])
    imp.sort_values().plot.barh(ax=ax2, color="#00b894")
    ax2.set_title("Feature Importance — Random Forest Regressor")
    plt.tight_layout()
    plt.savefig(PLOTS/"feature_importance.png", dpi=120)
    plt.close()
    print("  Saved: plots/feature_importance.png")

    return results, best

# ══════════════════════════════════════════════════════════════
# 4. CLASSIFICATION (Task 3 + Task 5)
# ══════════════════════════════════════════════════════════════
def run_classification(Xtr, Xte, ytr, yte):
    print("\n" + "="*60)
    print("TASK 3 — CLASSIFICATION  (target: IsDelayed 0/1)")
    print("="*60)

    n_neg = (ytr == 0).sum(); n_pos = (ytr == 1).sum()
    scale = round(n_neg / max(n_pos, 1), 2)

    # soft_scale=1.5 gives best accuracy+recall balance on this dataset
    # scale=1.0 collapses recall to ~0% (model always predicts On Time)
    # scale=4.1 (full balanced) drops accuracy to ~57%
    # 1.5 is the optimal trade-off: ~76% accuracy with meaningful F1
    soft_scale = min(scale, 1.5)

    cw = {0: 1.0, 1: soft_scale}
    print(f"  Class ratio (neg/pos): {scale}  — using soft weight {soft_scale}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=SEED, n_jobs=-1,
            class_weight=cw, C=0.5, solver="saga", penalty="l2"),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=15, random_state=SEED, class_weight=cw,
            min_samples_split=8, min_samples_leaf=4,
            max_features="sqrt", ccp_alpha=0.00005),
        "Random Forest":       RandomForestClassifier(
            n_estimators=400, max_depth=20, n_jobs=-1, random_state=SEED,
            class_weight=cw, min_samples_leaf=2, max_features="sqrt",
            min_samples_split=5),
        "XGBoost":             XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            gamma=0.02, reg_alpha=0.05, reg_lambda=2.0,
            n_jobs=-1, random_state=SEED, verbosity=0,
            eval_metric="logloss", scale_pos_weight=soft_scale),
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
        results[name] = {"model":m,"y_pred":yp,"prob":prob,
                         "Acc":acc,"Prec":prec,"Rec":rec,"F1":f1,
                         "train_time":train_time}

    best = max(results, key=lambda k: results[k]["F1"])
    print(f"\n Best Classifier: {best}  (F1={results[best]['F1']:.4f})")

    # Classification report for best model
    print(f"\nClassification Report — {best}:")
    print(classification_report(yte, results[best]["y_pred"],
                                 target_names=["On Time","Delayed"]))

    # Confusion matrix plot (all 4 models)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, res) in zip(axes.flat, results.items()):
        cm = confusion_matrix(yte, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["On Time","Delayed"],
                    yticklabels=["On Time","Delayed"])
        ax.set_title(f"{name}\nAcc={res['Acc']:.2%}  F1={res['F1']:.2%}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — All Classification Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS/"confusion_matrix.png", dpi=120)
    plt.close()
    print("  Saved: plots/confusion_matrix.png")

    return results, best

# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n== AIRLINE FLIGHT DELAY -- TRAINING PIPELINE ==")

    # Load dataset
    if DATA_PATH.exists():
        print(f"Loading real dataset from {DATA_PATH} ...")
        df = pd.read_csv(DATA_PATH, usecols=[
            "month","day_of_week","crs_dep_time",
            "op_unique_carrier","origin","dest","distance","dep_delay","cancelled"
        ], nrows=500_000)
        df.rename(columns={
            "month":               "Month",
            "day_of_week":         "DayOfWeek",
            "crs_dep_time":        "CRSDepTime",
            "op_unique_carrier":   "Airline",
            "origin":              "Origin",
            "dest":                "Dest",
            "distance":            "Distance",
            "dep_delay":           "DepDelay",
            "cancelled":           "Cancelled",
        }, inplace=True)
        df["Hour"] = (df["CRSDepTime"] // 100).clip(0, 23)
        df = df[df["Cancelled"] != 1]
        df = df.dropna(subset=["DepDelay","Origin","Dest","Airline",
                                "Hour","Distance","Month","DayOfWeek"])
        df["Distance"]   = df["Distance"].clip(0, 5000)
        df["Visibility"] = 7.0
        df["WindSpeed"]  = 10.0
        dataset_type = "real_2024"

    else:
        df = make_data(30_000)
        dataset_type = "synthetic"

    print(f"Dataset: {len(df):,} rows | type={dataset_type}")
    df["IsDelayed"] = (df["DepDelay"] > 15).astype(int)
    print(f"Delayed rate: {df['IsDelayed'].mean()*100:.1f}%")

    # Features
    X, les, scaler, feats = make_features(df)
    y_reg = df["DepDelay"].values.astype(float)
    y_clf = df["IsDelayed"].values

    # Split — same split for both tasks
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)

    # Run both tasks
    reg_res, best_reg = run_regression(X_tr, X_te, yr_tr, yr_te)
    clf_res, best_clf = run_classification(X_tr, X_te, yc_tr, yc_te)

    # ── Save models ────────────────────────────────────────────
    print("\n Saving models ...")
    clf_key_map = {
        "Logistic Regression": "logistic",
        "Decision Tree":       "decision_tree",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }
    for name, key in clf_key_map.items():
        if name in clf_res:
            joblib.dump(clf_res[name]["model"], MODELS/f"clf_{key}.joblib")

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

    # Confusion matrices + feature importance
    _, _, yc_tr2, yc_te2 = train_test_split(X, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)
    cms = {}; feat_imp = {}
    for name, res in clf_res.items():
        yp  = res["y_pred"]
        cm  = confusion_matrix(yc_te2, yp).tolist()
        cms[name] = {"TN":cm[0][0],"FP":cm[0][1],"FN":cm[1][0],"TP":cm[1][1]}
        m = res["model"]
        if hasattr(m, "feature_importances_"):
            feat_imp[name] = [
                {"feature": f, "importance": round(float(v), 5)}
                for f, v in sorted(zip(feats, m.feature_importances_),
                                   key=lambda x: -x[1])
            ]

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
                "mae":          round(r["MAE"],3),
                "r2_score":     round(r["R2"],4),
                "f1_score":     round(c["F1"],4),
                "accuracy":     round(c["Acc"],4),
            }).execute()
            print("  Logged to Supabase OK")
        except Exception as e:
            print(f"  Supabase log failed: {e}")

    print("\nDone! Run:  uvicorn backend.main:app --reload --port 8000")

if __name__ == "__main__":
    main()
