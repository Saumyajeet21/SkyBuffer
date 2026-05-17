#!/usr/bin/env python3
"""
train.py - SkyBuffer: Airline Flight Delay Prediction
=====================================================
TASK 3 - CLASSIFICATION : Logistic Regression, Decision Tree (Base Learners)
                          Random Forest, XGBoost             (Ensemble Methods)
  Metrics : Accuracy, Precision, Recall, F1-Score
  Plots   : Confusion Matrices, Metrics Comparison, ROC Curves,
            Metrics Heatmap, Precision-Recall Scatter, Training Time
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc as roc_auc
)
from xgboost import XGBClassifier

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
# 1. SYNTHETIC DATA FALLBACK
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
# 3. CLASSIFICATION  (Task 3 — Base + Task 5 — Ensemble)
# ══════════════════════════════════════════════════════════════
def run_classification(Xtr, Xte, ytr, yte):
    print("\n" + "="*60)
    print("TASK 3 — CLASSIFICATION  (target: IsDelayed 0/1)")
    print("="*60)

    n_neg = (ytr == 0).sum(); n_pos = (ytr == 1).sum()
    scale = round(n_neg / max(n_pos, 1), 2)

    # ── FAIR TRAINING: identical class weight for all 4 models ──
    sw = min(scale, 2.0)
    cw = {0: 1.0, 1: sw}
    print(f"  Class ratio (neg/pos): {scale}  — all models use identical weight={sw}")

    models = {
        # Base learners
        "Logistic Regression": LogisticRegression(
            max_iter=3000, random_state=SEED, n_jobs=-1,
            class_weight=cw, C=1.0, solver="saga", penalty="l2"),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=12, random_state=SEED, class_weight=cw,
            min_samples_split=8, min_samples_leaf=5,
            max_features="sqrt", ccp_alpha=0.0001),
        # Ensemble methods
        "Random Forest":       RandomForestClassifier(
            n_estimators=400, max_depth=18, n_jobs=-1, random_state=SEED,
            class_weight=cw, min_samples_leaf=3,
            max_features="sqrt", min_samples_split=6),
        # XGBoost — same weight, naturally outperforms via gradient boosting
        "XGBoost":             XGBClassifier(
            n_estimators=700, learning_rate=0.015, max_depth=8,
            subsample=0.85, colsample_bytree=0.80, min_child_weight=3,
            gamma=0.05, reg_alpha=0.1, reg_lambda=2.0,
            n_jobs=-1, random_state=SEED, verbosity=0,
            eval_metric="logloss", scale_pos_weight=sw),
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
    print(f"\nClassification Report — {best}:")
    print(classification_report(yte, results[best]["y_pred"],
                                 target_names=["On Time","Delayed"]))

    # ── SEABORN VISUALIZATION SUITE ───────────────────────────
    COLORS = ["#4F8EF7","#a29bfe","#00cec9","#00b894"]
    names  = list(results.keys())

    # 1. Confusion Matrices (2×2 grid)
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

    # 2. Grouped bar — all 4 metrics
    metrics_df = pd.DataFrame({
        "Model":  names * 4,
        "Metric": (["Accuracy"]*4 + ["Precision"]*4 + ["Recall"]*4 + ["F1-Score"]*4),
        "Score":  ([results[n]["Acc"]  for n in names] +
                   [results[n]["Prec"] for n in names] +
                   [results[n]["Rec"]  for n in names] +
                   [results[n]["F1"]   for n in names]),
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model",
                palette=COLORS, ax=ax, width=0.7)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.set_title("Model Comparison — Accuracy, Precision, Recall, F1-Score",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    for p in ax.patches:
        h = p.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(PLOTS/"metrics_comparison.png", dpi=120)
    plt.close()
    print("  Saved: plots/metrics_comparison.png")

    # 3. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, res), col in zip(results.items(), COLORS):
        if res["prob"] is not None:
            fpr, tpr, _ = roc_curve(yte, res["prob"])
            auc_score   = roc_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc_score:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random Classifier")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — All Classification Models")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS/"roc_curves.png", dpi=120)
    plt.close()
    print("  Saved: plots/roc_curves.png")

    # 4. Training time
    fig, ax = plt.subplots(figsize=(8, 5))
    times = [results[n]["train_time"] for n in names]
    bars  = ax.barh(names, times, color=COLORS, height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{t}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison — All Models", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS/"training_time.png", dpi=120)
    plt.close()
    print("  Saved: plots/training_time.png")

    # 5. Metrics heatmap
    hm_data = pd.DataFrame({
        n: {"Accuracy": results[n]["Acc"], "Precision": results[n]["Prec"],
            "Recall":   results[n]["Rec"], "F1-Score":  results[n]["F1"]}
        for n in names
    }).T
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="white", vmin=0, vmax=1,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Metrics Heatmap — Models vs Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS/"metrics_heatmap.png", dpi=120)
    plt.close()
    print("  Saved: plots/metrics_heatmap.png")

    # 6. Precision vs Recall scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, res), col in zip(results.items(), COLORS):
        ax.scatter(res["Prec"], res["Rec"], s=200, color=col, zorder=5, label=name)
        ax.annotate(name, (res["Prec"], res["Rec"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9, color=col)
    ax.set(xlabel="Precision", ylabel="Recall", title="Precision vs Recall — All Models")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS/"precision_recall.png", dpi=120)
    plt.close()
    print("  Saved: plots/precision_recall.png")

    # 7. Feature importance (XGBoost classifier)
    xgb_model = results["XGBoost"]["model"]
    feat_cols  = ["Hour_sin","Hour_cos","Mon_sin","Mon_cos","Dow_sin","Dow_cos",
                  "IsWeekend","IsRushHour","Distance","Visibility","WindSpeed",
                  "Airline","Origin","Dest"]
    imp = pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#00b894")
    ax.set_title("Feature Importance — XGBoost Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS/"feature_importance.png", dpi=120)
    plt.close()
    print("  Saved: plots/feature_importance.png")

    return results, best

# ══════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n== AIRLINE FLIGHT DELAY — CLASSIFICATION PIPELINE ==")

    # Load dataset
    if DATA_PATH.exists():
        print(f"Loading real dataset from {DATA_PATH} ...")
        df = pd.read_csv(DATA_PATH, usecols=[
            "month","day_of_week","crs_dep_time",
            "op_unique_carrier","origin","dest","distance","dep_delay","cancelled"
        ], nrows=500_000)
        df.rename(columns={
            "month":             "Month",
            "day_of_week":       "DayOfWeek",
            "crs_dep_time":      "CRSDepTime",
            "op_unique_carrier": "Airline",
            "origin":            "Origin",
            "dest":              "Dest",
            "distance":          "Distance",
            "dep_delay":         "DepDelay",
            "cancelled":         "Cancelled",
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

    # Features + split
    X, les, scaler, feats = make_features(df)
    y_clf = df["IsDelayed"].values
    X_tr, X_te, yc_tr, yc_te = train_test_split(
        X, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)

    # Run classification
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
    joblib.dump(scaler,                     MODELS/"scaler.joblib")
    joblib.dump(les,                        MODELS/"label_encoders.joblib")
    joblib.dump(feats,                      MODELS/"feature_columns.joblib")

    # Confusion matrices + feature importance dict
    cms = {}; feat_imp = {}
    for name, res in clf_res.items():
        cm  = confusion_matrix(yc_te, res["y_pred"]).tolist()
        cms[name] = {"TN":cm[0][0],"FP":cm[0][1],"FN":cm[1][0],"TP":cm[1][1]}
        m = res["model"]
        if hasattr(m, "feature_importances_"):
            feat_imp[name] = [
                {"feature": f, "importance": round(float(v), 5)}
                for f, v in sorted(zip(feats, m.feature_importances_),
                                   key=lambda x: -x[1])
            ]

    all_metrics = {
        "classification": {
            name: {
                "Accuracy":      round(res["Acc"],  4),
                "Precision":     round(res["Prec"], 4),
                "Recall":        round(res["Rec"],  4),
                "F1":            round(res["F1"],   4),
                "Training_Time": res["train_time"],
                "Type": "Ensemble" if name in ["Random Forest","XGBoost"] else "Base",
            } for name, res in clf_res.items()
        },
        "confusion_matrices": cms,
        "feature_importance": feat_imp,
        "best_clf":    best_clf,
        "dataset":     dataset_type,
        "n_samples":   int(len(df)),
        "delayed_rate": round(float(y_clf.mean()), 4),
    }
    joblib.dump(all_metrics, MODELS/"all_metrics.joblib")

    # Report (used by backend)
    joblib.dump({
        "best_clf":  best_clf,
        "clf_f1":    round(clf_res[best_clf]["F1"],  4),
        "clf_acc":   round(clf_res[best_clf]["Acc"], 4),
        "dataset":   dataset_type,
        "n_samples": len(df),
    }, MODELS/"report.joblib")

    print("  Saved to models/")

    # Supabase log
    if SUPABASE_ON and sb:
        try:
            c = clf_res[best_clf]
            sb.table("model_metadata").insert({
                "model_name":   best_clf,
                "dataset_type": dataset_type,
                "n_samples":    int(len(df)),
                "f1_score":     round(c["F1"],  4),
                "accuracy":     round(c["Acc"], 4),
            }).execute()
            print("  Logged to Supabase OK")
        except Exception as e:
            print(f"  Supabase log failed: {e}")

    print("\nDone! Run:  uvicorn backend.main:app --reload --port 8000")

if __name__ == "__main__":
    main()
