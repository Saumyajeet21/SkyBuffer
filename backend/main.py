"""
backend/main.py — SkyBuffer FastAPI Backend
============================================
Routes:
  POST /api/auth/login
  POST /api/auth/signup
  POST /api/auth/forgot-password
  POST /api/predict
  GET  /api/history
  GET  /api/airports
  GET  /api/model-info
"""

import os, time, math, requests, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
sys_path = str(Path(__file__).parent)
import sys; sys.path.insert(0, sys_path)
from flight_schedule import get_flights as _get_flights, AIRLINES as SCHED_AIRLINES
from airports_db import get_all_airports, search_airports as _search_airports

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="SkyBuffer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"
PLOTS  = ROOT / "plots"

# Serve plot images as static files at /plots/
if PLOTS.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS)), name="plots")

# ── Supabase ───────────────────────────────────────────────
try:
    from supabase import create_client
    _url = os.getenv("SUPABASE_URL", "")
    _key = os.getenv("SUPABASE_KEY", "")
    sb   = create_client(_url, _key) if _url and _key else None
except Exception:
    sb = None

# In-memory history fallback (used when Supabase is not configured)
_local_history: list = []


# ── Airport data ───────────────────────────────────────────
AIRPORTS = {
    "DEL":{"name":"Indira Gandhi Intl","city":"Delhi","country":"IN","lat":28.5562,"lon":77.1000,"icao":"VIDP"},
    "BOM":{"name":"Chhatrapati Shivaji Intl","city":"Mumbai","country":"IN","lat":19.0896,"lon":72.8656,"icao":"VABB"},
    "BLR":{"name":"Kempegowda Intl","city":"Bengaluru","country":"IN","lat":13.1986,"lon":77.7066,"icao":"VOBL"},
    "HYD":{"name":"Rajiv Gandhi Intl","city":"Hyderabad","country":"IN","lat":17.2403,"lon":78.4294,"icao":"VOHS"},
    "MAA":{"name":"Chennai Intl","city":"Chennai","country":"IN","lat":12.9900,"lon":80.1693,"icao":"VOMM"},
    "CCU":{"name":"Netaji S.C. Bose Intl","city":"Kolkata","country":"IN","lat":22.6549,"lon":88.4467,"icao":"VECC"},
    "COK":{"name":"Cochin Intl","city":"Kochi","country":"IN","lat":10.1520,"lon":76.4019,"icao":"VOCI"},
    "GOI":{"name":"Dabolim Airport","city":"Goa","country":"IN","lat":15.3808,"lon":73.8314,"icao":"VOGO"},
    "AMD":{"name":"Sardar Vallabhbhai Patel Intl","city":"Ahmedabad","country":"IN","lat":23.0772,"lon":72.6347,"icao":"VAAH"},
    "JAI":{"name":"Jaipur Intl","city":"Jaipur","country":"IN","lat":26.8242,"lon":75.8122,"icao":"VIJP"},
    "DXB":{"name":"Dubai Intl","city":"Dubai","country":"AE","lat":25.2532,"lon":55.3657,"icao":"OMDB"},
    "LHR":{"name":"Heathrow","city":"London","country":"GB","lat":51.4700,"lon":-0.4543,"icao":"EGLL"},
    "CDG":{"name":"Charles de Gaulle","city":"Paris","country":"FR","lat":49.0097,"lon":2.5478,"icao":"LFPG"},
    "SIN":{"name":"Changi Airport","city":"Singapore","country":"SG","lat":1.3644,"lon":103.9915,"icao":"WSSS"},
    "NRT":{"name":"Narita Intl","city":"Tokyo","country":"JP","lat":35.7720,"lon":140.3929,"icao":"RJAA"},
    "HKG":{"name":"Hong Kong Intl","city":"Hong Kong","country":"HK","lat":22.3080,"lon":113.9185,"icao":"VHHH"},
    "SYD":{"name":"Kingsford Smith","city":"Sydney","country":"AU","lat":-33.9461,"lon":151.1772,"icao":"YSSY"},
    "FRA":{"name":"Frankfurt Airport","city":"Frankfurt","country":"DE","lat":50.0379,"lon":8.5622,"icao":"EDDF"},
    "IST":{"name":"Istanbul Airport","city":"Istanbul","country":"TR","lat":41.2608,"lon":28.7418,"icao":"LTFM"},
    "DOH":{"name":"Hamad Intl","city":"Doha","country":"QA","lat":25.2731,"lon":51.6080,"icao":"OTHH"},
    "JFK":{"name":"John F. Kennedy Intl","city":"New York","country":"US","lat":40.6413,"lon":-73.7781,"icao":"KJFK"},
    "LAX":{"name":"Los Angeles Intl","city":"Los Angeles","country":"US","lat":33.9425,"lon":-118.4081,"icao":"KLAX"},
    "ORD":{"name":"O'Hare Intl","city":"Chicago","country":"US","lat":41.9742,"lon":-87.9073,"icao":"KORD"},
    "SFO":{"name":"San Francisco Intl","city":"San Francisco","country":"US","lat":37.6213,"lon":-122.3790,"icao":"KSFO"},
    "PEK":{"name":"Beijing Capital Intl","city":"Beijing","country":"CN","lat":40.0799,"lon":116.6031,"icao":"ZBAA"},
    "KUL":{"name":"Kuala Lumpur Intl","city":"Kuala Lumpur","country":"MY","lat":2.7456,"lon":101.7099,"icao":"WMKK"},
    "BKK":{"name":"Suvarnabhumi","city":"Bangkok","country":"TH","lat":13.6900,"lon":100.7501,"icao":"VTBS"},
}

AIRLINES = {
    "AI":"Air India","6E":"IndiGo","SG":"SpiceJet","UK":"Vistara",
    "IX":"Air India Express","QP":"Akasa Air","EK":"Emirates",
    "QR":"Qatar Airways","SQ":"Singapore Airlines","BA":"British Airways",
    "LH":"Lufthansa","AF":"Air France","AA":"American Airlines",
    "DL":"Delta","UA":"United Airlines","TK":"Turkish Airlines",
}

# ── Load models ────────────────────────────────────────────
CLF_KEYS = {
    "Logistic Regression": "logistic",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}

def load_models():
    try:
        base = {
            "scaler":   joblib.load(MODELS/"scaler.joblib"),
            "encoders": joblib.load(MODELS/"label_encoders.joblib"),
            "feats":    joblib.load(MODELS/"feature_columns.joblib"),
            "report":   joblib.load(MODELS/"report.joblib"),
            "clf":      joblib.load(MODELS/"best_classifier.joblib"),
        }
        # Try loading best regressor (optional)
        try: base["model"] = joblib.load(MODELS/"best_model.joblib")
        except: base["model"] = None
        # Load all 4 classifiers
        base["all_clfs"] = {}
        for name, key in CLF_KEYS.items():
            p = MODELS/f"clf_{key}.joblib"
            if p.exists():
                base["all_clfs"][name] = joblib.load(p)
            else:
                base["all_clfs"][name] = base["clf"]  # fallback
        # Load stored metrics
        try: base["all_metrics"] = joblib.load(MODELS/"all_metrics.joblib")
        except: base["all_metrics"] = {}
        return base
    except Exception as e:
        print(f"⚠️ Models not loaded: {e}")
        return None

arts = load_models()

# ── Weather helper ─────────────────────────────────────────
def fetch_weather(code: str, dep_date: str, hour: int) -> dict:
    try:
        a = AIRPORTS[code]
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": a["lat"], "longitude": a["lon"],
            "hourly": ["visibility","wind_speed_10m","precipitation"],
            "start_date": dep_date, "end_date": dep_date,
            "timezone": "auto", "wind_speed_unit": "mph"
        }, timeout=7)
        d = r.json()["hourly"]
        h = min(hour, len(d["visibility"])-1)
        return {
            "visibility": round(d["visibility"][h] * 0.000621371, 2),
            "windSpeed":  round(d["wind_speed_10m"][h], 1),
            "precip":     round(d["precipitation"][h], 2),
            "source":     "Open-Meteo Live"
        }
    except:
        return {"visibility": 7.0, "windSpeed": 10.0, "precip": 0.0, "source": "Mock fallback"}

# ── Congestion helper ──────────────────────────────────────
def fetch_congestion(code: str) -> dict:
    try:
        icao  = AIRPORTS[code]["icao"]
        now   = int(time.time())
        user  = os.getenv("OPENSKY_USERNAME","")
        pw    = os.getenv("OPENSKY_PASSWORD","")
        auth  = (user, pw) if user and pw else None
        r = requests.get(
            "https://opensky-network.org/api/flights/departure",
            params={"airport": icao, "begin": now-3600, "end": now},
            auth=auth, timeout=8
        )
        if r.status_code == 200:
            cnt = len(r.json() or [])
            return {"count": cnt, "index": round(min(cnt/60.0,1.0),3), "source": "OpenSky Live"}
        raise ValueError(f"HTTP {r.status_code}")
    except Exception:
        h   = datetime.now().hour
        est = 42 if (6<=h<=9 or 16<=h<=19) else 18
        idx = round(min(est/60.0, 1.0), 3)
        return {"count": est, "index": idx, "source": "Estimated (OpenSky unavailable)"}


# ── Feature builder ────────────────────────────────────────
def build_features(origin, dest, airline, dep_date_str, hour, vis, wind):
    if not arts: raise HTTPException(500, "Models not loaded. Run train.py first.")
    les=arts["encoders"]; scaler=arts["scaler"]; feats=arts["feats"]
    d   = date.fromisoformat(dep_date_str)
    month=d.month; dow=d.weekday()+1
    row = {
        "Hour_sin":  math.sin(2*math.pi*hour/24),
        "Hour_cos":  math.cos(2*math.pi*hour/24),
        "Mon_sin":   math.sin(2*math.pi*month/12),
        "Mon_cos":   math.cos(2*math.pi*month/12),
        "Dow_sin":   math.sin(2*math.pi*dow/7),
        "Dow_cos":   math.cos(2*math.pi*dow/7),
        "IsWeekend":  int(dow>=6),
        "IsRushHour": int((6<=hour<=9) or (16<=hour<=19)),
        "Distance": 1200.0, "Visibility": vis, "WindSpeed": wind,
    }
    for col, val in [("Airline",airline),("Origin",origin),("Dest",dest)]:
        le = les[col]; v = val if val in set(le.classes_) else le.classes_[0]
        row[col] = int(le.transform([v])[0])
    X = pd.DataFrame([row])[feats].copy()
    # scaler was fitted on all 14 features in train.py — transform the full array
    X_scaled = scaler.transform(X.values)
    return X_scaled


# ══════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════
class LoginReq(BaseModel):
    email: str; password: str

class SignupReq(BaseModel):
    email: str; password: str

class ForgotReq(BaseModel):
    email: str

class PredictReq(BaseModel):
    origin: str; dest: str; airline: str
    departure_date: str; departure_hour: int

# ══════════════════════════════════════════════════════════
# ROUTES — AUTH
# ══════════════════════════════════════════════════════════
@app.post("/api/auth/login")
def login(req: LoginReq):
    if not sb: raise HTTPException(503, "Supabase not configured")
    try:
        r = sb.auth.sign_in_with_password({"email":req.email,"password":req.password})
        return {"user":{"email":req.email,"id":str(r.user.id)},
                "access_token": r.session.access_token}
    except Exception as e:
        raise HTTPException(401, str(e))

@app.post("/api/auth/signup")
def signup(req: SignupReq):
    if not sb: raise HTTPException(503, "Supabase not configured")
    try:
        r = sb.auth.sign_up({"email":req.email,"password":req.password})
        return {"message":"Account created. Please verify your email.","user":{"email":req.email}}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/auth/forgot-password")
def forgot_password(req: ForgotReq):
    if not sb: raise HTTPException(503, "Supabase not configured")
    try:
        sb.auth.reset_password_for_email(req.email)
        return {"message":"Password reset link sent to your email."}
    except Exception as e:
        raise HTTPException(400, str(e))

# ══════════════════════════════════════════════════════════
# ROUTES — DATA
# ══════════════════════════════════════════════════════════
@app.get("/api/airports")
def get_airports():
    return {"airports": AIRPORTS, "airlines": AIRLINES}

@app.get("/api/model-comparison")
def model_comparison():
    """Return metrics for ALL models (regression + classification) for the comparison page."""
    try:
        data = joblib.load(MODELS / "all_metrics.joblib")
        # Attach plot URLs so frontend can display them
        data["plots"] = {
            "actual_vs_predicted": "/plots/actual_vs_predicted.png",
            "confusion_matrix":    "/plots/confusion_matrix.png",
            "feature_importance":  "/plots/feature_importance.png",
        }
        # confusion_matrices already stored in all_metrics.joblib
        return data
    except Exception as e:
        return {"error": str(e), "regression": {}, "classification": {}}


@app.get("/api/model-info")
def model_info():
    if not arts: return {"loaded": False, "message": "Run python train.py first"}
    rpt = arts.get("report", {})
    return {
        "loaded":      True,
        "best_reg":    rpt.get("best_reg", "Random Forest"),
        "best_clf":    rpt.get("best_clf", "Random Forest"),
        "mae":         rpt.get("reg_mae", "N/A"),
        "rmse":        rpt.get("reg_rmse", "N/A"),
        "r2":          rpt.get("reg_r2",  "N/A"),
        "f1":          rpt.get("clf_f1",  "N/A"),
        "dataset":     rpt.get("dataset", "synthetic"),
        "n_samples":   rpt.get("n_samples", 0),
    }

# ══════════════════════════════════════════════════════════
# ROUTES — PREDICT
# ══════════════════════════════════════════════════════════
@app.post("/api/predict")
def predict(req: PredictReq):
    if req.origin == req.dest:
        raise HTTPException(400, "Origin and destination cannot be the same.")

    # Live weather
    wx   = fetch_weather(req.origin, req.departure_date, req.departure_hour)
    cong = fetch_congestion(req.origin)

    # ML inference
    X = build_features(req.origin, req.dest, req.airline,
                       req.departure_date, req.departure_hour,
                       wx["visibility"], wx["windSpeed"])

    # Regression: predict delay minutes (XGBoost regressor)
    delay_minutes = 0.0
    if arts.get("model") is not None:
        try:
            delay_minutes = round(float(arts["model"].predict(X)[0]), 1)
        except: pass

    # Run ALL 4 classifiers and build comparison

    stored = arts.get("all_metrics", {}).get("classification", {})
    all_clfs = arts.get("all_clfs", {})
    model_comparison_out = {}
    for name, clf_model in all_clfs.items():
        try:
            pred_label = int(clf_model.predict(X)[0])
            pred_prob  = round(float(clf_model.predict_proba(X)[0][1]) * 100, 1)
            stored_m   = stored.get(name, {})
            model_comparison_out[name] = {
                "prediction":  "Delayed" if pred_label == 1 else "On Time",
                "probability": pred_prob,
                "is_delayed":  pred_label == 1,
                "Accuracy":    stored_m.get("Accuracy",    "N/A"),
                "Precision":   stored_m.get("Precision",   "N/A"),
                "Recall":      stored_m.get("Recall",      "N/A"),
                "F1":          stored_m.get("F1",          "N/A"),
                "Training_Time": stored_m.get("Training_Time", "N/A"),
                "Type":        stored_m.get("Type", "Base"),
            }
        except Exception as ex:
            model_comparison_out[name] = {"error": str(ex)}

    # Primary prediction = best classifier
    best_name = arts["report"].get("best_clf", "XGBoost")
    best_out  = model_comparison_out.get(best_name, {})
    prob  = best_out.get("probability", 50.0)
    status = best_out.get("prediction", "Unknown")

    # Consensus vote
    votes = sum(1 for v in model_comparison_out.values() if v.get("is_delayed", False))
    consensus = "Delayed" if votes >= 2 else "On Time"

    # Recommendation
    f1s = {n: v.get("F1", 0) for n, v in model_comparison_out.items() if isinstance(v.get("F1"), float)}
    recommended = max(f1s, key=f1s.get) if f1s else best_name
    rec_reasons = {
        "Logistic Regression": "Simple linear model — fast but limited for non-linear patterns.",
        "Decision Tree":       "Rule-based, interpretable — prone to overfitting without ensembling.",
        "Random Forest":       "Bagging ensemble — reduces variance, good generalization.",
        "XGBoost":             "Boosting ensemble — iteratively corrects errors, best F1 score.",
    }

    # Stored confusion matrices & feature importance
    all_meta = arts.get("all_metrics", {})

    # Alternate hours
    alternates = []
    for h in [req.departure_hour+2, req.departure_hour+4, req.departure_hour-2]:
        if 0 <= h <= 23:
            wx2 = fetch_weather(req.origin, req.departure_date, h)
            X2  = build_features(req.origin, req.dest, req.airline,
                                 req.departure_date, h, wx2["visibility"], wx2["windSpeed"])
            try:
                best_clf = arts["all_clfs"].get(recommended) or arts["clf"]
                p2 = round(float(best_clf.predict_proba(X2)[0][1])*100, 1)
                alternates.append({"hour":h, "prob":p2})
            except: pass
    alternates = sorted(alternates, key=lambda x: x["prob"])[:2]

    # Save to Supabase + local history
    record = {
        "created_at":       datetime.utcnow().isoformat() + "Z",
        "origin":           req.origin,
        "destination":      req.dest,
        "airline":          req.airline,
        "departure_date":   req.departure_date,
        "departure_hour":   req.departure_hour,
        "predicted_delay_min": delay_minutes,
        "delay_probability":float(prob / 100),
        "status":           status,
        "visibility":       float(wx["visibility"]),
        "wind_speed":       float(wx["windSpeed"]),
        "precipitation":    float(wx["precip"]),
        "weather_source":   wx["source"],
        "congestion_index": float(cong["index"]),
        "congestion_source":cong["source"],
        "model_name":       recommended,
    }
    # Always save to in-memory list
    _local_history.insert(0, record)
    if len(_local_history) > 200:
        _local_history.pop()
    # Also save to Supabase if connected
    if sb:
        try:
            sb.table("predictions").insert(record).execute()
        except Exception as e:
            print(f"Supabase save failed: {e}")


    return {
        "prob": prob, "status": status, "consensus": consensus,
        "delay_minutes": delay_minutes,
        "weather": wx, "congestion": cong, "alternates": alternates,
        "origin_info": AIRPORTS.get(req.origin, {}),
        "dest_info":   AIRPORTS.get(req.dest, {}),
        "model_comparison":      model_comparison_out,
        "recommended_model":     recommended,
        "recommendation_reason": rec_reasons.get(recommended, ""),
        "regression_metrics":    all_meta.get("regression", {}),
        "confusion_matrices":    all_meta.get("confusion_matrices", {}),
        "feature_importance":    all_meta.get("feature_importance", {}),
        "plots": {
            "actual_vs_predicted": "/plots/actual_vs_predicted.png",
            "confusion_matrix":    "/plots/confusion_matrix.png",
            "feature_importance":  "/plots/feature_importance.png",
        },
        "dataset_info": {
            "n_samples":    all_meta.get("n_samples", 0),
            "delayed_rate": all_meta.get("delayed_rate", 0),
            "dataset":      all_meta.get("dataset", "unknown"),
            "best_reg":     all_meta.get("best_reg", "XGBoost"),
        }
    }


@app.get("/api/history")
def get_history(search: str = "", limit: int = 30):
    # Try Supabase first
    if sb:
        try:
            q = (sb.table("predictions")
                   .select("created_at,origin,destination,airline,departure_date,"
                           "departure_hour,predicted_delay_min,status,visibility,wind_speed,congestion_index")
                   .order("created_at", desc=True).limit(limit).execute())
            data = q.data or []
            if search:
                s = search.upper()
                data = [r for r in data if s in r.get("origin","") or
                        s in r.get("destination","") or s in r.get("airline","") or
                        search.lower() in r.get("status","").lower()]
            return {"predictions": data}
        except Exception as e:
            print(f"Supabase history error: {e}")
    # Fallback to in-memory history
    data = _local_history[:limit]
    if search:
        s = search.upper()
        data = [r for r in data if s in r.get("origin","") or
                s in r.get("destination","") or s in r.get("airline","") or
                search.lower() in r.get("status","").lower()]
    return {"predictions": data}


@app.get("/api/airports/search")
def search_airports(q: str = Query(default="", min_length=0)):
    """Search airports by IATA code, city or name."""
    if not q or len(q) < 2:
        # Return all airports when empty
        return {"airports": [
            {"iata": k, **v} for k, v in get_all_airports().items()
        ]}
    return {"airports": _search_airports(q)}

@app.get("/api/flights/search")
def search_flights(origin: str = "", dest: str = "", date_str: str = ""):
    """Return flights on a route. Uses static schedule as primary source."""
    if not origin or not dest:
        return {"flights": [], "source": "none"}
    origin = origin.upper().strip()
    dest   = dest.upper().strip()
    flights = _get_flights(origin, dest)
    if not flights:
        return {"flights": [], "source": "none",
                "message": f"No scheduled flights found for {origin}-{dest}"}
    # Add day-of-week context if date provided
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dow = dt.weekday()  # 0=Mon ... 6=Sun
            # Weekends: keep all; Weekdays: some carriers reduce frequency — keep all for now
        except:
            pass
    return {"flights": flights, "source": "static_schedule", "count": len(flights)}

@app.get("/")
def root():
    return {"message":"SkyBuffer API running","docs":"/docs"}
