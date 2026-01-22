# infer_v6.py — Inference Pipeline for FOMC Forecast
import os, json, warnings, joblib, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from pytorch_forecasting import TemporalFusionTransformer

warnings.filterwarnings("ignore")

CONFIG = {
    "seq_len": 12,
    "threshold_hike": 0.25,
    "model_dir": "artifacts/",
    "required_cols": ["FEDFUNDS", "CPIAUCSL", "GDPC1", "UNRATE", "PCEPI", "VIXCLS", "T10Y2Y", "SP500", "sentiment"]
}

# Load artifacts
rf = joblib.load(os.path.join(CONFIG["model_dir"], "rf_model.pkl"))
xgb = joblib.load(os.path.join(CONFIG["model_dir"], "xgb_model.pkl"))
tft = TemporalFusionTransformer.load_from_checkpoint(
    os.path.join(CONFIG["model_dir"], "tft_checkpoint.ckpt"),
    map_location=torch.device("cpu")
)
scaler = joblib.load(os.path.join(CONFIG["model_dir"], "scaler.pkl"))
rfe = joblib.load(os.path.join(CONFIG["model_dir"], "rfe_selector.pkl"))
with open(os.path.join(CONFIG["model_dir"], "ensemble_weights.json")) as f:
    weights = json.load(f)

def fetch_fomc_dates():
    return pd.to_datetime(["2025-07-30", "2025-09-17", "2025-11-05"])

def fetch_fred_data():
    # REST (no fredapi dependency)
    dfs = []
    api_key = "4b9232177886c75f573c64f4ebe21c78"
    for sid in CONFIG["required_cols"][:-1]:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&api_key={api_key}&file_type=json"
        )
        import urllib.request, json as _json
        with urllib.request.urlopen(url) as r:
            js = _json.loads(r.read().decode())
        obs = js.get("observations", [])
        df = pd.DataFrame(
            {"DATE": [o["date"] for o in obs],
             sid: [float(o["value"]) if o["value"] != "." else np.nan for o in obs]}
        )
        df["DATE"] = pd.to_datetime(df["DATE"])
        dfs.append(df.set_index("DATE").resample("MS").mean())
    return pd.concat(dfs, axis=1)

def fetch_sentiment_scores(index_dates):
    cache_path = os.path.join(CONFIG["model_dir"], "sentiment_cache.json")
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except:
        cache = {}
    pipe = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    scores = {}
    for dt in tqdm(index_dates, desc="Scoring Sentiment"):
        key = dt.strftime("%Y-%m-%d")
        if key in cache:
            scores[dt] = cache[key]; continue
        try:
            result = pipe("The Federal Reserve stated its monetary outlook.")[0]
            label = result["label"].lower()
            score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(label, 0.0)
        except:
            score = 0.0
        cache[key] = score
        scores[dt] = score
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return pd.Series(scores)

def pad_sequence(seq, seq_len):
    if len(seq) >= seq_len:
        return seq[-seq_len:]
    pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)

def ensemble_predict(X):
    rf_p = rf.predict_proba(X)
    xgb_p = xgb.predict_proba(X)
    tft_p = np.zeros_like(rf_p)
    for i in range(len(X)):
        seq = pad_sequence(X[: i + 1], CONFIG["seq_len"])
        input_tensor = torch.tensor(seq.reshape(1, CONFIG["seq_len"], -1), dtype=torch.float32)
        with torch.no_grad():
            logits = tft(input_tensor)[0].numpy()[0]
        tft_p[i] = logits
    w_rf = weights.get("rf", 1/3); w_xgb = weights.get("xgb", 1/3); w_tft = weights.get("tft", 1/3)
    return w_rf*rf_p + w_xgb*xgb_p + w_tft*tft_p

def main():
    df = fetch_fred_data()
    df["rate_change"] = df["FEDFUNDS"].diff()
    df["sentiment"] = fetch_sentiment_scores(df.index)
    df.dropna(subset=CONFIG["required_cols"] + ["rate_change"], inplace=True)
    df["y"] = df["rate_change"].apply(lambda x: 2 if x > CONFIG["threshold_hike"] else 0 if x < -CONFIG["threshold_hike"] else 1)

    X = df[CONFIG["required_cols"]]
    y = df["y"]
    X_scaled = scaler.transform(X)
    X_sel = rfe.transform(X_scaled)

    # Backtest (fixed models, walk-forward evaluation of ensemble head)
    tscv = TimeSeriesSplit(n_splits=5)
    preds = []
    for _, test_idx in tscv.split(X_sel):
        pred = ensemble_predict(X_sel[test_idx])
        preds.extend(np.argmax(pred, axis=1))
    print("Backtest complete.")
    print(classification_report(y.iloc[-len(preds):], preds))

    # Forecast next meeting
    next_date = fetch_fomc_dates()[0]
    df_cut = df[df.index <= next_date - pd.Timedelta(days=7)]
    X_latest_scaled = scaler.transform(df_cut[CONFIG["required_cols"]].tail(CONFIG["seq_len"]))
    X_latest_sel = rfe.transform(X_latest_scaled)
    X_input = pad_sequence(X_latest_sel, CONFIG["seq_len"]).reshape(1, CONFIG["seq_len"], -1)

    rf_p = rf.predict_proba(X_latest_sel[-1:].reshape(1, -1))[0]
    xgb_p = xgb.predict_proba(X_latest_sel[-1:].reshape(1, -1))[0]
    tft_p = tft(torch.tensor(X_input, dtype=torch.float32)).detach().numpy()[0]

    w_rf = weights.get("rf", 1/3); w_xgb = weights.get("xgb", 1/3); w_tft = weights.get("tft", 1/3)
    probs = {
        "Cut":  round(float(rf_p[0]*w_rf + xgb_p[0]*w_xgb + tft_p[0]*w_tft), 4),
        "Hold": round(float(rf_p[1]*w_rf + xgb_p[1]*w_xgb + tft_p[1]*w_tft), 4),
        "Hike": round(float(rf_p[2]*w_rf + xgb_p[2]*w_xgb + tft_p[2]*w_tft), 4),
    }
    pred = max(probs, key=probs.get)
    result = {
        "Next FOMC": str(next_date.date()),
        "Prediction": pred,
        "Confidence": round(probs[pred]*100, 2),
        "Probabilities": probs
    }
    with open("forecast_result.json", "w") as f:
        json.dump(result, f)
    print("📊 Forecast:", result)

if __name__ == "__main__":
    main()

