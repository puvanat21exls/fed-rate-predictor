# train_v6.py — Clean version for Lightning 2.5 / PyTorch Forecasting 1.4.0

import os, json, joblib, urllib.request, warnings
import numpy as np, pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pytorch_forecasting import TimeSeriesDataSet
import pytorch_lightning as pl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

CONFIG = {
    "random_seed": 42,
    "cv_folds": 5,
    "seq_len": 12,
    "threshold_hike": 0.25,
    "model": {
        "n_features": 6,
        "oversampler": "borderline",     # "borderline" | "adasyn" | None
        "add_gaussian_noise": True,
        "tft": {"hidden_size": 16, "dropout": 0.1, "batch_size": 16, "epochs": 15}
    },
    "fred": {
        "start_date": "2000-01-01",
        "series_ids": ["FEDFUNDS","CPIAUCSL","GDPC1","UNRATE","PCEPI","VIXCLS","T10Y2Y","SP500"]
    },
    "required_cols": ["FEDFUNDS","CPIAUCSL","GDPC1","UNRATE","PCEPI","VIXCLS","T10Y2Y","SP500","sentiment"]
}

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(CONFIG["random_seed"])

def fetch_fred_data():
    dfs = []
    api_key = "4b9232177886c75f573c64f4ebe21c78"  # replace with your key if needed
    for sid in CONFIG["fred"]["series_ids"]:
        url = ( "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={sid}&api_key={api_key}&file_type=json&observation_start={CONFIG['fred']['start_date']}" )
        with urllib.request.urlopen(url) as r:
            js = json.loads(r.read().decode())
        obs = js.get("observations", [])
        df = pd.DataFrame({
            "DATE": [o["date"] for o in obs],
            sid: [float(o["value"]) if o["value"] != "." else np.nan for o in obs],
        })
        df["DATE"] = pd.to_datetime(df["DATE"])
        dfs.append(df.set_index("DATE").resample("MS").mean())
    return pd.concat(dfs, axis=1)



from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def fetch_sentiment_scores(dates):
    model_name = "yiyanghkust/finbert-tone"

    # Explicitly tell HF to load safetensors
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        local_files_only=False,
        use_safetensors=True  # ✅ This forces safetensors usage
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    # You need to replace "placeholder text" with your actual text column
    return [pipe("placeholder text")[0]["score"] for _ in dates]

    # Run the sentiment scoring
    return [pipe("placeholder text")[0]["score"] for _ in dates]  # Replace placeholder with actual text


from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def train_rf_xgb_tft(X, y, df):
    print("Training RandomForest and XGBoost...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X, y)

    xgb_model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    xgb_model.fit(X, y)

    print("Preparing data for TemporalFusionTransformer...")
    max_encoder_length = 12
    max_prediction_length = 3

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)

    print("Initializing TemporalFusionTransformer model...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print("Starting TFT training...")
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(tft, train_dataloader)

    # Save TFT checkpoint
    trainer.save_checkpoint("tft_checkpoint.ckpt")

    return rf_model, xgb_model, tft


def compute_shap(rf_model, xgb_model, X_sel):
    try:
        import shap
        sample = shap.sample(X_sel, 100)
        explainer = shap.KernelExplainer(
            lambda x: 0.5*rf_model.predict_proba(x)+0.5*xgb_model.predict_proba(x),
            sample
        )
        vals = explainer.shap_values(sample)
        shap.summary_plot(vals, sample, show=False)
        plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), bbox_inches="tight")
    except Exception as e:
        print(f"SHAP skipped: {e}")

def compute_weights(rf_model, xgb_model, tft_model, X_sel, y):
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=CONFIG["cv_folds"])
    scores = {"rf": [], "xgb": [], "tft": []}

    def tft_predict_seq(X_hist):
        import torch
        seq_len = CONFIG["seq_len"]
        if len(X_hist) >= seq_len:
            seq = X_hist[-seq_len:]
        else:
            pad = np.repeat(X_hist[:1], seq_len - len(X_hist), axis=0)
            seq = np.concatenate([pad, X_hist], axis=0)
        with torch.no_grad():
            logits = tft_model(torch.tensor(seq.reshape(1, seq_len, -1), dtype=torch.float32))[0].numpy()[0]
        return np.argmax(logits)

    for train_idx, test_idx in tscv.split(X_sel):
        X_test = X_sel[test_idx]
        y_test = y.iloc[test_idx]
        scores["rf"].append(f1_score(y_test, rf_model.predict(X_test), average="macro"))
        scores["xgb"].append(f1_score(y_test, xgb_model.predict(X_test), average="macro"))
        tpreds = [tft_predict_seq(X_sel[:test_idx.start+i+1]) for i in range(len(X_test))]
        scores["tft"].append(f1_score(y_test, tpreds, average="macro"))

    avg = {k: float(np.mean(v)) for k,v in scores.items()}
    total = sum(avg.values()) or 1.0
    weights = {k: round(avg[k]/total, 3) for k in avg}
    json.dump(weights, open(os.path.join(MODEL_DIR, "ensemble_weights.json"), "w"))
    print("Ensemble weights:", weights)
    return weights

def main():
    print("Start training…")
    df = fetch_fred_data()
    df["rate_change"] = df["FEDFUNDS"].diff()
    df["sentiment"] = fetch_sentiment_scores(df.index)
    df.dropna(subset=CONFIG["required_cols"] + ["rate_change"], inplace=True)
    df["y"] = df["rate_change"].apply(lambda x: 2 if x > CONFIG["threshold_hike"] else 0 if x < -CONFIG["threshold_hike"] else 1)

    X = df[CONFIG["required_cols"]]
    y = df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    rfe = RFE(RandomForestClassifier(), n_features_to_select=CONFIG["model"]["n_features"])
    X_sel = rfe.fit_transform(X_scaled, y)
    joblib.dump(rfe, os.path.join(MODEL_DIR, "rfe_selector.pkl"))

    df = df.reset_index().rename(columns={"DATE": "date"})
    df["time_idx"] = range(len(df))
    df["target"] = df["y"]  # Or whatever you want TFT to predict
    
    rf_model, xgb_model, tft_model = train_rf_xgb_tft(X_sel, y, df.assign(group=0))
    compute_shap(rf_model, xgb_model, X_sel)
    compute_weights(rf_model, xgb_model, tft_model, X_sel, y)
    print("Done.")

if __name__ == "__main__":
    main()
