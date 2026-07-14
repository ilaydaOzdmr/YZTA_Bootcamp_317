"""
ResilienceOS - Fatura Gecikme Tahmini (LightGBM Regresyon) v2
=============================================================
Dijital ikiz veritabanindan sales_invoices + customers tablolarini yukler,
zengin musteri + fatura ozellikleriyle (RFM, tarihsel agregat, temporal)
her faturayi KACGUN gecikecegini tahmin eden LightGBM modeli egitir.

v2 iyilestirmeleri:
  - RFM musteri ozellikleri (recency, frequency, monetary)
  - Musteri bazinda gecikme trendi (son 90g vs tum gecmis)
  - Fatura anomali skoru (musterinin ortalama faturasina oranla)
  - Temporal: donem sonu / ay sonu / ceyrek sonu bayraklari
  - 5-fold KFold CV ile daha guvenilir RMSE
  - Gec / zamaninda siniflandirma ek metrigi

Ciktilar:
  models/invoice_delay_model.txt
  reports/invoice_delay_metrics.json
  reports/invoice_delay_predictions.csv  <- acik faturalar icin tahminler

Calistirma: python src/models/train_invoice_delay.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)
from sklearn.model_selection import KFold

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "digital_twin.db"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH   = MODEL_DIR / "invoice_delay_model.txt"
METRICS_PATH = REPORT_DIR / "invoice_delay_metrics.json"
PRED_PATH    = REPORT_DIR / "invoice_delay_predictions.csv"

SEED    = 42
N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Veri Yukleme
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH)
    invoices  = pd.read_sql("SELECT * FROM sales_invoices", con)
    customers = pd.read_sql("SELECT * FROM customers", con)
    con.close()
    return invoices, customers


# --------------------------------------------------------------------------- #
# Feature Engineering
# --------------------------------------------------------------------------- #
def build_features(invoices: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Zengin ozellik kumesi:

    [Fatura]
      log_amount, disputed, default_payment_terms
      inv_month, inv_quarter, inv_weekday, inv_dayofyear
      days_to_month_end, is_quarter_end, is_month_end

    [Musteri segment]
      segment_enc, payment_reliability_score

    [RFM - gecmis settled faturalardan]
      recency_days       : son odemenin kac gun once yapildigi
      frequency          : toplam odeme sayisi
      monetary_avg       : ortalama fatura tutari

    [Tarihsel agregat]
      mean_days_late_hist   : gecmis ortalama gecikme
      std_days_late_hist    : gecikme std
      late_rate_hist        : gec odeme orani
      late_rate_90d         : son 90 gundeki gec odeme orani (trend)
      sum_amount_late_hist  : gec odenen toplam TL
      dispute_rate_hist     : gecmis itiraz orani

    [Anomali]
      amount_vs_cust_avg    : fatura tutarinin musterinin ortalama tutar'a orani
    """
    df = invoices.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"]     = pd.to_datetime(df["due_date"])

    # ---- Fatura zaman ozellikleri ----
    df["inv_month"]       = df["invoice_date"].dt.month
    df["inv_quarter"]     = df["invoice_date"].dt.quarter
    df["inv_weekday"]     = df["invoice_date"].dt.dayofweek
    df["inv_dayofyear"]   = df["invoice_date"].dt.dayofyear
    df["days_to_month_end"] = df["invoice_date"].apply(
        lambda d: (d + pd.offsets.MonthEnd(0) - d).days
    )
    df["is_month_end"]    = (df["days_to_month_end"] <= 3).astype(int)
    df["is_quarter_end"]  = (
        (df["inv_month"].isin([3, 6, 9, 12])) & df["is_month_end"]
    ).astype(int)
    df["log_amount"] = np.log1p(df["amount"])

    # ---- Musteri birlestir ----
    df = df.merge(
        customers[["customer_id", "segment", "payment_reliability_score",
                   "default_payment_terms"]],
        on="customer_id", how="left",
    )
    seg_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    df["segment_enc"] = df["segment"].map(seg_map).fillna(1)

    # ---- Sadece settled faturalardan tarihsel ozellikler ----
    settled = df[df["status"] == "settled"].copy()
    settled["is_late"] = (settled["days_late"] > 0).astype(int)
    ref_date = df["invoice_date"].max()

    # --- RFM ---
    rfm = settled.groupby("customer_id").agg(
        recency_days   = ("invoice_date", lambda x: (ref_date - x.max()).days),
        frequency      = ("invoice_id",   "count"),
        monetary_avg   = ("amount",        "mean"),
    ).reset_index()

    # --- Tarihsel agregat ---
    hist = settled.groupby("customer_id").agg(
        mean_days_late_hist  = ("days_late", "mean"),
        std_days_late_hist   = ("days_late", "std"),
        late_rate_hist       = ("is_late",   "mean"),
        sum_amount_late_hist = ("amount",    lambda x: x[settled.loc[x.index, "is_late"] == 1].sum()),
        dispute_rate_hist    = ("disputed",  "mean"),
    ).reset_index()

    # Son 90 gun gec odeme trendi
    cutoff_90 = ref_date - pd.Timedelta(days=90)
    settled_90 = settled[settled["invoice_date"] >= cutoff_90]
    late_90 = settled_90.groupby("customer_id")["is_late"].mean().reset_index()
    late_90.rename(columns={"is_late": "late_rate_90d"}, inplace=True)

    # Musteri ortalama fatura (anomali icin)
    cust_avg_amount = settled.groupby("customer_id")["amount"].mean().reset_index()
    cust_avg_amount.rename(columns={"amount": "cust_avg_amount"}, inplace=True)

    # --- Birlestir ---
    for extra in [rfm, hist, late_90, cust_avg_amount]:
        df = df.merge(extra, on="customer_id", how="left")

    # Eksik deger: yeni musteri -> global medyan
    fill_cols = [
        "mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
        "sum_amount_late_hist", "dispute_rate_hist",
        "recency_days", "frequency", "monetary_avg", "late_rate_90d",
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(df[col].median())
    df["cust_avg_amount"] = df["cust_avg_amount"].fillna(df["amount"].median())

    # Anomali: musterinin olagan tutarindan sapma
    df["amount_vs_cust_avg"] = df["amount"] / (df["cust_avg_amount"] + 1)

    return df


FEATURE_COLS = [
    # Fatura
    "log_amount", "disputed", "default_payment_terms",
    "inv_month", "inv_quarter", "inv_weekday", "inv_dayofyear",
    "days_to_month_end", "is_month_end", "is_quarter_end",
    # Musteri
    "segment_enc", "payment_reliability_score",
    # RFM
    "recency_days", "frequency", "monetary_avg",
    # Tarihsel
    "mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
    "late_rate_90d", "sum_amount_late_hist", "dispute_rate_hist",
    # Anomali
    "amount_vs_cust_avg",
]
TARGET = "days_late"


# --------------------------------------------------------------------------- #
# 5-Fold CV + Final Model
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    train_df = df[(df["status"] == "settled") & df[TARGET].notna()].copy()
    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    params = {
        "objective":         "regression",
        "metric":            ["rmse", "mae"],
        "learning_rate":     0.04,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.85,
        "bagging_freq":      5,
        "lambda_l1":         0.05,
        "lambda_l2":         0.1,
        "verbose":          -1,
        "seed":              SEED,
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rmse_list, mae_list, acc_list = [], [], []
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        m = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        y_pred = m.predict(X_val).clip(min=0)
        rmse_list.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        mae_list.append(float(mean_absolute_error(y_val, y_pred)))
        # Binary accuracy: gec (>0) mi degil mi
        acc_list.append(float(accuracy_score(y_val > 0, y_pred > 0)))
        best_iters.append(m.best_iteration)

    # Final model: tum veriyle, ortalama best_iter kadar
    best_n = int(np.mean(best_iters))
    dtrain_full = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)
    final_model = lgb.train(
        params, dtrain_full,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(0)],
    )

    # Baseline: ortalama ile tahmin
    baseline_rmse = float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean()))))
    improvement   = (baseline_rmse - np.mean(rmse_list)) / baseline_rmse * 100

    metrics = {
        "model":              "LightGBM Regressor v2",
        "target":             "days_late",
        "n_samples":          len(train_df),
        "n_features":         len(FEATURE_COLS),
        "cv_folds":           N_FOLDS,
        "best_iteration_avg": best_n,
        "cv_rmse_mean":       round(float(np.mean(rmse_list)), 4),
        "cv_rmse_std":        round(float(np.std(rmse_list)), 4),
        "cv_mae_mean":        round(float(np.mean(mae_list)), 4),
        "cv_late_accuracy":   round(float(np.mean(acc_list)), 4),
        "baseline_rmse":      round(baseline_rmse, 4),
        "improvement_pct":    round(improvement, 2),
        "feature_importance": dict(
            zip(FEATURE_COLS,
                final_model.feature_importance(importance_type="gain").tolist())
        ),
    }
    return final_model, metrics


# --------------------------------------------------------------------------- #
# Acik Fatura Risk Raporlama
# --------------------------------------------------------------------------- #
def predict_open_invoices(
    model: lgb.Booster,
    df: pd.DataFrame,
) -> pd.DataFrame:
    open_df = df[df["status"] == "open"].copy()
    if open_df.empty:
        return pd.DataFrame()

    preds = model.predict(open_df[FEATURE_COLS]).clip(min=0)
    open_df["predicted_days_late"] = preds.round(1)
    open_df["due_date"]            = pd.to_datetime(open_df["due_date"])
    open_df["estimated_settlement"] = (
        open_df["due_date"]
        + pd.to_timedelta(open_df["predicted_days_late"], unit="D")
    )

    # Risk skoru 0-100: gecikme + disputed + segment + tutar agirlikli
    open_df["risk_score"] = (
        (open_df["predicted_days_late"] / 30.0).clip(0, 1) * 45
        + open_df["disputed"] * 20
        + (open_df["segment_enc"] / 3.0) * 25
        + (open_df["amount_vs_cust_avg"].clip(1, 3) - 1) / 2.0 * 10
    ).round(1)

    # Risk etiketi
    def risk_label(s):
        if s >= 80:   return "KRITIK"
        if s >= 60:   return "YUKSEK"
        if s >= 40:   return "ORTA"
        if s >= 20:   return "DUSUK"
        return "GUVENLI"

    open_df["risk_label"] = open_df["risk_score"].apply(risk_label)

    return open_df[[
        "invoice_id", "customer_id", "segment", "due_date", "amount",
        "disputed", "predicted_days_late", "estimated_settlement",
        "risk_score", "risk_label",
    ]].sort_values("risk_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ana Akis
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 65)
    print("  ResilienceOS - Fatura Gecikme Tahmini  (LightGBM v2)")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"\n  HATA: {DB_PATH} bulunamadi.")
        print("  Once: python src/data_generator/generate_digital_twin.py")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("\n[1/4] Veri yukleniyor...")
    invoices, customers = load_data()
    print(f"  Fatura: {len(invoices):,}  |  Musteri: {len(customers):,}")

    print("[2/4] Ozellikler olusturuluyor  ({} ozellik)...".format(len(FEATURE_COLS)))
    df = build_features(invoices, customers)
    settled_n = (df["status"] == "settled").sum()
    open_n    = (df["status"] == "open").sum()
    print(f"  Egitim (settled): {settled_n:,}  |  Tahmin (acik): {open_n:,}")

    print(f"[3/4] {N_FOLDS}-fold KFold CV ile LightGBM egitiliyor...")
    model, metrics = train(df)

    print("[4/4] Acik fatura risk raporu uretiliyor...")
    preds = predict_open_invoices(model, df)

    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not preds.empty:
        preds.to_csv(PRED_PATH, index=False)

    # ----- Ozet Cikti -----
    print("\n" + "=" * 65)
    print("  SONUCLAR")
    print("=" * 65)
    print(f"  CV RMSE      : {metrics['cv_rmse_mean']} gun  "
          f"(+/- {metrics['cv_rmse_std']})")
    print(f"  CV MAE       : {metrics['cv_mae_mean']} gun")
    print(f"  Gec/zamaninda acc : {metrics['cv_late_accuracy']*100:.1f}%")
    print(f"  Baseline RMSE: {metrics['baseline_rmse']} gun")
    print(f"  Iyilestirme  : %{metrics['improvement_pct']}")
    print(f"  Best iter    : {metrics['best_iteration_avg']}")
    print("-" * 65)

    if not preds.empty:
        # Risk dagılımı
        rc = preds["risk_label"].value_counts()
        print("\n  Acik Fatura Risk Dagılımı:")
        for label in ["KRITIK", "YUKSEK", "ORTA", "DUSUK", "GUVENLI"]:
            cnt = rc.get(label, 0)
            if cnt > 0:
                print(f"    {label:<10}: {cnt} fatura")

        print(f"\n  En Riskli 5 Acik Fatura:")
        hdr = f"  {'FaturaID':<10} {'MstID':<6} {'Seg':<5} {'Tutar':>12}  {'Tahmin':>8}  {'Risk':>6}  Etiket"
        print(hdr)
        print(f"  {'-'*65}")
        for _, row in preds.head(5).iterrows():
            print(f"  {row['invoice_id']:<10} {row['customer_id']:<6} "
                  f"{row['segment']:<5} {row['amount']:>12,.0f}  "
                  f"{row['predicted_days_late']:>6.1f}g  "
                  f"{row['risk_score']:>5.1f}  {row['risk_label']}")

    print(f"\n  Model     -> {MODEL_PATH.name}")
    print(f"  Metrikler -> {METRICS_PATH.name}")
    print(f"  Tahminler -> {PRED_PATH.name}")
    print("=" * 65)

    # Ozellik onemliligi
    fi = sorted(metrics["feature_importance"].items(),
                key=lambda x: x[1], reverse=True)
    max_score = max(v for _, v in fi) or 1
    print("\n  Ozellik Onemliligi (Top 8):")
    for feat, score in fi[:8]:
        bar = "#" * int(score / max_score * 28)
        print(f"  {feat:<28} {bar}")
    print()


if __name__ == "__main__":
    main()
