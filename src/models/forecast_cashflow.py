"""Mimari: LightGBM ile gunluk net_flow tahmini -> cumsum ile bakiye projeksiyonu.
3 senaryo: iyimser / normal / kotumser (acik fatura gecikme carpani ile).
Ciktilar:
  models/cashflow_model.txt
  reports/cashflow_forecast.csv   (gunluk: 3 senaryo)
  reports/cashflow_metrics.json"""
from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "digital_twin.db"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH    = MODEL_DIR / "cashflow_model.txt"
METRICS_PATH  = REPORT_DIR / "cashflow_metrics.json"
FORECAST_PATH = REPORT_DIR / "cashflow_forecast.csv"
INV_PRED_PATH = REPORT_DIR / "invoice_delay_predictions.csv"

FORECAST_HORIZON = 90
MONTHLY_FIXED    = 150_000     # kira + maas (ayin 5'i)
SEED = 42

SCENARIO_DELAY = {
    "optimistic":  0.0,    # tum tahsilatlar zamaninda gelir
    "normal":      1.0,    # model tahmini kadar gecikme
    "pessimistic": 1.6,    # tahminlenen gecikmenin %60 daha uzunu
}

# Veri Yükleme
def load_daily_cashflow() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    ledger = pd.read_sql("SELECT * FROM bank_ledger ORDER BY date", con)
    con.close()

    ledger["date"]   = pd.to_datetime(ledger["date"])
    ledger["signed"] = np.where(ledger["type"] == "in",
                                 ledger["amount"], -ledger["amount"])

    daily = (
        ledger.groupby("date")
        .agg(
            net_flow  = ("signed",        "sum"),
            balance   = ("balance_after",  "last"),
            total_in  = ("signed",         lambda x: x[x > 0].sum()),
            total_out = ("signed",         lambda x: x[x < 0].abs().sum()),
            n_tx      = ("txn_id",         "count"),
        )
        .reset_index()
    )

    # Tam takvim (boş günleri doldur)
    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
             .reindex(full_idx)
             .rename_axis("date")
             .reset_index()
    )
    daily["net_flow"]  = daily["net_flow"].fillna(0.0)
    daily["balance"]   = daily["balance"].ffill()
    daily["total_in"]  = daily["total_in"].fillna(0.0)
    daily["total_out"] = daily["total_out"].fillna(0.0)
    daily["n_tx"]      = daily["n_tx"].fillna(0.0)
    return daily


def load_open_invoice_flows(delay_factor: float = 1.0) -> pd.DataFrame:
    """Acik fatura beklenen tahsilat tarihlerini yukler."""
    if INV_PRED_PATH.exists():
        df = pd.read_csv(INV_PRED_PATH, parse_dates=["due_date"])
        df["adj_days"] = (df["predicted_days_late"] * delay_factor).round(0)
        df["settle"]   = df["due_date"] + pd.to_timedelta(df["adj_days"], unit="D")
        flows = (
            df.groupby("settle")
            .agg(expected_inflow=("amount", "sum"))
            .reset_index()
            .rename(columns={"settle": "date"})
        )
    else:
        con = sqlite3.connect(DB_PATH)
        inv = pd.read_sql("SELECT * FROM sales_invoices WHERE status='open'", con)
        con.close()
        inv["date"] = pd.to_datetime(inv["due_date"]) + pd.Timedelta(days=5)
        flows = (
            inv.groupby("date")
            .agg(expected_inflow=("amount", "sum"))
            .reset_index()
        )
    return flows

# Feature Engineering
def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()

    # Lag'lar
    for lag in [1, 3, 7, 14, 30]:
        df[f"nf_lag{lag}"]  = df["net_flow"].shift(lag)
        df[f"bal_lag{lag}"] = df["balance"].shift(lag)

    # Rolling istatistikler
    for w in [7, 14, 30, 60]:
        df[f"nf_roll{w}_mean"] = df["net_flow"].rolling(w).mean()
        df[f"nf_roll{w}_std"]  = df["net_flow"].rolling(w).std().fillna(0)

    df["bal_roll7_min"]   = df["balance"].rolling(7).min()
    df["bal_roll30_mean"] = df["balance"].rolling(30).mean()

    # Cyclical encoding
    df["sin_dow"]   = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["sin_dom"]   = np.sin(2 * np.pi * df["date"].dt.day / 31)
    df["cos_dom"]   = np.cos(2 * np.pi * df["date"].dt.day / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Sabit gider bayraklari
    df["is_salary_day"] = (df["date"].dt.day == 5).astype(int)
    df["is_op_day"]     = (df["date"].dt.day == 20).astype(int)
    df["is_weekend"]    = (df["date"].dt.dayofweek >= 5).astype(int)

    df["in_out_ratio"] = df["total_in"] / (df["total_out"] + 1.0)
    df["n_tx"]         = df["n_tx"]

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = (
    [f"nf_lag{l}"  for l in [1, 3, 7, 14, 30]]
    + [f"bal_lag{l}" for l in [1, 3, 7, 14, 30]]
    + [f"nf_roll{w}_mean" for w in [7, 14, 30, 60]]
    + [f"nf_roll{w}_std"  for w in [7, 14, 30, 60]]
    + ["bal_roll7_min", "bal_roll30_mean"]
    + ["sin_dow", "cos_dow", "sin_dom", "cos_dom", "sin_month", "cos_month"]
    + ["is_salary_day", "is_op_day", "is_weekend"]
    + ["in_out_ratio", "n_tx"]
    # NOT: total_in / total_out FEATURE'dan cikarildi.
    # Model sadece tekrarlayan gunluk akis desenini ogrenir.
    # Acik fatura tahsilati ve sabit giderler forecast_scenario icinde
    # model tahminine AYRI OLARAK eklenir (cift sayma engellenir).
)
TARGET = "net_flow"

# Model Egitimi
def train_model(df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    """
    Egitim:
      - net_flow 1.-99. percentil araligina kirpilir (outlier => model bozmasini engeller)
      - 5-fold zaman serisi CV
    """
    p01 = float(df[TARGET].quantile(0.01))
    p99 = float(df[TARGET].quantile(0.99))
    df_clean = df.copy()
    df_clean[TARGET] = df_clean[TARGET].clip(p01, p99)

    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET]

    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "learning_rate":     0.03,
        "num_leaves":        31,
        "min_child_samples": 10,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.85,
        "bagging_freq":      5,
        "lambda_l2":         0.05,
        "verbose":          -1,
        "seed":              SEED,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_list, mae_list, best_iters = [], [], []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_val, label=y_val, reference=dt)
        m  = lgb.train(params, dt, num_boost_round=600,
                       valid_sets=[dv],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(0)])
        yp = m.predict(X_val)
        rmse_list.append(float(np.sqrt(mean_squared_error(y_val, yp))))
        mae_list.append(float(mean_absolute_error(y_val, yp)))
        best_iters.append(m.best_iteration)

    # Final model (tum temizlenmis veri)
    best_n = int(np.mean(best_iters))
    final  = lgb.train(params, lgb.Dataset(X, label=y),
                       num_boost_round=best_n,
                       callbacks=[lgb.log_evaluation(0)])

    metrics = {
        "model":           "LightGBM Time-Series v3 (outlier-robust, net_flow)",
        "target":          "net_flow (TL/gun)",
        "clip_range":      [round(p01, 2), round(p99, 2)],
        "n_train":         len(df_clean),
        "cv_folds":        5,
        "best_iter_avg":   best_n,
        "cv_rmse_mean":    round(float(np.mean(rmse_list)), 2),
        "cv_rmse_std":     round(float(np.std(rmse_list)), 2),
        "cv_mae_mean":     round(float(np.mean(mae_list)), 2),
    }
    return final, metrics

# 90 Gunluk Senaryo Projeksiyonu
def forecast_scenario(
    model: lgb.Booster,
    history: pd.DataFrame,
    open_flows: pd.DataFrame,
    scenario: str,
) -> list[dict]:
    """
    Iteratif ileriye projeksiyon — net_flow tahmini + cumsum.

    Her adimda:
      1. Gecmis tampona yeni beklenen gun satiri eklenir.
      2. Ozellikler hesaplanir.
      3. Model net_flow tahmin eder.
      4. Beklenen tahsilat + sabit giderler net_flow'a eklenir.
      5. Bakiye = onceki bakiye + bu gunun toplam net_flow'u.
    """
    last_date = history["date"].max()

    # Başlangıç bakiyesi: son bilinen bakiye kullan.
    # Negatifse (crunch sonrasi) son 60 günün medyanı
    last_balance = float(history["balance"].iloc[-1])
    if last_balance < 0:
        last_balance = float(history["balance"].tail(60).median())

    # Tahsilat haritas: tarih -> TL
    flow_map: dict[pd.Timestamp, float] = {}
    for _, row in open_flows.iterrows():
        d = pd.Timestamp(row["date"])
        flow_map[d] = flow_map.get(d, 0.0) + float(row["expected_inflow"])

    def fixed_outflow(d: pd.Timestamp) -> float:
        out = 0.0
        if d.day == 5:  out += MONTHLY_FIXED
        if d.day == 20: out += 12_000
        return out

    rows    = []
    buf     = history.copy()
    balance = last_balance

    for i in range(FORECAST_HORIZON):
        d      = last_date + timedelta(days=i + 1)
        d_ts   = pd.Timestamp(d)
        exp_in = flow_map.get(d_ts, 0.0)
        exp_out= fixed_outflow(d_ts)

        # Gecici satir: sifir in/out ile (model sadece pattern ogrenecek)
        tmp_row = pd.DataFrame({
            "date":      [d_ts],
            "net_flow":  [0.0],      # placeholder, asagida guncellenir
            "balance":   [balance],
            "total_in":  [0.0],
            "total_out": [0.0],
            "n_tx":      [1 if (exp_in + exp_out) > 0 else 0],
        })
        combined = pd.concat([buf, tmp_row], ignore_index=True)
        feat_row = build_features(combined).iloc[[-1]]

        # Eksik feature kontrolu
        for col in FEATURE_COLS:
            if col not in feat_row.columns:
                feat_row[col] = 0.0

        # Model: sadece tekrarlayan gunluk pattern'i tahmin eder
        # (maas gunu dip, hafta sonu dusuk, vb.)
        model_nf = float(model.predict(feat_row[FEATURE_COLS])[0])

        # Toplam günlük net akış:
        #   model_nf  = tekrarlayan pattern (kira/maas disindaki dalgalanma)
        #   exp_in    = açık fatura tahsilatı (senaryo bazlı)
        #   exp_out   = sabit giderler (kira + maas)
        total_nf = model_nf + exp_in - exp_out

        # Bakiye kumule et
        balance = balance + total_nf

        row_data = {
            "date":               d.isoformat()[:10],
            "scenario":           scenario,
            "model_net_flow":     round(model_nf, 2),
            "expected_inflow":    round(exp_in, 2),
            "expected_outflow":   round(exp_out, 2),
            "total_net_flow":     round(total_nf, 2),
            "predicted_balance":  round(balance, 2),
            "is_negative":        int(balance < 0),
            "is_low":             int(0 <= balance < 100_000),
        }
        rows.append(row_data)

        # Tampon guncelle (bir sonraki iterasyon icin)
        tmp_row["net_flow"] = total_nf
        tmp_row["balance"]  = balance
        buf = pd.concat([buf, tmp_row], ignore_index=True)

    return rows


def run_all_scenarios(
    model: lgb.Booster,
    history: pd.DataFrame,
) -> pd.DataFrame:
    all_rows = []
    for scenario, factor in SCENARIO_DELAY.items():
        open_flows = load_open_invoice_flows(delay_factor=factor)
        rows = forecast_scenario(model, history, open_flows, scenario)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)

# Kriz Analizi
def crunch_summary(forecast: pd.DataFrame) -> dict:
    summary = {}
    for sc in ["optimistic", "normal", "pessimistic"]:
        sub = forecast[forecast["scenario"] == sc]
        if sub.empty:
            continue
        min_idx = sub["predicted_balance"].idxmin()
        summary[sc] = {
            "min_balance":   round(float(sub["predicted_balance"].min()), 2),
            "min_date":      str(sub.loc[min_idx, "date"]),
            "negative_days": int(sub["is_negative"].sum()),
            "low_days":      int(sub["is_low"].sum()),
            "total_expected_inflow": round(float(sub["expected_inflow"].sum()), 2),
        }
    return summary




def main() -> None:
    print("=" * 65)
    print("  ResilienceOS - 90 Gunluk Nakit Akisi Projeksiyonu  (v3)")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"\n  HATA: {DB_PATH} bulunamadi.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("\n[1/5] Banka defteri yukleniyor...")
    history = load_daily_cashflow()
    raw_last_bal = float(history["balance"].iloc[-1])
    start_bal    = raw_last_bal if raw_last_bal >= 0 else float(history["balance"].tail(60).median())
    print(f"  {len(history):,} gunluk kayit  "
          f"({history['date'].min().date()} .. {history['date'].max().date()})")
    print(f"  Son bakiye (ham)   : {raw_last_bal:>12,.0f} TL")
    print(f"  Baslangic bakiyesi : {start_bal:>12,.0f} TL"
          + (" (medyan kullanildi)" if raw_last_bal < 0 else ""))

    print("[2/5] Zaman serisi ozellikleri olusturuluyor...")
    df_feat = build_features(history)
    p01 = float(history["net_flow"].quantile(0.01))
    p99 = float(history["net_flow"].quantile(0.99))
    print(f"  {len(df_feat):,} satir, {len(FEATURE_COLS)} ozellik  |  "
          f"net_flow clip: [{p01:,.0f}, {p99:,.0f}] TL")

    print("[3/5] LightGBM egitiliyor (5-fold TSCV, outlier-robust)...")
    model, metrics = train_model(df_feat)
    print(f"  CV RMSE: {metrics['cv_rmse_mean']:,.0f} TL/gun  "
          f"(+/- {metrics['cv_rmse_std']:,.0f})  |  "
          f"MAE: {metrics['cv_mae_mean']:,.0f} TL/gun")

    print("[4/5] 3 senaryo projeksiyonu uretiliyor "
          "(iyimser / normal / kotumser)...")
    forecast = run_all_scenarios(model, history)
    crunch   = crunch_summary(forecast)
    metrics["scenario_analysis"] = crunch

    print("[5/5] Kaydediliyor...")
    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    forecast.to_csv(FORECAST_PATH, index=False)

    # özet 
    print("\n" + "=" * 65)
    print("  SONUCLAR - 3 SENARYO KARSILASTIRMASI")
    print("=" * 65)
    hdr = (f"  {'Senaryo':<14} {'Min Bakiye (TL)':>18}  {'Tarih':<12}  "
           f"{'Neg.Gun':>8}  {'Dusuk.Gun':>10}  {'Beklenen Tahsilat':>18}")
    print(hdr)
    print(f"  {'-'*80}")
    for sc in ["optimistic", "normal", "pessimistic"]:
        s    = crunch.get(sc, {})
        flag = " <-- KRIZ" if s.get("negative_days", 0) > 0 else ""
        print(f"  {sc:<14} {s.get('min_balance',0):>18,.0f}  "
              f"{str(s.get('min_date','')):<12}  "
              f"{s.get('negative_days',0):>8}  "
              f"{s.get('low_days',0):>10}  "
              f"{s.get('total_expected_inflow',0):>18,.0f}"
              f"{flag}")

    print(f"\n  Projeksiyon -> {FORECAST_PATH.name}")
    print(f"  Model       -> {MODEL_PATH.name}")
    print(f"  Metrikler   -> {METRICS_PATH.name}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
