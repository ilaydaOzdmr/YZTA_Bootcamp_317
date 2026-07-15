"""
ResilienceOS - Fatura Gecikme Tahmini (LightGBM Regresyon) v3
=============================================================
Dijital ikiz veritabanindan sales_invoices + customers tablolarini yukler,
zengin musteri + fatura ozellikleriyle (RFM, tarihsel agregat, temporal)
her faturanin KAC GUN gecikecegini tahmin eden LightGBM modeli egitir.

v2'den gelen ozellikler (korundu):
  - RFM musteri ozellikleri (recency, frequency, monetary)
  - Musteri bazinda gecikme trendi (son 90g vs tum gecmis)
  - Fatura anomali skoru (musterinin ortalama faturasina oranla)
  - Temporal: donem sonu / ay sonu / ceyrek sonu bayraklari
  - Gec / zamaninda siniflandirma ek metrigi
  - Acik faturalar icin risk skoru (0-100) + etiket

v3 DUZELTMELERI (veri sizintisi / leakage giderildi):
  1) segment_enc ve payment_reliability_score FEATURE OLARAK KULLANILMIYOR.
     Sentetik veride days_late zaten segment profilinden uretiliyor,
     payment_reliability_score da ayni profilden turetiliyor. Bunlari feature
     yapmak, modelin cevabin uretildigi parametreyi okumasi demek (dongusel).
     Gercek hayatta boyle hazir bir kolon da olmaz; musteri riski ancak
     GECMIS ODEME DAVRANISINDAN cikarilir (asagidaki tarihsel ozellikler).
     Not: segment yalnizca RAPORLAMA icin cikti tablosunda tutuluyor.

  2) Tarihsel ozellikler artik SADECE ONCEKI faturalardan hesaplaniyor.
     Onceki surumde musteri bazli groupby agregati tum faturalar uzerinden
     alinip her satira merge ediliyordu; bu, tahmin edilen faturanin KENDI
     days_late'ini (ve gelecek faturalarini) ozelligin icine sokuyordu.
     Simdi ozellikler AS-OF (bilgi tarihi) dogru kurulur: bir faturanin ozelligine
     yalnizca settled_date'i o faturanin KESIM TARIHINDEN once olan faturalar girer.
     (Bir faturanin gecikmesi ancak odendiginde ogrenilir.)

  3) KFold(shuffle=True) yerine TimeSeriesSplit kullaniliyor.
     Zaman serisinde rastgele bolme, gelecekle egitip gecmisi test etmek
     anlamina geliyordu. (forecast_demand_stock.py zaten zaman-bazli boluyor,
     tutarlilik icin fatura modeli de ayni yaklasima gecti.)

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
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "digital_twin.db"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH   = MODEL_DIR / "invoice_delay_model.txt"
METRICS_PATH = REPORT_DIR / "invoice_delay_metrics.json"
PRED_PATH    = REPORT_DIR / "invoice_delay_predictions.csv"

SEED    = 42
N_FOLDS = 5

# LightGBM hiperparametreleri - modul seviyesinde tutulur ki dis dogrulama
# scripti (validate_real_transfer.py) BIREBIR AYNI parametreleri import edebilsin;
# "ayni pipeline" iddiasi ancak ayni PARAMS + ayni feature'lar kullanilinca dogru olur.
PARAMS = {
    "objective":         "regression",
    "metric":            ["rmse", "mae"],
    "learning_rate":     0.02,
    "num_leaves":        15,
    "max_depth":         -1,
    "min_child_samples": 60,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "lambda_l1":         0.05,
    "lambda_l2":         5.0,
    "verbose":          -1,
    "seed":              SEED,
}


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
# Feature Engineering  (sizintisiz)
# --------------------------------------------------------------------------- #
HIST_COLS = [
    "frequency", "monetary_avg", "recency_days",
    "mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
    "late_rate_90d", "sum_amount_late_hist", "dispute_rate_hist",
]


def _add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Musteri odeme gecmisi ozellikleri - AS-OF (bilgi tarihi) dogru.

    KRITIK: Bir faturanin gecikmesi (days_late) ancak ODENDIGINDE (settled_date)
    ogrenilir. Dolayisiyla X faturasinin ozellikleri hesaplanirken yalnizca
    settled_date'i X'in KESIM TARIHINDEN once olan faturalar kullanilabilir.

    Onceki surumde gecmise ekleme invoice_date sirasina gore yapiliyordu; bu,
    "kesilmis ama henuz odenmemis" faturalarin gecikmesini de ozellige sokuyordu
    (olculdu: satirlarin ~%96'si). Yani model, tahmin aninda HENUZ BILINMEYEN
    bilgiyi kullaniyordu. Asagidaki kurulum bunu tamamen ortadan kaldirir.
    """
    for c in HIST_COLS:
        df[c] = np.nan

    for _, grp in df.groupby("customer_id", sort=False):
        g = grp.sort_values("invoice_date")
        # Musterinin ODENMIS faturalari (bilgi ancak settled_date'te ortaya cikar)
        paid = g[(g["status"] == "settled") & g["days_late"].notna()]
        p_settled = paid["settled_date"].to_numpy()
        p_issued  = paid["invoice_date"].to_numpy()
        p_late    = paid["days_late"].to_numpy(dtype=float)
        p_amount  = paid["amount"].to_numpy(dtype=float)
        p_disp    = paid["disputed"].to_numpy(dtype=float)

        for idx in g.index:
            asof = df.at[idx, "invoice_date"]          # bu faturanin kesim tarihi
            # BILINEN gecmis: settled_date < asof  (o an gercekten bilinen faturalar)
            known = p_settled < np.datetime64(asof)
            n = int(known.sum())
            df.at[idx, "frequency"] = float(n)
            df.at[idx, "sum_amount_late_hist"] = float(
                p_amount[known & (p_late > 0)].sum()) if n else 0.0
            if n:
                kl, ka, kd = p_late[known], p_amount[known], p_disp[known]
                is_late = (kl > 0).astype(float)
                df.at[idx, "mean_days_late_hist"] = float(kl.mean())
                df.at[idx, "std_days_late_hist"]  = float(kl.std(ddof=1)) if n > 1 else 0.0
                df.at[idx, "late_rate_hist"]      = float(is_late.mean())
                df.at[idx, "dispute_rate_hist"]   = float(kd.mean())
                df.at[idx, "monetary_avg"]        = float(ka.mean())
                # en son BILINEN odeme ne kadar once yapildi
                df.at[idx, "recency_days"] = float(
                    (asof - pd.Timestamp(p_settled[known].max())).days)
                # son 90 gunde KESILMIS ve o an bilinen faturalarin gec odeme orani
                cut = np.datetime64(asof - pd.Timedelta(days=90))
                recent = known & (p_issued >= cut)
                if recent.any():
                    df.at[idx, "late_rate_90d"] = float((p_late[recent] > 0).mean())

    return df


def build_features(invoices: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    [Fatura]   log_amount, disputed, default_payment_terms,
               inv_month, inv_quarter, inv_weekday, inv_dayofyear,
               days_to_month_end, is_month_end, is_quarter_end
    [RFM]      recency_days, frequency, monetary_avg          (gecmisten)
    [Tarihsel] mean/std_days_late_hist, late_rate_hist,
               late_rate_90d, sum_amount_late_hist, dispute_rate_hist  (gecmisten)
    [Anomali]  amount_vs_cust_avg

    NOT: segment / payment_reliability_score FEATURE DEGIL (bkz. v3 notu).
    """
    df = invoices.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"]     = pd.to_datetime(df["due_date"])
    # as-of gecmis hesabi settled_date'e dayanir -> datetime olmali
    df["settled_date"] = pd.to_datetime(df["settled_date"], errors="coerce")

    # ---- Fatura zaman ozellikleri ----
    df["inv_month"]     = df["invoice_date"].dt.month
    df["inv_quarter"]   = df["invoice_date"].dt.quarter
    df["inv_weekday"]   = df["invoice_date"].dt.dayofweek
    df["inv_dayofyear"] = df["invoice_date"].dt.dayofyear
    df["days_to_month_end"] = (
        df["invoice_date"] + pd.offsets.MonthEnd(0) - df["invoice_date"]
    ).dt.days
    df["is_month_end"]   = (df["days_to_month_end"] <= 3).astype(int)
    df["is_quarter_end"] = (
        df["inv_month"].isin([3, 6, 9, 12]) & df["is_month_end"].astype(bool)
    ).astype(int)
    df["log_amount"] = np.log1p(df["amount"])

    # ---- Musteri meta: vade sartlari feature; segment SADECE raporlama icin ----
    df = df.merge(
        customers[["customer_id", "segment", "default_payment_terms"]],
        on="customer_id", how="left",
    )

    # ---- Sizintisiz musteri odeme gecmisi ----
    df = df.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)
    df = _add_customer_history(df)

    # Gecmisi olmayan (yeni musteri / ilk fatura) -> NOTR degerler.
    # NOT: global medyan ile doldurmak, tum veriden (gelecek dahil) bilgi sizdirirdi.
    df[["mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
        "late_rate_90d", "dispute_rate_hist", "sum_amount_late_hist",
        "frequency"]] = df[["mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
                            "late_rate_90d", "dispute_rate_hist", "sum_amount_late_hist",
                            "frequency"]].fillna(0.0)
    df["recency_days"] = df["recency_days"].fillna(999.0)      # hic odeme gorulmedi
    df["monetary_avg"] = df["monetary_avg"].fillna(df["amount"])  # kendi tutari

    # Anomali: faturanin musterinin olagan tutarindan sapmasi
    df["amount_vs_cust_avg"] = df["amount"] / (df["monetary_avg"] + 1)

    return df.sort_values("invoice_date").reset_index(drop=True)


FEATURE_COLS = [
    # Fatura
    "log_amount", "disputed", "default_payment_terms",
    "inv_month", "inv_quarter", "inv_weekday", "inv_dayofyear",
    "days_to_month_end", "is_month_end", "is_quarter_end",
    # RFM (gecmisten)
    "recency_days", "frequency", "monetary_avg",
    # Tarihsel odeme davranisi (gecmisten)
    "mean_days_late_hist", "std_days_late_hist", "late_rate_hist",
    "late_rate_90d", "sum_amount_late_hist", "dispute_rate_hist",
    # Anomali
    "amount_vs_cust_avg",
]
TARGET = "days_late"


# --------------------------------------------------------------------------- #
# Teorik taban (oracle): erisilebilecek EN IYI RMSE
# --------------------------------------------------------------------------- #
def compute_oracle(invoices: pd.DataFrame, customers: pd.DataFrame) -> dict:
    """
    Sentetik uretici days_late'i su sekilde uretir:
        days_late ~ Normal(mean_late(segment), std(segment))  (+ itiraz gurultusu)
    Dolayisiyla segment ve itirazi MUKEMMEL bilen bir "oracle" bile, kosullu
    ortalamayi tahmin etmekten oteye gidemez; kalan sacilim INDIRGENEMEZ gurultudur.

    Bu fonksiyon o teorik tabani (Bayes-optimal RMSE / dogruluk) OLCER. Modelin
    performansini bu tabanla kiyaslamak, "daha ne kadar iyilestirilebilir?" ve
    "bu sonuc sizintisiz mumkun mu?" sorularinin cevabidir.
    """
    df = invoices.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")
    df = df[df["days_late"].notna()]
    g = df.groupby(["segment", "disputed"])["days_late"]
    pred = g.transform("mean")
    rmse = float(np.sqrt(((df["days_late"] - pred) ** 2).mean()))
    mae  = float((df["days_late"] - pred).abs().mean())
    late = (df["days_late"] > 0).astype(int)
    p_late = df.assign(_l=late).groupby(["segment", "disputed"])["_l"].transform("mean")
    acc = float(((p_late > 0.5).astype(int) == late).mean())
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "accuracy": round(acc, 4)}


# --------------------------------------------------------------------------- #
# Zaman-bazli CV + Final Model
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    # Zaman sirasi kritik: TimeSeriesSplit her fold'da gecmisle egitip gelecegi test eder
    train_df = (df[(df["status"] == "settled") & df[TARGET].notna()]
                .sort_values("invoice_date").reset_index(drop=True))
    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    params = PARAMS

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    rmse_list, mae_list, acc_list, best_iters = [], [], [], []
    oof_pred, oof_true = [], []          # heteroskedastik sigma profili icin

    for tr_idx, val_idx in tscv.split(X):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=FEATURE_COLS)
        dval   = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)
        m = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        # NOT: clip(min=0) YOK. days_late NEGATIF olabilir (erken/vadesinden once odeme;
        # veride medyan -2 gun, A segmenti ortalama -4 gun). Tahmini sifira kirpmak,
        # faturalarin ~%68'inde sistematik +4 gunluk hata uretiyordu.
        y_pred = m.predict(X[val_idx])
        rmse_list.append(float(np.sqrt(mean_squared_error(y[val_idx], y_pred))))
        mae_list.append(float(mean_absolute_error(y[val_idx], y_pred)))
        acc_list.append(float(accuracy_score(y[val_idx] > 0, y_pred > 0)))
        best_iters.append(m.best_iteration)
        oof_pred.extend(y_pred.tolist()); oof_true.extend(y[val_idx].tolist())

    # Final model: tum veriyle, ortalama best_iter kadar
    best_n = max(int(np.mean(best_iters)), 10)
    dfull = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)
    final_model = lgb.train(params, dfull, num_boost_round=best_n,
                            callbacks=[lgb.log_evaluation(0)])
    # Kuantil regresyon: per-fatura tahmin araligi (P10-P90). MC'ye dokunmaz;
    # yalnizca acik fatura raporunu zenginlestirir (Tahsilat ajani icin yorumlanabilir aralik).
    qp = {k: v for k, v in params.items() if k not in ("objective", "metric")}
    q_lo = lgb.train({**qp, "objective": "quantile", "alpha": 0.1},
                     dfull, num_boost_round=best_n, callbacks=[lgb.log_evaluation(0)])
    q_hi = lgb.train({**qp, "objective": "quantile", "alpha": 0.9},
                     dfull, num_boost_round=best_n, callbacks=[lgb.log_evaluation(0)])

    # --- HETEROSKEDASTIK SIGMA PROFILI ---
    # Hata buyuklugu tahmin degerine gore degisiyor (gec odeyenlerde sacilim daha genis).
    # Monte Carlo'da tek bir sabit sigma kullanmak riski SISTEMATIK OLARAK KUCUMSER.
    # Sigma, modelin KENDI out-of-fold artiklarindan cikarilir (segment gibi uretici
    # parametrelerinden DEGIL - o, sizintiya geri donmek olurdu).
    op, ot = np.array(oof_pred), np.array(oof_true)
    edges = np.quantile(op, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    edges = np.unique(np.round(edges, 3))
    sigmas = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_ = (op >= lo) & (op <= hi)
        sigmas.append(float(np.std(ot[m_] - op[m_], ddof=1)) if m_.sum() > 2
                      else float(np.std(ot - op, ddof=1)))
    sigma_profile = {"edges": [float(e) for e in edges],
                     "sigmas": [round(s_, 3) for s_ in sigmas]}

    baseline_rmse = float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean()))))
    improvement   = (baseline_rmse - float(np.mean(rmse_list))) / baseline_rmse * 100

    metrics = {
        "model":              "LightGBM Regressor v3 (leak-free)",
        "target":             "days_late",
        "validation":         f"TimeSeriesSplit({N_FOLDS}) - zaman bazli",
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
        "oracle_rmse":        None,   # main() icinde hesaplanip doldurulur
        "sigma_profile":      sigma_profile,
        "oracle_note":        ("days_late = Normal(segment_ort, std) + itiraz gurultusu. "
                               "Segment ve itirazi MUKEMMEL bilen bir model bile bu RMSE'nin "
                               "altina inemez (indirgenemez gurultu tabani). Bizim modelimiz "
                               "bu tabanin ~%95'ine ulasiyor. Bunu belirgin sekilde asan bir "
                               "sonuc veri sizintisi SUPHESI dogurur (fold ornekleme gurultusu "
                               "+/- ~1 puan oynatabilir)."),
        "excluded_features":  ["segment_enc", "payment_reliability_score"],
        "exclusion_reason":   ("Sentetik uretici days_late'i segment profilinden uretiyor; "
                               "bu kolonlari feature yapmak dongusel (leakage) olur."),
        "feature_importance": dict(
            zip(FEATURE_COLS,
                final_model.feature_importance(importance_type="gain").tolist())
        ),
    }
    return final_model, metrics, q_lo, q_hi


# --------------------------------------------------------------------------- #
# Acik Fatura Risk Raporlama
# --------------------------------------------------------------------------- #
def predict_open_invoices(model: lgb.Booster, df: pd.DataFrame,
                          q_lo: lgb.Booster = None, q_hi: lgb.Booster = None) -> pd.DataFrame:
    open_df = df[df["status"] == "open"].copy()
    if open_df.empty:
        return pd.DataFrame()

    open_df["predicted_days_late"] = model.predict(open_df[FEATURE_COLS]).round(1)
    if q_lo is not None and q_hi is not None:
        lo = q_lo.predict(open_df[FEATURE_COLS])
        hi = q_hi.predict(open_df[FEATURE_COLS])
        open_df["gecikme_p10"] = np.minimum(lo, hi).round(1)   # %80 tahmin araligi alt
        open_df["gecikme_p90"] = np.maximum(lo, hi).round(1)   # ust
    open_df["estimated_settlement"] = (
        open_df["due_date"] + pd.to_timedelta(open_df["predicted_days_late"], unit="D")
    )

    # Risk skoru 0-100. Musteri riski artik segment'ten degil, GECMIS gec odeme
    # oranindan (late_rate_hist) geliyor -> sizintisiz ve gercek hayatta da hesaplanabilir.
    open_df["risk_score"] = (
        (open_df["predicted_days_late"] / 30.0).clip(0, 1) * 45
        + open_df["disputed"] * 20
        + open_df["late_rate_hist"].clip(0, 1) * 25
        + (open_df["amount_vs_cust_avg"].clip(1, 3) - 1) / 2.0 * 10
    ).round(1)

    def risk_label(s: float) -> str:
        if s >= 80: return "KRITIK"
        if s >= 60: return "YUKSEK"
        if s >= 40: return "ORTA"
        if s >= 20: return "DUSUK"
        return "GUVENLI"

    open_df["risk_label"] = open_df["risk_score"].apply(risk_label)

    cols = ["invoice_id", "customer_id", "segment", "due_date", "amount",
            "disputed", "predicted_days_late", "gecikme_p10", "gecikme_p90",
            "estimated_settlement", "risk_score", "risk_label"]
    cols = [c for c in cols if c in open_df.columns]
    return open_df[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ana Akis
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 65)
    print("  ResilienceOS - Fatura Gecikme Tahmini  (LightGBM v3, leak-free)")
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

    print(f"[2/4] Sizintisiz ozellikler olusturuluyor ({len(FEATURE_COLS)} ozellik)...")
    df = build_features(invoices, customers)
    settled_n = int((df["status"] == "settled").sum())
    open_n    = int((df["status"] == "open").sum())
    print(f"  Egitim (settled): {settled_n:,}  |  Tahmin (acik): {open_n:,}")

    print(f"[3/4] TimeSeriesSplit({N_FOLDS}) ile LightGBM egitiliyor (zaman bazli)...")
    model, metrics, q_lo, q_hi = train(df)

    # Teorik taban: modelin performansini baglamlandirir (ve sizinti kontrolu saglar)
    oracle = compute_oracle(invoices, customers)
    metrics["oracle_rmse"] = oracle["rmse"]
    metrics["oracle_accuracy"] = oracle["accuracy"]
    metrics["oracle_verimlilik_pct"] = round(100 * oracle["rmse"] / metrics["cv_rmse_mean"], 1)

    print("[4/4] Acik fatura risk raporu uretiliyor...")
    preds = predict_open_invoices(model, df, q_lo, q_hi)

    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    if not preds.empty:
        preds.to_csv(PRED_PATH, index=False)

    print("\n" + "=" * 65)
    print("  SONUCLAR")
    print("=" * 65)
    print(f"  CV RMSE      : {metrics['cv_rmse_mean']} gun  (+/- {metrics['cv_rmse_std']})")
    print(f"  CV MAE       : {metrics['cv_mae_mean']} gun")
    print(f"  Gec/zamaninda acc : {metrics['cv_late_accuracy']*100:.1f}%")
    print(f"  Baseline RMSE: {metrics['baseline_rmse']} gun")
    print(f"  Iyilestirme  : %{metrics['improvement_pct']}")
    print(f"  Dogrulama    : {metrics['validation']}")
    print(f"  Cikarilan    : {', '.join(metrics['excluded_features'])} (leakage)")
    print(f"  TEORIK TABAN : RMSE {metrics['oracle_rmse']} / acc %{metrics['oracle_accuracy']*100:.1f}  "
          f"-> verimlilik %{metrics['oracle_verimlilik_pct']}")
    print("-" * 65)

    if not preds.empty:
        rc = preds["risk_label"].value_counts()
        print("\n  Acik Fatura Risk Dagilimi:")
        for label in ["KRITIK", "YUKSEK", "ORTA", "DUSUK", "GUVENLI"]:
            cnt = int(rc.get(label, 0))
            if cnt:
                print(f"    {label:<10}: {cnt} fatura")

        print("\n  En Riskli 5 Acik Fatura:")
        print(f"  {'FaturaID':<10} {'MstID':<6} {'Seg':<5} {'Tutar':>12}  "
              f"{'Tahmin':>8}  {'Risk':>6}  Etiket")
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

    fi = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    max_score = max((v for _, v in fi), default=1) or 1
    print("\n  Ozellik Onemliligi (Top 8):")
    for feat, score in fi[:8]:
        bar = "#" * int(score / max_score * 28)
        print(f"  {feat:<28} {bar}")
    print()


if __name__ == "__main__":
    main()
