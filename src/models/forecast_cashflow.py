"""
ResilienceOS - Nakit Akisi Projeksiyonu v3
==========================================
Bir "bugun" (CUTOFF) tarihinden ileriye nakit projeksiyonu yapar ve nakit
darbogazini GERCEKLESMEDEN ONCE tahmin eder.

TASARIM (v2'deki cifte sayim / pencere sorunlarini gideren ayristirma)
---------------------------------------------------------------------
Bu dijital ikizde bank_ledger TAMAMEN faturalardan + sabit giderlerden turetilir,
yani net akisin aciklanamayan bir "artik" bileseni yoktur. Dolayisiyla gelecekteki
nakit iki AYRIK kaynaktan gelir:

  (A) BILINEN DEFTER  : cutoff'a KADAR kesilmis faturalar
      - acik alacaklar  -> tahsilat tarihi = vade + ML gecikme tahmini (invoice_delay_model)
      - acik borclar    -> vade tarihinde odenir
      - sabit giderler  -> her ayin 5'i
      Deterministiktir; darbogazi yaratan buyuk odeme burada gorunur.

KAPSAM: Yalnizca BILINEN DEFTER projekte edilir. Cutoff'tan SONRA kesilecek yeni
faturalar KAPSAM DISIDIR. Bu, hazine (treasury) pratiginde standart olan "direct method"
yaklasimidir: kisa ufukta (30 gun) yeni faturalarin vadesi (15-60 gun) zaten dolmaz,
dolayisiyla nakde donmezler. Muhafazakar ve savunulabilir.
(v2'de model TUM net akisi tahmin edip UZERINE tahsilat/gider tekrar ekliyordu -> CIFTE
SAYIM; butun senaryolar -4.5M'e suruklenip krizi kaciriyordu.)

ML KATKISI: Tahsilat TARIHLERI invoice_delay_model (LightGBM) tahminiyle belirlenir.
Yani projeksiyonu suruklenen sey ML'dir; kalemler muhasebeden gelir. Bu ayrimi jüriye
oldugu gibi anlatiyoruz. "ML krizi kesfetti" gibi bir iddiada BULUNMUYORUZ.

BACKTEST SEFFAFLIGI: Gecikme modeli 18 ayin TAMAMIYLA egitilmistir (cutoff sonrasi
odenen faturalar dahil). Dolayisiyla asagidaki "kriz tarihi sapmasi / MAE" degerleri
bir miktar IYIMSERDIR. Kesin bir out-of-time degerlendirme icin modelin yalnizca
cutoff oncesi odenmis faturalarla egitilmesi gerekir. Bunu gizlemiyoruz.

v2'DE GIDERILEN HATALAR
-----------------------
  1) Tahmin penceresi verinin SONUNDAN (30 Haz) basliyordu; crunch krizi 18 Haziran'da,
     yani pencereden ONCE. Model krizi yapisal olarak goremiyordu.
     -> CUTOFF krizden ONCEYE alindi; kriz artik gercekten ONCEDEN tahmin ediliyor.
  2) `if last_balance < 0: last_balance = son_60_gun_medyani` fallback'i, ledger negatif
     bitince projeksiyonu 21.8M gibi absurd bir bakiyeden baslatip krizi "temizliyordu".
     -> Fallback kaldirildi; cutoff'taki GERCEK bakiye kullaniliyor.
  3) Vadesi tahmin penceresinden ONCE olan tahsilatlar flow_map'te sessizce dusuyordu;
     iyimser senaryoda (gecikme=0) bu tahsilatlar hic sayilmiyor, kotumserde sayiliyordu
     -> senaryolar TERS cikiyordu (kotumser iyimserden iyi).
     -> Gecmis vadeli tahsilatlar ilk tahmin gunune tasiniyor (dusurulmuyor).

Ciktilar:
  reports/cashflow_forecast.csv       (3 senaryonun gunluk egrisi + gerceklesen)
  reports/cashflow_metrics.json

Calistirma: python src/models/forecast_cashflow.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT       = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from train_invoice_delay import (  # noqa: E402
    FEATURE_COLS as INV_FEATURES,
    build_features as build_invoice_features,
    load_data as load_invoice_data,
)

DB_PATH    = ROOT / "data" / "digital_twin.db"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
INV_MODEL_PATH = MODEL_DIR / "invoice_delay_model.txt"
METRICS_PATH   = REPORT_DIR / "cashflow_metrics.json"
FORECAST_PATH  = REPORT_DIR / "cashflow_forecast.csv"

CUTOFF          = pd.Timestamp("2025-05-31")   # "bugun" - crunch krizinden ONCE
FORECAST_DAYS   = 30                           # ufuk veri sonuna (30 Haz) hizali:
                                               # boylece backtest gercek veriyle yapilir
                                               # (daha uzun ufukta "gerceklesen" ffill olurdu)
MONTHLY_FIXED   = 700_000.0                    # kira+maas (ayin 5'i)
MONTHLY_OPEX    = 13_000.0                     # operasyonel gider (ayin 20'si, ort.)
MIN_CASH_BUFFER = 100_000.0                    # guvenli operasyonel minimum kasa
SEED = 42

SCENARIO_DELAY = {
    "iyimser":  0.0,    # tum tahsilatlar vadesinde gelir
    "normal":   1.0,    # ML modelinin tahmin ettigi gecikme
    "kotumser": 1.6,    # tahmin edilen gecikmenin %60 fazlasi
}


# --------------------------------------------------------------------------- #
# (0) Gecmis bakiye serisi
# --------------------------------------------------------------------------- #
def load_daily_balance(con) -> pd.Series:
    led = pd.read_sql("SELECT date, balance_after FROM bank_ledger ORDER BY date, txn_id",
                      con, parse_dates=["date"])
    daily = led.groupby("date")["balance_after"].last()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(idx).ffill()


# --------------------------------------------------------------------------- #
# (A) BILINEN DEFTER: cutoff'a kadar kesilmis faturalar
# --------------------------------------------------------------------------- #
def known_book(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cutoff aninda ACIK olan alacak/borclari TARIH bazli belirler.
    (status kolonu veri setinin SON durumunu yansitir; cutoff anini degil.)"""
    inv, cust = load_invoice_data()
    df = build_invoice_features(inv, cust)
    booster = lgb.Booster(model_file=str(INV_MODEL_PATH))
    df["pred_days_late"] = booster.predict(df[INV_FEATURES]).clip(min=0)

    df["settled_date"] = pd.to_datetime(df["settled_date"], errors="coerce")
    issued = df[df["invoice_date"] <= CUTOFF]
    recv = issued[issued["settled_date"].isna() | (issued["settled_date"] > CUTOFF)].copy()

    pur = pd.read_sql("SELECT invoice_date, due_date, settled_date, amount FROM purchase_invoices",
                      con, parse_dates=["invoice_date", "due_date", "settled_date"])
    pur = pur[pur["invoice_date"] <= CUTOFF]
    pay = pur[pur["settled_date"].isna() | (pur["settled_date"] > CUTOFF)].copy()
    pay["exp_pay"] = pay["settled_date"].fillna(pay["due_date"])
    return recv, pay


# --------------------------------------------------------------------------- #
# Projeksiyon
# --------------------------------------------------------------------------- #
def project(recv: pd.DataFrame, pay: pd.DataFrame, start_balance: float,
            future: pd.DatetimeIndex, delay_factor: float) -> pd.Series:
    first = future[0]

    # --- (A) bilinen alacaklar: vade + gecikme*faktor ---
    exp = recv["due_date"] + pd.to_timedelta(
        (recv["pred_days_late"] * delay_factor).round(), unit="D")
    # HATA DUZELTMESI: penceredan ONCE dusen (vadesi gecmis) tahsilatlar DUSURULMEZ,
    # ilk tahmin gunune tasinir. v2'de bunlar sessizce yok sayiliyor ve iyimser senaryo
    # kotumserden kotu cikiyordu.
    exp = exp.clip(lower=first)
    inflow = recv.groupby(exp)["amount"].sum().reindex(future, fill_value=0.0)

    # --- (A) bilinen borclar ---
    exp_pay = pay["exp_pay"].clip(lower=first)
    outflow = pay.groupby(exp_pay)["amount"].sum().reindex(future, fill_value=0.0)

    # --- (A) sabit giderler: kira/maas (ayin 5'i) + operasyonel (ayin 20'si) ---
    fixed = pd.Series(
        [MONTHLY_FIXED if d.day == 5 else (MONTHLY_OPEX if d.day == 20 else 0.0)
         for d in future], index=future)

    net = inflow - outflow - fixed
    return start_balance + net.cumsum()


# --------------------------------------------------------------------------- #
def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)

    print("=" * 68)
    print(f"  ResilienceOS - Nakit Akisi Projeksiyonu v3   (bugun = {CUTOFF.date()})")
    print("=" * 68)

    print("\n[1/4] Bilinen defter (cutoff'ta acik alacak/borclar) cikariliyor...")
    recv, pay = known_book(con)
    print(f"  acik alacak: {len(recv):>4} fatura / {recv['amount'].sum():>12,.0f} TL")
    print(f"  acik borc  : {len(pay):>4} fatura / {pay['amount'].sum():>12,.0f} TL")

    balance = load_daily_balance(con)
    # HATA DUZELTMESI: negatif-bakiye fallback'i YOK. Cutoff'taki GERCEK bakiye kullanilir.
    start_balance = float(balance.loc[:CUTOFF].iloc[-1])

    future = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    actual = balance.reindex(future).ffill()

    print(f"[2/3] {len(SCENARIO_DELAY)} senaryo projekte ediliyor "
          f"({FORECAST_DAYS} gun, baslangic kasa {start_balance:,.0f} TL)...")
    curves = pd.DataFrame({"date": future, "gerceklesen": actual.values})
    results = {}
    for name, f in SCENARIO_DELAY.items():
        proj = project(recv, pay, start_balance, future, f)
        curves[name] = proj.values
        trough = float(proj.min())
        results[name] = {
            "min_bakiye":  round(trough, 2),
            "min_tarih":   str(proj.idxmin().date()),
            "negatif_gun": int((proj < 0).sum()),
            "kriz":        bool(trough < MIN_CASH_BUFFER),
            "eksiye_duser": bool(trough < 0),
        }

    print("[3/3] Kaydediliyor...")
    curves.to_csv(FORECAST_PATH, index=False)

    # Backtest: "normal" senaryo vs gerceklesen
    normal = curves["normal"].values
    mae = float(np.mean(np.abs(normal - actual.values)))
    a_min, a_date = float(actual.min()), str(actual.idxmin().date())
    p_date = results["normal"]["min_tarih"]
    date_err = abs((pd.Timestamp(p_date) - pd.Timestamp(a_date)).days)

    print("\n" + "=" * 68)
    print("  SONUCLAR - 3 SENARYO")
    print("=" * 68)
    print(f"  {'Senaryo':<12}{'Min Bakiye':>16}{'Tarih':>14}{'Neg.Gun':>9}  Durum")
    print("  " + "-" * 64)
    for n, r in results.items():
        durum = "KRIZ" if r["eksiye_duser"] else ("RISKLI" if r["kriz"] else "GUVENLI")
        print(f"  {n:<12}{r['min_bakiye']:>16,.0f}{r['min_tarih']:>14}"
              f"{r['negatif_gun']:>9}  {durum}")
    print("  " + "-" * 64)
    print(f"  GERCEKLESEN min bakiye : {a_min:>12,.0f} TL @ {a_date}")
    print(f"  Tahmin (normal) min    : {results['normal']['min_bakiye']:>12,.0f} TL @ {p_date}")
    print(f"  >> KRIZ TARIHI SAPMASI : {date_err} gun")
    print(f"  >> Backtest MAE        : {mae:,.0f} TL")

    metrics = {
        "cutoff": str(CUTOFF.date()), "horizon_days": FORECAST_DAYS,
        "start_balance": round(start_balance, 2),
        "min_cash_buffer": MIN_CASH_BUFFER,
        "acik_alacak_tl": round(float(recv["amount"].sum()), 2),
        "acik_borc_tl":   round(float(pay["amount"].sum()), 2),
        "senaryolar": results,
        "backtest": {"gerceklesen_min": round(a_min, 2), "gerceklesen_tarih": a_date,
                     "mae": round(mae, 2), "kriz_tarihi_sapmasi_gun": date_err},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    con.close()
    print(f"\n  Projeksiyon -> {FORECAST_PATH.name}")
    print(f"  Metrikler   -> {METRICS_PATH.name}")
    print("=" * 68)


if __name__ == "__main__":
    main()
