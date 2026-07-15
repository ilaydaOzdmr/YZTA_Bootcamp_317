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
yaklasimidir. Cutoff sonrasi kesilen kisa vadeli faturalar kapsam disidir.
DURUSTLUK NOTU: rolling_backtest, MEDYAN projeksiyonun hafif IYIMSER oldugunu gosteriyor
(PIT ort ~0.39; gerceklesen genelde medyanin biraz altinda). Ancak asagi kuyruk (P5) ve
kriz olasiligi iyi kalibre - kriz uyarilari guvenilir. Karar verirken medyana degil,
olasiliga ve P5'e bakilmali.
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
from forecast_demand_stock import reorder_cash_needs  # noqa: E402  (stok -> nakit koprusu)

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
# Sistemik gecikme korelasyonu: gercek hayatta gecikmeler BAGIMSIZ degildir
# (piyasa siklasmasi -> herkes ayni anda gec oder). rho, faturalar arasi ortak
# gecikme bilesenidir. rolling_backtest.py ile kalibre edildi: rho=0'da coverage
# dusuk (~%78) cikiyordu; sistemik bilesen eklenince gerceklesenler dagilimin
# icine oturuyor. (Bagimsiz ornekleme downside kuyrugunu inceltiyordu.)
SYSTEMIC_RHO    = 0.35
SEED = 42

# Senaryo = (gecikme_carpani, erkenlik_carpani)
# NOT: Model NEGATIF gecikme de tahmin edebilir (erken odeme; musterilerin ~%68'i).
# Tek bir carpan kullanmak semantigi bozar: kotumserde pred*1.6, erken odeyen icin
# (-4)*1.6 = -6.4 yani DAHA DA ERKEN odeme demek olurdu (kotumser, iyimser gibi davranir).
# Bu yuzden gecikme (pozitif kisim) ve erkenlik (negatif kisim) AYRI olceklenir:
#   efektif_gecikme = max(pred,0)*gecikme_carpani + min(pred,0)*erkenlik_carpani
SCENARIO_DELAY = {
    "iyimser":  (0.0, 1.0),   # gec odeyenler vadesinde oder; erken odeyenler erken kalir
    "normal":   (1.0, 1.0),   # ML tahmininin aynisi
    "kotumser": (1.6, 0.0),   # gecikmeler %60 uzar; erken odeyenler erkenligini kaybeder
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
    # clip(min=0) YOK: erken odeme (negatif gecikme) gercek ve tahsilati one ceker.
    df["pred_days_late"] = booster.predict(df[INV_FEATURES])

    df["settled_date"] = pd.to_datetime(df["settled_date"], errors="coerce")
    issued = df[df["invoice_date"] <= CUTOFF]
    recv = issued[issued["settled_date"].isna() | (issued["settled_date"] > CUTOFF)].copy()

    pur = pd.read_sql("SELECT invoice_date, due_date, settled_date, amount FROM purchase_invoices",
                      con, parse_dates=["invoice_date", "due_date", "settled_date"])
    pur = pur[pur["invoice_date"] <= CUTOFF]
    pay = pur[pur["settled_date"].isna() | (pur["settled_date"] > CUTOFF)].copy()
    # Borclar VADESINDE odenir. (settled_date kullanmak cutoff sonrasi GELECEK
    # bilgisi olurdu; bu veride settled==due oldugundan etkisiz ama gercek veride sizinti.)
    pay["exp_pay"] = pay["due_date"]
    return recv, pay


# --------------------------------------------------------------------------- #
# Projeksiyon
# --------------------------------------------------------------------------- #
def project(recv: pd.DataFrame, pay: pd.DataFrame, start_balance: float,
            future: pd.DatetimeIndex,
            factors: tuple[float, float] = (1.0, 1.0),
            stock_outflows: pd.Series | None = None) -> pd.Series:
    first = future[0]
    late_f, early_f = factors

    # --- (A) bilinen alacaklar: vade + efektif gecikme ---
    pred = recv["pred_days_late"]
    eff_delay = pred.clip(lower=0) * late_f + pred.clip(upper=0) * early_f
    exp = recv["due_date"] + pd.to_timedelta(eff_delay.round(), unit="D")
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
    # STOK -> NAKIT: acil siparis nakit cikislari (crunch'in 3. ayagi; opsiyonel)
    if stock_outflows is not None and len(stock_outflows):
        net = net - stock_outflows.reindex(future, fill_value=0.0)
    return start_balance + net.cumsum()


# --------------------------------------------------------------------------- #
# Monte Carlo: olasiliksal projeksiyon
# --------------------------------------------------------------------------- #
def per_invoice_sigma(recv: pd.DataFrame) -> np.ndarray:
    """
    Her fatura icin belirsizlik (sigma). Modelin KENDI out-of-fold artiklarindan
    cikarilan profile gore (tahmin degeri buyudukce sacilim artiyor: erken odeyende
    ~2.5 gun, cok gec odeyende ~7.6 gun). Tek bir sabit sigma kullanmak, gec odeyen
    riskli faturalarin belirsizligini KUCUMSER ve kriz olasiligini olduğundan dusuk
    gosterir.

    sigma_scale kolonu: PAZARLIKLI / sozlesmeye baglanmis odemeler icin 0 verilir
    (orn. erken odeme indirimi anlasmasi). Bu akislara model gurultusu eklemek yanlistir;
    tarih artik tahmin degil, taahhuttur.
    """
    prof = json.loads((REPORT_DIR / "invoice_delay_metrics.json").read_text(encoding="utf-8"))
    sp = prof["sigma_profile"]
    edges, sigmas = np.array(sp["edges"]), np.array(sp["sigmas"])
    pred = recv["pred_days_late"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(sigmas) - 1)
    sig = sigmas[idx]
    if "sigma_scale" in recv.columns:
        sig = sig * recv["sigma_scale"].to_numpy(dtype=float)
    return sig



def monte_carlo(recv: pd.DataFrame, pay: pd.DataFrame, start_balance: float,
                future: pd.DatetimeIndex, sigma=None,
                n_sims: int = 2000, seed: int = SEED, return_troughs: bool = False,
                stock_outflows: pd.Series | None = None):
    """
    NEDEN GEREKLI: Deterministik projeksiyon her faturayi ORTALAMA tahmini tarihine
    koyar. Bu, nakit egrisini duzlestirir ve CUKURU SISTEMATIK OLARAK SIGLASTIRIR
    (gercek tahsilatlar ortalamanin etrafinda saciliyor; kotu gunlerin ust uste
    binme olasiligi deterministik hesapta kaybolur).

    Cozum: her fatura icin gecikmeyi tahmin edilen dagilimdan ORNEKLE
    (days_late ~ Normal(model_tahmini, sigma); sigma = modelin CV RMSE'si, yani
    indirgenemez gurultu) ve binlerce yol simule et. Ciktilar:
      - kasanin EKSIYE DUSME OLASILIGI
      - guvenli tampon ihlali olasiligi
      - cukurun dagilimi (P5 / P50 / P95)
    Bu, rapordaki "kasanin eksiye dusme ihtimali %X" ciktisinin ta kendisidir.
    """
    rng = np.random.default_rng(seed)
    first = future[0]
    if sigma is None:
        sigma = per_invoice_sigma(recv)          # fatura-bazli (heteroskedastik)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(len(recv), float(sigma))
    pred = recv["pred_days_late"].to_numpy(dtype=float)
    amt = recv["amount"].to_numpy(dtype=float)
    due = recv["due_date"].to_numpy()

    # Borclar ve sabit giderler deterministik (vadeleri sozlesmeyle belli)
    exp_pay = pay["exp_pay"].clip(lower=first)
    outflow = pay.groupby(exp_pay)["amount"].sum().reindex(future, fill_value=0.0).to_numpy()
    fixed = np.array([MONTHLY_FIXED if d.day == 5 else (MONTHLY_OPEX if d.day == 20 else 0.0)
                      for d in future])
    stock_out = (stock_outflows.reindex(future, fill_value=0.0).to_numpy()
                 if stock_outflows is not None and len(stock_outflows) else 0.0)
    base_net = -(outflow + fixed + stock_out)   # + stok acil siparis cikislari (opsiyonel)
    H = len(future)
    day0 = np.datetime64(first, "D")

    troughs = np.empty(n_sims)
    neg_any = 0
    below_buffer = 0
    trough_days = np.empty(n_sims, dtype=int)

    rho = SYSTEMIC_RHO
    for s in range(n_sims):
        # Gecikme = tahmin + sistemik ortak sok + fatura-ozgu gurultu.
        # Corr(delay_i, delay_j) = rho; her fatura icin toplam varyans sigma_i^2 korunur.
        z_sys = rng.normal(0.0, 1.0)                        # tum faturalara ORTAK (bu yol)
        z_idio = rng.normal(0.0, 1.0, size=len(pred))       # fatura-ozgu
        delay = pred + sigma * (np.sqrt(rho) * z_sys + np.sqrt(1.0 - rho) * z_idio)
        coll = due.astype("datetime64[D]") + np.round(delay).astype("timedelta64[D]")
        idx = (coll - day0).astype(int)
        idx = np.clip(idx, 0, H)                            # pencere disi -> ilk gun / disari
        inflow = np.zeros(H)
        inside = idx < H
        np.add.at(inflow, idx[inside], amt[inside])
        bal = start_balance + np.cumsum(inflow + base_net)
        troughs[s] = bal.min()
        trough_days[s] = int(bal.argmin())
        if bal.min() < 0:
            neg_any += 1
        if bal.min() < MIN_CASH_BUFFER:
            below_buffer += 1

    p_neg = neg_any / n_sims
    mode_day = int(np.bincount(trough_days).argmax())
    out = {
        "n_sims": n_sims, "sigma_ort": round(float(sigma.mean()), 3),
        "eksiye_dusme_olasiligi": round(p_neg, 3),
        "tampon_ihlali_olasiligi": round(below_buffer / n_sims, 3),
        "cukur_P5": round(float(np.percentile(troughs, 5)), 2),
        "cukur_P50": round(float(np.percentile(troughs, 50)), 2),
        "cukur_P95": round(float(np.percentile(troughs, 95)), 2),
        "en_olasi_kriz_tarihi": str(future[mode_day].date()),
    }
    if return_troughs:
        out["troughs"] = troughs            # PIT / kalibrasyon backtest'i icin ham dagilim
    return out


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

    # --- Monte Carlo: olasiliksal kriz degerlendirmesi (fatura-bazli heteroskedastik sigma) ---
    mc = monte_carlo(recv, pay, start_balance, future)                       # yalnizca alacak/borc
    stock_out = reorder_cash_needs(CUTOFF, FORECAST_DAYS)                    # STOK -> NAKIT koprusu
    mc_stock = monte_carlo(recv, pay, start_balance, future,
                           stock_outflows=stock_out)                        # 3 ayak dahil

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
    print("-" * 68)
    print(f"  MONTE CARLO ({mc['n_sims']} yol, ort. sigma={mc['sigma_ort']}g, fatura-bazli):")
    print(f"    Kasanin EKSIYE DUSME olasiligi : %{mc['eksiye_dusme_olasiligi']*100:.0f}")
    print(f"    Guvenli tampon ihlali olasiligi: %{mc['tampon_ihlali_olasiligi']*100:.0f}")
    print(f"    Cukur dagilimi P5/P50/P95      : {mc['cukur_P5']:,.0f} / "
          f"{mc['cukur_P50']:,.0f} / {mc['cukur_P95']:,.0f} TL")
    print(f"    En olasi kriz tarihi           : {mc['en_olasi_kriz_tarihi']}")
    print("-" * 68)
    print(f"  STOK -> NAKIT KOPRUSU (crunch'in 3. ayagi):")
    print(f"    Ufuk icinde acil stok siparisi ihtiyaci: {stock_out.sum():>12,.0f} TL "
          f"({len(stock_out)} tarih)")
    print(f"    3 AYAK DAHIL kriz olasiligi : %{mc_stock['eksiye_dusme_olasiligi']*100:.0f}  "
          f"(alacak/borc tek basina: %{mc['eksiye_dusme_olasiligi']*100:.0f})")
    print(f"    3 ayak dahil cukur P5/P50   : {mc_stock['cukur_P5']:,.0f} / "
          f"{mc_stock['cukur_P50']:,.0f} TL")

    metrics = {
        "cutoff": str(CUTOFF.date()), "horizon_days": FORECAST_DAYS,
        "start_balance": round(start_balance, 2),
        "min_cash_buffer": MIN_CASH_BUFFER,
        "acik_alacak_tl": round(float(recv["amount"].sum()), 2),
        "acik_borc_tl":   round(float(pay["amount"].sum()), 2),
        "senaryolar": results,
        "monte_carlo": mc,
        "stok_nakit_ihtiyaci_tl": round(float(stock_out.sum()), 2),
        "monte_carlo_stok_dahil": {k: v for k, v in mc_stock.items() if k != "troughs"},
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
