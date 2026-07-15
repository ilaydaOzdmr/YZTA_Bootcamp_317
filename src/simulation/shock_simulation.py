"""
ResilienceOS - Sok & Counterfactual Simulasyon Motoru
=====================================================
Nakit projeksiyonunun (forecast_cashflow) uzerine AKSIYON ve SOK senaryolari uygular;
her senaryonun KRIZ OLASILIGINI (Monte Carlo) hesaplar. Rapordaki "Aksiyonlari Uygula"
motoru: ajanlarin onerecegi aksiyonlar uygulaninca kriz olasiligi nasil degisiyor?

Senaryolar iki gruba ayrilir:
  AKSIYONLAR (isletmenin alabilecegi onlemler; kriz olasiligini DUSURUR):
    - ABC'ye %2 erken odeme indirimi (pazarlikli tahsilat)
    - Coklu tahsilat kampanyasi (en riskli birkac faturaya erken odeme tesviki)
    - Buyuk faturayi faktore satma (faktoring; ani nakit, iskontolu)
    - Tedarikci odemesini 2 taksite bolme
    - Kombine cozum
  SOKLAR (dis riskler; kriz olasiligini ARTIRIR - downside testi):
    - Genel tahsilat yavaslamasi (tum alacaklar +15 gun; resesyon/piyasa siklasmasi)
    - Kur soku (ithal ham madde odemesi artar) - GERCEK EVDS volatilitesinden turetilir

Calistirma:  python src/simulation/shock_simulation.py
Cikti:       reports/shock_scenarios.json, reports/shock_projection_curves.csv
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from forecast_cashflow import (  # noqa: E402
    CUTOFF, FORECAST_DAYS, MIN_CASH_BUFFER, DB_PATH, REPORT_DIR,
    known_book, load_daily_balance, project, monte_carlo,
)

# Oteleme suresi TAHMIN PENCERESI ICINDE kalmali (bkz. README olcum artefakti notu).
# 10 gun, tedarikci sozlesmelerindeki esneklik payi (3-15 gun) icinde.
SUPPLIER_DEFER_DAYS   = 10
DISCOUNT_RATE         = 0.02      # %2 erken odeme indirimi
FACTORING_FEE         = 0.05      # faktoring iskontosu (buyuk fatura ani nakde donerken)
CAMPAIGN_TOP_N        = 3         # tahsilat kampanyasinda kac fatura
CUSTOMER_DELAY_DAYS   = 15        # genel tahsilat yavaslamasi soku (tum alacaklar +15g)
FX_SHOCK_DEFAULT      = 0.10      # kur soku (EVDS'den turetilemezse yedek)

_FX_SHOCK = FX_SHOCK_DEFAULT      # main() icinde gercek EVDS verisinden guncellenir


# --------------------------------------------------------------------------- #
# Kur sokunu GERCEK makro veriye baglama
# --------------------------------------------------------------------------- #
def fx_shock_from_macro(con, stress_multiple: float = 2.0) -> float:
    """
    Kur sok buyuklugunu macro_daily'deki GERCEK USD/TRY hareketinden turetir:
    tarihsel en kotu 30-gunluk yukselisin `stress_multiple` kati (stres senaryosu).
    Boylece "kur %10 artarsa" sabiti yerine veriye dayali bir sok kullanilir ve
    toplanan EVDS verisi simulasyona GERCEKTEN girer.
    """
    try:
        m = pd.read_sql("SELECT date, usdtry FROM macro_daily ORDER BY date", con,
                        parse_dates=["date"])
        worst_30d = float(m["usdtry"].pct_change(30).max())
        if not np.isfinite(worst_30d) or worst_30d <= 0:
            return FX_SHOCK_DEFAULT
        return round(min(worst_30d * stress_multiple, 0.30), 3)   # makul ust sinir %30
    except Exception:
        return FX_SHOCK_DEFAULT


# --------------------------------------------------------------------------- #
# Aksiyon / sok donusumleri: (recv, pay) -> (recv', pay')
# Not: sabit tutar esigi yerine EN BUYUK alacak/odeme hedeflenir.
# --------------------------------------------------------------------------- #
def _ensure_scale(df):
    if "sigma_scale" not in df.columns:
        df = df.copy()
        df["sigma_scale"] = 1.0
    return df


def s_early_discount(recv, pay):
    """ABC'ye %2 erken odeme indirimi. PAZARLIK -> tahsilat tarihi taahhut,
    model gurultusu eklenmez (sigma_scale=0)."""
    r = _ensure_scale(recv.copy())
    idx = r["amount"].idxmax()
    r.loc[idx, "pred_days_late"] = 0.0
    r.loc[idx, "sigma_scale"] = 0.0
    r.loc[idx, "amount"] *= (1 - DISCOUNT_RATE)
    return r, pay


def s_collection_campaign(recv, pay):
    """En riskli (yuksek gecikme tahminli) ilk N acik faturaya erken odeme tesviki.
    Tek kalem yerine BIRDEN COK musteriye tahsilat baskisi - gercekci kampanya."""
    r = _ensure_scale(recv.copy())
    riskli = r[r["pred_days_late"] > 0].nlargest(CAMPAIGN_TOP_N, "pred_days_late").index
    r.loc[riskli, "pred_days_late"] = 0.0
    r.loc[riskli, "sigma_scale"] = 0.0
    r.loc[riskli, "amount"] *= (1 - DISCOUNT_RATE)
    return r, pay


def s_factoring(recv, pay):
    """Buyuk faturayi FAKTORE satma: ufkun ilk gunu ISKONTOLU ani nakit girisi.
    (Tahsilat riskini faktore devreder; belirsizlik yok.)"""
    r = _ensure_scale(recv.copy())
    idx = r["amount"].idxmax()
    r.loc[idx, "pred_days_late"] = -999.0        # project() clip(lower=first) -> ilk gun
    r.loc[idx, "sigma_scale"] = 0.0
    r.loc[idx, "amount"] *= (1 - FACTORING_FEE)
    return r, pay


def s_split_supplier(recv, pay):
    """Buyuk ham madde odemesini 2 esit taksite yay; 2. taksit +10 gun. Toplam korunur."""
    k = pay.copy()
    idx = k["amount"].idxmax()
    half = k.loc[idx, "amount"] / 2
    second = k.loc[idx].copy()
    k.loc[idx, "amount"] = half
    second["amount"] = half
    second["exp_pay"] = k.loc[idx, "exp_pay"] + pd.Timedelta(days=SUPPLIER_DEFER_DAYS)
    return recv, pd.concat([k, pd.DataFrame([second])], ignore_index=True)


def s_collection_slowdown(recv, pay):
    """SOK: genel tahsilat yavaslamasi (resesyon / piyasa siklasmasi).
    TUM acik alacaklar +CUSTOMER_DELAY_DAYS gun daha gec odenir - alacak tarafi
    sistemik downside testi. (Tek musteri mikro senaryosu bu veride etkisizdi;
    cunku en buyuk alacak zaten cukur tarihinden sonra tahsil ediliyor. Sistemik
    yavaslama hem gercekci bir makro sok hem de counterfactual olarak anlamli.)"""
    r = recv.copy()
    r["pred_days_late"] = r["pred_days_late"] + CUSTOMER_DELAY_DAYS
    return r, pay


def s_fx_shock(recv, pay):
    """SOK: kur artarsa ithal ham madde (en buyuk) odemesi artar. Sok buyuklugu
    _FX_SHOCK (gercek EVDS volatilitesinden turetilir)."""
    k = pay.copy()
    idx = k["amount"].idxmax()
    k.loc[idx, "amount"] *= (1 + _FX_SHOCK)
    return recv, k


def s_combined(recv, pay):
    """Ajanlarin onerecegi aksiyon paketi: erken odeme + tedarikci bolme."""
    r, k = s_early_discount(recv, pay)
    return s_split_supplier(r, k)


# key -> (etiket, fonksiyon, is_action)
SCENARIOS = {
    "0_baseline":        ("Baseline (aksiyon yok)", lambda r, p: (r, p), False),
    "1_erken_odeme":     ("ABC erken odeme (%2 indirim)", s_early_discount, True),
    "2_tahsilat_kampanya": (f"Tahsilat kampanyasi (en riskli {CAMPAIGN_TOP_N} fatura)",
                            s_collection_campaign, True),
    "3_faktoring":       ("Buyuk faturayi faktore sat (%5 iskonto)", s_factoring, True),
    "4_tedarikci_bolme": (f"Tedarikci odemesini 2 taksite bol (+{SUPPLIER_DEFER_DAYS}g)",
                          s_split_supplier, True),
    "5_kombine_cozum":   ("KOMBINE COZUM (erken odeme + tedarikci bolme)", s_combined, True),
    "6_tahsilat_yavaslama": (f"SOK: genel tahsilat yavaslamasi (tum alacaklar +{CUSTOMER_DELAY_DAYS}g)",
                             s_collection_slowdown, False),
    "7_kur_soku":        ("SOK: kur artisi (ithal odeme)", s_fx_shock, False),
}


def status_of(trough: float) -> str:
    """Beklenen (nokta) tahmine gore durum. Asil karar KRIZ OLASILIGI ile verilir."""
    if trough < 0:
        return "KRIZ"
    if trough < MIN_CASH_BUFFER:
        return "RISKLI"
    return "GUVENLI"


def main() -> None:
    global _FX_SHOCK
    REPORT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    recv0, pay0 = known_book(con)
    balance = load_daily_balance(con)
    _FX_SHOCK = fx_shock_from_macro(con)          # gercek EVDS verisinden
    con.close()

    start_balance = float(balance.loc[:CUTOFF].iloc[-1])
    future = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

    # etiketi guncel FX sok yuzdesiyle zenginlestir
    SCENARIOS["7_kur_soku"] = (f"SOK: kur +%{_FX_SHOCK*100:.0f} (ithal odeme, EVDS'den)",
                               s_fx_shock, False)

    curves = pd.DataFrame({"date": future})
    results = []
    for key, (label, fn, is_action) in SCENARIOS.items():
        r, p = fn(recv0.copy(), pay0.copy())
        proj = project(r, p, start_balance, future, (1.0, 1.0))
        curves[key] = proj.values
        trough = float(proj.min())
        mc = monte_carlo(r, p, start_balance, future, n_sims=2000)
        results.append({
            "key": key, "label": label, "is_action": is_action,
            "trough": round(trough, 0), "trough_date": str(proj.idxmin().date()),
            "status": status_of(trough),
            "kriz_olasiligi": mc["eksiye_dusme_olasiligi"],
            "cukur_P5": mc["cukur_P5"], "cukur_P50": mc["cukur_P50"],
            # nokta tahmini pozitif OLSA BILE olasilik yuksekse kriz cozulmus sayilmaz
            "crisis_resolved": bool(mc["eksiye_dusme_olasiligi"] < 0.10),
        })

    base = next(x for x in results if x["key"] == "0_baseline")
    print("=" * 84)
    print(f"SOK & COUNTERFACTUAL SIMULASYON  (bugun {CUTOFF.date()}, ufuk {FORECAST_DAYS}g)")
    print("=" * 84)
    print(f"{'Senaryo':<48}{'Beklenen kasa':>15}{'KRIZ OLAS.':>12}{'Durum':>9}")
    print("-" * 84)
    for grp, title in [(True, "AKSIYONLAR (kriz olasiligini dusurur)"),
                       (False, "SOKLAR / DOWNSIDE (kriz olasiligini artirir)")]:
        print(f"  -- {title} --")
        for x in results:
            if x["is_action"] != grp and x["key"] != "0_baseline":
                continue
            if x["key"] == "0_baseline" and not grp:
                continue
            print(f"  {x['label']:<46}{x['trough']:>11,.0f} TL{x['kriz_olasiligi']*100:>9.0f}%"
                  f"{x['status']:>9}")
    print("-" * 84)
    best = min((x for x in results if x["is_action"]), key=lambda x: x["kriz_olasiligi"])
    print(f"  >> Aksiyon yok    : kriz olasiligi %{base['kriz_olasiligi']*100:.0f}")
    print(f"  >> En iyi aksiyon : {best['label']} -> %{best['kriz_olasiligi']*100:.0f}")

    (REPORT_DIR / "shock_scenarios.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    curves.to_csv(REPORT_DIR / "shock_projection_curves.csv", index=False)
    print("\nOzet -> reports/shock_scenarios.json  |  Egriler -> reports/shock_projection_curves.csv")


if __name__ == "__main__":
    main()
