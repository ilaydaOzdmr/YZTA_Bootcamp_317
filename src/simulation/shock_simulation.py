"""
ResilienceOS - Sok & Counterfactual Simulasyon Motoru
=====================================================
Nakit projeksiyonunun (forecast_cashflow v3) uzerine AKSIYON senaryolari uygular ve
darbogazi cozer. Rapordaki "Aksiyonlari Uygula" motoru: ajanlarin onerecegi aksiyonlar
uygulaninca dijital ikiz yeniden calisir ve krizin kapandigi gosterilir.

forecast_cashflow senaryolari "gecikme faktoru" (iyimser/normal/kotumser) uzerineydi;
buradakiler ISLETMENIN ALABILECEGI AKSIYONLAR ve DIS SOKLAR:

  0. BASELINE                : mevcut tahmin (kriz)
  1. ABC'ye erken odeme (%2) : buyuk riskli alacagi vadesinde tahsil et (2% indirimle)
  2. Tedarikci odemesini bol : buyuk odemeyi 2 taksite yay (2. taksit +10 gun,
                               pencere ICINDE kalmali - bkz. README olcum artefakti notu)
  3. Kur +%10 sok            : ithal ham madde odemesi artar -> krizi DERINLESTIRIR
  4. KOMBINE COZUM (1+2)     : ajanlarin onerecegi aksiyon paketi

Calistirma:  python src/simulation/shock_simulation.py
Cikti:       reports/shock_scenarios.json, reports/shock_projection_curves.csv
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from forecast_cashflow import (  # noqa: E402
    CUTOFF, FORECAST_DAYS, MIN_CASH_BUFFER, DB_PATH, REPORT_DIR,
    known_book, load_daily_balance, project, monte_carlo,
)

# ONEMLI: oteleme suresi TAHMIN PENCERESI ICINDE kalmali. Aksi halde ertelenen taksit
# olcum penceresinin disina duser ve senaryo, krizi COZMUS gibi gorunur - oysa sadece
# odemeyi gorus alanindan cikarmis olur (olcum artefakti). 17 Haz + 10 gun = 27 Haz,
# pencere 30 Haz'da bitiyor -> taksit sayiliyor, iyilesme GERCEK.
# 10 gun, tedarikcilerin sozlesmesel esneklik payi (3-15 gun) icinde.
SUPPLIER_DEFER_DAYS = 10
DISCOUNT_RATE       = 0.02   # %2 erken odeme indirimi
FX_SHOCK            = 0.10   # kur +%10


# --- Aksiyon donusumleri: (recv, pay) -> (recv', pay') ---
# Not: sabit tutar esigi yerine EN BUYUK alacak/odeme hedeflenir (tutarlar degisse de bozulmaz).
def s_early_discount(recv, pay):
    """
    ABC'ye %2 erken odeme indirimi: vadesinde tahsil (gecikme 0), tutar %2 dusuk.

    ONEMLI: Bu bir PAZARLIK / SOZLESME sonucudur - tahsilat tarihi artik model
    tahmini degil, TAAHHUTtur. Bu yuzden Monte Carlo'da bu faturaya model
    gurultusu (sigma) EKLENMEZ: sigma_scale = 0.
    (Aksi halde "anlastik ama yine de %20 ihtimalle gec oder" gibi anlamsiz bir
     sonuc cikiyor ve aksiyon siralamasi bozuluyordu.)
    """
    r = recv.copy()
    if "sigma_scale" not in r.columns:
        r["sigma_scale"] = 1.0
    idx = r["amount"].idxmax()
    r.loc[idx, "pred_days_late"] = 0.0
    r.loc[idx, "sigma_scale"] = 0.0          # sozlesmeli tarih -> belirsizlik yok
    r.loc[idx, "amount"] = r.loc[idx, "amount"] * (1 - DISCOUNT_RATE)
    return r, pay


def s_split_supplier(recv, pay):
    """Buyuk ham madde odemesini 2 taksite yay; 2. taksit esneklik payi kadar otelenir."""
    k = pay.copy()
    idx = k["amount"].idxmax()
    half = k.loc[idx, "amount"] / 2
    second = k.loc[idx].copy()
    k.loc[idx, "amount"] = half
    second["amount"] = half
    second["exp_pay"] = k.loc[idx, "exp_pay"] + pd.Timedelta(days=SUPPLIER_DEFER_DAYS)
    return recv, pd.concat([k, pd.DataFrame([second])], ignore_index=True)


def s_fx_shock(recv, pay):
    """Kur +%10: ithal ham madde odemesi artar (downside riski)."""
    k = pay.copy()
    idx = k["amount"].idxmax()
    k.loc[idx, "amount"] = k.loc[idx, "amount"] * (1 + FX_SHOCK)
    return recv, k


def s_combined(recv, pay):
    r, k = s_early_discount(recv, pay)
    return s_split_supplier(r, k)


SCENARIOS = {
    "0_baseline":             ("Baseline (aksiyon yok)", lambda r, p: (r, p)),
    "1_erken_odeme":          ("ABC erken odeme (%2 indirim)", s_early_discount),
    "2_tedarikci_bolme":      (f"Tedarikci odemesini 2 taksite bol (+{SUPPLIER_DEFER_DAYS}g)",
                               s_split_supplier),
    "3_kur_soku":             ("Kur +%10 sok (downside)", s_fx_shock),
    "4_kombine_cozum":        ("KOMBINE COZUM (1+2)", s_combined),
}


def status_of(trough: float) -> str:
    """Beklenen (nokta) tahmine gore durum. Asil karar KRIZ OLASILIGI ile verilir."""
    if trough < 0:
        return "KRIZ"                    # beklenen kasa eksiye duser
    if trough < MIN_CASH_BUFFER:
        return "RISKLI"                  # pozitif ama guvenli tampon alti
    return "GUVENLI"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    recv0, pay0 = known_book(con)
    balance = load_daily_balance(con)
    con.close()

    start_balance = float(balance.loc[:CUTOFF].iloc[-1])
    future = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

    curves = pd.DataFrame({"date": future})
    results = []
    for key, (label, fn) in SCENARIOS.items():
        r, p = fn(recv0.copy(), pay0.copy())
        proj = project(r, p, start_balance, future, (1.0, 1.0))   # normal senaryo
        curves[key] = proj.values
        trough = float(proj.min())
        # OLASILIKSAL degerlendirme: aksiyon KRIZ OLASILIGINI ne kadar dusuruyor?
        # fatura-bazli heteroskedastik sigma; n_sims cashflow ile AYNI (tek kaynak)
        mc = monte_carlo(r, p, start_balance, future, n_sims=2000)
        results.append({
            "key": key, "label": label,
            "trough": round(trough, 0), "trough_date": str(proj.idxmin().date()),
            "status": status_of(trough),
            "kriz_olasiligi": mc["eksiye_dusme_olasiligi"],
            "cukur_P5": mc["cukur_P5"], "cukur_P50": mc["cukur_P50"],
            "crisis_resolved": bool(trough >= 0),
            "safe": bool(trough >= MIN_CASH_BUFFER),
        })

    base = next(x for x in results if x["key"] == "0_baseline")["trough"]
    print("=" * 74)
    print(f"SOK & COUNTERFACTUAL SIMULASYON  (bugun {CUTOFF.date()}, tampon "
          f"{MIN_CASH_BUFFER/1000:.0f}K TL)")
    print("=" * 74)
    print(f"{'Senaryo':<42}{'Beklenen kasa':>15}{'KRIZ OLASILIGI':>16}{'Durum':>12}")
    print("-" * 88)
    for x in results:
        print(f"{x['label']:<42}{x['trough']:>11,.0f} TL "
              f"{x['kriz_olasiligi']*100:>13.0f}%  {x['status']:>11}")
    print("-" * 88)
    bp = next(x for x in results if x["key"] == "0_baseline")["kriz_olasiligi"]
    bestx = min((x for x in results if x["key"] not in ("0_baseline", "3_kur_soku")),
                key=lambda x: x["kriz_olasiligi"])
    print(f"  >> Aksiyon yok    : kasanin eksiye dusme olasiligi %{bp*100:.0f}")
    print(f"  >> En iyi aksiyon : {bestx['label']} -> %{bestx['kriz_olasiligi']*100:.0f}")

    (REPORT_DIR / "shock_scenarios.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    curves.to_csv(REPORT_DIR / "shock_projection_curves.csv", index=False)
    print("\nOzet -> reports/shock_scenarios.json  |  Egriler -> reports/shock_projection_curves.csv")


if __name__ == "__main__":
    main()
