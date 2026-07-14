"""
ResilienceOS - Kalibrasyon Dogrulamasi
======================================
Sentetik dijital ikizin fatura gecikme dagilimini, GERCEK Kaggle veri setleriyle
(IBM Finance Factoring + HighRadius Payment Date) karsilastirir. Rapordaki
"gercek dagilima kalibre edildi" iddiasini kanitlar/olcer.

Calistirma:  python src/analysis/validate_calibration.py  (once fetch_kaggle.py)
Cikti:       reports/calibration_validation.json
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "data" / "kaggle"
IBM = K / "finance-factoring-ibm-late-payment-histories" / "WA_Fn-UseC_-Accounts-Receivable.csv"
HR = K / "payment-date-prediction-for-invoices-dataset" / "dataset.csv"
DB = ROOT / "data" / "digital_twin.db"


def stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return {
        "n": int(len(x)),
        "mean": round(float(x.mean()), 2),
        "std": round(float(x.std()), 2),
        "p50": round(float(x.quantile(0.50)), 1),
        "p90": round(float(x.quantile(0.90)), 1),
        "pct_late": round(float((x > 0).mean() * 100), 1),
    }


def main() -> None:
    (ROOT / "reports").mkdir(exist_ok=True)
    out = {}

    # --- Gercek 1: IBM Finance Factoring (DaysLate dogrudan) ---
    ibm = pd.read_csv(IBM)
    out["ibm_real"] = stats(ibm["DaysLate"])

    # --- Gercek 2: HighRadius (gecikme = clear_date - due_in_date, settled) ---
    hr = pd.read_csv(HR)
    cd = pd.to_datetime(hr["clear_date"], errors="coerce")
    dd = pd.to_datetime(hr["due_in_date"], format="%Y%m%d", errors="coerce")
    if dd.isna().mean() > 0.5:                       # format farkliysa infer
        dd = pd.to_datetime(hr["due_in_date"], errors="coerce")
    hr_delay = (cd - dd).dt.days
    out["highradius_real"] = stats(hr_delay)

    # --- Sentetik: digital_twin sales_invoices.days_late ---
    con = sqlite3.connect(DB)
    syn = pd.read_sql_query("SELECT days_late FROM sales_invoices WHERE days_late IS NOT NULL", con)
    con.close()
    out["synthetic"] = stats(syn["days_late"])

    # --- Degerlendirme: ortalama + medyan + %gec, gercek araliginda mi? (sıkı) ---
    real_means = [out["ibm_real"]["mean"], out["highradius_real"]["mean"]]
    real_late = [out["ibm_real"]["pct_late"], out["highradius_real"]["pct_late"]]
    lo, hi = min(real_means), max(real_means)
    syn = out["synthetic"]
    span = max(hi - lo, 5)
    mean_ok = (lo - 0.5 * span) <= syn["mean"] <= (hi + 0.5 * span)
    late_ok = (min(real_late) - 15) <= syn["pct_late"] <= (max(real_late) + 15)  # +/-15 puan tolerans
    median_ok = syn["p50"] <= 3                    # gercek medyan 0 -> sentetik de dusuk olmali
    ok = mean_ok and late_ok and median_ok
    out["assessment"] = {
        "real_mean_range": [lo, hi], "real_pct_late_range": [min(real_late), max(real_late)],
        "synthetic_mean": syn["mean"], "synthetic_pct_late": syn["pct_late"], "synthetic_p50": syn["p50"],
        "mean_ok": bool(mean_ok), "pct_late_ok": bool(late_ok), "median_ok": bool(median_ok),
        "synthetic_plausible": bool(ok),
        "note": ("Sentetik dagilim (ort+medyan+%gec) gercek kaynaklarla uyumlu -> kalibrasyon gecerli."
                 if ok else
                 "Uyumsuzluk var -> SEGMENT_PROFILES yeniden ayarlanmali (medyan/%gec kontrol et)."),
    }

    print("=" * 68)
    print("KALIBRASYON DOGRULAMASI - Fatura Gecikme Dagilimi (gun)")
    print("=" * 68)
    print(f"{'Kaynak':<22}{'n':>8}{'ort':>8}{'std':>8}{'p50':>7}{'p90':>7}{'%gec':>7}")
    print("-" * 68)
    for key in ["ibm_real", "highradius_real", "synthetic"]:
        s = out[key]
        print(f"{key:<22}{s['n']:>8,}{s['mean']:>8}{s['std']:>8}{s['p50']:>7}{s['p90']:>7}{s['pct_late']:>7}")
    print("-" * 68)
    a = out["assessment"]
    print(f"Gercek ort. araligi: {a['real_mean_range']}  |  Sentetik ort: {a['synthetic_mean']}")
    print(f">> {'MAKUL - kalibrasyon gecerli' if a['synthetic_plausible'] else 'ARALIK DISI - ayar gerek'}")
    print(f"   {a['note']}")

    (ROOT / "reports" / "calibration_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOzet -> reports/calibration_validation.json")


if __name__ == "__main__":
    main()
