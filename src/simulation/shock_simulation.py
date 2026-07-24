"""
ResilienceOS - Sok & Counterfactual Simulasyon Motoru
Aksiyon ve dis soklarin nakit akisina etkisini Monte Carlo ile olcer.
Cikti: reports/shock_scenarios.json, reports/shock_projection_curves.csv
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "digital_twin.db"
REPORT_DIR = ROOT / "reports"
INV_PRED   = REPORT_DIR / "invoice_delay_predictions.csv"

FORECAST_DAYS       = 90
MIN_CASH_BUFFER     = 100_000
SUPPLIER_DEFER_DAYS = 75
DISCOUNT_RATE       = 0.02
FACTORING_FEE       = 0.05
CAMPAIGN_TOP_N      = 3
CUSTOMER_DELAY_DAYS = 15
FX_SHOCK_DEFAULT    = 0.10
DAYS_LATE_SIGMA     = 5.0

_FX_SHOCK = FX_SHOCK_DEFAULT


def load_start_balance(con) -> tuple[float, pd.Timestamp]:
    bal = pd.read_sql(
        "SELECT date, balance_after FROM bank_ledger ORDER BY date, txn_id",
        con, parse_dates=["date"],
    ).groupby("date")["balance_after"].last()
    last_date = bal.index[-1]
    cutoff = last_date - pd.Timedelta(days=60)
    # En yakın mevcut tarihe snap
    cutoff = bal.index[bal.index <= cutoff][-1]
    start = float(bal.loc[cutoff])
    return start, cutoff



def load_known_book(con, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    if INV_PRED.exists():
        recv = pd.read_csv(INV_PRED, parse_dates=["due_date"])
        recv = recv.rename(columns={"predicted_days_late": "pred_days_late"})
    else:
        recv = pd.read_sql(
            "SELECT invoice_id, customer_id, amount, due_date "
            "FROM sales_invoices WHERE status='open'",
            con, parse_dates=["due_date"],
        )
        recv["pred_days_late"] = 0.0
    recv["sigma_scale"] = 1.0
    recv = recv[recv["due_date"] > cutoff].reset_index(drop=True)

    pay = pd.read_sql(
        "SELECT pinvoice_id, amount, due_date FROM purchase_invoices WHERE due_date > ?",
        con, params=(cutoff.date().isoformat(),), parse_dates=["due_date"],
    ).rename(columns={"due_date": "exp_pay"})


    return recv, pay


def fx_shock_from_macro(con, stress_multiple: float = 2.0) -> float:
    try:
        m = pd.read_sql(
            "SELECT date, usdtry FROM macro_daily ORDER BY date",
            con, parse_dates=["date"],
        )
        worst = float(m["usdtry"].pct_change(30).max())
        if not np.isfinite(worst) or worst <= 0:
            return FX_SHOCK_DEFAULT
        return round(min(worst * stress_multiple, 0.30), 3)
    except Exception:
        return FX_SHOCK_DEFAULT


def _settle(recv: pd.DataFrame) -> pd.Series:
    return recv["due_date"] + pd.to_timedelta(
        recv["pred_days_late"].round(0).astype(int), unit="D"
    )


def project(
    recv: pd.DataFrame,
    pay: pd.DataFrame,
    start: float,
    future: pd.DatetimeIndex,
) -> pd.Series:
    settle_map = recv.assign(settle=_settle(recv)).groupby("settle")["amount"].sum()
    pay_map    = pay.groupby("exp_pay")["amount"].sum()
    balance, result = start, {}
    for d in future:
        balance += settle_map.get(d, 0.0) - pay_map.get(d, 0.0)
        result[d] = balance
    return pd.Series(result)


def monte_carlo(
    recv: pd.DataFrame,
    pay: pd.DataFrame,
    start: float,
    future: pd.DatetimeIndex,
    n_sims: int = 2000,
) -> dict:
    rng = np.random.default_rng(42)
    troughs = []
    for _ in range(n_sims):
        r = recv.copy()
        noise = rng.normal(0, DAYS_LATE_SIGMA, size=len(r)) * r["sigma_scale"].values
        r["pred_days_late"] = (r["pred_days_late"] + noise).clip(lower=0)
        troughs.append(float(project(r, pay, start, future).min()))
    arr = np.array(troughs)
    return {
        "eksiye_dusme_olasiligi": float((arr < 0).mean()),
        "cukur_P5":  float(np.percentile(arr, 5)),
        "cukur_P50": float(np.percentile(arr, 50)),
    }


def _ensure_scale(df: pd.DataFrame) -> pd.DataFrame:
    if "sigma_scale" not in df.columns:
        df = df.copy()
        df["sigma_scale"] = 1.0
    return df


def s_early_discount(recv, pay):
    r = _ensure_scale(recv.copy())
    idx = r["amount"].idxmax()
    r.loc[idx, "pred_days_late"] = 0.0
    r.loc[idx, "sigma_scale"]    = 0.0
    r.loc[idx, "amount"]        *= (1 - DISCOUNT_RATE)
    return r, pay


def s_collection_campaign(recv, pay):
    r = _ensure_scale(recv.copy())
    riskli = r[r["pred_days_late"] > 0].nlargest(CAMPAIGN_TOP_N, "pred_days_late").index
    r.loc[riskli, "pred_days_late"] = 0.0
    r.loc[riskli, "sigma_scale"]    = 0.0
    r.loc[riskli, "amount"]        *= (1 - DISCOUNT_RATE)
    return r, pay


def s_factoring(recv, pay, cutoff: pd.Timestamp):
    r = _ensure_scale(recv.copy())
    idx = r["amount"].idxmax()
    r.loc[idx, "due_date"]      = cutoff + pd.Timedelta(days=1)
    r.loc[idx, "pred_days_late"] = 0.0
    r.loc[idx, "sigma_scale"]    = 0.0
    r.loc[idx, "amount"]        *= (1 - FACTORING_FEE)
    return r, pay


def s_split_supplier(recv, pay):
    k   = pay.copy()
    idx = k["amount"].idxmax()
    half   = k.loc[idx, "amount"] / 2
    second = k.loc[idx].copy()
    k.loc[idx, "amount"] = half
    second["amount"]  = half
    second["exp_pay"] = k.loc[idx, "exp_pay"] + pd.Timedelta(days=SUPPLIER_DEFER_DAYS)
    return recv, pd.concat([k, pd.DataFrame([second])], ignore_index=True)


def s_combined(recv, pay, cutoff: pd.Timestamp):
    r, k = s_early_discount(recv, pay)
    return s_split_supplier(r, k)


def s_collection_slowdown(recv, pay):
    r = recv.copy()
    r["pred_days_late"] = r["pred_days_late"] + CUSTOMER_DELAY_DAYS
    return r, pay


def s_fx_shock(recv, pay):
    k   = pay.copy()
    idx = k["amount"].idxmax()
    k.loc[idx, "amount"] *= (1 + _FX_SHOCK)
    return recv, k


def status_of(trough: float) -> str:
    if trough < 0:
        return "KRIZ"
    if trough < MIN_CASH_BUFFER:
        return "RISKLI"
    return "GUVENLI"


def main() -> None:
    global _FX_SHOCK
    REPORT_DIR.mkdir(exist_ok=True)

    if not DB_PATH.exists():
        print(f"HATA: {DB_PATH} bulunamadi.")
        return

    con = sqlite3.connect(DB_PATH)
    start_balance, CUTOFF = load_start_balance(con)
    recv0, pay0 = load_known_book(con, CUTOFF)
    _FX_SHOCK = fx_shock_from_macro(con)
    con.close()

    future = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

    SCENARIOS: dict[str, tuple] = {
        "0_baseline":           ("Baseline (aksiyon yok)",
                                 lambda r, p: (r, p), False),
        "1_erken_odeme":        ("ABC erken odeme (%2 indirim)",
                                 s_early_discount, True),
        "2_tahsilat_kampanya":  (f"Tahsilat kampanyasi (en riskli {CAMPAIGN_TOP_N} fatura)",
                                 s_collection_campaign, True),
        "3_faktoring":          ("Buyuk faturayi faktore sat (%5 iskonto)",
                                 lambda r, p: s_factoring(r, p, CUTOFF), True),
        "4_tedarikci_bolme":    (f"Tedarikci odemesini 2 taksite bol (+{SUPPLIER_DEFER_DAYS}g, Q4'e)",
                                 s_split_supplier, True),
        "5_kombine_cozum":      ("KOMBINE COZUM (erken odeme + tedarikci bolme)",
                                 lambda r, p: s_combined(r, p, CUTOFF), True),
        "6_tahsilat_yavaslama": (f"SOK: tahsilat yavaslamasi (+{CUSTOMER_DELAY_DAYS}g)",
                                 s_collection_slowdown, False),
        "7_kur_soku":           (f"SOK: kur +%{_FX_SHOCK*100:.0f} (EVDS'den)",
                                 s_fx_shock, False),
    }

    curves  = pd.DataFrame({"date": future.strftime("%Y-%m-%d")})
    results = []
    for key, (label, fn, is_action) in SCENARIOS.items():
        r, p  = fn(recv0.copy(), pay0.copy())
        proj  = project(r, p, start_balance, future)
        trough = float(proj.min())
        mc    = monte_carlo(r, p, start_balance, future, n_sims=2000)
        curves[key] = proj.values
        results.append({
            "key": key, "label": label, "is_action": is_action,
            "trough":       round(trough, 0),
            "trough_date":  str(proj.idxmin().date()),
            "status":       status_of(trough),
            "kriz_olasiligi":  mc["eksiye_dusme_olasiligi"],
            "cukur_P5":        mc["cukur_P5"],
            "cukur_P50":       mc["cukur_P50"],
            "crisis_resolved": bool(mc["eksiye_dusme_olasiligi"] < 0.10),
        })

    base = next(x for x in results if x["key"] == "0_baseline")
    best = min((x for x in results if x["is_action"]), key=lambda x: x["kriz_olasiligi"])

    print("=" * 84)
    print(f"SOK & COUNTERFACTUAL SIMULASYON  (cutoff: {CUTOFF.date()}, ufuk: {FORECAST_DAYS}g)")
    print("=" * 84)
    print(f"  Baslangic bakiye: {start_balance:>12,.0f} TL  |  "
          f"Acik alacak: {len(recv0)}  |  Acik odeme: {len(pay0)}")
    print(f"  {'Senaryo':<46}{'Beklenen kasa':>15}{'KRIZ OLAS.':>12}{'Durum':>9}")
    print("-" * 84)
    for grp, title in [(True, "AKSIYONLAR"), (False, "SOKLAR")]:
        print(f"  [{title}]")
        for x in results:
            if x["key"] == "0_baseline":
                if not grp:
                    continue
            elif x["is_action"] != grp:
                continue
            print(f"  {x['label']:<46}{x['trough']:>11,.0f} TL"
                  f"{x['kriz_olasiligi']*100:>9.0f}%{x['status']:>9}")
    print("-" * 84)
    print(f"  Aksiyon yok    : kriz olasiligi %{base['kriz_olasiligi']*100:.0f}")
    print(f"  En iyi aksiyon : {best['label']} -> %{best['kriz_olasiligi']*100:.0f}")
    print(f"\n  Ozet  -> reports/shock_scenarios.json")
    print(f"  Egri  -> reports/shock_projection_curves.csv")

    (REPORT_DIR / "shock_scenarios.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    curves.to_csv(REPORT_DIR / "shock_projection_curves.csv", index=False)


if __name__ == "__main__":
    main()
