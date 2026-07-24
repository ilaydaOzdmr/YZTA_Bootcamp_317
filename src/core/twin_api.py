"""
ResilienceOS - Dijital Ikiz API / Arac Katmani
==============================================
Modeller, projeksiyon ve sok simulasyonunu TEK temiz arayuzde toplar. Hem gelecekteki
kendi arayuzumuz hem de Sprint 2 ajanlari (CFO / Tahsilat / Tedarik / Risk Denetcisi)
bu fonksiyonlari cagirir. Framework-bagimsiz (CrewAI, LangChain veya duz Python).

Fonksiyonlar JSON-dostu (dict/list) doner:
  get_overview()          -> genel durum ozeti
  invoice_risk(top_n)     -> acik faturalarin gecikme/risk skoru       [Tahsilat Agent]
  cash_forecast()         -> nakit projeksiyonu + kriz uyarisi         [CFO Agent]
  stock_risk()            -> stok tukenme projeksiyonu                 [Tedarik Agent]
  simulate(scenario)      -> tek senaryo sok sonucu                    [Risk Denetcisi]
  recommend()             -> krizi cozen aksiyon(lar)                  [Orkestrator]
  situation_report()      -> hepsini birlestiren tam durum raporu

On kosul: once pipeline calistirilmis olmali (generate + train + fetch).
Demo:  python src/core/twin_api.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from train_invoice_delay import (  # noqa: E402
    FEATURE_COLS as FEATURES, build_features, load_data as load_invoices,
)
import shock_simulation as shock  # noqa: E402
from shock_simulation import (  # noqa: E402
    load_start_balance, load_known_book, project, monte_carlo,
    FORECAST_DAYS as HORIZON_DAYS, MIN_CASH_BUFFER,
)

DB = ROOT / "data" / "digital_twin.db"
INVOICE_MODEL = ROOT / "models" / "invoice_delay_model.txt"


def _con():
    return sqlite3.connect(DB)


def _projection_inputs():
    """Projeksiyon icin ortak girdiler (bilinen defter + baslangic kasa + cutoff)."""
    con = _con()
    start, cutoff = load_start_balance(con)
    recv, pay = load_known_book(con, cutoff)
    con.close()
    future = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    return recv, pay, start, cutoff, future


def get_overview() -> dict:
    """Genel durum: tablo sayilari + cutoff anindaki kasa."""
    con = _con()
    _, cutoff = load_start_balance(con)
    q = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    bal = con.execute(
        "SELECT balance_after FROM bank_ledger WHERE date<=? ORDER BY date DESC, txn_id DESC LIMIT 1",
        (cutoff.date().isoformat(),)).fetchone()
    result = {
        "cutoff": str(cutoff.date()),
        "musteri": q("customers"), "tedarikci": q("suppliers"), "urun": q("products"),
        "satis_faturasi": q("sales_invoices"), "alis_faturasi": q("purchase_invoices"),
        "guncel_kasa": round(bal[0], 2) if bal else None,
    }
    con.close()
    return result


def invoice_risk(top_n: int = 10) -> list[dict]:
    """Acik faturalarin tahmini gecikmesi (en riskliden). Tahsilat Agent icin."""
    con = _con()
    _, cutoff = load_start_balance(con)
    con.close()
    inv, cust = load_invoices()
    df = build_features(inv, cust)
    booster = lgb.Booster(model_file=str(INVOICE_MODEL))
    df["pred_days_late"] = booster.predict(df[FEATURES]).round(1)
    open_inv = df[(df["status"] == "open") & (df["invoice_date"] <= cutoff)]
    out = open_inv.sort_values("pred_days_late", ascending=False).head(top_n)
    return [{
        "invoice_id": int(r.invoice_id), "customer_id": int(r.customer_id),
        "amount": round(float(r.amount), 2), "due_date": str(r.due_date.date()),
        "disputed": bool(r.disputed), "pred_days_late": float(r.pred_days_late),
    } for r in out.itertuples()]


def cash_forecast() -> dict:
    """Nakit akisi ileri projeksiyonu + OLASILIKSAL kriz degerlendirmesi. CFO Agent icin."""
    recv, pay, start, cutoff, future = _projection_inputs()
    proj = project(recv, pay, start, future)
    trough = float(proj.min())
    mc = monte_carlo(recv, pay, start, future)

    # STOK -> NAKIT KOPRUSU:
    # Tedarik tarafinin (reorder-alti kritik urunler) urettigi plansiz acil siparis
    # nakit ihtiyaci CFO'nun nakit gorunumune eklenir.
    stok_koprusu = None
    stock_path = ROOT / "reports" / "stock_depletion_forecast.csv"
    if stock_path.exists():
        stock_df = pd.read_csv(stock_path)
        kritik = stock_df[stock_df["risk_level"].isin(["KRITIK", "YUKSEK"])]
        if not kritik.empty and "stockout_cost_30d" in kritik.columns:
            acil = float(kritik["stockout_cost_30d"].sum())
            stok_koprusu = {
                "acil_stok_ihtiyaci": round(acil, 2),
                "tarih_sayisi": int(len(kritik)),
                "kriz_olasiligi_3ayak": mc["eksiye_dusme_olasiligi"],
                "cukur_P5_3ayak": mc["cukur_P5"], "cukur_P50_3ayak": mc["cukur_P50"],
            }

    return {
        "start_balance": round(start, 2),
        "trough": round(trough, 2), "trough_date": str(proj.idxmin().date()),
        "kriz_olasiligi": mc["eksiye_dusme_olasiligi"],
        "tampon_ihlali_olasiligi": mc["tampon_ihlali_olasiligi"],
        "cukur_P5": mc["cukur_P5"], "cukur_P50": mc["cukur_P50"], "cukur_P95": mc["cukur_P95"],
        "en_olasi_kriz_tarihi": mc["en_olasi_kriz_tarihi"],
        "crisis": trough < MIN_CASH_BUFFER, "goes_negative": trough < 0,
        "buffer": MIN_CASH_BUFFER,
        "stok_koprusu": stok_koprusu,
        "curve": [{"date": str(d.date()), "balance": round(float(v), 2)}
                  for d, v in proj.items()],
    }


def stock_risk(top_n: int = 10) -> list[dict]:
    """
    Stok tukenme riski. Tedarik Agent icin.

    ONEMLI: Kendi naif gun-stok hesabini YAPMAZ; talep modelinin (LightGBM,
    ozyinelemeli tahmin) urettigi resmi tukenme raporunu okur. Boylece
    (a) tek kaynak, (b) ML gercekten cikti uretir, (c) yeniden siparis
    dinamigi hesaba katilir -> kritik urun dogru sekilde one cikar.
    On kosul: once `python src/models/forecast_demand_stock.py` calistirilmis olmali.
    """
    path = ROOT / "reports" / "stock_depletion_forecast.csv"
    if not path.exists():
        return [{"error": "once forecast_demand_stock.py calistirilmali"}]
    df = pd.read_csv(path)
    risk_order = {"KRITIK": 0, "YUKSEK": 1, "ORTA": 2, "DUSUK": 3, "GUVENLI": 4}
    df["_r"] = df["risk_level"].map(risk_order).fillna(9)
    df = df.sort_values(["_r", "days_to_zero"])
    cols = [c for c in ("product_id", "product_name", "category", "current_stock",
                        "ewma_daily_demand", "days_to_zero", "depletion_date",
                        "order_by_date", "stockout_cost_30d", "risk_level") if c in df.columns]
    return df[cols].head(top_n).to_dict("records")


def simulate(scenario: str) -> dict:
    """Tek sok senaryosunu calistir. Gecerli: shock.SCENARIOS anahtarlari."""
    recv, pay, start, cutoff, future = _projection_inputs()
    scenarios = shock.get_scenarios(cutoff)
    if scenario not in scenarios:
        return {"error": f"gecersiz senaryo. Secenekler: {list(scenarios)}"}
    label, fn, _is_action = scenarios[scenario]
    r, p = fn(recv.copy(), pay.copy())
    proj = project(r, p, start, future)
    trough = float(proj.min())
    mc = monte_carlo(r, p, start, future)
    status = "KRIZ" if trough < 0 else ("RISKLI" if trough < MIN_CASH_BUFFER else "GUVENLI")
    return {"scenario": scenario, "label": label, "trough": round(trough, 2),
            "trough_date": str(proj.idxmin().date()), "status": status,
            "kriz_olasiligi": mc["eksiye_dusme_olasiligi"],
            "cukur_P5": mc["cukur_P5"], "cukur_P50": mc["cukur_P50"]}


def recommend() -> dict:
    """Tum senaryolari calistirip krizi cozen aksiyonlari sirala. Orkestrator icin."""
    recv, pay, start, cutoff, future = _projection_inputs()
    scenarios = shock.get_scenarios(cutoff)
    results = [simulate(k) for k in scenarios]
    base = next(r for r in results if r["scenario"] == "0_baseline")
    # SADECE aksiyonlar onerilir (soklar degil) - is_action bayragiyla (string filtresi degil).
    actions = [r for r in results
               if r["scenario"] != "0_baseline" and scenarios[r["scenario"]][2]]
    # Aksiyonlar KRIZ OLASILIGINA gore siralanir (nokta tahmini degil - olasilik karar verir)
    improving = sorted([a for a in actions if a["kriz_olasiligi"] < base["kriz_olasiligi"]],
                       key=lambda x: (x["kriz_olasiligi"], -x["trough"]))
    return {
        "baseline_trough": base["trough"], "baseline_status": base["status"],
        "baseline_kriz_olasiligi": base["kriz_olasiligi"],
        "onerilen_aksiyonlar": improving,
        "en_iyi": improving[0] if improving else None,
    }


def situation_report() -> dict:
    """Orkestratorun toplayacagi tam durum raporu."""
    return {
        "genel": get_overview(),
        "nakit": cash_forecast() | {"curve": "..."},   # egriyi ozetle
        "riskli_faturalar": invoice_risk(3),
        "stok_riski": stock_risk(top_n=3),
        "aksiyon_onerisi": recommend(),
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=" * 60)
    print("DIJITAL IKIZ DURUM RAPORU  (cutoff", ov["cutoff"], ")")
    print("=" * 60)
    print(f"Kasa: {ov['guncel_kasa']:,.0f} TL | {ov['musteri']} musteri, "
          f"{ov['satis_faturasi']} satis faturasi")
    cf = cash_forecast()
    print(f"\nNAKIT: en dusuk {cf['trough']:,.0f} TL @ {cf['trough_date']} "
          f"-> {'KRIZ UYARISI' if cf['crisis'] else 'guvenli'}")
    print(f"  kriz olasiligi (alacak/borc): %{cf['kriz_olasiligi']*100:.0f}")
    if cf["stok_koprusu"]:
        sk = cf["stok_koprusu"]
        print(f"  + stok koprusu: acil siparis {sk['acil_stok_ihtiyaci']:,.0f} TL "
              f"-> 3-ayak kriz olasiligi %{sk['kriz_olasiligi_3ayak']*100:.0f}")
    print("\nEN RISKLI 3 FATURA:")
    for r in invoice_risk(3):
        print(f"  #{r['invoice_id']} musteri {r['customer_id']} | {r['amount']:,.0f} TL "
              f"-> {r['pred_days_late']} gun gec")
    print("\nSTOK TUKENME (ilk 3, model tabanli):")
    for r in stock_risk(top_n=3):
        print(f"  urun #{r['product_id']} ({r['category']}) -> {r['depletion_date']} "
              f"| {r['days_to_zero']:.0f} gun | risk: {r['risk_level']}")
    print("\nAKSIYON ONERISI:")
    rec = recommend()
    print(f"  Aksiyon yok: kriz olasiligi %{rec['baseline_kriz_olasiligi']*100:.0f}")
    for a in rec["onerilen_aksiyonlar"]:
        print(f"  -> {a['label']}: kriz olasiligi %{a['kriz_olasiligi']*100:.0f} "
              f"({a['trough']:,.0f} TL)")
