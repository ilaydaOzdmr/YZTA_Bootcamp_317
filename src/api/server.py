"""
ResilienceOS - API Backend (FastAPI)
=====================================
`src/core/twin_api.py`'deki TUM fonksiyonlari HTTP uzerinden servis eder:
  GET  /api/overview          -> genel durum ozeti
  GET  /api/cash-forecast     -> nakit projeksiyonu + kriz uyarisi
  GET  /api/invoice-risk      -> en riskli acik faturalar
  GET  /api/stock-risk        -> stok tukenme riski
  GET  /api/scenarios         -> mevcut sok/aksiyon senaryolarinin listesi
  GET  /api/simulate/{key}    -> tek senaryoyu calistir
  GET  /api/recommend         -> tum senaryolari calistirip en iyi aksiyonu bul

IKI MOD (her endpoint icin ayri ayri):
  - CANLI:   data/digital_twin.db + egitilmis modeller mevcutsa, twin_api'yi
             dogrudan cagirir (gercek zamanli sonuc).
  - OFFLINE: DB/model yoksa reports/ altindaki onceden hesaplanmis ciktilara
             (cashflow_metrics.json, invoice_delay_predictions.csv,
             stock_depletion_forecast.csv, shock_scenarios.json) duser.

Calistirma:
    pip install -r requirements.txt
    uvicorn src.api.server:app --reload --port 8000

Sonra tarayicida ac: http://localhost:8000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FRONTEND_DIR = ROOT / "frontend"
sys.path.insert(0, str(ROOT / "src" / "core"))

app = FastAPI(title="ResilienceOS API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _report(name: str) -> dict | list:
    path = REPORTS / name
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ne canli pipeline ne de reports/{name} bulunamadi. "
                f"Once README'deki 'Hizli Baslangic' adimlarini calistirin."
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _report_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"reports/{name} bulunamadi. Once ilgili modeli calistirin.",
        )
    return pd.read_csv(path)


def _try_live(fn_name: str, *args, **kwargs):
    """twin_api'den bir fonksiyonu guvenle cagirir; hata olursa None doner
    (caller bunu offline moda dusme sinyali olarak kullanir)."""
    try:
        import twin_api  # type: ignore  # src/core/twin_api.py
        fn = getattr(twin_api, fn_name)
        return fn(*args, **kwargs)
    except Exception:
        return None


@app.get("/api/overview")
def overview():
    live = _try_live("get_overview")
    if live is not None:
        live["source"] = "live"
        return live
    metrics = _report("cashflow_metrics.json")
    return {
        "source": "offline-reports",
        "cutoff": metrics["cutoff"],
        "musteri": None, "tedarikci": None, "urun": None,
        "satis_faturasi": None, "alis_faturasi": None,
        "guncel_kasa": round(metrics["start_balance"], 2),
    }


def _scenario_curves_from_csv() -> list[dict]:
    df = _report_csv("cashflow_forecast.csv")
    return [
        {
            "date": r.date,
            "gerceklesen": None if pd.isna(r.gerceklesen) else round(float(r.gerceklesen), 2),
            "iyimser": round(float(r.iyimser), 2),
            "normal": round(float(r.normal), 2),
            "kotumser": round(float(r.kotumser), 2),
        }
        for r in df.itertuples()
    ]


def _offline_cash_forecast() -> dict:
    metrics = _report("cashflow_metrics.json")
    normal = metrics["senaryolar"]["normal"]
    mc = metrics["monte_carlo"]
    scenario_curves = _scenario_curves_from_csv()
    return {
        "source": "offline-reports",
        "cutoff": metrics["cutoff"],
        "start_balance": round(metrics["start_balance"], 2),
        "trough": round(normal["min_bakiye"], 2),
        "trough_date": normal["min_tarih"],
        "kriz_olasiligi": mc["eksiye_dusme_olasiligi"],
        "tampon_ihlali_olasiligi": mc["tampon_ihlali_olasiligi"],
        "cukur_P5": mc["cukur_P5"], "cukur_P50": mc["cukur_P50"], "cukur_P95": mc["cukur_P95"],
        "en_olasi_kriz_tarihi": mc["en_olasi_kriz_tarihi"],
        "crisis": bool(normal["kriz"]), "goes_negative": bool(normal["eksiye_duser"]),
        "buffer": metrics["min_cash_buffer"],
        "acik_alacak_tl": metrics.get("acik_alacak_tl"),
        "acik_borc_tl": metrics.get("acik_borc_tl"),
        "stok_koprusu": (
            {
                "acil_stok_ihtiyaci": metrics["stok_nakit_ihtiyaci_tl"],
                "kriz_olasiligi_3ayak": metrics["monte_carlo_stok_dahil"]["eksiye_dusme_olasiligi"],
            }
            if metrics.get("stok_nakit_ihtiyaci_tl") is not None else None
        ),
        "curve": [{"date": r["date"], "balance": r["normal"]} for r in scenario_curves],
        "scenario_curves": scenario_curves,
    }


@app.get("/api/cash-forecast")
def cash_forecast():
    live = _try_live("cash_forecast")
    if live is not None:
        live["source"] = "live"
        try:
            live["scenario_curves"] = _scenario_curves_from_csv()
        except HTTPException:
            live["scenario_curves"] = []
        return live
    return _offline_cash_forecast()


@app.get("/api/invoice-risk")
def invoice_risk(top_n: int = 10):
    live = _try_live("invoice_risk", top_n)
    if live is not None:
        return {"source": "live", "items": live}

    df = _report_csv("invoice_delay_predictions.csv")
    sort_col = "risk_score" if "risk_score" in df.columns else "predicted_days_late"
    df = df.sort_values(sort_col, ascending=False).head(top_n)
    items = [
        {
            "invoice_id": int(r.invoice_id),
            "customer_id": int(r.customer_id),
            "amount": round(float(r.amount), 2),
            "due_date": str(r.due_date),
            "disputed": bool(r.disputed),
            "pred_days_late": float(r.predicted_days_late),
            "risk_score": float(r.risk_score) if "risk_score" in df.columns else None,
            "risk_label": str(r.risk_label) if "risk_label" in df.columns else None,
        }
        for r in df.itertuples()
    ]
    return {"source": "offline-reports", "items": items}


@app.get("/api/stock-risk")
def stock_risk(top_n: int = 10):
    live = _try_live("stock_risk", top_n)
    if live is not None:
        return {"source": "live", "items": live}

    df = _report_csv("stock_depletion_forecast.csv")
    risk_order = {"KRITIK": 0, "YUKSEK": 1, "ORTA": 2, "DUSUK": 3, "GUVENLI": 4}
    df["_r"] = df["risk_level"].map(risk_order).fillna(9)
    df = df.sort_values(["_r", "days_to_zero"])
    cols = [c for c in ("product_id", "product_name", "category", "current_stock",
                        "ewma_daily_demand", "days_to_zero", "depletion_date",
                        "order_by_date", "stockout_cost_30d", "risk_level") if c in df.columns]
    return {"source": "offline-reports", "items": df[cols].head(top_n).to_dict("records")}


@app.get("/api/scenarios")
def scenarios():
    try:
        import twin_api  # type: ignore
        import shock_simulation as shock  # type: ignore
        cutoff = pd.Timestamp(twin_api.get_overview()["cutoff"])
        live = shock.get_scenarios(cutoff)
        return {
            "source": "live",
            "items": [{"key": k, "label": v[0], "is_action": bool(v[2])} for k, v in live.items()],
        }
    except Exception:
        pass
    raw = _report("shock_scenarios.json")
    return {
        "source": "offline-reports",
        "items": [{"key": r["key"], "label": r["label"], "is_action": r["is_action"]} for r in raw],
    }


def _offline_scenario_lookup(key: str) -> dict:
    raw = _report("shock_scenarios.json")
    match = next((r for r in raw if r["key"] == key), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Senaryo bulunamadi: {key}")
    return {
        "source": "offline-reports",
        "scenario": match["key"], "label": match["label"],
        "trough": match["trough"], "trough_date": match["trough_date"],
        "status": match["status"], "kriz_olasiligi": match["kriz_olasiligi"],
        "cukur_P5": match["cukur_P5"], "cukur_P50": match["cukur_P50"],
    }


@app.get("/api/simulate/{key}")
def simulate(key: str):
    live = _try_live("simulate", key)
    if live is not None and "error" not in live:
        live["source"] = "live"
        return live
    return _offline_scenario_lookup(key)


@app.get("/api/recommend")
def recommend():
    live = _try_live("recommend")
    if live is not None:
        live["source"] = "live"
        return live

    raw = _report("shock_scenarios.json")
    baseline = next(r for r in raw if r["key"] == "0_baseline")
    actions = [r for r in raw if r["key"] != "0_baseline" and r["is_action"]]
    improving = sorted(
        [a for a in actions if a["kriz_olasiligi"] < baseline["kriz_olasiligi"]],
        key=lambda x: (x["kriz_olasiligi"], -x["trough"]),
    )
    to_public = lambda r: {
        "scenario": r["key"], "label": r["label"], "trough": r["trough"],
        "trough_date": r["trough_date"], "status": r["status"],
        "kriz_olasiligi": r["kriz_olasiligi"],
    }
    return {
        "source": "offline-reports",
        "baseline_trough": baseline["trough"], "baseline_status": baseline["status"],
        "baseline_kriz_olasiligi": baseline["kriz_olasiligi"],
        "onerilen_aksiyonlar": [to_public(r) for r in improving],
        "en_iyi": to_public(improving[0]) if improving else None,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Statik frontend'i sun (index.html, style.css, app.js).
# NOT: bu satir en sonda olmali, aksi halde /api/* rotalarini golgeler.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
