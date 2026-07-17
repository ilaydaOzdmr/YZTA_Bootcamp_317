"""
ResilienceOS - API Backend (FastAPI)
=====================================
`src/core/twin_api.py`'deki fonksiyonlari HTTP uzerinden servis eder.

IKI MOD:
  - CANLI:   data/digital_twin.db + egitilmis modeller mevcutsa, twin_api'yi
             dogrudan cagirir (gercek zamanli sonuc).
  - OFFLINE: DB/model yoksa reports/ altindaki onceden hesaplanmis ciktilara
             (cashflow_metrics.json, cashflow_forecast.csv) duser. Boylece
             pipeline'i calistirmamis biri bile dashboard'u gorebilir.

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

app = FastAPI(title="ResilienceOS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# CANLI MOD: twin_api mevcutsa ve pipeline calismissa kullan
# ------------------------------------------------------------------
def _live_cash_forecast() -> dict | None:
    try:
        import twin_api  # type: ignore  # src/core/twin_api.py
        return twin_api.cash_forecast()
    except Exception:
        # DB yok, model yok, veya pipeline hic calistirilmamis -> offline'a dus
        return None


# ------------------------------------------------------------------
# OFFLINE MOD: reports/ altindaki onceden hesaplanmis ciktilardan uret
# ------------------------------------------------------------------
def _scenario_curves_from_csv() -> list[dict]:
    csv_path = REPORTS / "cashflow_forecast.csv"
    df = pd.read_csv(csv_path)
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
    metrics_path = REPORTS / "cashflow_metrics.json"
    csv_path = REPORTS / "cashflow_forecast.csv"
    if not metrics_path.exists() or not csv_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Ne canli pipeline (data/digital_twin.db) ne de "
                "reports/cashflow_metrics.json bulunamadi. Once README'deki "
                "'Hizli Baslangic' adimlarini calistirin."
            ),
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
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
        "cukur_P5": mc["cukur_P5"],
        "cukur_P50": mc["cukur_P50"],
        "cukur_P95": mc["cukur_P95"],
        "en_olasi_kriz_tarihi": mc["en_olasi_kriz_tarihi"],
        "crisis": bool(normal["kriz"]),
        "goes_negative": bool(normal["eksiye_duser"]),
        "buffer": metrics["min_cash_buffer"],
        "acik_alacak_tl": metrics.get("acik_alacak_tl"),
        "acik_borc_tl": metrics.get("acik_borc_tl"),
        "stok_koprusu": (
            {
                "acil_stok_ihtiyaci": metrics["stok_nakit_ihtiyaci_tl"],
                "kriz_olasiligi_3ayak": metrics["monte_carlo_stok_dahil"]["eksiye_dusme_olasiligi"],
            }
            if metrics.get("stok_nakit_ihtiyaci_tl") is not None
            else None
        ),
        "curve": [{"date": r["date"], "balance": r["normal"]} for r in scenario_curves],
        "scenario_curves": scenario_curves,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cash-forecast")
def cash_forecast():
    """Nakit akisi projeksiyonu + kriz uyarisi. Canli varsa canli, yoksa offline."""
    live = _live_cash_forecast()
    if live is not None:
        live["source"] = "live"
        csv_path = REPORTS / "cashflow_forecast.csv"
        if csv_path.exists():
            live["scenario_curves"] = _scenario_curves_from_csv()
        return live
    return _offline_cash_forecast()


# Statik frontend'i sun (index.html, style.css, app.js).
# NOT: bu satir en sonda olmali, aksi halde /api/* rotalarini golgeler.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
