"""
ResilienceOS - CrewAI Tool Sarmalayicilari
twin_api fonksiyonlarini @tool dekoratoru ile ajanlara sunar.
"""
import json
import sys
from pathlib import Path
from crewai.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))

import twin_api


@tool("Nakit Akisi Raporu")
def nakit_raporu_tool(_: str = "") -> str:
    """CFO icin 90 gunluk nakit projeksiyonu ve kriz olasiligi."""
    result = twin_api.cash_forecast()
    result.pop("curve", None)   # egriyi cikart, token tasarrufu
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("Riskli Fatura Listesi")
def riskli_fatura_tool(_: str = "") -> str:
    """En riskli 10 acik faturanin gecikme tahmini."""
    return json.dumps(twin_api.invoice_risk(10), ensure_ascii=False, indent=2)


@tool("Stok Tukenme Riski")
def stok_riski_tool(_: str = "") -> str:
    """En kritik 10 urunun stok tukenme projeksiyonu."""
    return json.dumps(twin_api.stock_risk(10), ensure_ascii=False, indent=2)


@tool("Aksiyon Onerisi")
def aksiyon_onerisi_tool(_: str = "") -> str:
    """Tum senaryolari hesaplar, krizi cozen aksiyonlari sirali doner."""
    return json.dumps(twin_api.recommend(), ensure_ascii=False, indent=2)


@tool("Genel Durum Ozeti")
def genel_durum_tool(_: str = "") -> str:
    """Temel KPI'lar: kasa, musteri/tedarikci sayisi, acik fatura adedi."""
    return json.dumps(twin_api.get_overview(), ensure_ascii=False, indent=2)
