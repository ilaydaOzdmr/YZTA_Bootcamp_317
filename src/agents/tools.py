import sys
from pathlib import Path
from crewai.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))

import twin_api

@tool("Nakit Akisi Projeksiyonu")
def tool_cash_forecast() -> str:
    """Sistemin nakit akışı projeksiyonunu, kriz olasılığını ve kasa dibi tahminini getirir."""
    return str(twin_api.cash_forecast())

@tool("Riskli Faturalari Getir")
def tool_invoice_risk() -> str:
    """Gecikme riski en yüksek olan açık faturaları ve risk skorlarını getirir."""
    return str(twin_api.invoice_risk(top_n=3))

@tool("Stok Tukenme Riski Getir")
def tool_stock_risk() -> str:
    """Tükenme riski en yüksek olan ve acil sipariş gerektiren stokları getirir."""
    return str(twin_api.stock_risk(top_n=3))

@tool("Aksiyon Onerileri ve Kriz Cozumu")
def tool_recommend() -> str:
    """Sistemdeki krizi çözmek için simüle edilmiş en iyi aksiyonları ve kriz olasılığını nasıl düşürdüklerini getirir."""
    return str(twin_api.recommend())