import json
import sys
from pathlib import Path
from crewai.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))

import twin_api


@tool("Nakit Akisi Projeksiyonu")
def tool_cash_forecast() -> str:
    """Sistemin nakit akışı projeksiyonunu, kriz olasılığını ve kasa dibi tahminini getirir."""
    result = twin_api.cash_forecast()
    result.pop("curve", None)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool("Riskli Faturalari Getir")
def tool_invoice_risk() -> str:
    """Gecikme riski en yüksek olan açık faturaları ve risk skorlarını getirir."""
    return json.dumps(twin_api.invoice_risk(top_n=3), ensure_ascii=False, default=str)


@tool("Stok Tukenme Riski Getir")
def tool_stock_risk() -> str:
    """Tükenme riski en yüksek olan ve acil sipariş gerektiren stokları getirir."""
    return json.dumps(twin_api.stock_risk(top_n=3), ensure_ascii=False, default=str)


@tool("Aksiyon Onerileri ve Kriz Cozumu")
def tool_recommend() -> str:
    """Sistemdeki krizi çözmek için simüle edilmiş en iyi aksiyonları ve kriz olasılığını nasıl düşürdüklerini getirir."""
    return json.dumps(twin_api.recommend(), ensure_ascii=False, default=str)