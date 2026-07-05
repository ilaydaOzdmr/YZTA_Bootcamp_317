"""
ResilienceOS - TCMB EVDS Gercek Makro Veri Cekici
=================================================
USD/TRY, EUR/TRY, TUFE ve fonlama faizini TCMB EVDS'den ceker ve
data/digital_twin.db icindeki macro_daily tablosunu GERCEK veriyle gunceller
(placeholder yerine). Kur gunluk, TUFE aylik -> gunluk ffill.

ONCE: TCMB EVDS API key gerekli. Alma adimlari icin README veya sohbet notuna bak.
Key'i su yollardan biriyle ver:
  - ortam degiskeni:  set EVDS_API_KEY=xxxx  (Windows)  /  export EVDS_API_KEY=xxxx
  - veya proje kokunde .env dosyasi:  EVDS_API_KEY=xxxx

NOT: 05/04/2024 sonrasi EVDS key'i HTTP header'da gonderilir; guncel evds paketi bunu halleder.

Calistirma:  python src/data_ingest/fetch_evds.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "digital_twin.db"
ENV_PATH = ROOT / ".env"

# ResilienceOS makro katmani icin dogrulanmis EVDS seri kodlari
# (isim -> kod, frequency[1=gunluk,5=aylik], formulas[0=seviye,3=yillik % degisim])
SERIES = {
    "usdtry": ("TP.DK.USD.A.YTL", 1, 0),     # USD/TRY alis (gunluk)
    "eurtry": ("TP.DK.EUR.A.YTL", 1, 0),     # EUR/TRY alis (gunluk)
    "tufe_yoy": ("TP.FG.J0", 5, 3),          # TUFE yillik % degisim (aylik)
    "policy_rate": ("TP.APIFON4", 1, 0),     # Agirlikli ort. fonlama maliyeti (gunluk)
}
START = "01-01-2024"
END = "30-06-2025"


def parse_evds_date(series: pd.Series) -> pd.Series:
    """EVDS tarih stringlerini parse eder: gunluk 'dd-mm-yyyy' + aylik 'yyyy-m'."""
    d = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    m = d.isna()
    if m.any():
        d.loc[m] = pd.to_datetime(series[m], format="%Y-%m", errors="coerce")
    return d


def get_api_key() -> str:
    key = os.environ.get("EVDS_API_KEY", "").strip()
    if not key and ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("EVDS_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("HATA: EVDS_API_KEY bulunamadi.\n"
              "  1) https://evds3.tcmb.gov.tr adresinde ucretsiz hesap ac\n"
              "  2) Profil > 'API Key Kopyala'\n"
              "  3) set EVDS_API_KEY=<anahtar>  (veya .env dosyasina yaz)\n"
              "     ornek icin .env.example dosyasina bak")
        sys.exit(1)
    return key


def main() -> None:
    from evds import evdsAPI  # import burada: key kontrolunden sonra
    api = evdsAPI(get_api_key())

    frames = {}
    for name, (code, freq, formula) in SERIES.items():
        df = api.get_data([code], startdate=START, enddate=END,
                          frequency=freq, formulas=formula)
        df.columns = [c.replace(".", "_") for c in df.columns]
        val_col = [c for c in df.columns if c not in ("Tarih", "YEARWEEK")][0]
        s = df[["Tarih", val_col]].rename(columns={"Tarih": "date", val_col: name})
        s["date"] = parse_evds_date(s["date"])
        s = s.dropna(subset=["date"])
        s[name] = pd.to_numeric(s[name], errors="coerce")
        frames[name] = s
        print(f"  cekildi: {name} ({code}) - {len(s)} kayit")

    # Gunluk takvime hizala; aylik seriler -> gunluk ffill
    idx = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    out = pd.DataFrame({"date": idx})
    for name, s in frames.items():
        out = out.merge(s, on="date", how="left")
    out = out.ffill().bfill()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    con = sqlite3.connect(DB_PATH)
    out.to_sql("macro_daily", con, if_exists="replace", index=False)
    con.close()
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports" / "macro_evds.csv", index=False)
    print(f"\nmacro_daily GERCEK EVDS verisiyle guncellendi ({len(out)} gun).")
    print(f"  USD/TRY   {out['usdtry'].iloc[0]} -> {out['usdtry'].iloc[-1]}")
    print(f"  TUFE_yoy  %{out['tufe_yoy'].iloc[0]} -> %{out['tufe_yoy'].iloc[-1]}")
    print(f"  Faiz      %{out['policy_rate'].iloc[0]} -> %{out['policy_rate'].iloc[-1]}")


if __name__ == "__main__":
    main()
