"""
ResilienceOS - Kaggle Gercek Veri Seti Indirici
===============================================
Sentetik dijital ikizi GERCEK dagilimlarla kalibre etmek/karsilastirmak icin
oncelikli Kaggle veri setlerini indirir (docs/01_..._Workflow.md).

ONCE: Kaggle API token gerekli.
  1) https://www.kaggle.com -> hesap (Settings) -> 'Create New API Token'
  2) inen kaggle.json dosyasini su konuma koy:  C:\\Users\\<sen>\\.kaggle\\kaggle.json
     (veya ortam: set KAGGLE_USERNAME=... & set KAGGLE_KEY=...)

Calistirma:  python src/data_ingest/fetch_kaggle.py
Cikti:       data/kaggle/<dataset>/...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "kaggle"

# Oncelikli veri setleri (rapordaki dogrulanmis secimler)
DATASETS = [
    "pradumn203/payment-date-prediction-for-invoices-dataset",  # fatura gecikme (birincil)
    "hhenry/finance-factoring-ibm-late-payment-histories",       # fatura gecikme (baseline)
    "anirudhchauhan/retail-store-inventory-forecasting-dataset", # stok/talep
    "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",  # tedarik riski
]


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    if not have_credentials():
        print("HATA: Kaggle kimligi bulunamadi.\n"
              "  1) kaggle.com -> Settings -> 'Create New API Token'\n"
              f"  2) kaggle.json -> {Path.home() / '.kaggle' / 'kaggle.json'}\n"
              "     (veya set KAGGLE_USERNAME=... & set KAGGLE_KEY=...)")
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    DEST.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        folder = DEST / ds.split("/")[1]
        folder.mkdir(exist_ok=True)
        print(f"indiriliyor: {ds} ...")
        api.dataset_download_files(ds, path=str(folder), unzip=True, quiet=False)
        files = [f.name for f in folder.glob("*")][:5]
        print(f"  -> {folder}  ({', '.join(files)})")

    print("\nTum veri setleri indirildi ->", DEST)
    print("Sonraki adim: EDA notebook ile sentetik kalibrasyonu gercek dagilimla karsilastir.")


if __name__ == "__main__":
    main()
