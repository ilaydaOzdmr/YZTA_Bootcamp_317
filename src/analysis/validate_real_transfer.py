"""
ResilienceOS - Gercek Veri Dis Dogrulamasi (Sentetik -> Gercek Transfer)
=======================================================================
Ekip arkadasimiz Muhammet'in onerdigi fikir: "modelimiz sentetik tuhafliklari
ezberlemiyor, gercek veriye de genelliyor" iddiasini KANITLAMAK. Ancak gercek
veriyi egitim setine KARISTIRMIYORUZ - cunku kalibrasyon kanitlarimiz (oracle
teorik tabani, heteroskedastik sigma, Monte Carlo, rolling backtest) egitim
dagiliminin BILINIR olmasina dayaniyor. Gercek veri bizde egitim verisi degil,
SINAV KAGIDIDIR. Bu yuzden ana kalibre pipeline'a HIC dokunulmaz; bu ayri,
bagimsiz bir dogrulama scriptidir.

Veri: IBM "Late Payment Histories" (Kaggle faktoring veri seti), 2466 gercek
fatura, 100 gercek musteri, 2012-2013. Ana pipeline ile ayni sizintisiz feature
fonksiyonu (build_features), ayni FEATURE_COLS ve ayni PARAMS kullanilir - "ayni
pipeline" iddiasi ancak ayni fonksiyon cagrilirsa durustur.

Iki kol:
  KOL 1 (transfer):   Sentetikte egitilmis KAYITLI model, gercek veride sifir-atis
                      test edilir. "Sentetik ogrendigimiz gercege ne kadar tasiniyor?"
  KOL 2 (metodoloji): AYNI sizintisiz pipeline (as-of feature + TimeSeriesSplit +
                      klip yok) gercek veride SIFIRDAN egitilip test edilir.
                      "Metodolojimiz gercek veride de sizintisiz calisiyor mu?"

ONEMLI DURUSTLUK NOTLARI (raporda birebir yer alir):
  - RMSE'ler DOGRUDAN kiyaslanamaz: gercek verinin indirgenemez gurultusu daha
    buyuk (std ~12.3 vs sentetik ~7.3) ve tutar olcegi ~150x farkli (USD faktoring
    vs TL KOBI). Dogru kiyas: her domain'de "ortalama-tahminci baseline'a gore %
    iyilesme" ve gec/erken ISARET dogrulugu - ikisi de olcekten bagimsiz.
  - Kovaryat kaymasi: model, tutar-bazli feature'larda egitimde gormedigi bolgeye
    ekstrapolasyon yapar (KOL 1'de beklenen zayiflik; KOL 2 bunu kapatir).
  - Tek dis veri seti, tek sektor (faktoring), 2012-13. Iddia "BIR bagimsiz gercek
    veri setine transfer" ile sinirli; "her gercek veriye genellenir" DENMEZ.
  - days_late gercek veride HAM tarihlerden turetilir (SettledDate - DueDate).
    Hazir 'DaysLate' kolonu >=0'a kirpilmis; onu kullanmak verinin %61'inin
    (erken odeme) hedef bilgisini imha ederdi - Muhammet'in importer'indaki
    clip hatasini bilincli olarak duzeltiyoruz.
  - Oracle (teorik taban) gercek veri icin TANIMSIZDIR (uretici gurultu modeli
    bilinmiyor); gercek-veri sonucunun yanina oracle-verimlilik yazilamaz.

Calistirma: python src/analysis/validate_real_transfer.py
Cikti:      reports/external_validation_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from train_invoice_delay import (  # noqa: E402
    FEATURE_COLS, MODEL_PATH, N_FOLDS, PARAMS, TARGET, build_features,
)

CSV_PATH = (ROOT / "data" / "kaggle" / "finance-factoring-ibm-late-payment-histories"
            / "WA_Fn-UseC_-Accounts-Receivable.csv")
OUT_PATH = ROOT / "reports" / "external_validation_metrics.json"


# --------------------------------------------------------------------------- #
# Gercek veriyi bizim semaya map'le (egitim DB'sine YAZMAZ - sadece bellekte)
# --------------------------------------------------------------------------- #
def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(CSV_PATH)
    inv_date = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    due_date = pd.to_datetime(df["DueDate"], errors="coerce")
    set_date = pd.to_datetime(df["SettledDate"], errors="coerce")

    # customerID string ('0379-NEVHP') -> tamsayi
    cust_id = pd.factorize(df["customerID"])[0] + 1

    invoices = pd.DataFrame({
        "invoice_id":   np.arange(1, len(df) + 1),
        "customer_id":  cust_id,
        "invoice_date": inv_date,
        "due_date":     due_date,
        "settled_date": set_date,
        "amount":       df["InvoiceAmount"].astype(float),
        "disputed":     (df["Disputed"].astype(str).str.strip() == "Yes").astype(int),
        # HAM tarihlerden - KLIP YOK (erken odeme = negatif days_late korunur)
        "days_late":    (set_date - due_date).dt.days.astype(float),
        "status":       "settled",
    })
    invoices = invoices.dropna(subset=["invoice_date", "due_date", "settled_date",
                                       "days_late"]).reset_index(drop=True)

    # Musteri meta: gercek veride segment/risk yok -> segment None (feature DEGIL,
    # sadece raporlama icin merge edilir). Vade tum kayitlarda tam 30 gun (olculdu).
    customers = pd.DataFrame({
        "customer_id":           np.sort(invoices["customer_id"].unique()),
    })
    customers["segment"] = None
    customers["default_payment_terms"] = 30
    return invoices, customers


# --------------------------------------------------------------------------- #
# Olcekten bagimsiz metrikler
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    # gec/erken isaret dogrulugu (olcekten bagimsiz)
    sign_acc = float(accuracy_score(y_true > 0, y_pred > 0))
    # ortalama-tahminci baseline (bu domain'in indirgenemez gurultu tabani proxy'si)
    base = float(np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_true.mean()))))
    impr = (base - rmse) / base * 100 if base else 0.0
    return {
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "sign_accuracy": round(sign_acc, 4),
        "baseline_rmse": round(base, 4),
        "improvement_over_baseline_pct": round(impr, 2),
    }


def arm1_transfer(df_feat: pd.DataFrame) -> dict:
    """Sentetikte egitilmis KAYITLI model -> gercek veride sifir-atis test."""
    if not MODEL_PATH.exists():
        return {"error": f"once train_invoice_delay.py calistirilmali ({MODEL_PATH.name} yok)"}
    model = lgb.Booster(model_file=str(MODEL_PATH))
    y = df_feat[TARGET].to_numpy(dtype=float)
    pred = model.predict(df_feat[FEATURE_COLS])
    m = _metrics(y, pred)
    m["n"] = int(len(df_feat))
    m["aciklama"] = ("Sentetik-egitilmis model, hic gormedigi gercek faturalarda. "
                     "Mutlak RMSE'nin sentetik 4.30'dan yuksek olmasi BASARISIZLIK "
                     "DEGIL (gercek gurultu tabani daha yuksek); baseline'a gore "
                     "iyilesme ve isaret dogrulugu asil olctudur.")
    return m


def arm2_methodology(df_feat: pd.DataFrame) -> dict:
    """AYNI sizintisiz pipeline gercek veride sifirdan egitilir (TimeSeriesSplit)."""
    train_df = (df_feat[(df_feat["status"] == "settled") & df_feat[TARGET].notna()]
                .sort_values("invoice_date").reset_index(drop=True))
    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    rmse_l, mae_l, sign_l = [], [], []
    oof_pred, oof_true = [], []
    for tr_idx, val_idx in tscv.split(X):
        dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=FEATURE_COLS)
        dvl = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtr)
        mdl = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dvl],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])
        yp = mdl.predict(X[val_idx])   # KLIP YOK - ana pipeline ile ayni
        rmse_l.append(float(np.sqrt(mean_squared_error(y[val_idx], yp))))
        mae_l.append(float(mean_absolute_error(y[val_idx], yp)))
        sign_l.append(float(accuracy_score(y[val_idx] > 0, yp > 0)))
        oof_pred.extend(yp.tolist()); oof_true.extend(y[val_idx].tolist())

    base = float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean()))))
    cv_rmse = float(np.mean(rmse_l))
    return {
        "validation": f"TimeSeriesSplit({N_FOLDS}) - gercek veride sifirdan",
        "n": len(train_df),
        "cv_rmse_mean": round(cv_rmse, 4),
        "cv_rmse_std": round(float(np.std(rmse_l)), 4),
        "cv_mae_mean": round(float(np.mean(mae_l)), 4),
        "cv_sign_accuracy": round(float(np.mean(sign_l)), 4),
        "baseline_rmse": round(base, 4),
        "improvement_over_baseline_pct": round((base - cv_rmse) / base * 100, 2)
        if base else 0.0,
        "aciklama": ("Ayni sizintisiz metodoloji (as-of feature + zaman-bazli CV + "
                     "klip yok) gercek veride de calisiyor mu? Baseline'i belirgin "
                     "gecmesi metodolojinin veriye ozgu olmadigini gosterir."),
    }


def main() -> None:
    print("=" * 68)
    print("  ResilienceOS - Gercek Veri Dis Dogrulamasi (Sentetik -> Gercek)")
    print("=" * 68)
    if not CSV_PATH.exists():
        print(f"\n  HATA: gercek veri bulunamadi: {CSV_PATH}")
        print("  Kaggle veri setini data/kaggle/ altina indirin.")
        return

    print("\n[1/3] Gercek veri yukleniyor + semaya map'leniyor...")
    invoices, customers = load_real()
    dl = invoices["days_late"]
    print(f"  {len(invoices):,} gercek fatura | {invoices['customer_id'].nunique()} musteri | "
          f"{invoices['invoice_date'].min().date()} -> {invoices['invoice_date'].max().date()}")
    print(f"  days_late (ham): medyan {dl.median():.0f}g | erken odeme %{100*(dl<0).mean():.0f} "
          f"| std {dl.std():.1f}")

    print("[2/3] Ayni sizintisiz feature pipeline uygulaniyor...")
    df_feat = build_features(invoices, customers)

    print("[3/3] Iki kol calistiriliyor...")
    arm1 = arm1_transfer(df_feat)
    arm2 = arm2_methodology(df_feat)

    result = {
        "veri_seti": "IBM Late Payment Histories (Kaggle faktoring)",
        "amac": ("Sentetik-egitilmis modelin gercek veriye genellemesini KANITLAMAK "
                 "- gercek veriyi egitime karistirmadan (kalibrasyon kanitlari korunur)."),
        "kol1_transfer": arm1,
        "kol2_metodoloji": arm2,
        "durustluk_notlari": [
            "RMSE'ler domainler arasi DOGRUDAN kiyaslanamaz; gercek gurultu tabani "
            "daha yuksek (std ~12.3 vs sentetik ~7.3), tutar olcegi ~150x farkli.",
            "Asil olcut: baseline'a gore % iyilesme ve gec/erken isaret dogrulugu "
            "(ikisi de olcekten bagimsiz).",
            "Kovaryat kaymasi var: model tutar-bazli feature'larda ekstrapolasyon yapar.",
            "Tek dis veri seti, tek sektor (faktoring), 2012-13; iddia 'bir bagimsiz "
            "gercek veri setine transfer' ile sinirlidir.",
            "days_late ham tarihlerden turetildi (klip yok); erken odemeler korundu.",
            "Oracle/teorik taban gercek veri icin tanimsizdir.",
        ],
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print("  SONUCLAR")
    print("=" * 68)
    if "error" not in arm1:
        print(f"  KOL 1 (transfer, sentetik-egitilmis -> gercek):")
        print(f"    RMSE {arm1['rmse']} | isaret dogrulugu %{arm1['sign_accuracy']*100:.1f} | "
              f"baseline {arm1['baseline_rmse']} -> iyilesme %{arm1['improvement_over_baseline_pct']}")
    else:
        print(f"  KOL 1: {arm1['error']}")
    print(f"  KOL 2 (metodoloji, gercekte sifirdan egitim):")
    print(f"    CV RMSE {arm2['cv_rmse_mean']} (+/-{arm2['cv_rmse_std']}) | "
          f"isaret dogrulugu %{arm2['cv_sign_accuracy']*100:.1f} | "
          f"baseline {arm2['baseline_rmse']} -> iyilesme %{arm2['improvement_over_baseline_pct']}")
    print("-" * 68)
    print("  NOT: RMSE'ler domainler arasi dogrudan kiyaslanamaz (bkz. durustluk")
    print("       notlari); baseline'a gore iyilesme ve isaret dogrulugu esas alinir.")
    print(f"\n  Cikti -> {OUT_PATH.relative_to(ROOT)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
