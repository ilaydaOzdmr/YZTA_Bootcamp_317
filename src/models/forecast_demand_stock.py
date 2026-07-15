"""
ResilienceOS - Talep Tahmini ve Stok Tukenme Analizi v2
========================================================
v2 iyilestirmeleri:
  - EWMA (Exponential Weighted Moving Average): son gunlere daha fazla agirlik
  - Tedarikci lead_time entegrasyonu: ideal siparis tarihi = tukenme - lead_time
  - Stok tükenme maliyet hesabi: tukenme suresi × gunluk talep × birim maliyet
  - Urun kategorisi encode: kategoriye gore ortalama talep normalizasyonu
  - Model tahmini ile tukenme hesabi (basit ortalama yerine)
  - Daha zengin cikti: "ne zaman siparis ver?" + "ne kadara mal olur?"

Ciktilar:
  models/demand_model.txt
  reports/stock_depletion_forecast.csv
  reports/demand_metrics.json

Calistirma: python src/models/forecast_demand_stock.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "digital_twin.db"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH     = MODEL_DIR / "demand_model.txt"
METRICS_PATH   = REPORT_DIR / "demand_metrics.json"
DEPLETION_PATH = REPORT_DIR / "stock_depletion_forecast.csv"

FORECAST_DAYS = 30
SEED = 42

RISK_THRESHOLDS = {
    "KRITIK":  0,          # stok <= 0
    "YUKSEK":  1,          # stok <= reorder_point
    "ORTA":    14,         # tukenmeye <= 14 gun
    "DUSUK":   30,         # tukenmeye <= 30 gun
}


# --------------------------------------------------------------------------- #
# Veri Yukleme
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH)
    inv_log   = pd.read_sql("SELECT * FROM inventory_log ORDER BY date, product_id", con)
    products  = pd.read_sql("SELECT * FROM products", con)
    suppliers = pd.read_sql("SELECT * FROM suppliers", con)
    con.close()
    inv_log["date"] = pd.to_datetime(inv_log["date"])
    return inv_log, products, suppliers


def reorder_cash_needs(cutoff: pd.Timestamp, horizon: int = 30) -> pd.Series:
    """
    STOK -> NAKIT KOPRUSU.  Bir "bugun" (cutoff) itibariyle, onumuzdeki `horizon` gun
    icinde tukenecek urunler icin gereken ACIL SIPARIS nakit cikislarini doner.

    Rapordaki "3 olay ayni hafta" senaryosunun 3. ayagi (kritik stok tukenmesi) boylece
    nakit projeksiyonuna baglanir: uretimi surdurmek icin siparis verilmesi gerekir ve
    bu bir nakit cikisidir. (Tedarik Ajani -> CFO Ajani etkilesiminin sayisal karsiligi.)

    Yaklasim (cutoff vantajindan, son 14 gun talep hizi ile - forecast_demand_stock ile
    tutarli days-of-cover mantigi):
      - order_by = cutoff + (stok - reorder_point)/gunluk_talep - lead_time
      - order_by pencere icindeyse: maliyet = ~30 gunluk talep * birim_maliyet
    Doner: index=tarih, value=TL (ayni gune denk gelenler toplanir).
    """
    inv_log, products, suppliers = load_data()
    lead = suppliers.set_index("supplier_id")["lead_time_days"].to_dict()
    at_cut = (inv_log[inv_log["date"] <= cutoff].sort_values("date")
              .groupby("product_id").tail(1).set_index("product_id"))
    recent = (inv_log[(inv_log["date"] > cutoff - pd.Timedelta(days=14))
                      & (inv_log["date"] <= cutoff)]
              .groupby("product_id")["units_sold"].mean())

    # ONEMLI: Yalnizca reorder noktasinin ALTINA dusmus ve ufukta tukenecek urunler
    # sayilir. Reorder noktasinin ustundeki urunler NORMAL siparis dongusunde yeniden
    # siparis edilir (bu zaten purchase_invoices'ta -> cift saymamak icin haric).
    # Reorder altina dusup tukenmekte olan urun = PLANSIZ acil durum (rapordaki kritik
    # stok tukenmesi ayagi). Saglikli kalibre bir isletmede bu nadirdir.
    flows: dict[pd.Timestamp, float] = {}
    for _, p in products.iterrows():
        pid = p["product_id"]
        if pid not in at_cut.index:
            continue
        stock = float(at_cut.loc[pid, "closing_stock"])
        if stock >= p["reorder_point"]:            # rutin dongude -> haric (cift sayma)
            continue
        rate = max(float(recent.get(pid, 0.0)), 0.01)
        lt = int(lead.get(p["supplier_id"], 7))
        stockout_day = cutoff + pd.Timedelta(days=stock / rate)
        if stockout_day > cutoff + pd.Timedelta(days=horizon):
            continue
        # acil siparis: bir dongu (horizon gunluk) talebi yeniden stokla
        order_by = max(stockout_day - pd.Timedelta(days=lt), cutoff + pd.Timedelta(days=1))
        cost = round(rate * horizon * float(p["unit_cost"]), 2)
        flows[order_by] = flows.get(order_by, 0.0) + cost
    if not flows:
        return pd.Series(dtype=float)
    return pd.Series(flows).sort_index()


# --------------------------------------------------------------------------- #
# Feature Engineering
# --------------------------------------------------------------------------- #
def build_product_features(
    inv_log: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Urun bazinda zenginlestirilmis zaman serisi ozellikleri.
    EWMA, kategori normalize talep, stok/reorder orani, tedarikci guvenirligi.
    """
    # Urun meta bilgisi
    prod_meta = products.set_index("product_id")[
        ["category", "unit_cost", "sale_price", "reorder_point"]
    ].to_dict("index")

    # Kategori encode
    cats = products["category"].unique()
    cat_map = {c: i for i, c in enumerate(sorted(cats))}

    # ------------------------------------------------------------------ #
    # SIZINTI NOTU (v3)
    # Uretici kimligi:  units_sold = opening_stock + units_received - closing_stock
    # (veride %100 birebir gecerli). Bu yuzden asagidakiler FEATURE OLARAK KULLANILAMAZ:
    #   - units_received : uretici bunu "opening - units_sold < reorder" kosuluna gore
    #                      belirliyor, yani DOGRUDAN hedefe bagli.
    #   - closing_stock  : ayni gunun kapanis stogu; opening ve received ile birlikte
    #                      hedefi cebirsel olarak cozer.
    # Ayni sekilde units_sold uzerindeki rolling/EWMA/diff hesaplari SHIFT'SIZ
    # yapilirsa BUGUNUN hedefini icerir. Hepsi .shift(1) ile gecmise kaydirildi.
    # (Onceki surumde bu sizintilar RMSE'yi 0.30 gibi gercek disi bir seviyeye
    #  dusuruyordu; asagidaki sizintisiz kurulumda gercek performans olculur.)
    # ------------------------------------------------------------------ #
    frames = []
    for pid, grp in inv_log.groupby("product_id"):
        grp = grp.sort_values("date").copy()
        meta = prod_meta.get(pid, {})
        reorder_pt = meta.get("reorder_point", 50)

        past_sold  = grp["units_sold"].shift(1)      # yalnizca GECMIS satislar
        past_close = grp["closing_stock"].shift(1)   # yalnizca GECMIS kapanis stoku

        # Lag: satislar / stok
        for lag in [1, 3, 7, 14]:
            grp[f"sold_lag{lag}"]  = grp["units_sold"].shift(lag)
            grp[f"stock_lag{lag}"] = grp["closing_stock"].shift(lag)

        # Rolling ortalama (shift'li -> bugunu icermez)
        for w in [7, 14, 30]:
            grp[f"sold_roll{w}_mean"] = past_sold.rolling(w).mean()
            grp[f"sold_roll{w}_std"]  = past_sold.rolling(w).std()

        # EWMA (shift'li)
        grp["sold_ewma7"]  = past_sold.ewm(span=7,  adjust=False).mean()
        grp["sold_ewma14"] = past_sold.ewm(span=14, adjust=False).mean()
        grp["sold_ewma30"] = past_sold.ewm(span=30, adjust=False).mean()

        # Stok ozellikleri: gun BASINDA bilinen opening_stock + GECMIS kapanislar
        grp["stock_roll7_min"]  = past_close.rolling(7).min()
        grp["stock_vs_reorder"] = grp["opening_stock"] / (reorder_pt + 1)
        grp["stock_trend_7d"]   = past_close.diff(7)

        # Talep trendi (shift'li)
        grp["demand_trend_7d"] = past_sold.diff(7)
        grp["demand_accel"]    = past_sold.diff(1).diff(1)

        # Takvim
        grp["dayofweek"]  = grp["date"].dt.dayofweek
        grp["dayofmonth"] = grp["date"].dt.day
        grp["month"]      = grp["date"].dt.month
        grp["is_weekend"] = (grp["dayofweek"] >= 5).astype(int)
        grp["sin_dow"]    = np.sin(2 * np.pi * grp["dayofweek"] / 7)
        grp["cos_dow"]    = np.cos(2 * np.pi * grp["dayofweek"] / 7)

        # Meta
        grp["product_id"]   = pid
        grp["category_enc"] = cat_map.get(meta.get("category", ""), 0)
        grp["unit_cost"]    = meta.get("unit_cost", 0.0)
        grp["reorder_point"]= reorder_pt

        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = (
    [f"sold_lag{l}"   for l in [1, 3, 7, 14]]
    + [f"stock_lag{l}" for l in [1, 3, 7, 14]]
    + [f"sold_roll{w}_mean" for w in [7, 14, 30]]
    + [f"sold_roll{w}_std"  for w in [7, 14, 30]]
    + ["sold_ewma7", "sold_ewma14", "sold_ewma30"]
    + ["stock_roll7_min", "stock_vs_reorder", "stock_trend_7d"]
    + ["demand_trend_7d", "demand_accel"]
    + ["dayofweek", "dayofmonth", "month", "is_weekend", "sin_dow", "cos_dow"]
    + ["opening_stock"]                     # gun BASINDA bilinir -> mesru
    + ["product_id", "category_enc", "unit_cost", "reorder_point"]
)
# CIKARILANLAR (sizinti): units_received, stock_ratio, received_flag
#   ve ayni-gun closing_stock turevleri. Bkz. build_product_features icindeki not.
TARGET = "units_sold"


# --------------------------------------------------------------------------- #
# Model Egitimi
# --------------------------------------------------------------------------- #
def train_model(df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    # Zaman bazli bolme (%80 egitim, %20 test)
    dates  = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    X_tr = df.loc[df["date"] < cutoff, FEATURE_COLS]
    y_tr = df.loc[df["date"] < cutoff, TARGET]
    X_val= df.loc[df["date"] >= cutoff, FEATURE_COLS]
    y_val= df.loc[df["date"] >= cutoff, TARGET]

    params = {
        "objective":         "regression",
        "metric":            ["rmse", "mae"],
        "learning_rate":     0.04,
        "num_leaves":        63,
        "min_child_samples": 15,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.85,
        "bagging_freq":      5,
        "lambda_l1":         0.05,
        "lambda_l2":         0.1,
        "verbose":          -1,
        "seed":              SEED,
    }
    dt = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    dv = lgb.Dataset(X_val, label=y_val, reference=dt)
    m = lgb.train(
        params, dt, num_boost_round=800,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(0)],
    )
    y_pred = m.predict(X_val).clip(0)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae  = float(mean_absolute_error(y_val, y_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_tr.mean()))))
    improvement   = (baseline_rmse - rmse) / baseline_rmse * 100

    # Final model (tum veri)
    final = lgb.train(
        params,
        lgb.Dataset(df[FEATURE_COLS], label=df[TARGET]),
        num_boost_round=m.best_iteration,
        callbacks=[lgb.log_evaluation(0)],
    )

    metrics = {
        "model":            "LightGBM Demand v2 (EWMA + cyclical)",
        "target":           "units_sold",
        "n_train":          len(X_tr),
        "n_val":            len(X_val),
        "n_products":       df["product_id"].nunique(),
        "best_iteration":   m.best_iteration,
        "rmse":             round(rmse, 4),
        "mae":              round(mae, 4),
        "baseline_rmse":    round(baseline_rmse, 4),
        "improvement_pct":  round(improvement, 2),
        "feature_importance": dict(
            zip(FEATURE_COLS,
                final.feature_importance(importance_type="gain").tolist())
        ),
    }
    return final, metrics


# --------------------------------------------------------------------------- #
# Stok Tukenme Projeksiyonu
# --------------------------------------------------------------------------- #
def _feature_row(sold_hist: list[float], stock_hist: list[float], opening: float,
                 day: pd.Timestamp, pid: int, cat_enc: int,
                 unit_cost: float, reorder_pt: int) -> np.ndarray:
    """
    Gelecek bir gun icin FEATURE_COLS sirasina birebir uyan ozellik vektoru kurar.
    Tum satis/stok ozellikleri GECMISTEN (o gunden onceki degerlerden) hesaplanir -
    egitimdeki .shift(1) mantiginin aynisi.
    """
    s = pd.Series(sold_hist, dtype=float)
    c = pd.Series(stock_hist, dtype=float)

    def lag(series: pd.Series, k: int) -> float:
        return float(series.iloc[-k]) if len(series) >= k else float(series.iloc[0])

    vals: list[float] = []
    vals += [lag(s, k) for k in (1, 3, 7, 14)]                       # sold_lag*
    vals += [lag(c, k) for k in (1, 3, 7, 14)]                       # stock_lag*
    vals += [float(s.tail(w).mean()) for w in (7, 14, 30)]           # sold_roll*_mean
    vals += [float(s.tail(w).std(ddof=1)) if len(s) > 1 else 0.0
             for w in (7, 14, 30)]                                   # sold_roll*_std
    vals += [float(s.ewm(span=sp, adjust=False).mean().iloc[-1])
             for sp in (7, 14, 30)]                                  # sold_ewma*
    vals.append(float(c.tail(7).min()))                              # stock_roll7_min
    vals.append(float(opening / (reorder_pt + 1)))                   # stock_vs_reorder
    vals.append(float(c.iloc[-1] - c.iloc[-8]) if len(c) >= 8 else 0.0)   # stock_trend_7d
    vals.append(float(s.iloc[-1] - s.iloc[-8]) if len(s) >= 8 else 0.0)   # demand_trend_7d
    accel = ((s.iloc[-1] - s.iloc[-2]) - (s.iloc[-2] - s.iloc[-3])) if len(s) >= 3 else 0.0
    vals.append(float(accel))                                        # demand_accel
    dow = day.dayofweek
    vals += [float(dow), float(day.day), float(day.month),
             float(dow >= 5),
             float(np.sin(2 * np.pi * dow / 7)), float(np.cos(2 * np.pi * dow / 7))]
    vals.append(float(opening))                                      # opening_stock
    vals += [float(pid), float(cat_enc), float(unit_cost), float(reorder_pt)]
    return np.array(vals, dtype=float).reshape(1, -1)


def forecast_depletion(
    model: lgb.Booster,
    inv_log: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her urun icin:
      - Son 30 gun EWMA talep
      - Model tahminli gun basi talep (son feature satiri ile)
      - Tukenme tarihi, yeniden siparis tarihi (lead_time dahil)
      - Stok kesintisi maliyeti: tukenme_gun × talep × birim_maliyet
    """
    last_date = inv_log["date"].max()

    # Kategori kodu haritasi (egitimdeki build_product_features ile AYNI: sorted(cats))
    cat_map = {c: i for i, c in enumerate(sorted(products["category"].unique()))}

    # Tedarikci lead_time haritasi
    sup_prod = products.merge(
        suppliers[["supplier_id", "lead_time_days"]], on="supplier_id", how="left"
    ).set_index("product_id")[["name", "category", "unit_cost",
                                "sale_price", "reorder_point", "lead_time_days"]]

    results = []
    for pid in inv_log["product_id"].unique():
        grp = inv_log[inv_log["product_id"] == pid].sort_values("date")
        meta = sup_prod.loc[pid] if pid in sup_prod.index else pd.Series()

        current_stock = int(grp["closing_stock"].iloc[-1])
        reorder_pt    = int(meta.get("reorder_point", 50))
        unit_cost     = float(meta.get("unit_cost", 0.0))
        lead_time     = int(meta.get("lead_time_days", 7))

        # EWMA referansi (karsilastirma icin)
        ewma_demand = max(float(
            grp["units_sold"].tail(30).ewm(span=7, adjust=False).mean().iloc[-1]), 0.1)

        # --- MODEL TABANLI OZYINELEMELI TAHMIN ---
        # Onceki surumde `model_demand = ewma_demand` yaziliyordu; yani LightGBM
        # egitiliyor ama tukenme hesabinda HIC KULLANILMIYORDU. Simdi model,
        # ufuk boyunca gun gun talebi tahmin ediyor; her tahmin bir sonraki gunun
        # lag/rolling ozelliklerine besleniyor ("yeniden siparis verilmezse" senaryosu).
        sold_hist  = list(grp["units_sold"].astype(float))
        stock_hist = list(grp["closing_stock"].astype(float))
        stock      = float(current_stock)
        # kategori kodu products'tan (inv_log'da yok; egitimdeki cat_map ile AYNI siralama)
        _cat = meta.get("category", "") if isinstance(meta, dict) else meta["category"]
        cat_enc    = cat_map.get(_cat, 0)
        daily_preds, days_to_zero = [], float(FORECAST_DAYS)

        for h in range(1, FORECAST_DAYS + 1):
            day = last_date + pd.Timedelta(days=h)
            row = _feature_row(sold_hist, stock_hist, stock, day,
                               pid, cat_enc, unit_cost, reorder_pt)
            pred = float(max(0.0, model.predict(row)[0]))
            daily_preds.append(pred)
            sold_hist.append(pred)
            stock = max(0.0, stock - pred)
            stock_hist.append(stock)
            if stock <= 0 and days_to_zero == float(FORECAST_DAYS):
                days_to_zero = float(h)

        model_demand = max(float(np.mean(daily_preds)), 0.1)   # ufuk boyu ort. gunluk talep

        if current_stock <= 0:
            days_to_zero    = 0.0
            days_to_reorder = 0.0
        else:
            days_to_reorder = max(0.0, (current_stock - reorder_pt) / model_demand)

        depletion_date = (last_date + pd.Timedelta(days=int(days_to_zero))).date()
        # Siparis tarihi: tukenme tarihi - lead_time (siparisin ne zaman verilmesi lazim)
        order_by_date  = (last_date + pd.Timedelta(
            days=max(0, int(days_to_reorder) - lead_time)
        )).date()
        reorder_date   = (last_date + pd.Timedelta(days=int(days_to_reorder))).date()

        # Stok kesintisi maliyeti (tukenme gunu × gunluk talep × birim maliyet)
        projected_demand_30d = round(model_demand * FORECAST_DAYS, 1)
        shortage_if_no_order = max(0, projected_demand_30d - current_stock)
        stockout_cost        = round(shortage_if_no_order * unit_cost, 2)

        # Risk seviyesi
        if current_stock <= 0:
            risk = "KRITIK"
        elif current_stock <= reorder_pt:
            risk = "YUKSEK"
        elif days_to_zero <= RISK_THRESHOLDS["ORTA"]:
            risk = "ORTA"
        elif days_to_zero <= RISK_THRESHOLDS["DUSUK"]:
            risk = "DUSUK"
        else:
            risk = "GUVENLI"

        results.append({
            "product_id":           int(pid),
            "product_name":         str(meta.get("name", f"Urun-{pid}")),
            "category":             str(meta.get("category", "-")),
            "current_stock":        current_stock,
            "reorder_point":        reorder_pt,
            "lead_time_days":       lead_time,
            "ewma_daily_demand":    round(ewma_demand, 2),
            "projected_demand_30d": projected_demand_30d,
            "days_to_reorder":      round(days_to_reorder, 1),
            "days_to_zero":         round(days_to_zero, 1),
            "order_by_date":        str(order_by_date),
            "reorder_date":         str(reorder_date),
            "depletion_date":       str(depletion_date),
            "unit_cost":            unit_cost,
            "stockout_cost_30d":    stockout_cost,
            "risk_level":           risk,
        })

    df_out = pd.DataFrame(results).sort_values("days_to_zero").reset_index(drop=True)
    return df_out


# --------------------------------------------------------------------------- #
# Ana Akis
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 65)
    print("  ResilienceOS - Talep & Stok Tukenme Analizi  (v2)")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"\n  HATA: {DB_PATH} bulunamadi.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("\n[1/4] Stok, urun ve tedarikci verileri yukleniyor...")
    inv_log, products, suppliers = load_data()
    print(f"  Stok kaydi: {len(inv_log):,}  |  Urun: {len(products):,}  "
          f"|  Tedarikci: {len(suppliers):,}")

    print("[2/4] Talep ozellikleri olusturuluyor (EWMA + cyclical)...")
    df_feat = build_product_features(inv_log, products)
    print(f"  {len(df_feat):,} satir, {len(FEATURE_COLS)} ozellik")

    print("[3/4] LightGBM egitiliyor...")
    model, metrics = train_model(df_feat)

    print("[4/4] Stok tukenme projeksiyonu + maliyet analizi...")
    depletion = forecast_depletion(model, inv_log, products, suppliers)

    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    depletion.to_csv(DEPLETION_PATH, index=False)

    # ----- Ozet -----
    print("\n" + "=" * 65)
    print("  SONUCLAR")
    print("=" * 65)
    print(f"  RMSE        : {metrics['rmse']} adet")
    print(f"  MAE         : {metrics['mae']} adet")
    print(f"  Iyilestirme : %{metrics['improvement_pct']}")
    print(f"  Best iter   : {metrics['best_iteration']}")
    print("-" * 65)

    rc = depletion["risk_level"].value_counts()
    print("\n  Risk Dagılımı:")
    for lvl in ["KRITIK", "YUKSEK", "ORTA", "DUSUK", "GUVENLI"]:
        cnt = rc.get(lvl, 0)
        bar = "#" * cnt
        print(f"    {lvl:<10}: {bar} ({cnt})")

    # Acil urunler
    urgent = depletion[depletion["risk_level"].isin(["KRITIK", "YUKSEK", "ORTA"])]
    if not urgent.empty:
        print(f"\n  [!] DIKKAT GEREKEN URUNLER ({len(urgent)} adet):")
        hdr = (f"  {'Urun':<28} {'Stok':>6} {'Tukenme':>10} "
               f"{'Siparis':>10} {'Maliyet':>12}  Risk")
        print(hdr)
        print(f"  {'-'*75}")
        for _, row in urgent.head(8).iterrows():
            print(f"  {row['product_name'][:26]:<28} "
                  f"{row['current_stock']:>6}  "
                  f"{row['depletion_date']:>10}  "
                  f"{row['order_by_date']:>10}  "
                  f"{row['stockout_cost_30d']:>11,.0f}  "
                  f"{row['risk_level']}")

        total_cost = urgent["stockout_cost_30d"].sum()
        print(f"\n  Toplam stok kesintisi riski (30g): {total_cost:,.0f} TL")
    else:
        print("\n  [OK] Kritik stok riski yok.")

    print(f"\n  Stok Raporu -> {DEPLETION_PATH.name}")
    print(f"  Model       -> {MODEL_PATH.name}")
    print(f"  Metrikler   -> {METRICS_PATH.name}")
    print("=" * 65)

    fi = sorted(metrics["feature_importance"].items(),
                key=lambda x: x[1], reverse=True)
    max_s = max(v for _, v in fi) or 1
    print("\n  Ozellik Onemliligi (Top 8):")
    for feat, score in fi[:8]:
        bar = "#" * int(score / max_s * 28)
        print(f"  {feat:<28} {bar}")
    print()


if __name__ == "__main__":
    main()
