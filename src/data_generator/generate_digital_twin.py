"""
ResilienceOS - Sentetik Dijital Ikiz Veri Ureticisi
Tek kurgusal KOBI ("Anadolu Tekstil A.S.") icin iliskisel olarak tutarli,
18 aylik finansal + operasyonel veri uretir ve SQLite'a yazar.

Tasarim ilkeleri:
  - FK tutarliligi: customers -> sales_invoices, suppliers/products ->
    purchase_invoices, products -> inventory_log.
  - Gecikme lojigi IBM Finance Factoring istatistiklerine kalibre:
    ort ~3.4g, medyan 0, %gec ~36, p90 ~13.
  - bank_ledger BAGIMSIZ uretilmez; tahsilat (sales) + odeme (purchase) +
    duzenli giderlerden TURETILIR, kronolojik running-balance ile.
  - Demo darbogazi: 3 olay ayni haftaya denk gelir:
      1) ABC Tekstil'in 180K TL'lik faturasi gecikmeli
      2) Buyuk ham madde tedarikci odemesi
      3) Kritik stok tukenmesi
  - Makro katmani: gercekci USD/TRY yukselen trend + 15 Temmuz 2024 kur
    soku (simülasyon motorunun test edilmesi icin).
  - EVDS geldikten sonra macro_daily tablosu fetch_evds.py ile overwrite edilir.

Cikti:       data/digital_twin.db  (+ konsolda ozet ve kalibrasyon metrikleri)
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

# Konfigurasyon
SEED = 42
START_DATE = date(2024, 1, 1)
END_DATE   = date(2025, 6, 30)          # 18 ay geçmiş veri

N_CUSTOMERS = 40
N_SUPPLIERS  = 15
N_PRODUCTS   = 60

OPENING_CASH         = 650_000.0        # baslangic kasa (TL)
MONTHLY_FIXED_COST   = 150_000.0        # kira + maas (her ayin 5'i)
TARGET_CRUNCH_TROUGH = -60_000.0        # crunch haftasi hedef kasa dibi (TL)

# Demo darbogazi haftasi — 3 olay bu hafta cakisir
CRUNCH_WEEK_START = date(2025, 6, 16)
CRUNCH_WEEK_END   = date(2025, 6, 22)

# ABC Tekstil'in demo faturasi
ABC_INVOICE_AMOUNT = 180_000.0
ABC_INVOICE_DATE   = date(2025, 5, 12)
ABC_DUE_DATE       = date(2025, 6, 16)

# Makro seri: kur soku parametreleri
# 15-22 Temmuz 2024 arasinda ani kur yukselisi simule edilir
SHOCK_START    = date(2024, 7, 15)
SHOCK_END      = date(2024, 7, 22)
SHOCK_DAILY    = 0.18                   # TL/gun soku buyuklugu

# Musteri segmentlerine gore odeme davranisi
# IBM Finance Factoring istatistiklerine kalibre:
#   ort ~3.4g, medyan 0, gec ~%36, p90 ~13
SEGMENT_PROFILES = {
    "A": {"mean_late": -3.0, "std": 2.5, "dispute_rate": 0.02, "weight": 0.62},
    "B": {"mean_late":  6.0, "std": 5.0, "dispute_rate": 0.08, "weight": 0.22},
    "C": {"mean_late": 12.0, "std": 6.0, "dispute_rate": 0.15, "weight": 0.11},
    "D": {"mean_late": 18.0, "std": 8.0, "dispute_rate": 0.25, "weight": 0.05},
}

PRODUCT_CATEGORIES = ["Ham Kumas", "Iplik", "Boya/Kimyasal", "Aksesuar", "Hazir Giyim"]
PAYMENT_TERMS      = [15, 30, 45, 60]

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "digital_twin.db"

fake = Faker("tr_TR")

# Yardimcilar
def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def sample_days_late(rng: np.random.Generator, profile: dict, disputed: bool) -> int:
    """
    Vade asim gunu hesaplar (negatif = erken / zamaninda odeme).
    IBM Finance Factoring dagilimina kalibre: buyuk cogunluk negatif veya sifir,
    kotu segmentlerde sag-kuyruk uzun.
    """
    late = rng.normal(profile["mean_late"], profile["std"])
    if disputed:
        late += rng.normal(7.0, 3.0)    # itiraz gecikmeyi artirir
    return int(round(late))

# Tablo ureticileri
def gen_customers(rng: np.random.Generator) -> list[dict]:
    """40 musteri; segment dagilimi IBM istatistiklerine gore (A:62, B:22, C:11, D:5%)."""
    segments = list(SEGMENT_PROFILES.keys())
    weights  = np.array([SEGMENT_PROFILES[s]["weight"] for s in segments])
    weights  = weights / weights.sum()
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        seg  = rng.choice(segments, p=weights)
        prof = SEGMENT_PROFILES[seg]
        score = float(np.clip(1.0 - (prof["mean_late"] + 5) / 25.0, 0.05, 0.99))
        rows.append({
            "customer_id":               i,
            "name":                      fake.unique.company(),
            "segment":                   seg,
            "city":                      fake.city(),
            "payment_reliability_score": round(score, 3),
            "default_payment_terms":     int(rng.choice(PAYMENT_TERMS)),
        })
    return rows


def gen_suppliers(rng: np.random.Generator) -> list[dict]:
    """15 tedarikci; her birinin esneklik paylari ve teslim sureleri farkli."""
    rows = []
    for i in range(1, N_SUPPLIERS + 1):
        rows.append({
            "supplier_id":      i,
            "name":             fake.unique.company(),
            "category":         rng.choice(PRODUCT_CATEGORIES),
            "lead_time_days":   int(rng.integers(5, 30)),
            "reliability":      round(float(rng.uniform(0.6, 0.98)), 3),
            "flexibility_days": int(rng.integers(3, 15)),
        })
    return rows


def gen_products(rng: np.random.Generator, suppliers: list[dict]) -> list[dict]:
    """60 urun; her biri bir tedarikciye bagli (FK)."""
    supplier_ids = [s["supplier_id"] for s in suppliers]
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        cat       = rng.choice(PRODUCT_CATEGORIES)
        unit_cost = round(float(rng.uniform(20, 400)), 2)
        rows.append({
            "product_id":    i,
            "name":          f"{cat} - {fake.unique.bothify('??-####').upper()}",
            "category":      cat,
            "supplier_id":   int(rng.choice(supplier_ids)),
            "unit_cost":     unit_cost,
            "sale_price":    round(unit_cost * float(rng.uniform(1.25, 1.9)), 2),
            "reorder_point": int(rng.integers(40, 200)),
            "init_stock":    int(rng.integers(150, 600)),
        })
    return rows


def gen_sales_invoices(
    rng: np.random.Generator,
    customers: list[dict],
) -> tuple[list[dict], dict]:
    """
    Satis faturalari.
    - Gunluk Poisson dagiliminda fatura sayisi (hafta ici daha yogun).
    - Gecikme segmente gore; lognormal fatura tutarlari.
    - ABC Tekstil demo faturasi (180K, disputedli, vadesi crunch haftasinda) enjekte edilir.
    """
    rows   = []
    inv_id = 1

    # Segment D musterisi = ABC Tekstil rolu
    abc = next((c for c in customers if c["segment"] == "D"), customers[0])

    for d in daterange(START_DATE, END_DATE):
        n_invoices = rng.poisson(10.0 if d.weekday() < 5 else 2.5)
        for _ in range(n_invoices):
            cust     = customers[int(rng.integers(0, len(customers)))]
            prof     = SEGMENT_PROFILES[cust["segment"]]
            terms    = cust["default_payment_terms"]
            amount   = round(float(rng.lognormal(mean=8.95, sigma=0.5)), 2)
            disputed = bool(rng.random() < prof["dispute_rate"])
            due      = d + timedelta(days=terms)
            days_late = sample_days_late(rng, prof, disputed)
            settled  = due + timedelta(days=days_late)
            status   = "settled" if settled <= END_DATE else "open"

            rows.append({
                "invoice_id":   inv_id,
                "customer_id":  cust["customer_id"],
                "invoice_date": d.isoformat(),
                "due_date":     due.isoformat(),
                "amount":       amount,
                "disputed":     int(disputed),
                "settled_date": settled.isoformat() if status == "settled" else None,
                "days_late":    days_late if status == "settled" else None,
                "status":       status,
            })
            inv_id += 1

    #ABC Tekstil'in demo darbogazi faturasi 
    rows.append({
        "invoice_id":   inv_id,
        "customer_id":  abc["customer_id"],
        "invoice_date": ABC_INVOICE_DATE.isoformat(),
        "due_date":     ABC_DUE_DATE.isoformat(),
        "amount":       ABC_INVOICE_AMOUNT,
        "disputed":     1,
        "settled_date": None,
        "days_late":    None,
        "status":       "open",
    })

    return rows, abc


def gen_purchase_invoices(
    rng: np.random.Generator,
    suppliers: list[dict],
    products: list[dict],
) -> list[dict]:
    """
    Alis faturalari (~%35 gunluk alim olasiligi).
    Crunch haftasinda buyuk tedarikci odemesi dinamik olarak inject_crunch_payment()
    ile eklenir; bu fonksiyon sadece rutin alimlari uretir.
    """
    prod_by_supplier: dict[int, list[dict]] = {}
    for p in products:
        prod_by_supplier.setdefault(p["supplier_id"], []).append(p)

    rows = []
    pid  = 1
    for d in daterange(START_DATE, END_DATE):
        if rng.random() < 0.35:
            sup   = suppliers[int(rng.integers(0, len(suppliers)))]
            prods = prod_by_supplier.get(sup["supplier_id"], products)
            prod  = prods[int(rng.integers(0, len(prods)))]
            qty   = int(rng.integers(50, 400))
            amount = round(qty * prod["unit_cost"], 2)
            terms  = int(rng.choice([15, 30, 45]))
            due    = d + timedelta(days=terms)
            settled = due if due <= END_DATE else None
            rows.append({
                "pinvoice_id":  pid,
                "supplier_id":  sup["supplier_id"],
                "product_id":   prod["product_id"],
                "invoice_date": d.isoformat(),
                "due_date":     due.isoformat(),
                "quantity":     qty,
                "amount":       amount,
                "settled_date": settled.isoformat() if settled else None,
                "status":       "settled" if settled else "open",
            })
            pid += 1
    return rows


def inject_crunch_payment(
    purchases: list[dict],
    suppliers: list[dict],
    products: list[dict],
    crunch_amount: float,
) -> None:
    """
    Iki-gecisli hesapla belirlenen crunch_amount'u crunch haftasinin ilk is gunune
    vadeli buyuk bir tedarikci odemesi olarak ekler.
    Bu enjeksiyon kasa dibinin TARGET_CRUNCH_TROUGH'a (varsayilan -60K TL) ulasmasi icin
    analitik olarak boyutlandirilmistir.
    """
    big_due = CRUNCH_WEEK_START + timedelta(days=1)
    sup  = suppliers[0]
    prod = next(
        (p for p in products if p["supplier_id"] == sup["supplier_id"]),
        products[0],
    )
    qty = max(1, int(round(crunch_amount / max(prod["unit_cost"], 1.0))))
    purchases.append({
        "pinvoice_id":  max(p["pinvoice_id"] for p in purchases) + 1,
        "supplier_id":  sup["supplier_id"],
        "product_id":   prod["product_id"],
        "invoice_date": (big_due - timedelta(days=30)).isoformat(),
        "due_date":     big_due.isoformat(),
        "quantity":     qty,
        "amount":       round(crunch_amount, 2),
        "settled_date": big_due.isoformat(),
        "status":       "settled",
    })


def gen_inventory_log(
    rng: np.random.Generator,
    products: list[dict],
) -> tuple[list[dict], dict]:
    """
    Gunluk stok hareketi.
    Kritik urun (products[0]) crunch haftasinda kontrollu sekilde 0'a iner:
    - ramp_start (~30 gun oncesi) itibariyle stok 250'ye cekilir
    - Yeniden siparis durdurulur
    - Crunch haftasinda stok = 0  -> tukenmis
    Diger urunlerde normal demand + reorder lojigi.
    """
    rows       = []
    critical   = products[0]
    crit_id    = critical["product_id"]
    ramp_start = CRUNCH_WEEK_END - timedelta(days=30)
    stock      = {p["product_id"]: p["init_stock"] for p in products}

    for d in daterange(START_DATE, END_DATE):
        for p in products:
            pid     = p["product_id"]
            opening = stock[pid]

            if pid == crit_id and d >= ramp_start:
                if d == ramp_start:
                    opening = 250
                base_demand = max(1, int(rng.normal(10, 2)))
                units_sold  = min(opening, base_demand)
                received    = 0                 # siparis yok -> tukenir
            else:
                base_demand = max(0, int(rng.normal(8, 4)))
                units_sold  = min(opening, base_demand)
                received    = 0
                if opening - units_sold < p["reorder_point"]:
                    received = int(rng.integers(100, 400))

            closing   = max(0, opening - units_sold + received)
            stock[pid] = closing
            rows.append({
                "date":          d.isoformat(),
                "product_id":    pid,
                "opening_stock": opening,
                "units_sold":    units_sold,
                "units_received": received,
                "closing_stock": closing,
            })

    return rows, critical


def derive_bank_ledger(
    sales: list[dict],
    purchases: list[dict],
    rng: np.random.Generator,
    fixed_cost: float = MONTHLY_FIXED_COST,
    opening: float = OPENING_CASH,
) -> list[dict]:
    """
    bank_ledger = tahsilatlar (giris) + tedarikci odemeleri (cikis) + duzenli giderler.
    Kronolojik siraya gore running-balance hesaplanir.
    Not: bank_ledger BAGIMSIZ degerler iceremez; her satir bir fatura veya
         sabit gider kaleminden gelir (iliskisel tutarlilik).
    """
    events: list[tuple[date, str, float, str, str]] = []

    # Satis tahsilatlari
    for inv in sales:
        if inv["status"] == "settled" and inv["settled_date"]:
            events.append((
                date.fromisoformat(inv["settled_date"]),
                "in", inv["amount"],
                "tahsilat",
                f"musteri_{inv['customer_id']}",
            ))

    # Tedarikci odemeleri
    for p in purchases:
        if p["status"] == "settled" and p["settled_date"]:
            events.append((
                date.fromisoformat(p["settled_date"]),
                "out", p["amount"],
                "tedarikci_odeme",
                f"tedarikci_{p['supplier_id']}",
            ))

    # Duzenli giderler: her ayin 5'i kira+maas, 20'si operasyonel
    for d in daterange(START_DATE, END_DATE):
        if d.day == 5:
            events.append((d, "out", fixed_cost, "kira_maas", "sabit_gider"))
        if d.day == 20:
            events.append((d, "out",
                           round(float(rng.uniform(8_000, 18_000)), 2),
                           "operasyonel", "muhtelif"))

    events.sort(key=lambda e: e[0])

    rows    = []
    balance = opening
    for txn_id, (d, typ, amount, cat, cp) in enumerate(events, start=1):
        balance += amount if typ == "in" else -amount
        rows.append({
            "txn_id":       txn_id,
            "date":         d.isoformat(),
            "type":         typ,
            "amount":       amount,
            "category":     cat,
            "counterparty": cp,
            "balance_after": round(balance, 2),
        })
    return rows


def gen_macro_daily(rng: np.random.Generator) -> list[dict]:
    """
    Gercekci makro seri placeholder'i.
    - USD/TRY: yukselen trend (random walk) + 15 Temmuz 2024 kur soku (Script A'dan)
    - EUR/TRY: USD'ye sabit spread ile
    - TUFE_yoy: gercekci Turk enflasyon raylari (2024 bandi: ~65-68%)
    - policy_rate: Temmuz 2024 oncesi 50, sonrasi 45 (TCMB pivot)

    NOT: Bu tablo fetch_evds.py calistirildiktan sonra GERCEK EVDS verisiyle
    overwrite edilir. Buradaki degerler simülasyon ve crunch testi icindir.
    """
    rows = []
    usd  = 31.0
    eur  = 34.0
    tufe = 64.0

    for d in daterange(START_DATE, END_DATE):
        # Normal gunluk hareketlilik
        usd  *= (1 + rng.normal(0.0011, 0.004))
        eur  *= (1 + rng.normal(0.0010, 0.004))
        tufe += rng.normal(-0.02, 0.15)

        # Kur soku: 15-22 Temmuz 2024 (Script A'nin ozgunu simülasyon testi icin)
        if SHOCK_START <= d <= SHOCK_END:
            usd += SHOCK_DAILY
            eur += SHOCK_DAILY * 1.08

        rows.append({
            "date":        d.isoformat(),
            "usdtry":      round(usd, 4),
            "eurtry":      round(eur, 4),
            "tufe_yoy":    round(float(np.clip(tufe, 35, 75)), 2),
            "policy_rate": 50.0 if d < date(2024, 7, 1) else 45.0,
        })

    return rows

# SQLite sema ve yazimi
SCHEMA = """
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS sales_invoices;
DROP TABLE IF EXISTS purchase_invoices;
DROP TABLE IF EXISTS inventory_log;
DROP TABLE IF EXISTS bank_ledger;
DROP TABLE IF EXISTS macro_daily;

CREATE TABLE customers (
    customer_id               INTEGER PRIMARY KEY,
    name                      TEXT    NOT NULL,
    segment                   TEXT    NOT NULL CHECK(segment IN ('A','B','C','D')),
    city                      TEXT,
    payment_reliability_score REAL,
    default_payment_terms     INTEGER
);

CREATE TABLE suppliers (
    supplier_id     INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    category        TEXT,
    lead_time_days  INTEGER,
    reliability     REAL,
    flexibility_days INTEGER
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    category      TEXT,
    supplier_id   INTEGER REFERENCES suppliers(supplier_id),
    unit_cost     REAL,
    sale_price    REAL,
    reorder_point INTEGER,
    init_stock    INTEGER
);

CREATE TABLE sales_invoices (
    invoice_id   INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(customer_id),
    invoice_date TEXT    NOT NULL,
    due_date     TEXT    NOT NULL,
    amount       REAL    NOT NULL,
    disputed     INTEGER DEFAULT 0,
    settled_date TEXT,
    days_late    INTEGER,
    status       TEXT    CHECK(status IN ('settled','open'))
);

CREATE TABLE purchase_invoices (
    pinvoice_id  INTEGER PRIMARY KEY,
    supplier_id  INTEGER REFERENCES suppliers(supplier_id),
    product_id   INTEGER REFERENCES products(product_id),
    invoice_date TEXT    NOT NULL,
    due_date     TEXT    NOT NULL,
    quantity     INTEGER,
    amount       REAL    NOT NULL,
    settled_date TEXT,
    status       TEXT    CHECK(status IN ('settled','open'))
);

CREATE TABLE inventory_log (
    date            TEXT    NOT NULL,
    product_id      INTEGER REFERENCES products(product_id),
    opening_stock   INTEGER,
    units_sold      INTEGER,
    units_received  INTEGER,
    closing_stock   INTEGER,
    PRIMARY KEY (date, product_id)
);

CREATE TABLE bank_ledger (
    txn_id       INTEGER PRIMARY KEY,
    date         TEXT    NOT NULL,
    type         TEXT    CHECK(type IN ('in','out')),
    amount       REAL    NOT NULL,
    category     TEXT,
    counterparty TEXT,
    balance_after REAL
);

CREATE TABLE macro_daily (
    date        TEXT PRIMARY KEY,
    usdtry      REAL,
    eurtry      REAL,
    tufe_yoy    REAL,
    policy_rate REAL
);
"""


def write_db(tables: dict[str, list[dict]]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    for name, rows in tables.items():
        if not rows:
            continue
        cols         = list(rows[0].keys())
        placeholders = ",".join("?" * len(cols))
        con.executemany(
            f"INSERT INTO {name} ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in rows],
        )
    con.commit()
    con.close()


# Kalibrasyon metrikleri (IBM Finance Factoring kontrolu)
def print_calibration(sales: list[dict]) -> None:
    """
    Uretilen gecikme dagilimini ekrana yazar.
    Hedef (IBM Finance Factoring):
      - ort days_late ~ 3-4 gun
      - medyan ~ 0
      - %gec (days_late > 0) ~ %36
      - p90 ~ 13 gun
    """
    settled = [s for s in sales if s["status"] == "settled" and s["days_late"] is not None]
    if not settled:
        return
    dl = [s["days_late"] for s in settled]
    dl_arr = np.array(dl)
    print("\n  --- Kalibrasyon Metrikleri (IBM Finance Factoring karsilastirmasi) ---")
    print(f"  Ortalama days_late  : {dl_arr.mean():.2f}g  (hedef: ~3.4g)")
    print(f"  Medyan  days_late   : {float(np.median(dl_arr)):.1f}g   (hedef: ~0)")
    print(f"  Gec oran (>0)       : {(dl_arr > 0).mean() * 100:.1f}%  (hedef: ~%36)")
    print(f"  p90 days_late       : {float(np.percentile(dl_arr, 90)):.1f}g   (hedef: ~13g)")
    print(f"  Disputed fatura     : {sum(s['disputed'] for s in settled):,}")


# 13 Dogrulama Kontrolu (verify_digital_twin.py'a tasindi!; burasi ozet check)
def quick_verify(tables: dict) -> int:
    """
    Kritik tutarlilik kontrolleri. Gecen sayi dondurur.
    Tam 13-check icin: python src/data_generator/verify_digital_twin.py
    """
    checks = []

    # 1. Tum tablolar dolu mu?
    for t in ["customers","suppliers","products","sales_invoices",
              "purchase_invoices","inventory_log","bank_ledger","macro_daily"]:
        ok = len(tables.get(t, [])) > 0
        checks.append(("Tablo dolu: " + t, ok))

    # 2. FK tutarliligi: tum sales_invoices.customer_id customers'da mi?
    cust_ids = {c["customer_id"] for c in tables["customers"]}
    fk_ok    = all(s["customer_id"] in cust_ids for s in tables["sales_invoices"])
    checks.append(("FK: sales_invoices -> customers", fk_ok))

    # 3. ABC demo faturasi mevcut mu?
    abc_ok = any(
        s["amount"] == ABC_INVOICE_AMOUNT and s["status"] == "open"
        for s in tables["sales_invoices"]
    )
    checks.append(("Demo: ABC 180K acik fatura", abc_ok))

    # 4. Crunch haftasinda bank_ledger negatife dusiyor mu?
    crunch_min = min(
        (r["balance_after"] for r in tables["bank_ledger"]
         if CRUNCH_WEEK_START.isoformat() <= r["date"] <= CRUNCH_WEEK_END.isoformat()),
        default=0.0,
    )
    checks.append(("Crunch kasasi negatife duser", crunch_min < 0))

    # 5. Makro kur soku macro_daily'de var mi?
    shock_vals = [
        r["usdtry"] for r in tables["macro_daily"]
        if SHOCK_START.isoformat() <= r["date"] <= SHOCK_END.isoformat()
    ]
    pre_shock  = next(
        (r["usdtry"] for r in tables["macro_daily"]
         if r["date"] == (SHOCK_START - timedelta(days=1)).isoformat()),
        None,
    )
    shock_ok = bool(shock_vals and pre_shock and max(shock_vals) > pre_shock + 0.5)
    checks.append(("Makro kur soku tespiti", shock_ok))

    # Ozet
    print("\n  --- Hizli Dogrulama (5 kontrol) ---")
    passed = 0
    for name, ok in checks:
        status = "[OK] " if ok else "[XX]"
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    return passed

# Ana akis
def main() -> None:
    print("=" * 65)
    print("  ResilienceOS Dijital Ikiz Veri Ureticisi v3")
    print(f"  Simülasyon: {START_DATE} .. {END_DATE}  (18 ay)")
    print(f"  Crunch haftasi: {CRUNCH_WEEK_START} .. {CRUNCH_WEEK_END}")
    print(f"  Kur soku: {SHOCK_START} .. {SHOCK_END}  (+{SHOCK_DAILY} TL/gun)")
    print("=" * 65)

    Faker.seed(SEED)
    fake.unique.clear()
    rng = np.random.default_rng(SEED)

    print("\n[1/8] Musteriler uretiliyor...")
    customers = gen_customers(rng)

    print("[2/8] Tedarikciler uretiliyor...")
    suppliers = gen_suppliers(rng)

    print("[3/8] Urunler uretiliyor...")
    products = gen_products(rng, suppliers)

    print("[4/8] Satis faturalari uretiliyor (18 ay, ~32K kayit bekleniyor)...")
    sales, abc = gen_sales_invoices(rng, customers)

    print("[5/8] Alis faturalari uretiliyor...")
    purchases = gen_purchase_invoices(rng, suppliers, products)

    print("[6/8] Stok hareketi uretiliyor (60 urun x 18 ay)...")
    inventory, critical = gen_inventory_log(rng, products)

    print("[7/8] Banka defteri turetiliyor + crunch boyutlandirmasi...")
    # Iki-gecisli crunch enjeksiyonu:
    # 1. gecis: crunch buyuk odemesi OLMADAN ledger'i hesapla
    prelim       = derive_bank_ledger(sales, purchases, np.random.default_rng(SEED + 1))
    big_due_iso  = (CRUNCH_WEEK_START + timedelta(days=1)).isoformat()
    crunch_end_iso = CRUNCH_WEEK_END.isoformat()

    # Crunch haftasindaki minimum on-bakiye (odeme olmadan)
    crunch_week_balances = [r["balance_after"] for r in prelim
                            if big_due_iso <= r["date"] <= crunch_end_iso]
    M = min(crunch_week_balances) if crunch_week_balances else OPENING_CASH

    # P = M - hedef_dip  => crunch odemesi bu kadar olmali
    # %10 buffer: simülasyon her seferinde biraz farkli cikabilir
    crunch_amount = round((M - TARGET_CRUNCH_TROUGH) * 1.10 / 1_000) * 1_000
    crunch_amount = max(50_000.0, crunch_amount)   # minimum 50K TL
    inject_crunch_payment(purchases, suppliers, products, crunch_amount)

    # 2. gecis: final ledger (operasyonel giderler ayni rng ile tutarli)
    bank  = derive_bank_ledger(sales, purchases, np.random.default_rng(SEED + 1))

    print("[8/8] Makro seri uretiliyor (kur soku dahil)...")
    macro = gen_macro_daily(rng)

    # --- Veritabani yazimi ---
    tables = {
        "customers":        customers,
        "suppliers":        suppliers,
        "products":         products,
        "sales_invoices":   sales,
        "purchase_invoices": purchases,
        "inventory_log":    inventory,
        "bank_ledger":      bank,
        "macro_daily":      macro,
    }
    write_db(tables)

    # --- Ozet ---
    print("\n" + "=" * 65)
    print(f"  Dijital Ikiz uretildi -> {DB_PATH}")
    print("=" * 65)
    print(f"  customers          : {len(customers):>8,}")
    print(f"  suppliers          : {len(suppliers):>8,}")
    print(f"  products           : {len(products):>8,}")
    print(f"  sales_invoices     : {len(sales):>8,}")
    print(f"  purchase_invoices  : {len(purchases):>8,}")
    print(f"  inventory_log      : {len(inventory):>8,}")
    print(f"  bank_ledger        : {len(bank):>8,}")
    print(f"  macro_daily        : {len(macro):>8,}")
    print("-" * 65)
    print(f"  Crunch odemesi boyutu : {crunch_amount:,.0f} TL")
    crunch_min = min(
        (r["balance_after"] for r in bank
         if big_due_iso <= r["date"] <= crunch_end_iso),
        default=0.0,
    )
    print(f"  Crunch haftasi kasa dibi: {crunch_min:,.0f} TL")
    print(f"  ABC Tekstil (D seg)    : #{abc['customer_id']} {abc['name']}")
    print(f"  Kritik tukenen urun    : #{critical['product_id']} {critical['name']}")

    # Kalibrasyon
    print_calibration(sales)

    # Hizli dogrulama
    passed = quick_verify(tables)
    print(f"\n  Toplam: {passed}/5 hizli kontrol gecti.")
    print("  Tam dogrulama: python src/data_generator/verify_digital_twin.py")
    print("=" * 65)
    print("\n  NOT: macro_daily su an placeholder.")
    print("  Gercek EVDS verisi icin:")
    print("    python src/data_ingest/fetch_evds.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
