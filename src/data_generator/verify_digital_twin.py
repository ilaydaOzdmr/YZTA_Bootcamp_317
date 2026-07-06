"""
ResilienceOS - Dijital Ikiz Dogrulayici 
generate_digital_twin.py calistirildiktan sonra bu scripti calistir.
Tum 13 kontrolun GECTIGI gorulmeli.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "digital_twin.db"

CHECKS_PASSED = 0
CHECKS_TOTAL  = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    icon = "[OK] " if condition else "[XX]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  [{CHECKS_TOTAL:02d}] {name}{suffix}")
    if condition:
        CHECKS_PASSED += 1


def main() -> None:
    print("=" * 65)
    print("  ResilienceOS - 13 Dogrulama Kontrolu")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"\n  HATA: {DB_PATH} bulunamadi.")
        print("  Once calistir: python src/data_generator/generate_digital_twin.py")
        return

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    def q(sql: str, *args):
        return con.execute(sql, args).fetchall()

    def qone(sql: str, *args):
        row = con.execute(sql, args).fetchone()
        return row[0] if row else None

    print("\n  [BOLUM A] Tablo Varligi ve Satir Sayilari")
    # 1
    counts = {t: qone(f"SELECT COUNT(*) FROM {t}")
              for t in ["customers","suppliers","products","sales_invoices",
                        "purchase_invoices","inventory_log","bank_ledger","macro_daily"]}
    all_nonempty = all(v > 0 for v in counts.values())
    check("Tum 8 tablo dolu", all_nonempty,
          " | ".join(f"{t}:{v}" for t, v in counts.items()))

    # 2
    si_count = counts["sales_invoices"]
    check("sales_invoices >= 3000 kayit (KOBI gercekci hacim)", si_count >= 3_000, f"{si_count:,} kayit")

    # 3
    inv_count = counts["inventory_log"]
    expected  = 60 * 546   # 60 urun x ~546 gun (18 ay)
    check("inventory_log satir sayisi mantikli (>20000)", inv_count > 20_000,
          f"{inv_count:,} kayit")

    print("\n  [BOLUM B] FK ve Iliskisel Tutarlilik")
    # 4
    orphan_sales = qone("""
        SELECT COUNT(*) FROM sales_invoices s
        WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = s.customer_id)
    """)
    check("FK: sales_invoices.customer_id -> customers", orphan_sales == 0,
          f"{orphan_sales} orphan")

    # 5
    orphan_pinv = qone("""
        SELECT COUNT(*) FROM purchase_invoices p
        WHERE NOT EXISTS (SELECT 1 FROM suppliers s WHERE s.supplier_id = p.supplier_id)
    """)
    check("FK: purchase_invoices.supplier_id -> suppliers", orphan_pinv == 0,
          f"{orphan_pinv} orphan")

    # 6
    orphan_inv_log = qone("""
        SELECT COUNT(*) FROM inventory_log il
        WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = il.product_id)
    """)
    check("FK: inventory_log.product_id -> products", orphan_inv_log == 0,
          f"{orphan_inv_log} orphan")

    print("\n  [BOLUM C] Gecikme Dagilimi Kalibrasyonu (IBM Finance Factoring)")
    settled = q("""
        SELECT days_late FROM sales_invoices
        WHERE status = 'settled' AND days_late IS NOT NULL
    """)
    dl = np.array([r[0] for r in settled], dtype=float)

    # 7 - Ortalama 0-8 gun araliginda olmali
    mean_dl = dl.mean() if len(dl) > 0 else 999
    check("Ort days_late 0-8 gun araliginda", 0 <= mean_dl <= 8, f"{mean_dl:.2f}g")

    # 8 - Medyan 0 veya -1 olmali (cogunluk zamaninda/erken)
    med_dl = float(np.median(dl)) if len(dl) > 0 else 999
    check("Medyan days_late <= 2", med_dl <= 2, f"{med_dl:.1f}g")

    # 9 - Gec oran %20-%55 araliginda
    late_rate = (dl > 0).mean() * 100 if len(dl) > 0 else 0
    check("Gec oran (>0) %20-%55", 20 <= late_rate <= 55, f"{late_rate:.1f}%")

    # 10 - p90 5-20 gun araliginda
    p90 = float(np.percentile(dl, 90)) if len(dl) > 0 else 999
    check("p90 days_late 5-20 gun araliginda", 5 <= p90 <= 20, f"{p90:.1f}g")

    print("\n  [BOLUM D] Demo Darbogazi Senaryosu")
    # 11 - ABC Tekstil 180K acik fatura
    abc_ok = qone("""
        SELECT COUNT(*) FROM sales_invoices
        WHERE amount = 180000.0 AND status = 'open'
    """)
    check("ABC Tekstil 180K TL acik fatura mevcut", abc_ok > 0, f"{abc_ok} fatura")

    # 12 - Crunch haftasinda kasa negatif
    crunch_min = qone("""
        SELECT MIN(balance_after) FROM bank_ledger
        WHERE date BETWEEN '2025-06-16' AND '2025-06-22'
    """)
    check("Crunch haftasinda kasa negatife dusuyor",
          crunch_min is not None and crunch_min < 0,
          f"min: {crunch_min:,.0f} TL" if crunch_min else "veri yok")

    # 13 - Kritik stok tukenmesi (product_id=1, crunch haftasinda closing=0)
    stock_zero = qone("""
        SELECT MIN(closing_stock) FROM inventory_log
        WHERE product_id = 1 AND date BETWEEN '2025-06-16' AND '2025-06-22'
    """)
    check("Kritik urun crunch haftasinda tukeniyor (closing=0)",
          stock_zero is not None and stock_zero == 0,
          f"min stok: {stock_zero}")

    con.close()

    print("\n" + "=" * 65)
    print(f"  Sonuc: {CHECKS_PASSED}/{CHECKS_TOTAL} kontrol gecti.")
    if CHECKS_PASSED == CHECKS_TOTAL:
        print("  Dijital Ikiz dogrulama BASARILI. Sprint 2'ye gecebilirsiniz.")
    else:
        print("  Bazi kontroller basarisiz. generate_digital_twin.py'i gozden gecirin.")
    print("=" * 65)


if __name__ == "__main__":
    main()
