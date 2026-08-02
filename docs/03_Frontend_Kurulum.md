# Önyüz Kurulumu — Nakit Akışı Dashboard'u

## Mimari

```
src/api/server.py   FastAPI backend. twin_api.py'yi 8 endpoint ile HTTP'ye açar.
frontend/            Statik dashboard (index.html açılış + dashboard.html), Chart.js — Node.js gerekmez.
```

Backend iki modda çalışır:

- **Canlı mod**: `data/digital_twin.db` ve eğitilmiş modeller mevcutsa (yani
  README'deki "Hızlı Başlangıç" pipeline'ı çalıştırılmışsa), `twin_api.py`
  üzerinden gerçek zamanlı sonuç döner.
- **Offline mod**: DB/model yoksa, `reports/cashflow_metrics.json` ve
  `reports/cashflow_forecast.csv`'deki önceden hesaplanmış çıktılara düşer.
  Böylece pipeline'ı çalıştırmamış biri de dashboard'u görebilir (demo/review
  için pratik).

Dashboard'un sağ üstünde hangi modda olduğunuzu gösteren bir rozet var
("CANLI VERİ" / "OFFLINE / DEMO VERİSİ").

## Çalıştırma

```bash
pip install -r requirements.txt
uvicorn src.api.server:app --reload --port 8000
```

Tarayıcıda aç: **http://localhost:8000**

## Kapsam (güncel)

Dashboard, `twin_api`'nin tüm çıktılarını gösterir — 8 endpoint canlı:

**Ana ekran (nakit + kriz):**
- Kriz olasılığı göstergesi (yarım daire gauge)
- 30 günlük kasa projeksiyonu grafiği (iyimser/normal/kötümser + gerçekleşen, Chart.js)
- Özet kartlar: başlangıç bakiye, en düşük nokta, tampon ihlali olasılığı,
  en olası kriz tarihi, açık alacak/borç
- Stok→nakit köprüsü uyarısı

**Risk tabloları:**
- Riskli faturalar tablosu (`/api/invoice-risk`)
- Stok tükenme tablosu (`/api/stock-risk`)

**Şok simülatörü ("Aksiyonları Uygula"):**
- Senaryo listesi (`/api/scenarios`), tek senaryo (`/api/simulate/{key}`),
  en iyi aksiyon önerisi (`/api/recommend`)

**Endpoint özeti:** `/api/overview`, `/api/cash-forecast`, `/api/invoice-risk`,
`/api/stock-risk`, `/api/scenarios`, `/api/simulate/{key}`, `/api/recommend`, `/api/health`

Her endpoint canlı/offline iki modludur; DB/model yoksa `reports/` çıktılarına düşer.
