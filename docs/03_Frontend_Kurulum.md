# Önyüz Kurulumu — Nakit Akışı Dashboard'u

## Mimari

```
src/api/server.py   FastAPI backend. twin_api.py'yi HTTP'ye açar.
frontend/            Statik HTML/CSS/JS dashboard (Node.js gerekmez).
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

## Şu an kapsam

Sprint 3'ün ilk adımı olarak sadece **nakit akışı + kriz uyarısı** ekranı
hazır:

- Kriz olasılığı göstergesi (yarım daire gauge)
- 30 günlük kasa projeksiyonu grafiği (iyimser/normal/kötümser + gerçekleşen)
- Özet kartlar: başlangıç bakiye, en düşük nokta, tampon ihlali olasılığı,
  en olası kriz tarihi, açık alacak/borç
- Stok→nakit köprüsü uyarısı (varsa)

## Sıradaki genişletmeler

`src/core/twin_api.py`'de zaten hazır, backend'e eklenmeyi bekleyen
fonksiyonlar:

- `invoice_risk(top_n)` → riskli faturalar tablosu
- `stock_risk(top_n)` → stok tükenme tablosu
- `simulate(scenario)` / `recommend()` → "Aksiyonları Uygula" şok simülatörü

Her biri için `src/api/server.py`'ye yeni bir `@app.get(...)` rotası ve
`frontend/`'e karşılık gelen bir bölüm eklemek yeterli — mevcut desen
(canlı/offline fallback) aynı şekilde tekrar kullanılabilir.
